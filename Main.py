import sys
import os
import glob
import threading
import time
from typing import Tuple
from math import sqrt
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fix for Windows terminal encoding issues
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

app = Flask(__name__)

# Base directory and dataset path
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Dataset path (absolute preferred)
DATA_PATH = r"e:\Price Predictor\master.csv"

# Exact feature set required by the frontend and models
FEATURES = [
    'temp_mean', 'rainfall_mm', 'humidity', 'wind_speed',
    'price_lag_7d', 'price_lag_30d', 'rainfall_mm_7d_sum',
    'month', 'day_of_year', 'drought_indicator', 'flood_indicator'
]

# In-memory model store and training flags
models = {}
metrics = {}
training_lock = threading.Lock()
is_training = False
class DataHandler:
    @staticmethod
    def load(path: str) -> pd.DataFrame:
        # Prefer provided path; fall back to master.csv in project root
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            fallback = os.path.join(BASE_DIR, 'master.csv')
            if os.path.exists(fallback):
                print(f"Dataset not found at {path}; falling back to {fallback}")
                df = pd.read_csv(fallback)
            else:
                raise FileNotFoundError(f"Dataset not found at {path} and fallback {fallback} missing")

        # Fill forward/backward to handle missing values as requested
        df = df.ffill().bfill()
        return df


class ModelTrainer:
    def __init__(self, features):
        self.features = features

    def train_for(self, crop: str, df: pd.DataFrame):
        crop_df = df[df.get('Commodity', '').astype(str).str.strip().str.lower() == crop.lower()].copy()
        if crop_df.empty or 'Modal_Price' not in crop_df.columns:
            raise ValueError(f"No data for crop {crop} or 'Modal_Price' missing")

        # Ensure all feature columns exist
        for f in self.features:
            if f not in crop_df.columns:
                crop_df[f] = 0

        X = crop_df[self.features].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = crop_df['Modal_Price'].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(sqrt(mean_squared_error(y_test, preds)))

        return model, mae, rmse


class Predictor:
    def __init__(self, models_map, metrics_map, features):
        self.models = models_map
        self.metrics = metrics_map
        self.features = features

    def validate_input(self, payload: dict) -> Tuple[str, np.ndarray]:
        commodity = payload.get('commodity')
        if not commodity or not isinstance(commodity, str):
            raise ValueError('commodity field is required and must be a string')

        # Accept either 'values' array or named fields
        values = payload.get('values')
        if values is not None:
            if not isinstance(values, (list, tuple)) or len(values) != len(self.features):
                raise ValueError(f"'values' must be an array of length {len(self.features)}")
            arr = np.array([float(v) for v in values], dtype=float).reshape(1, -1)
        else:
            arr = []
            for f in self.features:
                v = payload.get(f, 0)
                arr.append(float(v))
            arr = np.array([arr], dtype=float)

        return commodity, arr

    def predict(self, commodity: str, X: np.ndarray):
        model = self.models.get(commodity)
        if model is None:
            raise ValueError(f"Model for '{commodity}' not trained yet")
        pred = float(model.predict(X)[0])
        m = self.metrics.get(commodity, {})
        return {'price': round(pred, 2), 'mae': m.get('mae'), 'rmse': m.get('rmse')}


def background_train():
    global is_training
    with training_lock:
        if is_training:
            return
        is_training = True
    try:
        print('Background training started')
        try:
            df = DataHandler.load(DATA_PATH)
        except FileNotFoundError as fe:
            print(f"Background training aborted: {fe}")
            return

        trainer = ModelTrainer(FEATURES)
        for crop in ['Onion', 'Tomato']:
            try:
                print(f"Training model for {crop}...")
                model, mae, rmse = trainer.train_for(crop, df)
                models[crop] = model
                metrics[crop] = {'mae': round(mae, 3), 'rmse': round(rmse, 3)}
                print(f"Trained {crop}: MAE={mae:.3f}, RMSE={rmse:.3f}")
            except Exception as e:
                print(f"Failed training for {crop}: {e}")
        print('Background training finished')
    finally:
        with training_lock:
            is_training = False


def load_persisted_models():
    """Load any joblib models from the models directory into memory and compute quick metrics if dataset available."""
    global models, metrics
    os.makedirs(MODEL_DIR, exist_ok=True)
    files = glob.glob(os.path.join(MODEL_DIR, '*.joblib')) + glob.glob(os.path.join(MODEL_DIR, '*.pkl'))
    if not files:
        print('No persisted models found in', MODEL_DIR)
        return
    print('Loading persisted models:', files)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        # try to infer crop name
        crop = None
        for c in ['Onion', 'Tomato']:
            if c.lower() in name.lower():
                crop = c
                break
        try:
            m = joblib.load(f)
            if crop:
                models[crop] = m
                metrics[crop] = {'mae': None, 'rmse': None}
                # try quick evaluation (small sample) to populate metrics
                try:
                    if os.path.exists(DATA_PATH):
                        sdf = pd.read_csv(DATA_PATH, nrows=100000)
                        sdf = sdf.ffill().bfill()
                        cdf = sdf[sdf['Commodity'].astype(str).str.strip().str.lower() == crop.lower()].copy()
                        if not cdf.empty and 'Modal_Price' in cdf.columns:
                            X = cdf[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
                            y = cdf['Modal_Price'].astype(float)
                            if len(X) > 10:
                                from sklearn.model_selection import train_test_split
                                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                                preds = m.predict(X_te)
                                mae = float(mean_absolute_error(y_te, preds))
                                rmse = float(sqrt(mean_squared_error(y_te, preds)))
                                metrics[crop] = {'mae': round(mae,3), 'rmse': round(rmse,3)}
                except Exception as ee:
                    print('Quick eval failed for', crop, ee)
            else:
                print('Loaded model', name, 'but could not infer crop; model available as', name)
        except Exception as e:
            print('Failed to load persisted model', f, e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status', methods=['GET'])
def status():
    return jsonify({'training': is_training, 'models': {k: ('ready' if k in models else 'missing') for k in ['Onion', 'Tomato']}})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json(force=True)
        predictor = Predictor(models, metrics, FEATURES)
        commodity, X = predictor.validate_input(payload)
        result = predictor.predict(commodity, X)
        # Build a 7-day synthetic forecast centered on the predicted value
        # The model and dataset use price per quintal; expose both quintal and kg prices
        base_quintal = result['price']
        price_per_kg = round(base_quintal / 100.0, 2)
        forecast_quintal = []
        forecast_kg = []
        for i in range(7):
            jitter = (np.sin(i / 2.0) * 0.01 + (np.random.randn() * 0.005))
            fq = round(base_quintal * (1 + jitter), 2)
            fk = round(fq / 100.0, 2)
            forecast_quintal.append(fq)
            forecast_kg.append(fk)

        # Attempt to provide feature importance (drivers)
        drivers = {}
        try:
            model = models.get(commodity)
            if hasattr(model, 'feature_importances_'):
                fi = model.feature_importances_
                drivers = {f: float(round(float(w), 4)) for f, w in zip(FEATURES, fi)}
        except Exception:
            drivers = {}

        # Choose asset image based on commodity
        image_map = {'Onion': '/static/images/onion.svg', 'Tomato': '/static/images/tomato.svg'}

        return jsonify({
            'status': 'Success',
            'price_per_quintal': base_quintal,
            'price_per_kg': price_per_kg,
            'mae': result['mae'],
            'rmse': result['rmse'],
            'forecast_quintal': forecast_quintal,
            'forecast_kg': forecast_kg,
            'drivers': drivers,
            'image': image_map.get(commodity, '')
        })
    except ValueError as ve:
        return jsonify({'status': 'Error', 'message': str(ve)}), 400
    except Exception as e:
        return jsonify({'status': 'Error', 'message': str(e)}), 500


@app.route('/analytics', methods=['GET'])
def analytics():
    """Return analytics based on master dataset: recent series and driver correlations."""
    try:
        df = DataHandler.load(DATA_PATH)
        # Ensure Date column exists
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except Exception:
                pass

        days = 14
        end = df['Date'].max() if 'Date' in df.columns else None
        if end is None:
            # fallback: use last N rows
            recent = df.tail(days)
            labels = [f"Row {i+1}" for i in range(len(recent))]
        else:
            start = end - pd.Timedelta(days=days-1)
            window = df[df['Date'] >= start]
            # group by Date and take median Modal_Price per crop
            labels = [(start + pd.Timedelta(days=i)).strftime('%b %d') for i in range(days)]

        series = {'Onion': [], 'Tomato': []}
        for i in range(days):
            day = (pd.to_datetime(end) - pd.Timedelta(days=(days-1-i))) if end is not None else None
            if day is not None:
                day_mask = df['Date'].dt.floor('D') == day.floor('D')
                day_df = df[day_mask]
            else:
                day_df = df.iloc[i:i+1]
            for crop in ['Onion', 'Tomato']:
                vals = day_df[day_df['Commodity'].astype(str).str.strip().str.lower() == crop.lower()]['Modal_Price'].dropna()
                series[crop].append(float(vals.median()) if len(vals) else None)

        # KPIs
        kpis = {}
        for crop in ['Onion', 'Tomato']:
            latest_vals = df[df['Commodity'].astype(str).str.strip().str.lower() == crop.lower()]['Modal_Price'].dropna()
            latest = float(latest_vals.iloc[-1]) if len(latest_vals) else None
            prev = float(latest_vals.iloc[-2]) if len(latest_vals) > 1 else None
            change = round(((latest - prev) / prev * 100), 2) if prev else 0
            acc = metrics.get(crop, {}).get('mae')
            kpis[crop.lower()] = {'latest': latest, 'change_pct': change, 'mae': acc}

        # compute driver correlations for each crop (pearson with features)
        drivers = {}
        for crop in ['Onion', 'Tomato']:
            crop_df = df[df['Commodity'].astype(str).str.strip().str.lower() == crop.lower()].copy()
            weights = {}
            if not crop_df.empty:
                for f in FEATURES:
                    if f in crop_df.columns:
                        try:
                            corr = crop_df['Modal_Price'].astype(float).corr(pd.to_numeric(crop_df[f], errors='coerce'))
                            weights[f] = 0 if pd.isna(corr) else abs(float(corr))
                        except Exception:
                            weights[f] = 0
                    else:
                        weights[f] = 0
                # normalize to 0-1
                vals = np.array(list(weights.values()), dtype=float)
                maxv = vals.max() if len(vals) else 1
                if maxv == 0:
                    norm = {k:0 for k in weights}
                else:
                    norm = {k: float(round(v/maxv, 4)) for k,v in weights.items()}
                drivers[crop] = norm
            else:
                drivers[crop] = {f:0 for f in FEATURES}

        return jsonify({'labels': labels, 'series': series, 'kpis': kpis, 'drivers': drivers, 'generated_at': pd.Timestamp.utcnow().isoformat() + 'Z'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Return a quick health summary for models and analytics readiness."""
    models_ready = all(k in models for k in ['Onion', 'Tomato'])
    analytics_ready = os.path.exists(DATA_PATH) or os.path.exists(os.path.join(BASE_DIR, 'master.csv'))
    return jsonify({'models_ready': models_ready, 'analytics_ready': analytics_ready, 'training': is_training})


@app.route('/dataset_preview', methods=['GET'])
def dataset_preview():
    """Return a small preview (columns + first N rows) without loading entire large CSV."""
    try:
        n = int(request.args.get('nrows', 20))
        # try primary path first
        path = DATA_PATH if os.path.exists(DATA_PATH) else os.path.join(BASE_DIR, 'master.csv')
        if not os.path.exists(path):
            return jsonify({'error': 'dataset not found'}), 404
        # read only N rows to avoid huge memory usage
        df = pd.read_csv(path, nrows=n)
        cols = list(df.columns)
        rows = df.fillna('').head(n).to_dict(orient='records')
        return jsonify({'columns': cols, 'rows': rows})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load any persisted models so /predict can work immediately
    try:
        load_persisted_models()
    except Exception as e:
        print('Loading persisted models failed:', e)
    
    # Gate background training via environment variable (default: disabled on Render free tier; set ENABLE_BACKGROUND_TRAIN=1 for local dev)
    enable_bg_training = os.getenv('ENABLE_BACKGROUND_TRAIN', '0') == '1'
    if enable_bg_training:
        t = threading.Thread(target=background_train, daemon=True)
        t.start()
        print('Background training enabled.')
    else:
        print('Background training disabled (set ENABLE_BACKGROUND_TRAIN=1 to enable).')
    
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_DEBUG', 'False') == 'True'
    
    print(f'Starting web server on http://0.0.0.0:{port}')
    app.run(host='0.0.0.0', port=port, debug=debug_mode)