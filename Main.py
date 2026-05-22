import sys
import os
import glob
from typing import Tuple
from math import sqrt
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import threading
import shutil
from datetime import datetime

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
# Expanded features including engineered ones. When calling /predict, the payload must include
# these fields (or reasonable defaults) so the model receives the same features it was trained on.
FEATURES = [
    # Core weather / market features
    'temp_mean', 'rainfall_mm', 'humidity', 'wind_speed',
    'temp_min', 'temp_max', 'temp_range',
    # Recent price history (keep only the most informative lags)
    'price_lag_1d                   ', 'price_lag_7d',
    # Rolling / momentum stats
    'rolling_mean_7d', 'rolling_std_7d', 'price_momentum_7d',
    # Aggregated rainfall
    'rainfall_7d_avg',
    # Date / cyclic features
    'month', 'day_of_year', 'week_sin', 'week_cos',
    # Domain heuristics
    'festival_indicator', 'harvest_season_indicator',
    # Supply indicator
    'arrival_qty'
]

# In-memory model store and training flags
models = {}
metrics = {}
is_training = False
training_lock = threading.Lock()
class DataHandler:
    @staticmethod
    def ingest_data(path: str, raw_dir: str = 'data/raw', anomalies_dir: str = 'data/anomalies') -> tuple:
        """Create an immutable raw copy of the incoming dataset and run a lightweight anomaly check.

        Returns tuple: (raw_copy_path, anomalies_path_or_None, df)
        This helper is intentionally conservative and non-destructive.
        """
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(anomalies_dir, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        raw_name = f'master_raw_{ts}.csv'
        raw_path = os.path.join(raw_dir, raw_name)
        try:
            shutil.copy2(path, raw_path)
        except Exception:
            raw_path = None

        # lightweight read for anomaly detection
        try:
            df = pd.read_csv(path)
        except Exception:
            return raw_path, None, None

        # normalize date column if present
        if 'date' in df.columns and 'Date' not in df.columns:
            df = df.rename(columns={'date': 'Date'})
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except Exception:
                pass

        # simple anomaly detection on Modal_Price: z-score per commodity
        anomalies = []
        anom_path = None
        if 'Modal_Price' in df.columns and 'Commodity' in df.columns:
            try:
                df['Modal_Price_num'] = pd.to_numeric(df['Modal_Price'], errors='coerce')
                for crop in df['Commodity'].dropna().unique():
                    try:
                        cdf = df[df['Commodity'].astype(str).str.strip().str.lower() == str(crop).strip().lower()]
                        vals = cdf['Modal_Price_num'].dropna()
                        if len(vals) < 5:
                            continue
                        m = vals.mean(); s = vals.std()
                        if s == 0 or pd.isna(s):
                            continue
                        z = (cdf['Modal_Price_num'] - m).abs() / s
                        anom_idx = z[z > 3].index.tolist()
                        anomalies.extend(anom_idx)
                    except Exception:
                        continue
                if anomalies:
                    anom_df = df.loc[sorted(set(anomalies))]
                    anom_path = os.path.join(anomalies_dir, f'anomalies_{ts}.csv')
                    try:
                        anom_df.to_csv(anom_path, index=False)
                    except Exception:
                        anom_path = None
            except Exception:
                anom_path = None
            finally:
                if 'Modal_Price_num' in df.columns:
                    df = df.drop(columns=['Modal_Price_num'], errors='ignore')

        return raw_path, anom_path, df

    @staticmethod
    def load(path: str, ingest_raw: bool = False) -> pd.DataFrame:
        # Prefer provided path; fall back to master.csv in project root
        if os.path.exists(path):
            if ingest_raw:
                try:
                    _, _, df = DataHandler.ingest_data(path)
                    if df is None:
                        df = pd.read_csv(path)
                except Exception:
                    df = pd.read_csv(path)
            else:
                df = pd.read_csv(path)
        else:
            fallback = os.path.join(BASE_DIR, 'master.csv')
            if os.path.exists(fallback):
                print(f"Dataset not found at {path}; falling back to {fallback}")
                df = pd.read_csv(fallback)
            else:
                raise FileNotFoundError(f"Dataset not found at {path} and fallback {fallback} missing")
        # Normalize date column name if present
        if 'date' in df.columns and 'Date' not in df.columns:
            df = df.rename(columns={'date': 'Date'})

        # Restrict dataset to only Tomato and Onion (case-insensitive)
        if 'Commodity' not in df.columns:
            raise ValueError("Dataset missing 'Commodity' column")
        df['Commodity'] = df['Commodity'].astype(str).str.strip()
        df = df[df['Commodity'].str.lower().isin(['onion', 'tomato'])].copy()

        # Keep only identifiers + model FEATURES + target/date
        id_cols = ['STATE', 'district', 'Market Name', 'Commodity', 'Variety', 'Grade']
        target_cols = ['Date', 'Modal_Price']
        keep_cols = id_cols + target_cols + FEATURES

        # Ensure columns exist; add missing as NaN
        for c in keep_cols:
            if c not in df.columns:
                df[c] = np.nan

        return df[keep_cols]



def feature_engineer(df: pd.DataFrame, group_col: str = None, crop: str = None) -> pd.DataFrame:
    """Add engineered time-series features per group (market) or per commodity if group_col is None.
    All rolling/lag operations are shifted to avoid future leakage.
    """
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Commodity', 'Date'] if 'Commodity' in df.columns else ['Date'])
        df['month'] = df['Date'].dt.month
        try:
            df['week_of_year'] = df['Date'].dt.isocalendar().week
        except Exception:
            df['week_of_year'] = df['Date'].dt.week
        df['day_of_year'] = df['Date'].dt.dayofyear
    else:
        df['month'] = np.nan
        df['week_of_year'] = np.nan
        df['day_of_year'] = np.nan

    grp = group_col if group_col in df.columns else 'Commodity'
    grouped = df.groupby(grp)

    # Price lags (keep only 1d and 7d to reduce feature count)
    df['price_lag_1d'] = grouped['Modal_Price'].shift(1)
    df['price_lag_7d'] = grouped['Modal_Price'].shift(7)

    # Rolling stats (shifted to avoid leakage)
    df['rolling_mean_7d'] = grouped['Modal_Price'].rolling(7, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    df['rolling_std_7d'] = grouped['Modal_Price'].rolling(7, min_periods=1).std().shift(1).reset_index(level=0, drop=True)

    # Momentum
    df['price_momentum_7d'] = grouped['Modal_Price'].pct_change(periods=7).shift(1)

    # Derived weather
    if 'temp_max' in df.columns and 'temp_min' in df.columns:
        df['temp_range'] = pd.to_numeric(df['temp_max'], errors='coerce') - pd.to_numeric(df['temp_min'], errors='coerce')
    else:
        df['temp_range'] = np.nan

    if 'rainfall_mm' in df.columns:
        df['rainfall_7d_avg'] = grouped['rainfall_mm'].rolling(7, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    else:
        df['rainfall_7d_avg'] = np.nan

    # Week cyclical transforms
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # Festival and harvest heuristics
    df['festival_indicator'] = df['month'].isin({10, 11, 12}).astype(int)
    if crop:
        hmap = {'onion': {3,4,5,6}, 'tomato': {8,9,10}}
        df['harvest_season_indicator'] = df['month'].isin(hmap.get(crop.lower(), set())).astype(int)
    else:
        df['harvest_season_indicator'] = 0

    # arrival quantity (if present)
    arrival_cols = [c for c in df.columns if c.lower() in ('arrival', 'arrival_qty', 'arrival_quantity', 'total_arrival')]
    if arrival_cols:
        df['arrival_qty'] = pd.to_numeric(df[arrival_cols[0]], errors='coerce')
    else:
        df['arrival_qty'] = np.nan

    return df


class ModelTrainer:
    def __init__(self, features):
        self.features = features

    def train_for(self, crop: str, df: pd.DataFrame):
        crop_df = df[df.get('Commodity', '').astype(str).str.strip().str.lower() == crop.lower()].copy()
        if crop_df.empty or 'Modal_Price' not in crop_df.columns:
            raise ValueError(f"No data for crop {crop} or 'Modal_Price' missing")

        # Feature engineering (per market to avoid leakage)
        group_col = 'Market Name' if 'Market Name' in crop_df.columns else 'Commodity'
        crop_df = feature_engineer(crop_df, group_col=group_col, crop=crop)

        # Build final feature matrix using the trainer's feature list plus newly engineered features
        for f in self.features:
            if f not in crop_df.columns:
                crop_df[f] = np.nan

        # Also ensure engineered features exist (match trimmed FEATURES)
        engineered = ['price_lag_1d', 'price_lag_7d', 'rolling_mean_7d', 'rolling_std_7d', 'price_momentum_7d',
                  'week_sin', 'week_cos', 'festival_indicator', 'harvest_season_indicator',
                  'temp_range', 'rainfall_7d_avg', 'arrival_qty']
        for f in engineered:
            if f not in crop_df.columns:
                crop_df[f] = np.nan

        feature_cols = list(dict.fromkeys(self.features + engineered))
        X_all = crop_df[feature_cols].apply(pd.to_numeric, errors='coerce')
        y_all = crop_df['Modal_Price'].astype(float)

        # Drop rows with NA in primary lag features (require 1d and 7d lags)
        if 'price_lag_1d' in X_all.columns and 'price_lag_7d' in X_all.columns:
            mask = X_all[['price_lag_1d', 'price_lag_7d']].notna().all(axis=1)
            X_all = X_all[mask]
            y_all = y_all[mask]

        if len(X_all) < 50:
            # fallback to simpler model if too little data
            X_train, X_test, y_train, y_test = train_test_split(X_all.fillna(0), y_all, test_size=0.2, shuffle=False)
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
        else:
            # Time-series aware split: use first 80% for train, last 20% for test
            split = int(len(X_all) * 0.8)
            X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]
            y_train, y_test = y_all.iloc[:split], y_all.iloc[split:]

            # Optimized: Reduced hyperparameter search for faster training
            # Reduced n_iter from 10 to 5, cv splits from 4 to 2, and parameter space
            tscv = TimeSeriesSplit(n_splits=2)
            base = RandomForestRegressor(random_state=42, n_jobs=-1, warm_start=False)
            param_dist = {
                'n_estimators': [100, 200],  # Removed 400
                'max_depth': [8, 12],  # Reduced options
                'min_samples_split': [5, 10],  # Removed 2
                'min_samples_leaf': [2, 4],  # Removed 1
                'max_features': ['sqrt', 'log2']  # Removed 0.3 and None
            }
            rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=5, cv=tscv, 
                                   scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, 
                                   verbose=0, error_score='raise')
            rs.fit(X_train, y_train)
            best = rs.best_estimator_
            # Fit best on training data
            best.fit(X_train, y_train)
            model = best

        preds = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(sqrt(mean_squared_error(y_test, preds)))

        return model, mae, rmse


class FeatureAutoFiller:
    """Intelligently fills missing features using historical dataset statistics."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_stats = {}
        self.compute_stats()
    
    def compute_stats(self):
        """Pre-compute feature statistics per commodity."""
        for crop in ['onion', 'tomato']:
            crop_df = self.df[self.df['Commodity'].astype(str).str.strip().str.lower() == crop]
            if crop_df.empty:
                self.feature_stats[crop] = {}
                continue
            stats = {}
            for col in FEATURES:
                if col in crop_df.columns:
                    numeric = pd.to_numeric(crop_df[col], errors='coerce')
                    stats[col] = {
                        'mean': numeric.mean(),
                        'median': numeric.median(),
                        'std': numeric.std()
                    }
                else:
                    stats[col] = {'mean': 0, 'median': 0, 'std': 0}
            self.feature_stats[crop] = stats
    
    def fill_features(self, commodity: str, partial_data: dict) -> dict:
        """Fill missing features with intelligent defaults."""
        crop = commodity.lower()
        stats = self.feature_stats.get(crop, {})
        filled = partial_data.copy()
        
        # If no date provided, use today
        if 'date' not in filled:
            now = pd.Timestamp.now()
            filled['month'] = int(now.month)
            filled['day_of_year'] = int(now.dayofyear)
            filled['week_of_year'] = int(now.isocalendar().week)
        
        # Cyclic encoding for week
        week = filled.get('week_of_year', 26)
        if 'week_sin' not in filled:
            filled['week_sin'] = float(np.sin(2 * np.pi * week / 52))
        if 'week_cos' not in filled:
            filled['week_cos'] = float(np.cos(2 * np.pi * week / 52))
        
        # Festival and harvest indicators
        month = filled.get('month', 1)
        if 'festival_indicator' not in filled:
            filled['festival_indicator'] = 1 if month in (10, 11, 12) else 0
        if 'harvest_season_indicator' not in filled:
            hmap = {'onion': {3, 4, 5, 6}, 'tomato': {8, 9, 10}}
            filled['harvest_season_indicator'] = 1 if month in hmap.get(crop, set()) else 0

        # Derive temperature range when possible so the model receives the same schema it was trained on
        if 'temp_range' not in filled:
            if 'temp_max' in filled and 'temp_min' in filled:
                try:
                    filled['temp_range'] = float(filled['temp_max']) - float(filled['temp_min'])
                except Exception:
                    filled['temp_range'] = 6.0
            elif 'temp_mean' in filled:
                filled['temp_min'] = float(filled.get('temp_min', float(filled['temp_mean']) - 3))
                filled['temp_max'] = float(filled.get('temp_max', float(filled['temp_mean']) + 3))
                filled['temp_range'] = float(filled['temp_max']) - float(filled['temp_min'])
            else:
                filled['temp_range'] = 6.0
        
        # Fill all other missing features with historical medians
        for feature in FEATURES:
            if feature not in filled or filled[feature] is None or filled[feature] == '':
                default_val = stats.get(feature, {}).get('median', 0)
                filled[feature] = float(default_val) if pd.notna(default_val) else 0.0
        
        return filled


class Predictor:
    def __init__(self, models_map, metrics_map, features, feature_filler=None):
        self.models = models_map
        self.metrics = metrics_map
        self.features = features
        self.filler = feature_filler

    def validate_input(self, payload: dict) -> Tuple[str, np.ndarray]:
        """Accept minimal input: only commodity, price_lag_7d, and temp_mean required."""
        commodity_raw = payload.get('commodity', '').strip()
        commodity = commodity_raw.title()
        if not commodity_raw or commodity.lower() not in ['onion', 'tomato']:
            raise ValueError('commodity must be "onion" or "tomato"')

        # Core inputs: price from 7 days ago and current temperature
        price_lag_7d = payload.get('price_lag_7d')
        temp_mean = payload.get('temp_mean')
        
        if price_lag_7d is None or price_lag_7d == '':
            raise ValueError('price_lag_7d is required (price 7 days ago)')
        if temp_mean is None or temp_mean == '':
            raise ValueError('temp_mean is required (current temperature)')
        
        try:
            price_lag_7d = float(price_lag_7d)
            temp_mean = float(temp_mean)
        except ValueError:
            raise ValueError('price_lag_7d and temp_mean must be numbers')
        
        # Build minimal data dict
        data = {
            'commodity': commodity,
            'price_lag_7d': price_lag_7d,
            'temp_mean': temp_mean,
        }
        
        # Use auto-filler to intelligently fill remaining features
        if self.filler:
            data = self.filler.fill_features(commodity, data)
        else:
            # Fallback: basic filling without filler
            now = pd.Timestamp.now()
            week = int(now.isocalendar().week)
            month = int(now.month)
            data.update({
                'month': month,
                'day_of_year': int(now.dayofyear),
                'week_of_year': week,
                'week_sin': float(np.sin(2 * np.pi * week / 52)),
                'week_cos': float(np.cos(2 * np.pi * week / 52)),
                'festival_indicator': 1 if month in (10, 11, 12) else 0,
                'harvest_season_indicator': 1 if month in ({3,4,5,6} if commodity.lower()=='onion' else {8,9,10}) else 0,
                'price_lag_1d': price_lag_7d,  # Use 7d as proxy for 1d
                'rolling_mean_7d': price_lag_7d,
                'rolling_std_7d': 0.0,
                'price_momentum_7d': 0.0,
                'rainfall_mm': 0.0,
                'humidity': 60.0,
                'wind_speed': 5.0,
                'temp_min': temp_mean - 3,
                'temp_max': temp_mean + 3,
                'temp_range': 6.0,
                'rainfall_7d_avg': 0.0,
                'arrival_qty': 100.0,
            })
        
        # Build feature array in correct order
        arr = np.array([[float(data.get(f, 0)) for f in self.features]], dtype=float)
        return commodity, arr

    def predict(self, commodity: str, X: np.ndarray):
        commodity_key = commodity.strip().title()
        model = self.models.get(commodity_key)
        if model is None:
            raise ValueError(f"Model for '{commodity_key.lower()}' is still training. Please wait a moment and try again.")
        pred = float(model.predict(X)[0])
        m = self.metrics.get(commodity_key, {})
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


def load_persisted_models(evaluate: bool = True):
    """Load any joblib models from the models directory into memory.

    When evaluate is True, also compute quick metrics from the dataset.
    Render startup should use evaluate=False to avoid delaying port binding.
    """
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
                if evaluate:
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
                                    rmse = float(mean_squared_error(y_te, preds) ** 0.5)
                                    metrics[crop] = {'mae': round(mae, 3), 'rmse': round(rmse, 3)}
                    except Exception as ee:
                        print('Quick eval failed for', crop, ee)
            else:
                print('Loaded model', name, 'but could not infer crop; model available as', name)
        except Exception as e:
            print('Failed to load persisted model', f, e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_ui')
def predict_ui():
    return render_template('predict.html')


@app.route('/status', methods=['GET'])
def status():
    return jsonify({'training': is_training, 'models': {k: ('ready' if k in models else 'missing') for k in ['Onion', 'Tomato']}})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json(force=True)
        
        # Initialize feature filler with current dataset if available
        filler = None
        try:
            df = DataHandler.load(DATA_PATH)
            filler = FeatureAutoFiller(df)
        except:
            pass
        
        predictor = Predictor(models, metrics, FEATURES, feature_filler=filler)
        commodity, X = predictor.validate_input(payload)
        result = predictor.predict(commodity, X)
        
        # Calculate confidence (0-100)
        mae = result.get('mae')
        rmse = result.get('rmse')
        # Normalize: lower error = higher confidence
        # Assume MAE of 100 = 50% confidence, MAE of 50 = 75% confidence
        if mae:
            confidence = max(10, min(95, 100 - (mae / 2)))
        else:
            confidence = 75
        
        # Determine trend based on momentum
        price_7d_ago = float(payload.get('price_lag_7d', 0))
        predicted = result['price']
        trend = 'up' if predicted > price_7d_ago else ('down' if predicted < price_7d_ago else 'stable')
        change_pct = round(((predicted - price_7d_ago) / price_7d_ago * 100), 1) if price_7d_ago else 0
        
        # Generate simple 7-day forecast
        # Build a trend-aware path from the submitted 7-day lag price to the model output.
        # This gives the frontend a more natural line instead of a flat synthetic curve.
        forecast = []
        start_price = price_7d_ago if price_7d_ago > 0 else predicted
        end_price = predicted
        steps = 7
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 1.0
            smooth = t * t * (3 - 2 * t)
            value = start_price + (end_price - start_price) * smooth
            spread = abs(end_price - start_price)
            if trend == 'up':
                value += spread * 0.025 * t
            elif trend == 'down':
                value -= spread * 0.025 * t
            elif trend == 'stable':
                value += (spread * 0.01) * np.sin(t * np.pi)
            forecast.append(round(max(value, 0), 2))
        
        return jsonify({
            'status': 'Success',
            'price': round(predicted, 2),
            'price_per_kg': round(predicted / 100.0, 2),
            'price_per_quintal': round(predicted, 2),
            'trend': trend,
            'change_pct': change_pct,
            'confidence': round(confidence, 1),
            'forecast': forecast,  # 7-day forecast
            'mae': round(mae, 2) if mae else None,
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
        # use DataHandler.load which filters to Onion and Tomato
        df = DataHandler.load(path)
        cols = list(df.columns)
        rows = df.fillna('').head(n).to_dict(orient='records')
        return jsonify({'columns': cols, 'rows': rows})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # First try to load persisted models to avoid retraining
    print('=' * 60)
    print('Loading persisted models...')
    load_persisted_models()
    print(f'Models loaded: {list(models.keys())}')
    print('=' * 60)
    
    # Only train if models are missing
    if 'Onion' not in models or 'Tomato' not in models:
        print('⚠️  Missing models detected. Training on startup...')
        print('=' * 60)
        try:
            print('Loading dataset...')
            df = DataHandler.load(DATA_PATH)
            print(f'Dataset loaded: {len(df)} rows, columns: {list(df.columns)[:5]}...')
            
            trainer = ModelTrainer(FEATURES)
            for crop in ['Onion', 'Tomato']:
                if crop in models:
                    print(f'✅ Skipping {crop} (already loaded)')
                    continue
                try:
                    print(f'\n🔄 Training model for {crop}...')
                    model, mae, rmse = trainer.train_for(crop, df)
                    models[crop] = model
                    metrics[crop] = {'mae': round(mae, 3), 'rmse': round(rmse, 3)}
                    print(f"✅ Trained {crop}: MAE={mae:.3f}, RMSE={rmse:.3f}")
                    
                    # Persist trained model
                    model_path = os.path.join(MODEL_DIR, f'{crop.lower()}_model.joblib')
                    joblib.dump(model, model_path)
                    print(f"💾 Saved {crop} model to {model_path}")
                except Exception as e:
                    print(f'❌ Failed training for {crop}: {str(e)}')
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f'❌ Training failed: {str(e)}')
            import traceback
            traceback.print_exc()
        print('=' * 60)
    else:
        print('✅ All models loaded successfully')
        print('=' * 60)

    # Check final status
    print('\n📊 FINAL STATUS:')
    print(f'  Onion model: {"✅ Ready" if "Onion" in models else "❌ Missing"}')
    print(f'  Tomato model: {"✅ Ready" if "Tomato" in models else "❌ Missing"}')
    print('=' * 60)

    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_DEBUG', 'False') == 'True'

    print(f'\n🚀 Starting web server on http://0.0.0.0:{port}')
    app.run(host='0.0.0.0', port=port, debug=debug_mode)