"""
Improved RandomForest training pipeline for Onion and Tomato price prediction.

Features implemented:
- price_lag_1d, price_lag_3d
- rolling_mean_7d, rolling_std_7d
- price_momentum (7-day pct change)
- week_of_year
- festival_indicator (heuristic by month)
- harvest_season_indicator (heuristic per crop)
- temp_range, rainfall_7d_avg
- use arrival quantity if column exists (heuristic column names)

Tuning & Validation:
- TimeSeriesSplit for cross-validation
- RandomizedSearchCV followed by GridSearchCV refinement
- Walk-forward evaluation
- Permutation importance (and SHAP if available)

Usage:
    python scripts\improve_rf_training.py --data "e:\\Price Predictor\\master.csv" --out models --crops Onion Tomato

Note: script avoids shuffling, prevents leakage by shifting/rolling with shift(1).
"""
import os
import argparse
import warnings
from pprint import pprint
from math import sqrt
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

# Default features used by the lightweight app; we'll extend these
BASE_FEATURES = [
    'temp_mean', 'rainfall_mm', 'humidity', 'wind_speed',
    'price_lag_7d', 'price_lag_30d', 'rainfall_mm_7d_sum',
    'month', 'day_of_year', 'drought_indicator', 'flood_indicator'
]

# Heuristic festival months (example): Oct-Nov-Dec
FESTIVAL_MONTHS = {10, 11, 12}

# Heuristic harvest months per crop (adjustable)
HARVEST_SEASONS = {
    'onion': {3, 4, 5, 6},
    'tomato': {8, 9, 10}
}


def load_and_filter(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if 'date' in df.columns and 'Date' not in df.columns:
        df = df.rename(columns={'date': 'Date'})
    if 'Commodity' not in df.columns:
        raise ValueError("Dataset missing 'Commodity' column")
    df['Commodity'] = df['Commodity'].astype(str).str.strip()
    df = df[df['Commodity'].str.lower().isin(['onion', 'tomato'])].copy()
    # Ensure Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        # Attempt common date column names
        for c in ['DATE', 'Date', 'date']:
            if c in df.columns:
                df['Date'] = pd.to_datetime(df[c])
                break
    # Sort
    if 'Market Name' in df.columns:
        df = df.sort_values(['Market Name', 'Date'])
    else:
        df = df.sort_values(['Commodity', 'Date'])
    return df


def feature_engineer(df: pd.DataFrame, crop: str) -> pd.DataFrame:
    """Add derived features; compute lags and rolling stats per Market Name to avoid cross-market leakage."""
    df = df.copy()
    # Basic date features
    if 'Date' in df.columns:
        df['month'] = df['Date'].dt.month
        # isocalendar() returns DataFrame in pandas >= 1.1
        try:
            df['week_of_year'] = df['Date'].dt.isocalendar().week
        except Exception:
            df['week_of_year'] = df['Date'].dt.week
        df['day_of_year'] = df['Date'].dt.dayofyear
    else:
        df['month'] = np.nan
        df['week_of_year'] = np.nan
        df['day_of_year'] = np.nan

    # Ensure numeric price column
    df['Modal_Price'] = pd.to_numeric(df['Modal_Price'], errors='coerce')

    group_cols = ['Market Name'] if 'Market Name' in df.columns else ['Commodity']
    grouped = df.groupby(group_cols)

    # Price lags and rolling stats
    df['price_lag_1d'] = grouped['Modal_Price'].shift(1)
    df['price_lag_3d'] = grouped['Modal_Price'].shift(3)
    df['price_lag_7d'] = grouped['Modal_Price'].shift(7)
    df['price_lag_30d'] = grouped['Modal_Price'].shift(30)

    df['rolling_mean_7d'] = grouped['Modal_Price'].rolling(window=7, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    df['rolling_std_7d'] = grouped['Modal_Price'].rolling(window=7, min_periods=1).std().shift(1).reset_index(level=0, drop=True)

    # Price momentum: 7-day pct change (previous value)
    df['price_momentum_7d'] = grouped['Modal_Price'].pct_change(periods=7).shift(1)

    # Derived weather
    if 'temp_max' in df.columns and 'temp_min' in df.columns:
        df['temp_range'] = pd.to_numeric(df['temp_max'], errors='coerce') - pd.to_numeric(df['temp_min'], errors='coerce')
    else:
        df['temp_range'] = np.nan

    if 'rainfall_mm' in df.columns:
        df['rainfall_7d_avg'] = grouped['rainfall_mm'].rolling(window=7, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    else:
        df['rainfall_7d_avg'] = np.nan

    # Week of year already added; normalize week_of_year into sine/cos if desired (cyclical)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # Festival indicator (heuristic)
    df['festival_indicator'] = df['month'].isin(FESTIVAL_MONTHS).astype(int)

    # Harvest season indicator per crop (heuristic)
    hmonths = HARVEST_SEASONS.get(crop.lower(), set())
    df['harvest_season_indicator'] = df['month'].isin(hmonths).astype(int)

    # Arrival quantity if present (try common column names)
    arrival_cols = [c for c in df.columns if c.lower() in ('arrival', 'arrival_qty', 'arrival_quantity', 'total_arrival')]
    if arrival_cols:
        df['arrival_qty'] = pd.to_numeric(df[arrival_cols[0]], errors='coerce')
    else:
        df['arrival_qty'] = np.nan

    # Fill missing numeric features conservatively (do not leak future info)
    # We'll keep NaNs for rows where lags are missing so they can be dropped before training

    return df


def prepare_training_data(df: pd.DataFrame, crop: str, features: list):
    dfc = df[df['Commodity'].astype(str).str.lower() == crop.lower()].copy()
    if dfc.empty:
        raise ValueError(f'No data for {crop}')
    # Feature engineer
    dfc = feature_engineer(dfc, crop)

    # Define final feature list: combine provided features and new engineered ones
    engineered = ['price_lag_1d', 'price_lag_3d', 'rolling_mean_7d', 'rolling_std_7d', 'price_momentum_7d',
                  'week_sin', 'week_cos', 'festival_indicator', 'harvest_season_indicator', 'temp_range',
                  'rainfall_7d_avg', 'arrival_qty']

    final_features = list(dict.fromkeys(features + engineered))  # unique preserve order

    # Ensure exists
    for f in final_features:
        if f not in dfc.columns:
            dfc[f] = np.nan

    # Drop rows where target or the primary lag is missing to avoid leakage
    dfc = dfc.sort_values('Date')
    dfc = dfc.dropna(subset=['Modal_Price', 'price_lag_1d', 'price_lag_3d'])

    X = dfc[final_features].apply(pd.to_numeric, errors='coerce')
    y = dfc['Modal_Price'].astype(float)

    # Final drop rows with any NaN in X
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    return X, y, dfc


def walk_forward_eval(model, X, y, n_splits=5):
    """Simple expanding window walk-forward evaluation."""
    n = len(X)
    if n_splits < 2:
        n_splits = 2
    sizes = np.linspace(0.2, 0.8, n_splits)
    results = []
    for frac in sizes:
        split = int(n * frac)
        X_tr, y_tr = X.iloc[:split], y.iloc[:split]
        X_te, y_te = X.iloc[split:split + int(n * (1 - frac) / 5) + 1], y.iloc[split:split + int(n * (1 - frac) / 5) + 1]
        if len(X_te) < 1:
            continue
        m = model
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        rmse = sqrt(mean_squared_error(y_te, preds))
        mae = mean_absolute_error(y_te, preds)
        results.append({'rmse': rmse, 'mae': mae, 'n_test': len(y_te)})
    return results


def tune_and_train(X, y, random_state=42):
    tscv = TimeSeriesSplit(n_splits=5)
    base = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    param_dist = {
        'n_estimators': [100, 200, 400, 600],
        'max_depth': [6, 8, 12, 16, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, None]
    }

    rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=20, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, random_state=random_state, verbose=1)
    rs.fit(X, y)
    print('RandomizedSearch best:', rs.best_params_, 'score', rs.best_score_)

    # refine with GridSearch around best params
    best = rs.best_params_
    grid = {
        'n_estimators': [best.get('n_estimators', 200)],
        'max_depth': [best.get('max_depth', None), None],
        'min_samples_split': [best.get('min_samples_split', 2), max(2, best.get('min_samples_split', 2)//2)],
        'min_samples_leaf': [best.get('min_samples_leaf', 1), max(1, best.get('min_samples_leaf', 1)//2)],
        'max_features': [best.get('max_features', 'auto')]
    }
    gs = GridSearchCV(RandomForestRegressor(random_state=random_state, n_jobs=-1), grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    gs.fit(X, y)
    print('GridSearch best:', gs.best_params_, 'score', gs.best_score_)

    final_model = gs.best_estimator_
    return final_model, rs, gs


def permutation_imp(model, X, y, n_repeats=10):
    res = permutation_importance(model, X, y, n_repeats=n_repeats, n_jobs=-1, random_state=42)
    imp = {f: v for f, v in zip(X.columns, res.importances_mean)}
    return imp


def try_shap(model, X):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        return shap, shap_vals
    except Exception as e:
        print('SHAP not available or failed:', e)
        return None, None


def run_for_crop(path, crop, out_dir, base_features=BASE_FEATURES):
    df = load_and_filter(path)
    X, y, dfc = prepare_training_data(df, crop, base_features)
    print(f'{crop}: Prepared data shape: X={X.shape}, y={y.shape}')

    model, rs, gs = tune_and_train(X, y)

    # Walk-forward with the final model
    wf = walk_forward_eval(model, X, y)
    print('Walk-forward results:', wf)

    # Permutation importance
    pi = permutation_imp(model, X, y)
    sorted_pi = dict(sorted(pi.items(), key=lambda kv: kv[1], reverse=True))
    print('Permutation importance (top-10):')
    pprint(list(sorted_pi.items())[:10])

    # SHAP (optional)
    shap_mod, shap_vals = try_shap(model, X)
    if shap_mod is not None:
        print('SHAP computed')

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f'{crop}_rf_tuned.joblib')
    joblib.dump(model, model_path, compress=3)
    print('Saved model to', model_path, 'size(MB):', os.path.getsize(model_path) / 1024.0 / 1024.0)

    # Save a small report
    report = {
        'crop': crop,
        'model_path': model_path,
        'best_params': gs.best_params_,
        'random_search_best': rs.best_params_,
        'walk_forward': wf,
        'permutation_importance_top': list(sorted_pi.items())[:20]
    }
    report_path = os.path.join(out_dir, f'{crop}_training_report.json')
    import json
    with open(report_path, 'w') as fh:
        json.dump(report, fh, indent=2)
    print('Saved report to', report_path)

    return model, report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='models')
    p.add_argument('--crops', nargs='+', default=['Onion', 'Tomato'])
    args = p.parse_args()

    for c in args.crops:
        try:
            run_for_crop(args.data, c, args.out)
        except Exception as e:
            print('Failed for', c, e)
