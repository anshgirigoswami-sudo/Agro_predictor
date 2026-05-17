"""
Train compressed, compact RandomForest models aiming for a target serialized size.
Usage:
    python scripts/make_compact_models.py --crops Onion Tomato --min_mb 50 --max_mb 60

Note: Training may be slow depending on dataset size and machine. This script iteratively reduces
`n_estimators` until the joblib file is within the max size or a minimum estimator threshold is reached.
"""
import os
import argparse
from math import sqrt
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if os.path.basename(__file__) == 'make_compact_models.py' else os.path.dirname(__file__)
DATA_PATH = r"e:\Price Predictor\master.csv"
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FEATURES = [
    'temp_mean', 'rainfall_mm', 'humidity', 'wind_speed',
    'price_lag_7d', 'price_lag_30d', 'rainfall_mm_7d_sum',
    'month', 'day_of_year', 'drought_indicator', 'flood_indicator'
]


def ensure_features(df):
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0


def train_compact_for_crop(crop, min_mb=50, max_mb=60, compress_level=3, start_estimators=100, min_estimators=10):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.ffill().bfill()
    crop_df = df[df['Commodity'].astype(str).str.strip().str.lower() == crop.lower()].copy()
    if crop_df.empty or 'Modal_Price' not in crop_df.columns:
        raise ValueError(f"No data for crop {crop} or 'Modal_Price' missing in dataset")

    ensure_features(crop_df)
    X = crop_df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = crop_df['Modal_Price'].astype(float)

    n_estimators = start_estimators
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp_path = os.path.join(MODEL_DIR, f"{crop}_compact_tmp.joblib")
    final_path = os.path.join(MODEL_DIR, f"{crop}_compact.joblib")

    while True:
        print(f"Training {crop} with n_estimators={n_estimators}...")
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=12, random_state=42, n_jobs=-1)
        model.fit(X, y)
        # dump with gzip compression
        joblib.dump(model, tmp_path, compress=('gzip', compress_level))
        size_mb = os.path.getsize(tmp_path) / 1024.0 / 1024.0
        print(f"Serialized size: {size_mb:.2f} MB")

        if size_mb <= max_mb or n_estimators <= min_estimators:
            os.replace(tmp_path, final_path)
            print(f"Saved compact model to {final_path} ({size_mb:.2f} MB)")
            break

        # reduce estimators and retry
        # reduce by 10 or by 20% whichever keeps progress
        dec = max(10, int(n_estimators * 0.2))
        n_estimators = max(min_estimators, n_estimators - dec)

    return final_path, size_mb


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train compact joblib models within size bounds')
    p.add_argument('--crops', nargs='+', default=['Onion', 'Tomato'])
    p.add_argument('--min_mb', type=int, default=50)
    p.add_argument('--max_mb', type=int, default=60)
    p.add_argument('--compress', type=int, default=3)
    p.add_argument('--start_estimators', type=int, default=100)
    p.add_argument('--min_estimators', type=int, default=10)
    args = p.parse_args()

    for c in args.crops:
        try:
            path, sz = train_compact_for_crop(c, min_mb=args.min_mb, max_mb=args.max_mb, compress_level=args.compress, start_estimators=args.start_estimators, min_estimators=args.min_estimators)
            print(f"Done: {c} -> {path} ({sz:.2f} MB)")
        except Exception as e:
            print(f"Failed for {c}: {e}")
