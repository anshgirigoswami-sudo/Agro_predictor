# Agro Predictor — Crop Price Forecasting (Onion & Tomato)

Ansh Giri Goswami — MCA (AI/ML) student

Agro Predictor is a lightweight, production-ready web application and training pipeline for short-term price forecasting of two commodities: Onion and Tomato. It combines time-series feature engineering, robust Random Forest models, and a simple Flask web UI + JSON API to provide price predictions, confidence estimates, and short forecasts.

Key goals
- Provide an easy-to-run demo for commodity price forecasting.
- Ship compact, persisted models (`models/`) for quick predictions.
- Offer training utilities to improve and re-train models from historical market data (`master.csv`).

Highlights
- Predict endpoint: `/predict` (JSON API)
- Simple web UI: root `/` and `/predict_ui` for manual inputs
- Training scripts: `scripts/improve_rf_training.py` and `scripts/make_compact_models.py`

--

## Quick Demo — Run locally

1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the app (development):

```powershell
python app.py
```

The server listens on port `5000` by default. Visit `http://localhost:5000` in your browser to open the UI.

Notes:
- `app.py` imports and starts the Flask `app` defined in `Main.py`.
- On startup the app will try to load persisted models from `models/` (e.g. `onion_model.joblib`, `tomato_model.joblib`). If those are missing it will train models using `master.csv`.

--

## API Usage

Predict (POST JSON)

- Endpoint: `/predict`
- Required minimal JSON fields: `commodity` ("Onion" or "Tomato"), `price_lag_7d` (price 7 days ago), `temp_mean` (current mean temperature)

Example cURL:

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"commodity":"Onion","price_lag_7d":120,"temp_mean":28}'
```

Response contains:
- `price` (predicted modal price), `price_per_kg`, `trend`, `change_pct`, `confidence`, `forecast` (7-day), and error metrics when available (`mae`).

Other useful endpoints:
- `/status` — training & model readiness
- `/analytics` — small analytics + driver correlations
- `/dataset_preview` — preview rows from `master.csv`
- `/predict_ui` — web UI for manual prediction

--

## Project structure (important files)

- `app.py` — small launcher that imports `Main.app` and runs the server.
- `Main.py` — core Flask app, data handling, feature engineering, model life-cycle, and API routes.
- `master.csv` — historical dataset (project root). Used for training and analytics.
- `models/` — persisted models (joblib) (e.g. `onion_model.joblib`, `tomato_model.joblib`).
- `scripts/` — training utilities:
	- `improve_rf_training.py` — improved training pipeline with tuning and reports.
	- `make_compact_models.py` — helper to create compact persisted models for the app.
- `templates/` & `static/` — Flask UI files.

--

## Dataset & Features

- The app expects a CSV (`master.csv`) containing market data including at minimum: `Date`, `Commodity`, `Modal_Price`, `Market Name` (or similar identifiers). The project filters to `onion` and `tomato` (case-insensitive).
- Feature engineering performed by the code includes:
	- time features: month, week, day_of_year, cyclical transforms (`week_sin`, `week_cos`)
	- lag features: `price_lag_1d`, `price_lag_7d`, etc.
	- rolling statistics: `rolling_mean_7d`, `rolling_std_7d`
	- momentum: `price_momentum_7d`
	- weather/supply heuristics: `temp_range`, `rainfall_7d_avg`, `arrival_qty`
	- festival/harvest indicators

The exact feature set used by the app is defined in `Main.py` in the `FEATURES` list.

--

## Training & Improving Models

The project ships compact, ready-to-load models in `models/`. To retrain or produce improved models:

- Quick training on startup: the app trains automatically on startup if persisted models are missing (reads `master.csv`).
- For rigorous training and hyperparameter tuning, use:

```powershell
python scripts\improve_rf_training.py --data "E:\Price Predictor\master.csv" --out models --crops Onion Tomato
```

This script performs time-series-aware tuning (RandomizedSearchCV + GridSearchCV), walk-forward evaluation, permutation importance, and writes a training report.

After training locally, save compact models to `models/` (joblib) so the web app can load them quickly on startup.

--

## Deployment

- The repository includes `Procfile`, `runtime.txt`, and `render.yaml` to simplify deployment on platforms like Heroku or Render.
- Use environment variable `PORT` to bind to a platform-provided port; the app reads `PORT` automatically.

--

## Development notes & troubleshooting

- If the app prints `No persisted models found in models`, ensure `master.csv` exists and contains both `Onion` and `Tomato` rows.
- If training is slow, run `scripts/improve_rf_training.py` on a machine with more CPU and memory, then copy compact joblib files to `models/`.
- The feature autofiller (`FeatureAutoFiller`) fills missing inputs intelligently to allow minimal JSON payloads for prediction.

--

## Contributing

Contributions and improvements are welcome. Suggested ways to help:
- Add more commodities and extend feature engineering.
- Improve model explainability (SHAP summaries) and production monitoring.
- Add unit tests for preprocessing and the API.

--

## License

This repository is provided for educational use. Add your preferred license (e.g., MIT) if you wish to publish it publicly.

--

## Author

Ansh Giri Goswami

Student — MCA (Artificial Intelligence & Machine Learning)

Contact: Open to collaboration — raise an issue or create a PR.
