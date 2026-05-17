# Render Deployment Guide

## Prerequisites
- GitHub repository with Price Predictor code
- Render account (free tier available)
- Pre-trained models in `models/` directory (Onion_best.joblib, Tomato_best.joblib)

## Deployment Steps

### 1. Push to GitHub
```bash
git add -A
git commit -m "Deployment-ready: Render free tier setup"
git push origin main
```

### 2. Create New Web Service on Render

1. Go to [render.com](https://render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository (authorize if needed)
4. Select the Price Predictor repository
5. Fill in service details:
   - **Name**: `price-predictor` (or custom)
   - **Environment**: Python 3
   - **Region**: Choose closest to you
   - **Branch**: main
   - **Build Command**: `pip install -r requirements.txt`
  - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 60`

  If you add the included `render.yaml` file to the repo, Render can use that automatically instead of manual dashboard settings.

### 3. Configure Environment Variables

In the Render dashboard, add these environment variables:

- **ENABLE_BACKGROUND_TRAIN**: `0` (disabled on free tier to save resources)
- **FLASK_DEBUG**: `False` (production mode)

Optional (if using external data source):
- **DATA_PATH**: Path to your dataset (if hosted elsewhere)

### 4. Set Instance Type

- **Pricing Plan**: Free tier ($0/month)
  - 0.5 CPU, 512 MB RAM
  - Auto-sleep after 15 min inactivity (free tier)
  - ~3 second cold start when woken

### 5. Deploy

1. Click **"Create Web Service"**
2. Render builds and deploys (2-3 min)
3. Once green checkmark appears, service is live

Your app will be available at:
```
https://<service-name>.onrender.com
```

## Testing Live Deployment

```bash
curl -X GET https://<service-name>.onrender.com/
# Returns: index.html dashboard

curl -X POST https://<service-name>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"commodity":"Onion","values":[25,50,60,2,1200,1100,350,6,180,0,0]}'
# Returns: prediction with both price units
```

## First Request (Cold Start)

On free tier, first request after 15 min inactivity may take 3-5 seconds (auto-spin-up). Subsequent requests are instant.

## Persisted Models

Pre-trained models are included in the repo:
- `models/Onion_best.joblib` — RandomForest trained on historical onion prices
- `models/Tomato_best.joblib` — RandomForest trained on historical tomato prices

## Recommended Render Files

- `app.py` exposes the Flask app through the standard `app:app` entrypoint.
- `render.yaml` pins the build and start commands for repeatable deploys.
- `runtime.txt` pins the Python version used by Render.

Models load at startup; no training happens on Render free tier (ENABLE_BACKGROUND_TRAIN=0).

## Database / Dataset

Currently the app uses `master.csv` (included in repo). If dataset grows:

1. Option A: Replace `master.csv` in repo (re-deploy)
2. Option B: Use Render PostgreSQL add-on (paid tier)
3. Option C: Link to external CSV storage (S3 bucket, Google Drive, etc.)

## Troubleshooting

### Service won't start
- Check build logs in Render dashboard
- Verify `requirements.txt` has all dependencies
- Ensure `Main.py` has correct import order

### Predictions return errors
- Cold start may timeout; wait 10 seconds and retry
- Check `/health` endpoint: `GET https://<service-name>.onrender.com/health`

### High latency
- Free tier has limited resources; upgrade to paid for better performance
- Reduce Three.js mesh count in frontend if ambience is slow

## Upgrading from Free Tier

If you need:
- Always-on service (no auto-sleep): Starter plan ($12/month)
- Better performance: Standard plan ($25/month)
- Custom domain: Available on any paid plan

In Render dashboard, modify Instance Type under **Settings** → **Pricing Plan**.

## Environment Variables Summary

| Variable | Default | Purpose |
|----------|---------|---------|
| PORT | 5000 | HTTP server port (set by Render) |
| ENABLE_BACKGROUND_TRAIN | 0 | Training enabled (use 0 on free tier) |
| FLASK_DEBUG | False | Debug mode (keep False in production) |
| DATA_PATH | e:\Price Predictor\master.csv | Dataset location |

## Endpoints

- `GET /` — Dashboard UI
- `GET /health` — Service health check
- `POST /predict` — Get price prediction
- `GET /analytics` — Analytics data
- `GET /status` — Model status
- `GET /dataset_preview?nrows=20` — Sample dataset rows

## Support

For Render deployment issues: [render.com/docs](https://render.com/docs)
For Price Predictor issues: Check `Main.py` logs in Render dashboard
