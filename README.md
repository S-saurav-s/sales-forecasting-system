# Sales Forecasting System
**End-to-end time series forecasting · 44 US states · 8-week horizon**

---

## Project structure

```
forecasting_system/
├── data/
│   └── clean_weekly.csv          ← your cleaned CSV goes here
├── models/
│   ├── sarima_model.py           ← SARIMA / Auto-ARIMA wrapper
│   ├── prophet_model.py          ← Facebook Prophet wrapper
│   ├── xgboost_model.py          ← XGBoost with recursive forecasting
│   └── lstm_model.py             ← 2-layer stacked LSTM
├── utils/
│   ├── feature_engineering.py   ← lag, rolling, calendar, holiday features
│   └── metrics.py               ← MAPE, RMSE, MAE + model selection
├── api/
│   └── app.py                   ← FastAPI REST service
├── train.py                      ← Training orchestrator (CLI)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Setup

```bash
# 1. Create virtualenv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your cleaned weekly CSV
#    Required columns: State | Date (YYYY-MM-DD, weekly Mon) | Total (numeric)
cp /path/to/clean_weekly.csv data/clean_weekly.csv
```

---

## Expected input format (after your cleaning step)

```
State,Date,Total
Alabama,2019-01-14,109574036
Alabama,2019-01-21,109574036
...
California,2019-01-14,444766891
```

- `Date` — already parsed to `YYYY-MM-DD`, weekly frequency (Mondays), **no gaps, no nulls**
- `Total` — integer or float, **no commas**, no nulls
- One row per (State, Week) combination

---

## Step 1 — Train all models

```bash
# Train all states, val window = last 10 weeks
python train.py \
    --data_path  data/clean_weekly.csv \
    --output_dir registry \
    --val_weeks  10

# Train a subset of states only
python train.py --states California Texas Florida
```

**What happens:**
1. Loads cleaned CSV
2. For each state, trains SARIMA, Prophet, XGBoost, LSTM
3. Evaluates each model on the held-out validation window (no leakage)
4. Selects best model by MAPE (ties broken by RMSE)
5. Saves all models to `registry/<State>/`
6. Writes `registry/registry.json` (model paths + metrics)

**Expected output:**
```
10:00:01  INFO     [California]  SARIMA   MAPE=6.10%  (12.3s)
10:00:14  INFO     [California]  Prophet  MAPE=4.83%  (8.1s)
10:00:22  INFO     [California]  XGBoost  MAPE=3.91%  (2.4s)
10:00:25  INFO     [California]  LSTM     MAPE=4.44%  (45.2s)
10:01:10  INFO     [California]  ★ Best → XGBOOST (MAPE=3.91%)
```

---

## Step 2 — Start the API

```bash
# Set API key (default: dev-secret-key-change-in-prod)
export FORECAST_API_KEY="your-secure-key"
export REGISTRY_DIR="registry"

uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Docs available at: `http://localhost:8000/docs`

---

## API Reference

All endpoints (except `/health`) require:
```
X-API-Key: your-secure-key
```

### GET `/health`
```json
{
  "status": "ok",
  "states_loaded": 44,
  "registry_dir": "registry",
  "timestamp": "2022-08-07T10:30:00+00:00"
}
```

### GET `/states`
Returns all states with best model + MAPE.

### POST `/forecast`
```json
// Request
{
  "states": ["California", "Texas"],
  "horizon_weeks": 8
}

// Response
[
  {
    "state": "California",
    "best_model": "xgboost",
    "generated_at": "2022-08-07T10:30:00Z",
    "horizon_weeks": 8,
    "last_train_date": "2022-07-25",
    "forecast": [
      {
        "week": 1,
        "date": "2022-08-01",
        "predicted": 921450000,
        "lower_95": 874000000,
        "upper_95": 968000000,
        "delta_ow_pct": null
      },
      ...
    ],
    "model_metrics": {
      "mape": 3.91,
      "rmse": 27000000,
      "mae": 21000000
    }
  }
]
```

### GET `/forecast/{state}`
Shortcut for single-state 8-week forecast.

### GET `/models/{state}`
Returns validation metrics for ALL 4 models on that state.

### POST `/retrain`
```json
// Request (omit states to retrain all)
{ "states": ["California"] }

// Response
{
  "job_id": "a3f7c2b1",
  "states": ["California"],
  "status": "queued"
}
```

---

## Feature engineering summary

| Feature | Description | Purpose |
|---|---|---|
| `lag_1` | Sales 1 week ago | Short-term momentum |
| `lag_7` | Sales 7 weeks ago | Quarterly cycle |
| `lag_30` | Sales 30 weeks ago | Semi-annual cycle |
| `roll_mean_4w` | 4-week rolling mean | Short trend |
| `roll_mean_12w` | 12-week rolling mean | Medium trend |
| `roll_mean_26w` | 26-week rolling mean | Long trend |
| `roll_std_4w` | 4-week rolling std | Volatility regime |
| `momentum_ratio` | roll_mean_4w / roll_mean_26w | Trend acceleration |
| `week_of_year` | 1–52 | Annual seasonality |
| `month` | 1–12 | Monthly pattern |
| `quarter` | 1–4 | Quarterly split |
| `is_holiday` | US federal holiday in week | Demand spike |
| `is_peak_week` | Thanksgiving / Christmas / July4 | Retail peaks |
| `sin_week / cos_week` | Fourier terms | Smooth cyclical encoding |
| `trend` | Days since series start | Linear trend |

All lag/rolling features shift by ≥1 to prevent data leakage.

---

## Model selection logic

```
Best model = argmin(MAPE) over {SARIMA, Prophet, XGBoost, LSTM}
             evaluated on the last 10 weeks (validation set)
             using strict chronological split — no random shuffling
```

---

## Docker deployment

```bash
# Build
docker build -t forecasting-api .

# Train (mount your data)
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/registry:/app/registry \
    forecasting-api python train.py --data_path data/clean_weekly.csv

# Serve
docker run -p 8000:8000 \
    -e FORECAST_API_KEY=your-key \
    -v $(pwd)/registry:/app/registry \
    forecasting-api
```

---

## Tech stack

| Layer | Library |
|---|---|
| SARIMA | `pmdarima` (auto-ARIMA) |
| Prophet | `prophet` (Meta / Facebook) |
| XGBoost | `xgboost` |
| LSTM | `tensorflow` / Keras |
| API | `FastAPI` + `uvicorn` |
| Validation | `pydantic` v2 |
| Feature eng. | `pandas`, `numpy`, `holidays` |
| Serialisation | `joblib` |
| Container | Docker |
