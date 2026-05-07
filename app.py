"""
Forecasting REST API
---------------------
FastAPI service that loads trained models from registry and serves predictions.

Endpoints:
  GET  /health                      → service health + loaded states
  GET  /states                      → list all available states with best model
  POST /forecast                    → forecast 1..N states for next 8 weeks
  GET  /forecast/{state}            → cached latest forecast for one state
  GET  /models/{state}              → validation metrics for all models of a state
  POST /retrain                     → trigger background retraining job

Run locally:
  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Model imports ──────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sarima_model  import SARIMAModel
from prophet_model import ProphetModel
from xgboost_model import XGBoostModel
from lstm_model    import LSTMModel

log = logging.getLogger("uvicorn.error")

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

REGISTRY_DIR  = Path(os.getenv("REGISTRY_DIR", "registry"))
API_KEY       = os.getenv("FORECAST_API_KEY", "dev-secret-key-change-in-prod")
HORIZON_WEEKS = 8

MODEL_LOADERS = {
    "sarima":   SARIMAModel.load,
    "prophet":  ProphetModel.load,
    "xgboost":  XGBoostModel.load,
    "lstm":     LSTMModel.load,
}


# ─────────────────────────────────────────────────────────────────
# Registry loader (in-memory cache)
# ─────────────────────────────────────────────────────────────────

class ModelRegistry:
    """Loads and caches the best model per state from disk."""

    def __init__(self):
        self._meta:   Dict[str, Dict] = {}    # registry.json contents
        self._models: Dict[str, Any]  = {}    # {state: loaded model object}

    def load_registry(self, registry_dir: Path) -> None:
        reg_path = registry_dir / "registry.json"
        if not reg_path.exists():
            log.warning(f"Registry file not found at {reg_path}. Run train.py first.")
            return
        with open(reg_path) as f:
            self._meta = json.load(f)
        log.info(f"Registry loaded: {len(self._meta)} states")

    def get_model(self, state: str) -> Any:
        """Lazy-load model on first request for that state."""
        if state not in self._meta:
            raise KeyError(f"State '{state}' not in registry.")
        if state not in self._models:
            best    = self._meta[state]["best_model"]
            path    = self._meta[state]["model_paths"][best]
            loader  = MODEL_LOADERS[best]
            self._models[state] = loader(path)
            log.info(f"Loaded {best.upper()} for {state}")
        return self._models[state]

    def get_meta(self, state: str) -> Dict:
        if state not in self._meta:
            raise KeyError(f"State '{state}' not in registry.")
        return self._meta[state]

    @property
    def states(self) -> List[str]:
        return sorted(self._meta.keys())

    def reload(self, registry_dir: Path) -> None:
        self._models.clear()
        self.load_registry(registry_dir)


registry = ModelRegistry()


# ─────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sales Forecasting API",
    description="8-week state-level beverage sales forecasting — SARIMA / Prophet / XGBoost / LSTM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    registry.load_registry(REGISTRY_DIR)


# ─────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(default="test")):
    return x_api_key


# ─────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    states: List[str] = Field(..., example=["California", "Texas"])
    horizon_weeks: int = Field(default=8, ge=1, le=52)

    @field_validator("states")
    @classmethod
    def states_not_empty(cls, v):
        if not v:
            raise ValueError("At least one state required.")
        return v


class WeekForecast(BaseModel):
    week:         int
    date:         str
    predicted:    float
    lower_95:     float
    upper_95:     float
    delta_ow_pct: Optional[float]


class ModelMetrics(BaseModel):
    mape: float
    rmse: float
    mae:  float


class StateForecast(BaseModel):
    state:          str
    best_model:     str
    generated_at:   str
    horizon_weeks:  int
    last_train_date: str
    forecast:       List[WeekForecast]
    model_metrics:  ModelMetrics


class StateInfo(BaseModel):
    state:           str
    best_model:      str
    mape:            float
    last_train_date: str
    n_train_weeks:   int


class RetrainRequest(BaseModel):
    states: Optional[List[str]] = None   # None = all states


class RetrainResponse(BaseModel):
    job_id: str
    states: List[str]
    status: str


# ─────────────────────────────────────────────────────────────────
# Forecast builder
# ─────────────────────────────────────────────────────────────────

def _build_forecast(state: str, horizon: int) -> StateForecast:
    meta  = registry.get_meta(state)
    model = registry.get_model(state)
    best  = meta["best_model"]

    # Generate forecast + CI
    try:
        preds, lower, upper = model.predict_with_ci(n_periods=horizon)
    except AttributeError:
        preds = model.predict(n_periods=horizon)
        lower = preds * 0.95
        upper = preds * 1.05

    future_dates = pd.date_range(
        start=pd.Timestamp(meta["forecast_start"]),
        periods=horizon,
        freq="W-MON",
    )

    weeks = []
    for i, (d, p, lo, hi) in enumerate(zip(future_dates, preds, lower, upper)):
        prev = float(preds[i - 1]) if i > 0 else None
        dow  = round((p - prev) / prev * 100, 2) if prev else None
        weeks.append(WeekForecast(
            week=i + 1,
            date=str(d.date()),
            predicted=round(float(p), 2),
            lower_95=round(float(lo), 2),
            upper_95=round(float(hi), 2),
            delta_ow_pct=dow,
        ))

    best_metrics = meta["metrics"][best]
    return StateForecast(
        state=state,
        best_model=best,
        generated_at=datetime.now(timezone.utc).isoformat(),
        horizon_weeks=horizon,
        last_train_date=meta["last_train_date"],
        forecast=weeks,
        model_metrics=ModelMetrics(
            mape=best_metrics["mape"],
            rmse=best_metrics["rmse"],
            mae=best_metrics["mae"],
        ),
    )


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "states_loaded": len(registry.states),
        "registry_dir": str(REGISTRY_DIR),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/states", response_model=List[StateInfo], tags=["States"])
def list_states(_: str = Depends(verify_api_key)):
    """Return all states with their best model and key metrics."""
    out = []
    for state in registry.states:
        meta = registry.get_meta(state)
        best = meta["best_model"]
        out.append(StateInfo(
            state=state,
            best_model=best,
            mape=meta["metrics"][best]["mape"],
            last_train_date=meta["last_train_date"],
            n_train_weeks=meta["n_train_weeks"],
        ))
    return out


@app.post("/forecast", response_model=List[StateForecast], tags=["Forecast"])
def forecast_batch(
    req: ForecastRequest,
    _: str = Depends(verify_api_key),
):
    """
    Generate 8-week (or custom horizon) forecasts for one or more states.
    Uses the best model per state as determined during training.
    """
    results = []
    errors  = []
    for state in req.states:
        if state not in registry.states:
            errors.append(f"'{state}' not found in registry.")
            continue
        try:
            results.append(_build_forecast(state, req.horizon_weeks))
        except Exception as e:
            errors.append(f"'{state}': {str(e)}")

    if errors and not results:
        raise HTTPException(status_code=404, detail="; ".join(errors))
    if errors:
        log.warning("Partial errors: " + "; ".join(errors))
    return results


@app.get("/forecast/{state}", response_model=StateForecast, tags=["Forecast"])
def forecast_single(state: str, _: str = Depends(verify_api_key)):
    """Get 8-week forecast for a single state."""
    if state not in registry.states:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")
    return _build_forecast(state, HORIZON_WEEKS)


@app.get("/models/{state}", tags=["Models"])
def get_model_metrics(state: str, _: str = Depends(verify_api_key)):
    """Return validation metrics for ALL models trained on this state."""
    if state not in registry.states:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")
    meta = registry.get_meta(state)
    return {
        "state":      state,
        "best_model": meta["best_model"],
        "metrics":    meta["metrics"],
    }


@app.post("/retrain", response_model=RetrainResponse, tags=["System"])
def retrain(
    req: RetrainRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_api_key),
):
    """
    Trigger background retraining.
    Runs train.py as a subprocess — in production, replace with Celery task.
    """
    target_states = req.states or registry.states
    job_id = str(uuid.uuid4())[:8]

    def _run_retrain(states: List[str], jid: str):
        log.info(f"[job {jid}] Retraining {len(states)} states ...")
        cmd = [
            "python", "train.py",
            "--data_path", "data/clean_weekly.csv",
            "--output_dir", str(REGISTRY_DIR),
        ]
        if states != registry.states:
            cmd += ["--states"] + states
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            registry.reload(REGISTRY_DIR)
            log.info(f"[job {jid}] Retrain complete.")
        else:
            log.error(f"[job {jid}] Retrain failed:\n{result.stderr}")

    background_tasks.add_task(_run_retrain, target_states, job_id)
    return RetrainResponse(
        job_id=job_id,
        states=target_states,
        status="queued",
    )


# ─────────────────────────────────────────────────────────────────
# Dev runner
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
