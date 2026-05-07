from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from feature_engineering import build_features, train_val_split
from metrics import compare_models

from sarima_model import SARIMAModel
from prophet_model import ProphetModel
from xgboost_model import XGBoostModel
from lstm_model import LSTMModel

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)

HORIZON_WEEKS = 8
VAL_WEEKS = 10


# =========================================================
# DATA LOADING
# =========================================================

def load_clean_data(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    # Fix dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Fix numeric column
    df["Total"] = (
        df["Total"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=["Date", "Total", "State"])

    # Sort properly
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)

    log.info(f"Loaded {len(df):,} rows · {df['State'].nunique()} states")

    return df


# =========================================================
# STATE SERIES
# =========================================================

def _series_for_state(df: pd.DataFrame, state: str) -> pd.Series:

    sub = df[df["State"] == state][["Date", "Total"]].copy()

    sub = sub.set_index("Date").sort_index()

    series = sub["Total"].astype(float)

    # Safety cleanup
    series = series.replace([np.inf, -np.inf], np.nan)
    series = series.dropna()

    return series


# =========================================================
# FUTURE DATES
# =========================================================

def _future_dates(last_date: pd.Timestamp, n: int = HORIZON_WEEKS):

    dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1),
        periods=n,
        freq="W-MON",
    )

    return [str(d.date()) for d in dates]


# =========================================================
# TRAIN SINGLE STATE
# =========================================================

def train_state(
    state: str,
    series: pd.Series,
    output_dir: Path,
    val_weeks: int = VAL_WEEKS,
) -> Dict:

    log.info(f"[{state}] {len(series)} weeks of data")

    state_dir = output_dir / state.replace(" ", "_")
    state_dir.mkdir(parents=True, exist_ok=True)

    train_raw = series.iloc[:-val_weeks]
    val_raw = series.iloc[-val_weeks:]

    feat_df = build_features(
        series.to_frame("Total"),
        target="Total",
        log_transform=True,
    )

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.bfill().ffill().fillna(0)

    train_feat, val_feat = train_val_split(
        feat_df,
        val_weeks=val_weeks,
    )

    metrics = {}
    saved_paths = {}

    # =====================================================
    # SARIMA
    # =====================================================

    try:

        t0 = time.time()

        model = SARIMAModel()

        model.fit(train_raw, log_transform=True)

        m = model.validate(val_raw)

        metrics["sarima"] = m

        path = str(state_dir / "sarima.joblib")

        model.save(path)

        saved_paths["sarima"] = path

        log.info(
            f"[{state}] SARIMA "
            f"MAPE={m['mape']:.2f}% "
            f"({time.time()-t0:.1f}s)"
        )

    except Exception as e:
        log.warning(f"[{state}] SARIMA failed: {e}")

    # =====================================================
    # PROPHET
    # =====================================================

    try:

        t0 = time.time()

        model = ProphetModel()

        model.fit(train_raw, log_transform=True)

        m = model.validate(val_raw)

        metrics["prophet"] = m

        path = str(state_dir / "prophet.joblib")

        model.save(path)

        saved_paths["prophet"] = path

        log.info(
            f"[{state}] Prophet "
            f"MAPE={m['mape']:.2f}% "
            f"({time.time()-t0:.1f}s)"
        )

    except Exception as e:
        log.warning(f"[{state}] Prophet failed: {e}")

    # =====================================================
    # XGBOOST
    # =====================================================

    try:

        t0 = time.time()

        model = XGBoostModel()

        model.fit(
            train_feat,
            target="Total",
            log_transform=True,
            val_df=val_feat,
        )

        m = model.validate(
            val_feat,
            target="Total",
        )

        metrics["xgboost"] = m

        path = str(state_dir / "xgboost.joblib")

        model.save(path)

        saved_paths["xgboost"] = path

        log.info(
            f"[{state}] XGBoost "
            f"MAPE={m['mape']:.2f}% "
            f"({time.time()-t0:.1f}s)"
        )

    except Exception as e:
        log.warning(f"[{state}] XGBoost failed: {e}")

    # =====================================================
    # LSTM
    # =====================================================

    try:

        t0 = time.time()

        model = LSTMModel()

        model.fit(
            train_raw,
            log_transform=True,
            val_series=val_raw,
        )

        m = model.validate(val_raw)

        metrics["lstm"] = m

        path = str(state_dir / "lstm.joblib")

        model.save(path)

        saved_paths["lstm"] = path

        log.info(
            f"[{state}] LSTM "
            f"MAPE={m['mape']:.2f}% "
            f"({time.time()-t0:.1f}s)"
        )

    except Exception as e:
        log.warning(f"[{state}] LSTM failed: {e}")

    if not metrics:
        raise RuntimeError(f"All models failed for state: {state}")

    best_name = compare_models(metrics)

    last_date = series.index.max()

    return {
        "state": state,
        "best_model": best_name,
        "model_paths": saved_paths,
        "metrics": metrics,
        "last_train_date": str(last_date.date()),
        "forecast_start": str((last_date + pd.Timedelta(weeks=1)).date()),
        "n_train_weeks": len(train_raw),
        "n_val_weeks": val_weeks,
        "future_dates": _future_dates(last_date),
    }


# =========================================================
# FULL TRAINING
# =========================================================

def run_training(
    data_path: str,
    output_dir: str = "registry",
    val_weeks: int = VAL_WEEKS,
    states: Optional[list[str]] = None,
):

    df = load_clean_data(data_path)

    out = Path(output_dir)

    out.mkdir(parents=True, exist_ok=True)

    all_states = states or sorted(df["State"].unique())

    registry = {}

    log.info(f"Training {len(all_states)} states")

    for state in all_states:

        try:

            series = _series_for_state(df, state)

            if len(series) < val_weeks + 15:
                log.warning(f"[{state}] Too few rows")
                continue

            meta = train_state(
                state,
                series,
                out,
                val_weeks,
            )

            registry[state] = meta

        except Exception as e:

            log.error(f"[{state}] FAILED: {e}")

    registry_path = out / "registry.json"

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    log.info(f"Registry saved → {registry_path}")

    # =====================================================
    # SUMMARY
    # =====================================================

    rows = []

    for state, meta in registry.items():

        best = meta["best_model"]

        rows.append({
            "State": state,
            "Best Model": best.upper(),
            "MAPE (%)": meta["metrics"][best]["mape"],
            "RMSE": meta["metrics"][best]["rmse"],
        })

    summary = pd.DataFrame(rows)

    if not summary.empty:

        summary = summary.sort_values("MAPE (%)")

        print("\nModel Selection Summary\n")

        print(summary.to_string(index=False))

    else:

        print("\nNo models trained successfully.\n")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default="data/clean_weekly.csv",
    )

    parser.add_argument(
        "--output_dir",
        default="registry",
    )

    parser.add_argument(
        "--val_weeks",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--states",
        nargs="*",
    )

    args = parser.parse_args()

    run_training(
        data_path=args.data_path,
        output_dir=args.output_dir,
        val_weeks=args.val_weeks,
        states=args.states,
    )