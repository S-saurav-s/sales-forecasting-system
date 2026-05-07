"""
SARIMA Model
-------------
Uses pmdarima.auto_arima to find optimal (p,d,q)(P,D,Q,m=52) orders.
Seasonal period m=52 for weekly data (annual cycle).

Input  : raw (unlogged) weekly pd.Series with DatetimeIndex
Output : 8-week forecast array + validation metrics
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from pmdarima import auto_arima
    PMDARIMA_OK = True
except ImportError:
    PMDARIMA_OK = False


class SARIMAModel:
    MODEL_NAME = "sarima"

    def __init__(self, seasonal_period: int = 52):
        self.m = seasonal_period
        self.model_ = None
        self.fitted_ = None
        self.log_transform_ = True

    # ------------------------------------------------------------------
    def fit(self, series: pd.Series, log_transform: bool = True) -> "SARIMAModel":
        """
        Fit auto-ARIMA on the training portion of `series`.

        Parameters
        ----------
        series        : raw weekly sales values
        log_transform : log1p-transform before fitting (stabilises variance)
        """
        if not PMDARIMA_OK:
            raise ImportError("Install pmdarima: pip install pmdarima")

        self.log_transform_ = log_transform
        y = np.log1p(series.values) if log_transform else series.values

        self.model_ = auto_arima(
            y,
            seasonal=True,
            m=self.m,
            stepwise=True,          # faster grid search
            information_criterion="aic",
            max_p=3, max_q=3,
            max_P=2, max_Q=2,
            max_d=2, max_D=1,
            error_action="ignore",
            suppress_warnings=True,
        )
        self.fitted_ = self.model_
        return self

    # ------------------------------------------------------------------
    def predict(self, n_periods: int = 8) -> np.ndarray:
        """Return n_periods-step ahead forecast in original scale."""
        if self.fitted_ is None:
            raise RuntimeError("Call fit() before predict().")
        preds = self.fitted_.predict(n_periods=n_periods)
        if self.log_transform_:
            preds = np.expm1(preds)
        return np.maximum(preds, 0)  # clip negatives

    # ------------------------------------------------------------------
    def predict_with_ci(
        self, n_periods: int = 8, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (forecast, lower_95, upper_95)."""
        preds, conf = self.fitted_.predict(
            n_periods=n_periods, return_conf_int=True, alpha=alpha
        )
        if self.log_transform_:
            preds = np.expm1(preds)
            conf  = np.expm1(conf)
        preds = np.maximum(preds, 0)
        lower = np.maximum(conf[:, 0], 0)
        upper = conf[:, 1]
        return preds, lower, upper

    # ------------------------------------------------------------------
    def validate(self, val_series: pd.Series) -> dict:
        """One-step-ahead rolling forecast over validation window."""
        from metrics import evaluate
        preds = self.predict(n_periods=len(val_series))
        return evaluate(val_series.values, preds, model_name=self.MODEL_NAME)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SARIMAModel":
        return joblib.load(path)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        if self.fitted_:
            return str(self.fitted_.summary())
        return "Model not fitted yet."
