"""
Facebook Prophet Model
-----------------------
Additive decomposition: trend + yearly seasonality + weekly seasonality + US holidays.
Prophet expects a DataFrame with columns [ds, y].

Input  : raw (unlogged) weekly pd.Series with DatetimeIndex
Output : 8-week forecast DataFrame + validation metrics
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False


class ProphetModel:
    MODEL_NAME = "prophet"

    def __init__(self):
        self.model_: Prophet | None = None
        self.log_transform_ = True
        self.last_train_date_: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    def _to_prophet_df(self, series: pd.Series) -> pd.DataFrame:
        df = series.reset_index()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        if self.log_transform_:
            df["y"] = np.log1p(df["y"])
        return df

    # ------------------------------------------------------------------
    def fit(self, series: pd.Series, log_transform: bool = True) -> "ProphetModel":
        """
        Fit Prophet on training series.

        Parameters
        ----------
        series        : raw weekly sales (DatetimeIndex)
        log_transform : log1p-transform y before fitting
        """
        if not PROPHET_OK:
            raise ImportError("Install prophet: pip install prophet")

        self.log_transform_ = log_transform
        self.last_train_date_ = series.index.max()

        prophet_df = self._to_prophet_df(series)

        self.model_ = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,   # weekly data — no intra-week pattern
            daily_seasonality=False,
            seasonality_mode="multiplicative",   # handles growing amplitude
            changepoint_prior_scale=0.1,         # flexibility of trend
            seasonality_prior_scale=10.0,
            interval_width=0.95,
        )
        self.model_.add_country_holidays(country_name="US")
        # Custom monthly seasonality (Fourier order 3)
        self.model_.add_seasonality(
            name="monthly", period=30.5, fourier_order=3
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(prophet_df)

        return self

    # ------------------------------------------------------------------
    def _make_future(self, n_periods: int) -> pd.DataFrame:
        """Build future DataFrame starting the week after last training date."""
        future_dates = pd.date_range(
            start=self.last_train_date_ + pd.Timedelta(weeks=1),
            periods=n_periods,
            freq="W-MON",
        )
        return pd.DataFrame({"ds": future_dates})

    # ------------------------------------------------------------------
    def predict(self, n_periods: int = 8) -> np.ndarray:
        """Return point forecasts in original scale."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")
        future = self._make_future(n_periods)
        forecast = self.model_.predict(future)
        yhat = forecast["yhat"].values
        if self.log_transform_:
            yhat = np.expm1(yhat)
        return np.maximum(yhat, 0)

    # ------------------------------------------------------------------
    def predict_with_ci(
        self, n_periods: int = 8
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (forecast, lower_95, upper_95) in original scale."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")
        future = self.model_.make_future_dataframe(
            periods=n_periods, freq="W-MON", include_history=False
        )
        forecast = self.model_.predict(future)
        yhat  = forecast["yhat"].values
        lower = forecast["yhat_lower"].values
        upper = forecast["yhat_upper"].values
        if self.log_transform_:
            yhat  = np.expm1(yhat)
            lower = np.expm1(lower)
            upper = np.expm1(upper)
        return np.maximum(yhat, 0), np.maximum(lower, 0), upper

    # ------------------------------------------------------------------
    def validate(self, val_series: pd.Series) -> dict:
        from metrics import evaluate

        val_dates = val_series.index
        future_df = pd.DataFrame({"ds": val_dates})
        forecast  = self.model_.predict(future_df)
        preds = forecast["yhat"].values
        if self.log_transform_:
            preds = np.expm1(preds)
        preds = np.maximum(preds, 0)
        return evaluate(val_series.values, preds, model_name=self.MODEL_NAME)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ProphetModel":
        return joblib.load(path)
