"""
XGBoost Forecasting Model
--------------------------
Gradient-boosted trees on engineered lag + calendar features.
Uses recursive multi-step forecasting for 8-week horizon:
  - Predict week t+1, append to history, predict t+2, etc.

Input  : feature-engineered DataFrame from feature_engineering.build_features()
Output : 8-week forecast array + validation metrics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from typing import Optional

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False

from feature_engineering import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    FEATURE_COLS,
)
from metrics import evaluate


class XGBoostModel:
    MODEL_NAME = "xgboost"

    def __init__(self):
        self.model_: Optional[XGBRegressor] = None
        self.log_transform_ = True
        self.feature_cols_: list[str] = []
        self.train_tail_: Optional[pd.Series] = None   # last 30 rows for recursive forecasting

    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        target: str = "Total",
        log_transform: bool = True,
        n_estimators: int = 400,
        max_depth: int = 5,
        learning_rate: float = 0.04,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 30,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "XGBoostModel":
        """
        Fit XGBoost regressor.

        Parameters
        ----------
        train_df  : feature-engineered training DataFrame (from build_features)
        target    : name of target column (already log-transformed in build_features)
        val_df    : optional validation set for early stopping
        """
        if not XGB_OK:
            raise ImportError("Install xgboost: pip install xgboost")

        self.log_transform_ = log_transform
        # Keep only features that exist in this DataFrame
        self.feature_cols_ = [c for c in FEATURE_COLS if c in train_df.columns]

        X_train = train_df[self.feature_cols_]
        y_train = train_df[target]

        eval_set = None
        if val_df is not None and len(val_df) > 0:
            X_val = val_df[self.feature_cols_]
            y_val = val_df[target]
            eval_set = [(X_val, y_val)]

        self.model_ = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            tree_method="hist",          # fast histogram method
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            eval_metric="rmse",
            verbosity=0,
            random_state=42,
        )
        self.model_.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Cache tail of training series (raw Target values, pre-log) for recursive forecasting
        self.train_tail_ = train_df[target].tail(30).copy()
        return self

    # ------------------------------------------------------------------
    def _build_one_row(
        self, history: pd.Series, date: pd.Timestamp
    ) -> pd.DataFrame:
        """Build a single-row feature vector for `date` using `history`."""
        row = pd.DataFrame({"Total": [np.nan]}, index=[date])
        temp = pd.concat([history.to_frame("Total"), row])

        temp = add_calendar_features(temp)
        temp = add_lag_features(temp, "Total")
        temp = add_rolling_features(temp, "Total")
        # Return only the last row (the new date)
        return temp.iloc[[-1]]

    # ------------------------------------------------------------------
    def predict(self, n_periods: int = 8) -> np.ndarray:
        """
        Recursive multi-step forecast.
        Each predicted value is appended to history before the next step.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")

        history = self.train_tail_.copy()
        last_date = history.index.max()
        preds = []

        for i in range(n_periods):
            next_date = last_date + pd.Timedelta(weeks=i + 1)
            row = self._build_one_row(history, next_date)

            feat_cols = [c for c in self.feature_cols_ if c in row.columns]
            # Fill any missing feature columns with 0
            for c in self.feature_cols_:
                if c not in row.columns:
                    row[c] = 0.0

            pred_log = self.model_.predict(row[self.feature_cols_])[0]
            preds.append(pred_log)

            # Append predicted value to history for next iteration
            new_entry = pd.Series(
                [pred_log], index=[next_date], name="Total"
            )
            history = pd.concat([history, new_entry])

        preds = np.array(preds)
        if self.log_transform_:
            preds = np.expm1(preds)
        return np.maximum(preds, 0)

    # ------------------------------------------------------------------
    def predict_with_ci(
        self, n_periods: int = 8, ci_pct: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        XGBoost has no native CI — approximate with ±5% of point forecast.
        For production, replace with quantile regression or conformal prediction.
        """
        preds = self.predict(n_periods)
        margin = preds * ci_pct * 2
        return preds, np.maximum(preds - margin, 0), preds + margin

    # ------------------------------------------------------------------
    def validate(self, val_df: pd.DataFrame, target: str = "Total") -> dict:
        """Evaluate on pre-built validation feature DataFrame."""
        feat_cols = [c for c in self.feature_cols_ if c in val_df.columns]
        X_val = val_df[feat_cols]
        y_val_log = val_df[target].values
        preds_log = self.model_.predict(X_val)

        if self.log_transform_:
            y_val = np.expm1(y_val_log)
            preds = np.expm1(preds_log)
        else:
            y_val, preds = y_val_log, preds_log

        return evaluate(y_val, np.maximum(preds, 0), model_name=self.MODEL_NAME)

    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.Series:
        if self.model_ is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self.model_.feature_importances_,
            index=self.feature_cols_,
        ).sort_values(ascending=False)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "XGBoostModel":
        return joblib.load(path)
