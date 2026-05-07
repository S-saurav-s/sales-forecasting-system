"""
Feature Engineering Pipeline
------------------------------
Input  : clean weekly DataFrame per state with columns [Date, State, Total]
         (Date = Monday, no gaps, no nulls — cleaning already done upstream)
Output : feature-enriched DataFrame ready for XGBoost / LSTM / tree models
         SARIMA and Prophet consume the raw series directly (no lag features needed)
"""

import numpy as np
import pandas as pd
import holidays


US_HOLIDAYS = holidays.UnitedStates()

# Major US retail peak weeks (month, week-of-month approximate)
PEAK_WEEKS = {
    (11, 4): "thanksgiving",   # last week Nov
    (12, 3): "christmas_pre",  # 3rd week Dec
    (12, 4): "christmas_pre",
    (7,  1): "july4",
    (1,  1): "new_year",
}


def _holiday_flag(date: pd.Timestamp) -> int:
    """1 if any US federal holiday falls in the Mon–Sun week of `date`."""
    week_days = pd.date_range(date, periods=7, freq="D")
    return int(any(d.date() in US_HOLIDAYS for d in week_days))


def _peak_flag(date: pd.Timestamp) -> int:
    """1 for high-retail weeks (Thanksgiving, Christmas build-up, July 4th, New Year)."""
    key = (date.month, (date.day - 1) // 7 + 1)
    return int(key in PEAK_WEEKS)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time/calendar features.
    Expects df.index to be a DatetimeIndex (weekly Mon frequency).
    """
    df = df.copy()
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["month"]        = df.index.month
    df["quarter"]      = df.index.quarter
    df["year"]         = df.index.year
    df["day_of_week"]  = df.index.dayofweek          # always 0 (Mon) for weekly
    df["trend"]        = (df.index - df.index.min()).days  # numeric time index

    df["is_holiday"]   = df.index.map(_holiday_flag)
    df["is_peak_week"] = df.index.map(_peak_flag)

    # Fourier terms for annual seasonality (better than raw week_of_year for linear models)
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    return df


def add_lag_features(df: pd.DataFrame, target: str = "Total") -> pd.DataFrame:
    """
    Lag features: t-1 (prev week), t-7 (7 weeks ago ~quarter back), t-30 (30 weeks ~half-year).
    All lags are shifted to avoid data leakage (shift(1) minimum).
    """
    df = df.copy()
    for lag in [1, 7, 30]:
        df[f"lag_{lag}"] = df[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "Total") -> pd.DataFrame:
    """
    Rolling statistics computed on already-shifted series (no leakage).
    Windows: 4 weeks (short-term), 12 weeks (medium-term), 26 weeks (long-term).
    """
    df = df.copy()
    shifted = df[target].shift(1)   # shift before rolling to prevent leakage

    for window in [4, 12, 26]:
        df[f"roll_mean_{window}w"] = shifted.rolling(window, min_periods=2).mean()
        df[f"roll_std_{window}w"]  = shifted.rolling(window, min_periods=2).std()

    df["roll_min_4w"] = shifted.rolling(4, min_periods=2).min()
    df["roll_max_4w"] = shifted.rolling(4, min_periods=2).max()

    # Momentum: ratio of short vs long rolling mean
    df["momentum_ratio"] = df["roll_mean_4w"] / (df["roll_mean_26w"] + 1e-9)

    return df


def build_features(
    df: pd.DataFrame,
    target: str = "Total",
    log_transform: bool = True,
) -> pd.DataFrame:
    """
    Full feature pipeline.

    Parameters
    ----------
    df           : DataFrame with DatetimeIndex and at least a `target` column
    target       : name of the sales column
    log_transform: apply log1p to target before building lag features (recommended)

    Returns
    -------
    feature-rich DataFrame; NaN rows from lag/rolling windows are dropped.
    """
    df = df.copy()

    if log_transform:
        df[target] = np.log1p(df[target])

    df = add_calendar_features(df)
    df = add_lag_features(df, target)
    df = add_rolling_features(df, target)

    # Drop rows where lag / rolling windows couldn't be computed
    df.dropna(inplace=True)
    return df


def train_val_split(
    df: pd.DataFrame,
    val_weeks: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict chronological split — NO shuffling, NO random_state.
    val_weeks : number of trailing weeks held out for validation.
    """
    if len(df) <= val_weeks:
        raise ValueError(
            f"Series too short ({len(df)} rows) for val_weeks={val_weeks}"
        )
    return df.iloc[:-val_weeks], df.iloc[-val_weeks:]


FEATURE_COLS = [
    "lag_1", "lag_7", "lag_30",
    "roll_mean_4w", "roll_mean_12w", "roll_mean_26w",
    "roll_std_4w",  "roll_std_12w",  "roll_std_26w",
    "roll_min_4w",  "roll_max_4w",   "momentum_ratio",
    "week_of_year", "month", "quarter", "year", "trend",
    "is_holiday", "is_peak_week",
    "sin_week", "cos_week",
]
