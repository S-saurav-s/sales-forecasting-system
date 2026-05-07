"""
Evaluation utilities
---------------------
All metrics work on raw (unlogged) values.
"""

import numpy as np
import pandas as pd
from typing import Dict


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%). Returns 9999 if y_true has zeros."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return 9999.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> Dict[str, float]:
    """Return dict with MAPE, RMSE, MAE for a single model."""
    return {
        "model": model_name,
        "mape":  round(mape(y_true, y_pred), 4),
        "rmse":  round(rmse(y_true, y_pred), 2),
        "mae":   round(mae(y_true, y_pred), 2),
    }


def compare_models(results: Dict[str, Dict]) -> str:
    """
    Given {model_name: metrics_dict}, return name of best model by MAPE.
    Ties broken by RMSE.
    """
    df = pd.DataFrame(results).T
    df = df.sort_values(["mape", "rmse"])
    return df.index[0]
