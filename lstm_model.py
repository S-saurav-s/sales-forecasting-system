"""
LSTM Forecasting Model  (PyTorch backend)
------------------------------------------
TensorFlow does not support Python 3.13 as of 2024.
PyTorch 2.4+ has full 3.13 wheels, so we use it here instead.

Architecture : 2-layer stacked LSTM → Linear head
Training     : Huber loss, Adam optimiser, early stopping on val loss
Multi-step   : Recursive prediction (predict t+1, feed back, predict t+2 ...)
CI           : Monte Carlo Dropout (50 forward passes with dropout active)

Input  : raw weekly pd.Series with DatetimeIndex
Output : 8-week forecast array + validation metrics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from typing import Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

from metrics import evaluate


SEQ_LEN      = 12
LSTM_HIDDEN1 = 64
LSTM_HIDDEN2 = 32
DROPOUT_RATE = 0.2
BATCH_SIZE   = 16
MAX_EPOCHS   = 100
PATIENCE     = 15
LR           = 1e-3


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int = 1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, LSTM_HIDDEN1, batch_first=True)
        self.drop1 = nn.Dropout(DROPOUT_RATE)
        self.lstm2 = nn.LSTM(LSTM_HIDDEN1, LSTM_HIDDEN2, batch_first=True)
        self.drop2 = nn.Dropout(DROPOUT_RATE)
        self.fc1   = nn.Linear(LSTM_HIDDEN2, 16)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


def _make_sequences(values: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMModel:
    MODEL_NAME = "lstm"

    def __init__(self, seq_len: int = SEQ_LEN):
        self.seq_len        = seq_len
        self.net_: Optional[_LSTMNet] = None
        self.log_transform_ = True
        self.scale_mean_    = 0.0
        self.scale_std_     = 1.0
        self.train_tail_: Optional[np.ndarray] = None
        self.device_        = "cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu"

    def fit(self, series: pd.Series, log_transform: bool = True,
            val_series: Optional[pd.Series] = None) -> "LSTMModel":
        if not TORCH_OK:
            raise ImportError("Install pytorch: pip install torch")

        self.log_transform_ = log_transform
        values = np.log1p(series.values) if log_transform else series.values.astype(np.float32)

        self.scale_mean_ = float(values.mean())
        self.scale_std_  = float(values.std()) + 1e-9
        normed = ((values - self.scale_mean_) / self.scale_std_).astype(np.float32)

        X_tr, y_tr = _make_sequences(normed, self.seq_len)
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_tr).unsqueeze(-1).to(self.device_),
                torch.tensor(y_tr).to(self.device_),
            ),
            batch_size=BATCH_SIZE, shuffle=True,
        )

        val_loader = None
        if val_series is not None and len(val_series) > 0:
            val_values = np.log1p(val_series.values) if log_transform else val_series.values.astype(np.float32)
            val_normed = ((val_values - self.scale_mean_) / self.scale_std_).astype(np.float32)
            combined   = np.concatenate([normed[-self.seq_len:], val_normed])
            X_v, y_v   = _make_sequences(combined, self.seq_len)
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_v).unsqueeze(-1).to(self.device_),
                    torch.tensor(y_v).to(self.device_),
                ),
                batch_size=BATCH_SIZE,
            )

        self.net_ = _LSTMNet().to(self.device_)
        optimiser  = torch.optim.Adam(self.net_.parameters(), lr=LR)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=7, factor=0.5)
        criterion  = nn.HuberLoss()
        best_val   = float("inf")
        patience_c = 0
        best_state = None

        for _ in range(MAX_EPOCHS):
            self.net_.train()
            for xb, yb in train_loader:
                optimiser.zero_grad()
                criterion(self.net_(xb), yb).backward()
                optimiser.step()

            if val_loader:
                self.net_.eval()
                vl = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        vl.append(criterion(self.net_(xb), yb).item())
                val_loss = float(np.mean(vl))
                scheduler.step(val_loss)
                if val_loss < best_val:
                    best_val   = val_loss
                    patience_c = 0
                    best_state = {k: v.cpu().clone() for k, v in self.net_.state_dict().items()}
                else:
                    patience_c += 1
                    if patience_c >= PATIENCE:
                        break

        if best_state:
            self.net_.load_state_dict(best_state)
        self.train_tail_ = normed[-self.seq_len:].copy()
        return self

    def _run(self, n_periods: int, training_mode: bool = False) -> np.ndarray:
        self.net_.train(training_mode)
        seq = self.train_tail_.tolist()
        preds = []
        with torch.no_grad():
            for _ in range(n_periods):
                x = torch.tensor(
                    np.array(seq[-self.seq_len:], dtype=np.float32)
                ).unsqueeze(0).unsqueeze(-1).to(self.device_)
                p = self.net_(x).item()
                preds.append(p)
                seq.append(p)
        return np.array(preds)

    def predict(self, n_periods: int = 8) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("Call fit() before predict().")
        p = self._run(n_periods) * self.scale_std_ + self.scale_mean_
        return np.maximum(np.expm1(p) if self.log_transform_ else p, 0)

    def predict_with_ci(self, n_periods: int = 8, n_samples: int = 50):
        if self.net_ is None:
            raise RuntimeError("Call fit() before predict().")
        runs = []
        for _ in range(n_samples):
            r = self._run(n_periods, training_mode=True) * self.scale_std_ + self.scale_mean_
            runs.append(np.maximum(np.expm1(r) if self.log_transform_ else r, 0))
        runs = np.array(runs)
        return np.mean(runs, 0), np.percentile(runs, 2.5, 0), np.percentile(runs, 97.5, 0)

    def validate(self, val_series: pd.Series) -> dict:
        return evaluate(val_series.values, self.predict(len(val_series)), model_name=self.MODEL_NAME)

    def save(self, path: str) -> None:
        state = self.net_.state_dict() if self.net_ else None
        self.net_ = None
        joblib.dump({"wrapper": self, "net_state": state}, path)
        if state:
            self.net_ = _LSTMNet().to(self.device_)
            self.net_.load_state_dict(state)

    @staticmethod
    def load(path: str) -> "LSTMModel":
        payload = joblib.load(path)
        obj: LSTMModel = payload["wrapper"]
        if payload["net_state"]:
            obj.net_ = _LSTMNet().to(obj.device_)
            obj.net_.load_state_dict(payload["net_state"])
        return obj
