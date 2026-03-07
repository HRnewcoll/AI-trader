"""
DLinear + PatchTST Transformer for time-series forecasting.
Both models are trained locally — no external API needed.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


_NAN_THRESHOLD = 0.3  # Drop feature columns with more than this fraction of NaN values


# ─────────────────────────────────────────────────────────────────────────────
# DLinear (Decomposition + Linear) — fast & strong baseline
# ─────────────────────────────────────────────────────────────────────────────

class MovingAvgDecomposition(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        # Pad to maintain sequence length
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, features)
        pad_size = self.kernel_size - 1
        x_pad = nn.functional.pad(x.permute(0, 2, 1), (pad_size // 2, pad_size - pad_size // 2), mode="replicate")
        trend = self.avg(x_pad).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """Decomposition Linear model — SOTA on many time-series benchmarks."""

    def __init__(
        self,
        seq_len: int = 60,
        pred_len: int = 1,
        n_features: int = 50,
        kernel_size: int = 25,
    ):
        super().__init__()
        self.decomp = MovingAvgDecomposition(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.output = nn.Linear(n_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        seasonal, trend = self.decomp(x)
        # Apply linear across time dimension
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        out = seasonal_out + trend_out  # (batch, pred_len, n_features)
        out = self.output(out.squeeze(1))  # (batch, 1)
        return self.sigmoid(out).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# PatchTST (Patched Time Series Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, n_features: int, patch_len: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x: torch.Tensor, patch_len: int) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        batch, seq_len, n_feat = x.shape
        n_patches = seq_len // patch_len
        x = x[:, :n_patches * patch_len, :]
        x = x.reshape(batch, n_patches, patch_len * n_feat)
        return self.projection(x)  # (batch, n_patches, d_model)


class PatchTST(nn.Module):
    """PatchTST — Patched Time-Series Transformer for forex direction."""

    def __init__(
        self,
        seq_len: int = 60,
        patch_len: int = 12,
        n_features: int = 50,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        n_patches = seq_len // patch_len
        self.patch_embed = PatchEmbedding(n_features, patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        patches = self.patch_embed(x, self.patch_len)
        patches = patches + self.pos_embed
        encoded = self.transformer(patches)
        out = encoded.mean(dim=1)
        return self.head(out).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TransformerTrainer:
    """Train DLinear or PatchTST on Forex feature data."""

    def __init__(
        self,
        model_type: str = "dlinear",
        seq_len: int = 60,
        n_features: int = 50,
        device: str | None = None,
        artifacts_dir: str = "artifacts/models",
    ):
        self.model_type = model_type
        self.seq_len = seq_len
        self.n_features = n_features
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model: nn.Module | None = None
        self.feature_names: list[str] = []
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.is_trained = False
        self.train_metrics: dict = {}

    def _build_model(self) -> nn.Module:
        if self.model_type == "dlinear":
            return DLinear(seq_len=self.seq_len, n_features=self.n_features)
        else:
            return PatchTST(seq_len=self.seq_len, n_features=self.n_features)

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            xs.append(X[i - self.seq_len: i])
            ys.append(y[i])
        return (
            torch.FloatTensor(np.array(xs)),
            torch.FloatTensor(np.array(ys)),
        )

    def _preprocess(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        data = df[feature_cols].copy()
        valid_cols = [c for c in data.columns if data[c].isna().mean() < _NAN_THRESHOLD]
        self.feature_names = valid_cols
        self.n_features = len(valid_cols)
        data = data[valid_cols].ffill().fillna(0).values
        if self.mean_ is None:
            self.mean_ = data.mean(axis=0)
            self.std_ = data.std(axis=0) + 1e-8
        return (data - self.mean_) / self.std_

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "direction",
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 3e-4,
        val_split: float = 0.2,
    ) -> dict:
        logger.info("Training %s (%d rows, %d features)", self.model_type, len(df), len(feature_cols))

        X = self._preprocess(df, feature_cols)
        y = df[target_col].values.astype(np.float32)

        X_t, y_t = self._make_sequences(X, y)
        split = int(len(X_t) * (1 - val_split))
        train_ds = TensorDataset(X_t[:split], y_t[:split])
        val_ds = TensorDataset(X_t[split:], y_t[split:])
        # pin_memory speeds up GPU transfers; use parallel data loading if multiple threads available
        _pin = self.device.type == "cuda"
        _MIN_THREADS_FOR_WORKERS = 2
        _workers = _MIN_THREADS_FOR_WORKERS if torch.get_num_threads() > _MIN_THREADS_FOR_WORKERS else 0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                  pin_memory=_pin, num_workers=_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                pin_memory=_pin, num_workers=_workers)

        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        best_state = None
        val_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    val_loss += criterion(pred, yb).item()
                    correct += ((pred > 0.5).float() == yb).sum().item()
                    total += len(yb)

            val_acc = correct / (total + 1e-8)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d/%d — val_loss: %.4f, val_acc: %.3f",
                            epoch + 1, epochs, val_loss / len(val_loader), val_acc)

        if best_state:
            self.model.load_state_dict(best_state)

        self.is_trained = True
        self.train_metrics = {
            "best_val_loss": float(best_val_loss),
            "final_val_acc": float(val_acc),
            "n_features": len(self.feature_names),
        }
        return self.train_metrics

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        X = (df[self.feature_names].ffill().fillna(0).values - self.mean_) / self.std_
        X_t, _ = self._make_sequences(X, np.zeros(len(X)))
        self.model.eval()
        with torch.no_grad():
            proba = self.model(X_t.to(self.device)).cpu().numpy()
        direction = (proba > 0.5).astype(int)
        return direction, proba

    def predict_latest(self, df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < self.seq_len + 1:
            return 1, 0.5
        directions, probas = self.predict(df)
        return int(directions[-1]), float(probas[-1])

    def save(self, name: str | None = None) -> Path:
        if name is None:
            name = self.model_type
        path = self.artifacts_dir / f"{name}.pt"
        torch.save({
            "state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "seq_len": self.seq_len,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "mean": self.mean_,
            "std": self.std_,
            "metrics": self.train_metrics,
        }, path)
        logger.info("%s saved to %s", self.model_type, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TransformerTrainer":
        data = torch.load(path, map_location="cpu")
        obj = cls(
            model_type=data["model_type"],
            seq_len=data["seq_len"],
            n_features=data["n_features"],
        )
        obj.feature_names = data["feature_names"]
        obj.mean_ = data["mean"]
        obj.std_ = data["std"]
        obj.train_metrics = data.get("metrics", {})
        obj.model = obj._build_model()
        obj.model.load_state_dict(data["state_dict"])
        obj.is_trained = True
        logger.info("Transformer model loaded from %s", path)
        return obj

    def export_onnx(self, name: str | None = None, onnx_dir: str = "artifacts/onnx") -> Path:
        if name is None:
            name = self.model_type
        onnx_path = Path(onnx_dir)
        onnx_path.mkdir(parents=True, exist_ok=True)
        out_path = onnx_path / f"{name}.onnx"
        dummy = torch.zeros(1, self.seq_len, self.n_features)
        torch.onnx.export(
            self.model.cpu(), dummy, str(out_path),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}},
            opset_version=17,
        )
        logger.info("ONNX exported to %s", out_path)
        return out_path
