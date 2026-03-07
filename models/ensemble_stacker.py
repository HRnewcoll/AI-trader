"""
Ensemble stacker — meta-learner that combines predictions from all base models
(XGBoost, DLinear, PatchTST, RL agent) into a single signal.

Uses a second-level logistic regression trained on held-out OOF (out-of-fold)
predictions. Falls back to weighted average when insufficient data.

Architecture:
  base models → probability predictions → logistic regression → final signal
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class EnsembleStacker:
    """
    Meta-learner that stacks base model predictions.

    Base model predictions (probabilities in [0, 1]) are used as features
    for a second-level logistic regression. The stacker is trained using
    out-of-fold predictions to prevent overfitting.

    The stacker can also operate in "weighted average" fallback mode when
    fewer than `min_samples_to_fit` labelled examples are available.
    """

    def __init__(
        self,
        model_names: list[str] | None = None,
        min_samples_to_fit: int = 50,
        artifacts_dir: str = "artifacts/models",
    ):
        self.model_names = model_names or ["xgboost", "dlinear", "patchtst", "rl"]
        self.min_samples_to_fit = min_samples_to_fit
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Meta-learner
        self._meta: Optional[LogisticRegression] = None
        self._scaler = StandardScaler()
        self._is_fitted = False

        # Weights for fallback weighted-average (updated from recent performance)
        self._weights: dict[str, float] = {n: 1.0 for n in self.model_names}
        self._last_auroc: dict[str, float] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────

    def fit(
        self,
        base_predictions: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> dict:
        """
        Train the meta-learner on out-of-fold predictions from base models.

        Parameters
        ----------
        base_predictions : {model_name: probability_array}  shape (n_samples,)
        labels           : 0/1 array  shape (n_samples,)

        Returns
        -------
        dict with training metrics
        """
        n = len(labels)
        if n < self.min_samples_to_fit:
            logger.info("EnsembleStacker: insufficient data (%d < %d) — using weighted avg", n, self.min_samples_to_fit)
            self._update_weights_from_auroc(base_predictions, labels)
            return {"status": "weighted_avg", "n_samples": n}

        # Build stacking feature matrix
        X = self._build_feature_matrix(base_predictions)
        if X is None or X.shape[1] == 0:
            return {"status": "no_base_models"}

        y = np.asarray(labels)

        # OOF cross-validation
        n_splits = min(5, max(2, n // 20))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros(n)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_val_sc = scaler.transform(X_val)
            clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            clf.fit(X_tr_sc, y_tr)
            oof_preds[val_idx] = clf.predict_proba(X_val_sc)[:, 1]

        oof_auc = roc_auc_score(y, oof_preds)

        # Fit final meta-learner on all data
        X_sc = self._scaler.fit_transform(X)
        self._meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self._meta.fit(X_sc, y)
        self._is_fitted = True

        # Update model weights from per-model AUROC
        self._update_weights_from_auroc(base_predictions, labels)

        metrics = {
            "status": "fitted",
            "oof_auroc": round(oof_auc, 4),
            "n_samples": n,
            "n_splits": n_splits,
            "model_weights": {k: round(v, 3) for k, v in self._weights.items()},
        }
        logger.info("EnsembleStacker trained: OOF AUC=%.4f", oof_auc)
        return metrics

    # ─────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, base_predictions: dict[str, float]) -> tuple[int, float]:
        """
        Predict final signal from base model predictions.

        Parameters
        ----------
        base_predictions : {model_name: probability_float}  (single sample)

        Returns
        -------
        (direction: 1|0|-1, confidence: 0-1)
          direction 1 = buy, -1 = sell, 0 = hold
        """
        if self._is_fitted and self._meta is not None:
            return self._meta_predict(base_predictions)
        return self._weighted_avg_predict(base_predictions)

    def _meta_predict(self, base_predictions: dict[str, float]) -> tuple[int, float]:
        """Use fitted logistic regression."""
        row = np.array(
            [base_predictions.get(name, 0.5) for name in self.model_names],
            dtype=np.float64,
        ).reshape(1, -1)
        try:
            row_sc = self._scaler.transform(row)
            proba = float(self._meta.predict_proba(row_sc)[0, 1])
        except Exception as e:
            logger.debug("Meta predict error: %s", e)
            return self._weighted_avg_predict(base_predictions)

        if proba > 0.60:
            return 1, round(proba, 3)
        elif proba < 0.40:
            return -1, round(1 - proba, 3)
        return 0, round(max(proba, 1 - proba), 3)

    def _weighted_avg_predict(self, base_predictions: dict[str, float]) -> tuple[int, float]:
        """Weighted average fallback."""
        weighted_sum = 0.0
        weight_total = 0.0
        for name in self.model_names:
            if name in base_predictions:
                w = self._weights.get(name, 1.0)
                weighted_sum += base_predictions[name] * w
                weight_total += w

        if weight_total == 0:
            return 0, 0.5

        proba = weighted_sum / weight_total
        if proba > 0.60:
            return 1, round(proba, 3)
        elif proba < 0.40:
            return -1, round(1 - proba, 3)
        return 0, round(max(proba, 1 - proba), 3)

    # ─────────────────────────────────────────────────────────────────────
    # Weight management
    # ─────────────────────────────────────────────────────────────────────

    def update_weights_from_pnl(self, pnl_per_model: dict[str, list[float]]) -> None:
        """
        Update model weights based on recent P&L.
        Models with better recent Sharpe get higher weights.
        """
        new_weights: dict[str, float] = {}
        for name in self.model_names:
            pnls = pnl_per_model.get(name, [])
            if len(pnls) >= 5:
                arr = np.array(pnls, dtype=float)
                std = np.std(arr) + 1e-8
                sharpe = float(np.mean(arr) / std)
                new_weights[name] = max(sharpe, 0.1)
            else:
                new_weights[name] = self._weights.get(name, 1.0)

        # Normalise
        total = sum(new_weights.values()) + 1e-8
        self._weights = {k: v / total for k, v in new_weights.items()}
        logger.info("EnsembleStacker weights updated: %s", self._weights)

    def _update_weights_from_auroc(
        self, base_predictions: dict[str, np.ndarray], labels: np.ndarray
    ) -> None:
        """Update internal model weights from per-model AUROC scores."""
        auroc_scores: dict[str, float] = {}
        for name, preds in base_predictions.items():
            try:
                auc = roc_auc_score(labels, preds)
                auroc_scores[name] = float(auc)
            except Exception:
                auroc_scores[name] = 0.5

        # Softmax-like normalisation
        scores = np.array([auroc_scores.get(n, 0.5) for n in self.model_names])
        exp_scores = np.exp(scores - scores.max())
        norm = exp_scores.sum() + 1e-8
        for i, name in enumerate(self.model_names):
            self._weights[name] = float(exp_scores[i] / norm)

        self._last_auroc = auroc_scores
        logger.info("Per-model AUROC: %s", {k: round(v, 4) for k, v in auroc_scores.items()})

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _build_feature_matrix(
        self, base_predictions: dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        cols = [base_predictions[n] for n in self.model_names if n in base_predictions]
        if not cols:
            return None
        return np.column_stack(cols)

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def save(self, pair: str) -> Path:
        """Save the stacker to disk."""
        path = self.artifacts_dir / f"stacker_{pair}.pkl"
        state = {
            "meta": self._meta,
            "scaler": self._scaler,
            "is_fitted": self._is_fitted,
            "weights": self._weights,
            "model_names": self.model_names,
            "last_auroc": self._last_auroc,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("EnsembleStacker saved to %s", path)
        return path

    @classmethod
    def load(cls, pair: str, artifacts_dir: str = "artifacts/models") -> "EnsembleStacker":
        """Load a saved stacker from disk."""
        path = Path(artifacts_dir) / f"stacker_{pair}.pkl"
        if not path.exists():
            stacker = cls(artifacts_dir=artifacts_dir)
            return stacker

        with open(path, "rb") as f:
            state = pickle.load(f)

        stacker = cls(
            model_names=state.get("model_names") or ["xgboost", "dlinear", "patchtst", "rl"],
            artifacts_dir=artifacts_dir,
        )
        stacker._meta = state.get("meta")
        stacker._scaler = state.get("scaler", StandardScaler())
        stacker._is_fitted = state.get("is_fitted", False)
        stacker._weights = state.get("weights", {})
        stacker._last_auroc = state.get("last_auroc", {})
        return stacker

    def get_model_weights(self) -> dict[str, float]:
        """Return current normalised model weights."""
        return {k: round(v, 4) for k, v in self._weights.items()}

    def summary(self) -> dict:
        """Human-readable summary."""
        return {
            "is_fitted": self._is_fitted,
            "model_names": self.model_names,
            "weights": self.get_model_weights(),
            "last_auroc": {k: round(v, 4) for k, v in self._last_auroc.items()},
        }
