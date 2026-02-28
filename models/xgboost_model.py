"""
XGBoost model — trained on technical features.
Supports training, evaluation, ONNX export, and inference.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.error("xgboost not installed")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


_NAN_THRESHOLD = 0.3  # Drop feature columns with more than this fraction of NaN values


class XGBoostForexModel:
    """
    Profit-aware XGBoost classifier for Forex direction prediction.
    Uses walk-forward cross-validation and optional Optuna hyperparameter tuning.
    """

    DEFAULT_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "device": "cpu",
    }

    def __init__(self, params: dict | None = None, artifacts_dir: str = "artifacts/models"):
        if not _HAS_XGB:
            raise ImportError("xgboost is required")
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: xgb.XGBClassifier | None = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.is_trained = False
        self.train_metrics: dict = {}

    def _prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "direction",
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = df[feature_cols].copy()
        # Drop columns with too many NaN
        valid_cols = [c for c in data.columns if data[c].isna().mean() < _NAN_THRESHOLD]
        data = data[valid_cols].ffill().fillna(0)
        self.feature_names = valid_cols
        X = self.scaler.fit_transform(data.values)
        y = df[target_col].values
        return X, y

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "direction",
        n_splits: int = 5,
        tune_hyperparams: bool = False,
    ) -> dict:
        """Walk-forward training with optional Optuna tuning."""
        logger.info("Training XGBoost model (%d rows, %d features)", len(df), len(feature_cols))

        X, y = self._prepare_data(df, feature_cols, target_col)

        if tune_hyperparams and _HAS_OPTUNA:
            logger.info("Running Optuna hyperparameter search...")
            best_params = self._tune_hyperparams(X, y, n_trials=50)
            self.params.update(best_params)

        # Walk-forward cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=30,
            )

            preds = model.predict(X_val)
            proba = model.predict_proba(X_val)[:, 1]
            acc = accuracy_score(y_val, preds)
            try:
                auc = roc_auc_score(y_val, proba)
            except Exception:
                auc = 0.5
            cv_scores.append({"fold": fold, "accuracy": acc, "auc": auc})
            logger.info("Fold %d — accuracy: %.3f, AUC: %.3f", fold, acc, auc)

        # Final training on full dataset
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, verbose=False)

        self.is_trained = True
        self.train_metrics = {
            "mean_accuracy": float(np.mean([s["accuracy"] for s in cv_scores])),
            "mean_auc": float(np.mean([s["auc"] for s in cv_scores])),
            "n_features": len(self.feature_names),
            "n_samples": len(df),
        }
        logger.info("XGBoost trained. Mean AUC: %.3f", self.train_metrics["mean_auc"])
        return self.train_metrics

    def _tune_hyperparams(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> dict:
        from sklearn.model_selection import cross_val_score

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": 42,
                "n_jobs": -1,
            }
            model = xgb.XGBClassifier(**params)
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X, y, cv=tscv, scoring="roc_auc")
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (predicted_direction, confidence_proba)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        data = df[self.feature_names].ffill().fillna(0)
        X = self.scaler.transform(data.values)
        proba = self.model.predict_proba(X)[:, 1]
        direction = (proba > 0.5).astype(int)
        return direction, proba

    def predict_latest(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Predict on latest bar. Returns (direction, confidence)."""
        directions, probas = self.predict(df.tail(1))
        return int(directions[0]), float(probas[0])

    def get_feature_importance(self) -> dict[str, float]:
        if not self.is_trained:
            return {}
        importance = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importance.tolist()),
            key=lambda x: x[1], reverse=True
        ))

    def save(self, name: str = "xgboost_forex") -> Path:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        path = self.artifacts_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "feature_names": self.feature_names,
                         "params": self.params,
                         "metrics": self.train_metrics}, f)
        logger.info("XGBoost model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostForexModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.feature_names = data["feature_names"]
        obj.params = data["params"]
        obj.train_metrics = data.get("metrics", {})
        obj.is_trained = True
        logger.info("XGBoost model loaded from %s", path)
        return obj

    def export_onnx(self, name: str = "xgboost_forex", onnx_dir: str = "artifacts/onnx") -> Path:
        """Export to ONNX for fast inference."""
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            logger.error("skl2onnx not installed — cannot export ONNX")
            return None

        onnx_path = Path(onnx_dir)
        onnx_path.mkdir(parents=True, exist_ok=True)
        n_features = len(self.feature_names)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        model_onnx = convert_sklearn(self.model, initial_types=initial_type)
        out_path = onnx_path / f"{name}.onnx"
        with open(out_path, "wb") as f:
            f.write(model_onnx.SerializeToString())
        logger.info("ONNX model exported to %s", out_path)
        return out_path
