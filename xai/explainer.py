"""
XAI explainer — SHAP global/local + LIME per-trade explanations.
Wraps every model and logs explanations for every live trade.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    logger.warning("shap not installed — XAI disabled")

try:
    from lime import lime_tabular
    _HAS_LIME = True
except ImportError:
    _HAS_LIME = False


class ForexExplainer:
    """
    SHAP + LIME explainer for any sklearn/XGBoost/Torch model.
    Generates per-trade explanations and global feature importance.
    """

    def __init__(
        self,
        model,
        feature_names: list[str],
        model_type: str = "xgboost",
        background_data: np.ndarray | None = None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self._shap_explainer = None
        self._lime_explainer = None
        self._background = background_data

        if _HAS_SHAP and model is not None:
            try:
                if model_type == "xgboost":
                    self._shap_explainer = shap.TreeExplainer(model)
                elif background_data is not None:
                    self._shap_explainer = shap.KernelExplainer(
                        lambda x: model.predict_proba(x)[:, 1],
                        shap.sample(background_data, 50),
                    )
                logger.info("SHAP explainer initialised for %s", model_type)
            except Exception as e:
                logger.warning("SHAP init failed: %s", e)

        if _HAS_LIME and background_data is not None:
            try:
                self._lime_explainer = lime_tabular.LimeTabularExplainer(
                    background_data,
                    feature_names=feature_names,
                    mode="classification",
                    discretize_continuous=True,
                )
            except Exception as e:
                logger.warning("LIME init failed: %s", e)

    def explain_prediction(
        self,
        X: np.ndarray,
        direction: int,
        confidence: float,
        pair: str = "EURUSD",
        top_k: int = 5,
    ) -> dict:
        """
        Generate SHAP + LIME explanation for a single prediction.
        Returns structured explanation dict.
        """
        explanation = {
            "pair": pair,
            "direction": "BUY" if direction == 1 else "SELL",
            "confidence": round(confidence, 3),
            "shap_values": {},
            "top_features": [],
            "narrative": "",
        }

        # SHAP
        if self._shap_explainer is not None:
            try:
                shap_vals = self._shap_explainer.shap_values(X)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # positive class
                if len(shap_vals.shape) > 1:
                    shap_vals = shap_vals[0]

                shap_dict = dict(zip(self.feature_names, shap_vals.tolist()))
                # Top features by |SHAP|
                top = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
                explanation["shap_values"] = {k: round(v, 4) for k, v in top}
                explanation["top_features"] = [k for k, _ in top]
            except Exception as e:
                logger.debug("SHAP explain error: %s", e)

        # LIME
        lime_text = ""
        if self._lime_explainer is not None:
            try:
                lime_exp = self._lime_explainer.explain_instance(
                    X[0], lambda x: self.model.predict_proba(x), num_features=top_k
                )
                lime_contributions = dict(lime_exp.as_list())
                explanation["lime_values"] = {k: round(v, 4) for k, v in lime_contributions.items()}
            except Exception as e:
                logger.debug("LIME explain error: %s", e)

        # Generate narrative
        explanation["narrative"] = self._build_narrative(
            explanation, pair, direction, confidence
        )
        return explanation

    def _build_narrative(
        self, explanation: dict, pair: str, direction: int, confidence: float
    ) -> str:
        """Build human-readable explanation."""
        action = "BUY" if direction == 1 else "SELL"
        top = list(explanation.get("shap_values", {}).items())[:3]

        if not top:
            return f"{action} {pair} with {confidence:.0%} confidence."

        reasons = []
        for feat, shap_val in top:
            sign = "↑" if shap_val > 0 else "↓"
            reasons.append(f"{feat} {sign} (SHAP: {shap_val:+.3f})")

        return (
            f"Signal: {action} {pair} | Confidence: {confidence:.0%}\n"
            f"Top drivers: {' | '.join(reasons)}"
        )

    def get_global_importance(self, X: np.ndarray) -> dict[str, float]:
        """Global feature importance from SHAP."""
        if self._shap_explainer is None:
            return {}
        try:
            shap_vals = self._shap_explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            importance = np.abs(shap_vals).mean(axis=0)
            return dict(sorted(
                zip(self.feature_names, importance.tolist()),
                key=lambda x: x[1], reverse=True
            ))
        except Exception as e:
            logger.error("SHAP global importance error: %s", e)
            return {}
