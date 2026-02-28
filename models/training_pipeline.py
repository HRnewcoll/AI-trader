"""
End-to-end training pipeline — fetches data, computes features,
trains all models, exports to ONNX.
Run this to produce actual trained model artifacts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from data_pipeline.market_data import fetch_ohlcv
from features.technical_indicators import compute_all_indicators, get_feature_columns
from models.xgboost_model import XGBoostForexModel
from models.transformer_model import TransformerTrainer
from utils.config_loader import load_config

logger = logging.getLogger(__name__)


def train_all_models(
    pairs: list[str] | None = None,
    timeframe: str = "H4",
    bars: int = 2000,
    cfg: dict | None = None,
    tune: bool = False,
) -> dict[str, dict]:
    """
    Main training entry point.
    Fetches data, engineers features, trains XGBoost + DLinear + PatchTST.
    Returns dict of {model_name: metrics}.
    """
    if cfg is None:
        try:
            cfg = load_config()
        except Exception:
            cfg = {}

    if pairs is None:
        pairs = cfg.get("pairs", {}).get("majors", ["EURUSD", "GBPUSD", "USDJPY"])

    results = {}
    artifacts_dir = cfg.get("models", {}).get("models_dir", "artifacts/models")
    onnx_dir = cfg.get("models", {}).get("onnx_dir", "artifacts/onnx")
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(onnx_dir).mkdir(parents=True, exist_ok=True)

    # Train per pair
    for pair in pairs:
        logger.info("=" * 60)
        logger.info("Training models for %s %s", pair, timeframe)
        logger.info("=" * 60)

        # 1. Fetch data
        df = fetch_ohlcv(pair, timeframe, bars, cfg)
        if df.empty or len(df) < 200:
            logger.warning("Insufficient data for %s — skipping", pair)
            continue

        # 2. Feature engineering
        df = compute_all_indicators(df, pair)
        feature_cols = get_feature_columns(df)

        # Drop NaN rows (from rolling indicators)
        df = df.dropna(subset=feature_cols + ["direction"])
        logger.info("Clean dataset: %d rows × %d features", len(df), len(feature_cols))

        pair_results = {}

        # 3. XGBoost
        try:
            xgb_model = XGBoostForexModel(artifacts_dir=artifacts_dir)
            metrics = xgb_model.train(df, feature_cols, tune_hyperparams=tune)
            xgb_model.save(f"xgboost_{pair}")
            try:
                xgb_model.export_onnx(f"xgboost_{pair}", onnx_dir)
            except Exception as e:
                logger.warning("XGBoost ONNX export failed: %s", e)
            pair_results["xgboost"] = metrics
            logger.info("XGBoost %s: %s", pair, metrics)
        except Exception as e:
            logger.error("XGBoost training failed for %s: %s", pair, e)

        # 4. DLinear
        try:
            dlinear = TransformerTrainer(
                model_type="dlinear",
                seq_len=60,
                n_features=len(feature_cols),
                artifacts_dir=artifacts_dir,
            )
            metrics = dlinear.train(df, feature_cols, epochs=30)
            dlinear.save(f"dlinear_{pair}")
            try:
                dlinear.export_onnx(f"dlinear_{pair}", onnx_dir)
            except Exception as e:
                logger.warning("DLinear ONNX export failed: %s", e)
            pair_results["dlinear"] = metrics
            logger.info("DLinear %s: %s", pair, metrics)
        except Exception as e:
            logger.error("DLinear training failed for %s: %s", pair, e)

        # 5. PatchTST
        try:
            patchtst = TransformerTrainer(
                model_type="patchtst",
                seq_len=60,
                n_features=len(feature_cols),
                artifacts_dir=artifacts_dir,
            )
            metrics = patchtst.train(df, feature_cols, epochs=30)
            patchtst.save(f"patchtst_{pair}")
            try:
                patchtst.export_onnx(f"patchtst_{pair}", onnx_dir)
            except Exception as e:
                logger.warning("PatchTST ONNX export failed: %s", e)
            pair_results["patchtst"] = metrics
            logger.info("PatchTST %s: %s", pair, metrics)
        except Exception as e:
            logger.error("PatchTST training failed for %s: %s", pair, e)

        results[pair] = pair_results

    logger.info("Training complete. Results: %s", results)
    return results


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils.logger import setup_logging
    setup_logging()

    # Quick training run — can be called directly:
    # python models/training_pipeline.py
    results = train_all_models(
        pairs=["EURUSD"],
        timeframe="H4",
        bars=2000,
        tune=False,
    )
    print("Training results:", results)
