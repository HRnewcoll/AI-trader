#!/usr/bin/env python3
"""
ForexAI Main Orchestrator — entry point for the full system.
Starts all services, runs signal loop, schedules retraining.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path

# Ensure project root on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logging
from utils.config_loader import load_config

setup_logging()
logger = logging.getLogger(__name__)


def run_training(cfg: dict, pairs: list[str] | None = None) -> None:
    """Run full training pipeline (called on startup if no models exist + weekends)."""
    logger.info("=" * 70)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 70)
    try:
        from models.training_pipeline import train_all_models
        train_all_models(pairs=pairs, cfg=cfg)
        logger.info("Training pipeline complete")
    except Exception as e:
        logger.error("Training pipeline failed: %s", e)


def run_genetic_optimization(cfg: dict) -> None:
    """Run genetic optimizer on weekends."""
    logger.info("Starting genetic optimization...")
    try:
        from models.genetic_optimizer import GeneticOptimizer, GeneticConfig
        from data_pipeline.market_data import fetch_ohlcv
        from features.technical_indicators import compute_all_indicators, get_feature_columns
        from backtesting.walk_forward import vectorised_backtest

        pair = "EURUSD"
        df = fetch_ohlcv(pair, "H4", 1000, cfg)
        if df.empty:
            logger.warning("No data for genetic optimization")
            return

        df = compute_all_indicators(df, pair)
        feature_cols = get_feature_columns(df)

        def fitness_fn(params: dict) -> tuple:
            try:
                from models.xgboost_model import XGBoostForexModel
                model = XGBoostForexModel(params={
                    "max_depth": int(params.get("xgb_max_depth", 6)),
                    "learning_rate": float(params.get("xgb_learning_rate", 0.05)),
                    "n_estimators": int(params.get("xgb_n_estimators", 200)),
                })
                metrics = model.train(df, feature_cols)
                signals = df["direction"].shift(-1).fillna(0).astype(int) * 2 - 1
                result = vectorised_backtest(df, signals, pair=pair)
                sharpe = result.sharpe
                sortino = result.sortino
                calmar = result.calmar
                neg_dd = -result.max_drawdown
                return (max(sharpe, 0.0), max(sortino, 0.0), max(calmar, 0.0), neg_dd)
            except Exception:
                return (0.0, 0.0, 0.0, -0.5)

        gc = GeneticConfig(population_size=20, generations=10)
        opt = GeneticOptimizer(fitness_fn=fitness_fn, cfg=gc)
        best = opt.evolve()
        logger.info("Genetic optimization best params: %s", best)

        # Save best params
        import json
        Path("artifacts").mkdir(exist_ok=True)
        Path("artifacts/best_genetic_params.json").write_text(json.dumps(best, indent=2))

    except Exception as e:
        logger.error("Genetic optimization failed: %s", e)


def schedule_jobs(orchestrator, cfg: dict) -> None:
    """Set up scheduled tasks: weekly retrain, daily health, hourly weights."""
    try:
        import schedule

        pairs = orchestrator.pairs
        schedule.every().saturday.at("06:00").do(run_training, cfg=cfg, pairs=pairs)
        schedule.every().saturday.at("08:00").do(run_genetic_optimization, cfg=cfg)
        schedule.every().day.at("00:00").do(orchestrator.risk_manager.reset_daily)
        schedule.every().hour.do(orchestrator.update_ensemble_weights)
        schedule.every(60).seconds.do(
            lambda: orchestrator.healer.run_health_check(
                equity=orchestrator.broker.get_account_equity()
            )
        )
        logger.info("Scheduled jobs registered")
    except ImportError:
        logger.warning("schedule library not available — no automated scheduling")


def main() -> None:
    logger.info("=" * 70)
    logger.info("  ForexAI — Self-Healing Autonomous Forex Trader 2026")
    logger.info("=" * 70)

    # Load config
    cfg = load_config()
    pairs = cfg.get("pairs", {}).get("majors", ["EURUSD"])[:3]  # Start with 3 pairs

    # Create artifact dirs
    for d in ["artifacts/models", "artifacts/onnx", "artifacts/chromadb", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Train if no models exist
    models_dir = Path(cfg.get("models", {}).get("models_dir", "artifacts/models"))
    trained_models = list(models_dir.glob("xgboost_*.pkl"))
    if not trained_models:
        logger.info("No trained models found — running initial training...")
        run_training(cfg, pairs=pairs)

    # Start orchestrator
    from agents.orchestrator import ForexOrchestrator
    orchestrator = ForexOrchestrator(cfg)

    # Schedule background jobs
    schedule_jobs(orchestrator, cfg)

    # Graceful shutdown
    _running = [True]
    def _shutdown(sig, frame):
        logger.info("Shutdown signal received")
        _running[0] = False
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    notifier = orchestrator.notifier
    notifier.send("🚀 ForexAI Started", f"Mode: {orchestrator.mode} | Pairs: {pairs}", level="info")

    logger.info("Starting main signal loop...")
    cycle = 0
    while _running[0]:
        try:
            # Run pending scheduled tasks
            try:
                import schedule
                schedule.run_pending()
            except ImportError:
                pass

            # Signal cycle (H4 = every 4 hours; but we poll every 5 min for responsiveness)
            trades = orchestrator.run_cycle()
            if trades:
                logger.info("Cycle %d: %d trades executed", cycle, len(trades))

            cycle += 1
            # Sleep 5 minutes between cycles
            for _ in range(300):
                if not _running[0]:
                    break
                time.sleep(1)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error("Main loop error (cycle %d): %s", cycle, e)
            time.sleep(30)

    notifier.send("⏹️ ForexAI Stopped", f"Completed {cycle} cycles", level="info")
    logger.info("ForexAI shutdown complete")


if __name__ == "__main__":
    main()
