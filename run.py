#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ForexAI — ONE-CLICK LAUNCHER                                       ║
║  python run.py                  → starts everything in paper-trading mode   ║
║  python run.py --mode live      → live trading (needs MT5 credentials)      ║
║  python run.py --train-only     → train models then exit                    ║
║  python run.py --dashboard-only → only start the Streamlit dashboard        ║
║  python run.py --backtest       → run a backtest on configured pairs        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — dependency check / auto-install
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_PACKAGES: list[str] = [
    "numpy", "pandas", "xgboost", "torch", "sklearn",
    "stable_baselines3", "gymnasium", "vaderSentiment",
    "feedparser", "bs4", "yfinance", "schedule",
    "yaml", "deap", "prometheus_client", "psutil",
]


def check_and_install_deps() -> None:
    missing = []
    for mod in REQUIRED_PACKAGES:
        try:
            __import__(mod.replace("-", "_"))
        except ImportError:
            missing.append(mod)

    if missing:
        print(f"[setup] Installing missing packages: {', '.join(missing)}")
        req = ROOT / "requirements.txt"
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(req), "--quiet"]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("[setup] ⚠ pip install returned non-zero; proceeding anyway")
    else:
        print("[setup] ✓ All dependencies satisfied")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — env / config bootstrapping
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_env() -> None:
    """Copy .env.example → .env if no .env exists."""
    env_file = ROOT / ".env"
    example = ROOT / ".env.example"
    if not env_file.exists() and example.exists():
        import shutil
        shutil.copy(example, env_file)
        print("[setup] ✓ Created .env from .env.example — edit it to add credentials")
    # Load env vars
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — model training (first-time or forced)
# ─────────────────────────────────────────────────────────────────────────────

def needs_training() -> bool:
    artifact_dir = ROOT / "artifacts"
    if not artifact_dir.exists():
        return True
    xgb_models = list(artifact_dir.glob("xgb_*.pkl"))
    return len(xgb_models) == 0


def run_training() -> None:
    print("\n[train] No trained models found — starting initial training...")
    print("[train] This may take several minutes depending on your hardware.")
    from models.training_pipeline import train_all_models
    from utils.config_loader import load_config
    cfg = load_config()
    train_all_models(cfg)
    print("[train] ✓ Training complete — models saved to artifacts/")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — run modes
# ─────────────────────────────────────────────────────────────────────────────

def run_dashboard() -> subprocess.Popen:
    print("\n[dashboard] Starting Streamlit dashboard at http://localhost:8501")
    cmd = [sys.executable, "-m", "streamlit", "run",
           str(ROOT / "dashboard" / "streamlit_app.py"),
           "--server.headless", "true",
           "--server.port", "8501"]
    return subprocess.Popen(cmd)


def run_backtest() -> None:
    print("\n[backtest] Running walk-forward backtest...")
    from utils.config_loader import load_config
    from backtesting.walk_forward import WalkForwardBacktester
    from data_pipeline.market_data import fetch_ohlcv
    from features.technical_indicators import compute_all_indicators, get_feature_columns
    import pandas as pd

    cfg = load_config()
    pairs = cfg.get("pairs", {}).get("majors", ["EURUSD"])[:3]
    results = {}

    for pair in pairs:
        print(f"  Backtesting {pair}...")
        df = fetch_ohlcv(pair, timeframe="H4", limit=2000)
        if df.empty:
            print(f"  ⚠ No data for {pair}")
            continue
        df = compute_all_indicators(df)
        feature_cols = get_feature_columns(df)
        bt = WalkForwardBacktester(feature_cols=feature_cols)
        report = bt.run(df)
        results[pair] = report
        sharpe = report.get("sharpe_ratio", 0)
        dd = report.get("max_drawdown_pct", 0)
        pf = report.get("profit_factor", 0)
        print(f"  {pair}: Sharpe={sharpe:.2f}, MaxDD={dd:.1%}, PF={pf:.2f}")

    print("\n[backtest] ✓ Done")


def run_trader(mode: str = "paper") -> None:
    import os
    os.environ.setdefault("FOREX_MODE", mode)
    print(f"\n[trader] Starting ForexAI in {mode.upper()} mode...")
    print("[trader] Press Ctrl+C to stop\n")
    import main_orchestrator
    main_orchestrator.main()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ForexAI one-click launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                        help="Trading mode (default: paper)")
    parser.add_argument("--train-only", action="store_true",
                        help="Train models then exit")
    parser.add_argument("--dashboard-only", action="store_true",
                        help="Only launch the Streamlit dashboard")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest then exit")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force model retraining even if artifacts exist")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Skip launching the dashboard alongside the trader")
    args = parser.parse_args()

    print("=" * 62)
    print("   ForexAI — Self-Healing Autonomous Forex Trader")
    print("=" * 62)

    # Step 1: deps
    check_and_install_deps()

    # Step 2: env
    bootstrap_env()

    # Step 3: train
    if args.force_retrain or needs_training():
        run_training()
    else:
        print("[setup] ✓ Trained models found — skipping training (use --force-retrain to retrain)")

    if args.train_only:
        print("\n[run] --train-only flag set — exiting after training.")
        return

    if args.backtest:
        run_backtest()
        return

    if args.dashboard_only:
        proc = run_dashboard()
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return

    # Normal mode: start dashboard + trader together
    dash_proc = None
    if not args.no_dashboard:
        try:
            import streamlit  # noqa: F401
            dash_proc = run_dashboard()
            time.sleep(2)  # let Streamlit boot
        except ImportError:
            print("[dashboard] streamlit not installed — skipping dashboard")

    try:
        run_trader(mode=args.mode)
    except KeyboardInterrupt:
        print("\n[run] Stopped by user")
    finally:
        if dash_proc:
            dash_proc.terminate()
            print("[dashboard] Stopped")


if __name__ == "__main__":
    main()
