"""
Walk-forward backtesting with realistic slippage simulation.
Uses custom Gym env for RL + direct vectorized for XGBoost/Transformer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from models.rl_environment import ForexTradingEnv
from features.technical_indicators import compute_all_indicators, get_feature_columns

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    pair: str
    timeframe: str
    total_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    profit_factor: float
    calmar: float
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)


def _add_slippage(price: float, direction: int, spread_pips: float, slippage_pips: float, pip: float) -> float:
    """Add spread + random slippage in direction of trade."""
    total = (spread_pips + slippage_pips * np.random.uniform(0.5, 1.5)) * pip
    return price + total * direction


def vectorised_backtest(
    df: pd.DataFrame,
    signals: pd.Series,   # +1 buy, -1 sell, 0 hold
    initial_equity: float = 10_000.0,
    spread_pips: float = 1.5,
    slippage_pips: float = 0.5,
    commission_pct: float = 0.0001,
    pip_value: float = 10.0,
    pair: str = "EURUSD",
    position_size_lots: float = 0.1,
) -> BacktestResult:
    """Fast vectorised backtest for supervised models."""
    pip = 0.0001 if "JPY" not in pair.upper() else 0.01

    equity = initial_equity
    peak = initial_equity
    position = 0
    entry_price = 0.0
    equity_curve = [equity]
    trades = []
    returns = []

    for i in range(len(df)):
        row = df.iloc[i]
        close = float(row["close"])
        signal = int(signals.iloc[i]) if i < len(signals) else 0

        # Close existing
        if position != 0 and (signal != position or i == len(df) - 1):
            exit_price = _add_slippage(close, -position, spread_pips, slippage_pips, pip)
            raw_pnl = (exit_price - entry_price) * position * position_size_lots * pip_value / pip
            commission = equity * commission_pct
            pnl = raw_pnl - commission
            equity += pnl
            returns.append(pnl / (equity - pnl + 1e-8))
            trades.append({
                "idx": i,
                "pair": pair,
                "direction": "long" if position == 1 else "short",
                "entry": entry_price,
                "exit": exit_price,
                "pnl": round(pnl, 2),
            })
            position = 0

        # Open new
        if signal != 0 and position == 0:
            entry_price = _add_slippage(close, signal, spread_pips, slippage_pips, pip)
            position = signal

        if equity > peak:
            peak = equity
        equity_curve.append(equity)

    # Metrics
    total_return = (equity - initial_equity) / initial_equity
    max_dd = max((peak - e) / peak for e in equity_curve) if equity_curve else 0.0

    ret_arr = np.array(returns)
    sharpe = 0.0
    sortino = 0.0
    if len(ret_arr) > 1 and ret_arr.std() > 0:
        sharpe = float(ret_arr.mean() / ret_arr.std() * np.sqrt(252))
        neg = ret_arr[ret_arr < 0]
        if len(neg) > 0 and neg.std() > 0:
            sortino = float(ret_arr.mean() / neg.std() * np.sqrt(252))

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / max(len(trades), 1)
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0.0
    avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 1.0
    profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses) + 1e-8)
    calmar = total_return / (max_dd + 1e-8)

    return BacktestResult(
        pair=pair,
        timeframe="H4",
        total_return=round(total_return, 4),
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        max_drawdown=round(max_dd, 4),
        win_rate=round(win_rate, 3),
        n_trades=len(trades),
        profit_factor=round(profit_factor, 3),
        calmar=round(calmar, 3),
        equity_curve=equity_curve,
        trade_log=trades,
    )


def walk_forward_backtest(
    df: pd.DataFrame,
    model_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    n_splits: int = 5,
    train_pct: float = 0.8,
    pair: str = "EURUSD",
    **kwargs,
) -> list[BacktestResult]:
    """
    Walk-forward validation:
    train on rolling 80%, test on next 20%.
    """
    results = []
    fold_size = len(df) // n_splits

    for i in range(n_splits - 1):
        start = 0
        train_end = (i + 1) * fold_size
        test_end = min(train_end + fold_size, len(df))

        train_df = df.iloc[start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        if len(train_df) < 100 or len(test_df) < 20:
            continue

        logger.info("Walk-forward fold %d: train=%d, test=%d", i + 1, len(train_df), len(test_df))

        try:
            signals = model_fn(train_df, test_df)
            result = vectorised_backtest(test_df, signals, pair=pair, **kwargs)
            results.append(result)
            logger.info(
                "Fold %d: return=%.2f%%, sharpe=%.2f, DD=%.2f%%",
                i + 1, result.total_return * 100, result.sharpe, result.max_drawdown * 100
            )
        except Exception as e:
            logger.error("Walk-forward fold %d error: %s", i + 1, e)

    return results


def summarise_backtest_results(results: list[BacktestResult]) -> dict:
    """Aggregate metrics across walk-forward folds."""
    if not results:
        return {}

    returns = [r.total_return for r in results]
    sharpes = [r.sharpe for r in results]
    drawdowns = [r.max_drawdown for r in results]

    return {
        "n_folds": len(results),
        "mean_return": round(float(np.mean(returns)), 4),
        "mean_sharpe": round(float(np.mean(sharpes)), 3),
        "mean_max_dd": round(float(np.mean(drawdowns)), 4),
        "best_sharpe": round(float(np.max(sharpes)), 3),
        "worst_dd": round(float(np.max(drawdowns)), 4),
        "total_trades": sum(r.n_trades for r in results),
        "mean_win_rate": round(float(np.mean([r.win_rate for r in results])), 3),
    }
