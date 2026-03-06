"""
Performance analytics — comprehensive trading metrics.
Calculates Sharpe, Sortino, Calmar, MAR, win rate, expectancy, profit factor,
maximum adverse/favourable excursion and more from a list of trade returns.
All functions are pure (no side-effects) and vectorised with NumPy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Data class — all metrics in one place
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradingMetrics:
    """All performance metrics for one backtest window or live session."""

    # P&L
    total_return_pct: float = 0.0         # net return as percentage
    total_pnl: float = 0.0                # net P&L in currency units

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0             # annualised Sharpe (risk-free=0)
    sortino_ratio: float = 0.0            # annualised Sortino (downside only)
    calmar_ratio: float = 0.0             # CAGR / MaxDD
    mar_ratio: float = 0.0               # mean annual return / MaxDD
    omega_ratio: float = 0.0             # probability-weighted gain vs loss ratio

    # Drawdown
    max_drawdown_pct: float = 0.0         # maximum peak-to-trough drawdown (%)
    avg_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0        # bars in max drawdown

    # Trade stats
    n_trades: int = 0
    win_rate: float = 0.0                 # fraction of profitable trades
    loss_rate: float = 0.0
    avg_win: float = 0.0                  # avg winning trade P&L
    avg_loss: float = 0.0                 # avg losing trade P&L (negative)
    profit_factor: float = 0.0            # gross profit / gross loss
    expectancy: float = 0.0              # expected P&L per trade
    avg_rr: float = 0.0                  # average realised risk-reward ratio

    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Tail risk
    var_95: float = 0.0                   # 95% Value at Risk (per-trade)
    cvar_95: float = 0.0                  # 95% CVaR / Expected Shortfall
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Extra
    annualised_return_pct: float = 0.0
    trades_per_day: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_TRADING_DAYS_PER_YEAR = 252
_BARS_PER_YEAR = {
    "M1": 252 * 24 * 60,
    "M5": 252 * 24 * 12,
    "M15": 252 * 24 * 4,
    "M30": 252 * 24 * 2,
    "H1": 252 * 24,
    "H4": 252 * 6,
    "D1": 252,
    "W1": 52,
}


def _annualisation_factor(timeframe: str = "H4") -> float:
    return float(_BARS_PER_YEAR.get(timeframe, 252 * 6)) ** 0.5


def _max_drawdown(equity_curve: np.ndarray) -> tuple[float, int]:
    """Return (max_drawdown_fraction, duration_in_bars)."""
    if len(equity_curve) < 2:
        return 0.0, 0
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / (peak + 1e-10)
    max_dd = float(drawdown.max())

    # Drawdown duration (bars from last peak to recovery)
    in_dd = drawdown > 0
    max_dur = 0
    cur_dur = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0

    return max_dd, max_dur


def _consecutive_streaks(pnl_arr: np.ndarray) -> tuple[int, int]:
    """Return (max_consecutive_wins, max_consecutive_losses)."""
    max_wins = max_losses = 0
    cur_wins = cur_losses = 0
    for p in pnl_arr:
        if p > 0:
            cur_wins += 1
            cur_losses = 0
        elif p < 0:
            cur_losses += 1
            cur_wins = 0
        else:
            cur_wins = cur_losses = 0
        max_wins = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    pnl_series: Sequence[float],
    initial_equity: float = 10_000.0,
    timeframe: str = "H4",
    n_trading_days: float | None = None,
) -> TradingMetrics:
    """
    Compute all performance metrics from a sequence of trade P&Ls.

    Parameters
    ----------
    pnl_series      : Iterable of per-trade P&L in currency units.
    initial_equity  : Starting equity.
    timeframe       : Used to annualise ratios (H4, H1, D1 etc.).
    n_trading_days  : If provided, overrides timeframe-based annualisation.
    """
    pnl = np.asarray(pnl_series, dtype=np.float64)
    if len(pnl) == 0:
        return TradingMetrics()

    m = TradingMetrics()
    m.n_trades = int(len(pnl))

    # ── Equity curve ────────────────────────────────────────────────────────
    equity = np.concatenate([[initial_equity], initial_equity + np.cumsum(pnl)])
    m.total_pnl = float(equity[-1] - initial_equity)
    m.total_return_pct = float(m.total_pnl / initial_equity * 100)

    # ── Returns per trade ────────────────────────────────────────────────────
    rets = pnl / initial_equity  # fractional returns per trade

    # ── Annualisation ───────────────────────────────────────────────────────
    ann = (
        (_TRADING_DAYS_PER_YEAR / n_trading_days) ** 0.5
        if n_trading_days
        else _annualisation_factor(timeframe)
    )

    # ── Sharpe ──────────────────────────────────────────────────────────────
    std_ret = float(np.std(rets, ddof=1)) if len(rets) > 1 else 1e-8
    mean_ret = float(np.mean(rets))
    m.sharpe_ratio = round(mean_ret / (std_ret + 1e-10) * ann, 4)

    # ── Sortino ─────────────────────────────────────────────────────────────
    downside = rets[rets < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-8
    m.sortino_ratio = round(mean_ret / (downside_std + 1e-10) * ann, 4)

    # ── Drawdown ────────────────────────────────────────────────────────────
    max_dd, dd_dur = _max_drawdown(equity)
    m.max_drawdown_pct = round(max_dd * 100, 4)
    m.max_drawdown_duration = dd_dur

    # Avg drawdown
    peak = np.maximum.accumulate(equity)
    all_dd = (peak - equity) / (peak + 1e-10)
    m.avg_drawdown_pct = round(float(all_dd.mean()) * 100, 4)

    # ── Calmar (annualised return / max drawdown) ────────────────────────────
    ann_ret = mean_ret * ann * ann   # ann² = bars_per_year
    m.annualised_return_pct = round(ann_ret * 100, 4)
    m.calmar_ratio = round(ann_ret / (max_dd + 1e-10), 4)
    m.mar_ratio = m.calmar_ratio  # same calculation

    # ── Win/loss ────────────────────────────────────────────────────────────
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    m.win_rate = round(len(wins) / len(pnl), 4)
    m.loss_rate = round(len(losses) / len(pnl), 4)
    m.avg_win = round(float(wins.mean()), 4) if len(wins) else 0.0
    m.avg_loss = round(float(losses.mean()), 4) if len(losses) else 0.0

    # ── Profit factor ────────────────────────────────────────────────────────
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 1e-8
    m.profit_factor = round(gross_profit / (gross_loss + 1e-10), 4)

    # ── Expectancy ───────────────────────────────────────────────────────────
    m.expectancy = round(float(pnl.mean()), 4)

    # ── Average R:R (absolute avg_win / avg_loss) ───────────────────────────
    if m.avg_loss != 0:
        m.avg_rr = round(abs(m.avg_win / (m.avg_loss + 1e-10)), 4)

    # ── Streaks ──────────────────────────────────────────────────────────────
    m.max_consecutive_wins, m.max_consecutive_losses = _consecutive_streaks(pnl)

    # ── Omega ratio ─────────────────────────────────────────────────────────
    m.omega_ratio = round(gross_profit / (gross_loss + 1e-10), 4)

    # ── Tail risk ───────────────────────────────────────────────────────────
    if len(rets) >= 10:
        m.var_95 = round(float(np.percentile(rets, 5)), 6)
        cvar_mask = rets <= m.var_95
        m.cvar_95 = round(float(rets[cvar_mask].mean()) if cvar_mask.any() else m.var_95, 6)

    if len(rets) >= 4:
        from scipy.stats import skew, kurtosis  # type: ignore[import-untyped]
        m.skewness = round(float(skew(rets)), 4)
        m.kurtosis = round(float(kurtosis(rets)), 4)

    return m


def metrics_to_dict(m: TradingMetrics) -> dict:
    """Convert TradingMetrics to plain dict for JSON / DataFrame serialisation."""
    return {k: getattr(m, k) for k in m.__dataclass_fields__}


def equity_series_from_pnl(
    pnl_series: Sequence[float],
    initial_equity: float = 10_000.0,
    timestamps: Sequence | None = None,
) -> pd.Series:
    """
    Build a pandas Series equity curve from P&L series.
    If timestamps provided, uses them as the index.
    """
    pnl = np.asarray(pnl_series, dtype=np.float64)
    equity = initial_equity + np.concatenate([[0.0], np.cumsum(pnl)])
    if timestamps is not None and len(timestamps) == len(equity):
        return pd.Series(equity, index=timestamps)
    return pd.Series(equity)


def rolling_sharpe(
    pnl_series: Sequence[float],
    window: int = 20,
    timeframe: str = "H4",
) -> pd.Series:
    """Rolling Sharpe ratio over a window of trades."""
    ann = _annualisation_factor(timeframe)
    s = pd.Series(np.asarray(pnl_series, dtype=np.float64))
    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std(ddof=1).replace(0, np.nan)
    return (roll_mean / roll_std * ann).fillna(0.0)


def compare_periods(
    pnl_before: Sequence[float],
    pnl_after: Sequence[float],
    initial_equity: float = 10_000.0,
    timeframe: str = "H4",
) -> dict:
    """
    Compare metrics between two periods (e.g. before/after model retraining).
    Returns dict with 'before', 'after', and 'delta' sub-dicts.
    """
    m_before = metrics_to_dict(compute_metrics(pnl_before, initial_equity, timeframe))
    m_after = metrics_to_dict(compute_metrics(pnl_after, initial_equity, timeframe))
    delta = {k: round(m_after[k] - m_before[k], 6) for k in m_before if isinstance(m_before[k], float)}
    return {"before": m_before, "after": m_after, "delta": delta}
