"""
Market regime detector — identifies trending, ranging, volatile, and quiet regimes.
Used to switch model weights and adjust risk parameters automatically.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    TRENDING_UP   = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING       = "ranging"
    VOLATILE      = "volatile"
    QUIET         = "quiet"
    UNKNOWN       = "unknown"


@dataclass
class RegimeState:
    regime: Regime
    confidence: float        # 0–1
    volatility_level: str    # "low" | "medium" | "high"
    trend_strength: float    # 0–1 (ADX-based)
    vol_percentile: float    # current vol vs history (0–1)
    description: str


# ─────────────────────────────────────────────────────────────────────────────
# Core detection logic
# ─────────────────────────────────────────────────────────────────────────────

def detect_regime(df: pd.DataFrame, lookback: int = 20) -> RegimeState:
    """
    Detect current market regime from OHLCV + pre-computed indicators.
    Works best with H4 or D1 timeframe data.
    Requires at least: close, atr_14 or will compute them inline.
    """
    if len(df) < lookback + 10:
        return RegimeState(Regime.UNKNOWN, 0.0, "medium", 0.0, 0.5, "Insufficient data")

    close = df["close"]

    # ── Volatility ──────────────────────────────────────────────────────────
    if "atr_14" in df.columns:
        atr = df["atr_14"]
    else:
        high, low = df["high"], df["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()

    current_atr = atr.iloc[-1]
    atr_history = atr.iloc[-100:] if len(atr) >= 100 else atr
    vol_percentile = float((atr_history < current_atr).mean())

    if vol_percentile > 0.8:
        vol_level = "high"
    elif vol_percentile < 0.3:
        vol_level = "low"
    else:
        vol_level = "medium"

    # ── Trend strength (ADX proxy) ────────────────────────────────────────
    if "adx_14" in df.columns:
        adx = float(df["adx_14"].iloc[-1])
    else:
        # Quick ADX approximation
        returns = close.pct_change()
        pos_returns = returns.clip(lower=0).rolling(lookback).mean()
        neg_returns = (-returns.clip(upper=0)).rolling(lookback).mean()
        dm_ratio = (pos_returns - neg_returns).abs() / (pos_returns + neg_returns + 1e-10)
        adx = float(dm_ratio.iloc[-1] * 100)

    trend_strength = float(np.clip(adx / 50.0, 0.0, 1.0))

    # ── Direction (slope of EMA) ──────────────────────────────────────────
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()
    slope_fast = float((ema_fast.iloc[-1] - ema_fast.iloc[-lookback]) / (ema_fast.iloc[-lookback] + 1e-10))
    ema_crossover = ema_fast.iloc[-1] > ema_slow.iloc[-1]

    # ── Classify ──────────────────────────────────────────────────────────
    confidence = 0.5

    if vol_percentile > 0.85:
        regime = Regime.VOLATILE
        confidence = vol_percentile
        desc = f"High volatility spike (ATR {vol_percentile:.0%} percentile)"

    elif vol_percentile < 0.2:
        regime = Regime.QUIET
        confidence = 1 - vol_percentile
        desc = f"Low volatility squeeze (ATR {vol_percentile:.0%} percentile)"

    elif adx > 25 and slope_fast > 0 and ema_crossover:
        regime = Regime.TRENDING_UP
        confidence = min(0.5 + trend_strength * 0.5, 0.95)
        desc = f"Uptrend (ADX={adx:.1f}, slope={slope_fast:.4f})"

    elif adx > 25 and slope_fast < 0 and not ema_crossover:
        regime = Regime.TRENDING_DOWN
        confidence = min(0.5 + trend_strength * 0.5, 0.95)
        desc = f"Downtrend (ADX={adx:.1f}, slope={slope_fast:.4f})"

    else:
        regime = Regime.RANGING
        confidence = 0.6
        desc = f"Ranging market (ADX={adx:.1f})"

    return RegimeState(
        regime=regime,
        confidence=round(confidence, 3),
        volatility_level=vol_level,
        trend_strength=round(trend_strength, 3),
        vol_percentile=round(vol_percentile, 3),
        description=desc,
    )


def add_regime_features(df: pd.DataFrame, rolling_window: int = 20, stride: int = 5) -> pd.DataFrame:
    """
    Add regime indicator columns to the DataFrame.
    Applied rolling so each row has a regime label.
    stride: compute regime every N rows and forward-fill (much faster for large DataFrames).
    """
    df = df.copy()
    n = len(df)

    regimes = [Regime.UNKNOWN.value] * n
    vol_percs = [0.5] * n
    trend_strengths = [0.0] * n

    # Compute at every 'stride' rows, then forward-fill
    for i in range(0, n, stride):
        start = max(0, i - rolling_window - 10)
        sub = df.iloc[start: i + 1]
        if len(sub) < 15:
            continue
        state = detect_regime(sub, lookback=min(rolling_window, len(sub) - 5))
        # Fill this and the next (stride-1) rows
        end = min(i + stride, n)
        for j in range(i, end):
            regimes[j] = state.regime.value
            vol_percs[j] = state.vol_percentile
            trend_strengths[j] = state.trend_strength

    df["regime"] = regimes
    df["regime_vol_pct"] = vol_percs
    df["regime_trend_strength"] = trend_strengths

    # Encode regime as numeric features
    df["regime_trending_up"]   = (df["regime"] == Regime.TRENDING_UP.value).astype(float)
    df["regime_trending_down"] = (df["regime"] == Regime.TRENDING_DOWN.value).astype(float)
    df["regime_ranging"]       = (df["regime"] == Regime.RANGING.value).astype(float)
    df["regime_volatile"]      = (df["regime"] == Regime.VOLATILE.value).astype(float)
    df["regime_quiet"]         = (df["regime"] == Regime.QUIET.value).astype(float)

    logger.debug("Regime features added (%d rows, stride=%d)", n, stride)
    return df


def get_regime_columns() -> list[str]:
    return [
        "regime_vol_pct", "regime_trend_strength",
        "regime_trending_up", "regime_trending_down",
        "regime_ranging", "regime_volatile", "regime_quiet",
    ]
