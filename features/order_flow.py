"""
Order flow analysis features.

Computes:
  - VWAP (Volume Weighted Average Price) and deviation bands
  - Volume delta (estimated buy vs sell volume per bar)
  - Cumulative Delta (running buy-sell imbalance)
  - Buy/Sell pressure ratio (bar-by-bar)
  - Volume profile (price level with highest traded volume)
  - Order Block detection (large-volume imbalance zones)
  - Absorption (volume spike without price progress)

These features are appended to the feature DataFrame and can be
fed directly into ML models as additional columns.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# VWAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_vwap(df: pd.DataFrame, rolling_window: int = 20) -> pd.DataFrame:
    """
    Add rolling VWAP and deviation bands (+/- 1σ, +/- 2σ).

    Columns added:
      vwap, vwap_upper_1, vwap_lower_1, vwap_upper_2, vwap_lower_2,
      vwap_dev  (close - vwap, normalised by vwap)
    """
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tpv = typical_price * df["volume"]

    # Rolling VWAP
    df["vwap"] = tpv.rolling(rolling_window).sum() / (df["volume"].rolling(rolling_window).sum() + 1e-8)

    # VWAP standard deviation bands
    squared_dev = ((typical_price - df["vwap"]) ** 2 * df["volume"]).rolling(rolling_window).sum()
    vol_sum = df["volume"].rolling(rolling_window).sum() + 1e-8
    vwap_std = np.sqrt(squared_dev / vol_sum)

    df["vwap_upper_1"] = df["vwap"] + vwap_std
    df["vwap_lower_1"] = df["vwap"] - vwap_std
    df["vwap_upper_2"] = df["vwap"] + 2 * vwap_std
    df["vwap_lower_2"] = df["vwap"] - 2 * vwap_std

    # Normalised deviation: positive = above VWAP, negative = below
    df["vwap_dev"] = (df["close"] - df["vwap"]) / (df["vwap"] + 1e-8)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Volume delta (buy / sell pressure)
# ─────────────────────────────────────────────────────────────────────────────

def compute_volume_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate buy vs. sell volume per bar using the close position within the bar.

    bar_range = high - low
    buy_pct   = (close - low) / bar_range   → fraction of range that was bullish
    sell_pct  = (high - close) / bar_range

    Columns added:
      buy_volume, sell_volume, volume_delta, volume_delta_ratio,
      cumulative_delta, delta_divergence
    """
    df = df.copy()
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)

    buy_pct = ((df["close"] - df["low"]) / bar_range).fillna(0.5)
    sell_pct = ((df["high"] - df["close"]) / bar_range).fillna(0.5)

    df["buy_volume"] = df["volume"] * buy_pct
    df["sell_volume"] = df["volume"] * sell_pct
    df["volume_delta"] = df["buy_volume"] - df["sell_volume"]
    df["volume_delta_ratio"] = df["volume_delta"] / (df["volume"] + 1e-8)

    # Cumulative delta (running imbalance)
    df["cumulative_delta"] = df["volume_delta"].cumsum()

    # Delta divergence: price going up but delta negative (bearish), vice versa
    price_direction = np.sign(df["close"].diff().fillna(0))
    delta_direction = np.sign(df["volume_delta"])
    df["delta_divergence"] = (price_direction != delta_direction).astype(float)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Buy / Sell pressure rolling ratios
# ─────────────────────────────────────────────────────────────────────────────

def compute_pressure_ratios(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Rolling buy/sell pressure ratio over a window.

    Columns added:
      pressure_ratio   (rolling buy_volume / total_volume)
      pressure_ma      (SMA of pressure_ratio)
      pressure_signal  (+1 buy dominant, -1 sell dominant, 0 neutral)
    """
    df = df.copy()
    if "buy_volume" not in df.columns:
        df = compute_volume_delta(df)

    rolling_buy = df["buy_volume"].rolling(window).sum()
    rolling_vol = df["volume"].rolling(window).sum() + 1e-8
    df["pressure_ratio"] = rolling_buy / rolling_vol
    df["pressure_ma"] = df["pressure_ratio"].rolling(window).mean()
    df["pressure_signal"] = df["pressure_ratio"].apply(
        lambda x: 1 if x > 0.55 else (-1 if x < 0.45 else 0)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Volume profile (price level analysis)
# ─────────────────────────────────────────────────────────────────────────────

def compute_volume_profile(
    df: pd.DataFrame,
    n_bins: int = 20,
    lookback: int = 100,
) -> dict:
    """
    Compute a volume profile (point of control, value area high/low).

    Returns dict with:
      poc (Point of Control — price with max volume),
      vah (Value Area High — top 70% of volume),
      val (Value Area Low — bottom 70% of volume),
      profile (dict: price_level -> volume)
    """
    subset = df.tail(lookback)
    if subset.empty:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "profile": {}}

    price_min = float(subset["low"].min())
    price_max = float(subset["high"].max())
    if price_max <= price_min:
        return {"poc": float(subset["close"].iloc[-1]), "vah": price_max, "val": price_min, "profile": {}}

    bins = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_volumes = np.zeros(n_bins)

    bar_lows = subset["low"].values
    bar_highs = subset["high"].values
    bar_vols = subset["volume"].values

    for i in range(len(subset)):
        bar_low = bar_lows[i]
        bar_high = bar_highs[i]
        bar_vol = bar_vols[i]
        overlap_mask = (bins[1:] >= bar_low) & (bins[:-1] <= bar_high)
        n_overlap = overlap_mask.sum()
        if n_overlap > 0:
            bin_volumes[overlap_mask] += bar_vol / n_overlap

    poc_idx = int(np.argmax(bin_volumes))
    poc = float(bin_centers[poc_idx])

    # Value area: 70% of total volume around POC
    total_vol = bin_volumes.sum()
    va_target = total_vol * 0.70
    va_vol = bin_volumes[poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx

    while va_vol < va_target:
        expand_up = hi_idx + 1 < n_bins
        expand_dn = lo_idx - 1 >= 0
        if not expand_up and not expand_dn:
            break
        up_vol = bin_volumes[hi_idx + 1] if expand_up else -1
        dn_vol = bin_volumes[lo_idx - 1] if expand_dn else -1
        if up_vol >= dn_vol:
            hi_idx += 1
            va_vol += bin_volumes[hi_idx]
        else:
            lo_idx -= 1
            va_vol += bin_volumes[lo_idx]

    return {
        "poc": poc,
        "vah": float(bin_centers[hi_idx]),
        "val": float(bin_centers[lo_idx]),
        "profile": {round(float(c), 5): round(float(v), 1) for c, v in zip(bin_centers, bin_volumes)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Order block detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_order_blocks(
    df: pd.DataFrame,
    vol_percentile: float = 90.0,
    lookback: int = 50,
) -> list[dict]:
    """
    Detect order blocks — zones where large-volume candles cause a price
    reversal. A bullish OB is the last down-candle before a strong up-move.
    A bearish OB is the last up-candle before a strong down-move.

    Returns list of dicts: {type, price_high, price_low, volume, bar_idx}
    """
    subset = df.tail(lookback).copy()
    if len(subset) < 10:
        return []

    vol_threshold = float(np.percentile(subset["volume"].values, vol_percentile))
    close_arr = subset["close"].values
    open_arr = subset["open"].values
    high_arr = subset["high"].values
    low_arr = subset["low"].values
    vol_arr = subset["volume"].values

    blocks = []
    for i in range(1, len(subset) - 2):
        if vol_arr[i] < vol_threshold:
            continue

        is_bearish_candle = close_arr[i] < open_arr[i]
        is_bullish_candle = close_arr[i] > open_arr[i]

        # Bullish OB: big down candle followed by strong up-move
        if is_bearish_candle:
            next_range = close_arr[i + 1] - open_arr[i + 1]
            if next_range > abs(close_arr[i] - open_arr[i]) * 0.5:
                blocks.append({
                    "type": "bullish_ob",
                    "price_high": float(high_arr[i]),
                    "price_low": float(low_arr[i]),
                    "volume": float(vol_arr[i]),
                    "bar_idx": int(i),
                })

        # Bearish OB: big up candle followed by strong down-move
        if is_bullish_candle:
            next_range = open_arr[i + 1] - close_arr[i + 1]
            if next_range > abs(close_arr[i] - open_arr[i]) * 0.5:
                blocks.append({
                    "type": "bearish_ob",
                    "price_high": float(high_arr[i]),
                    "price_low": float(low_arr[i]),
                    "volume": float(vol_arr[i]),
                    "bar_idx": int(i),
                })

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Absorption detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_absorption(
    df: pd.DataFrame,
    vol_percentile: float = 85.0,
    price_move_pct: float = 0.001,
    lookback: int = 20,
) -> pd.Series:
    """
    Detect volume absorption — high volume bar with small price move.
    Returns boolean Series (True = absorption detected).
    """
    subset = df.tail(lookback)
    vol_threshold = float(np.percentile(subset["volume"].values, vol_percentile))

    high_vol = df["volume"] >= vol_threshold
    bar_range = (df["high"] - df["low"]) + 1e-8
    mid_price = (df["high"] + df["low"]) / 2 + 1e-8
    small_move = (bar_range / mid_price) < price_move_pct
    return (high_vol & small_move).rename("absorption")


# ─────────────────────────────────────────────────────────────────────────────
# Main function: compute all order flow features
# ─────────────────────────────────────────────────────────────────────────────

def compute_order_flow_features(
    df: pd.DataFrame,
    vwap_window: int = 20,
    pressure_window: int = 14,
) -> pd.DataFrame:
    """
    Compute all order flow features and append them to df.
    Returns the augmented DataFrame.
    """
    df = compute_vwap(df, rolling_window=vwap_window)
    df = compute_volume_delta(df)
    df = compute_pressure_ratios(df, window=pressure_window)

    # Absorption as a float column
    absorption = detect_absorption(df)
    df["absorption"] = absorption.astype(float).reindex(df.index, fill_value=0.0)

    # Near order block flag (is current price inside any recent OB?)
    blocks = detect_order_blocks(df)
    current_price = float(df["close"].iloc[-1])
    in_bullish_ob = any(
        b["price_low"] <= current_price <= b["price_high"]
        for b in blocks if b["type"] == "bullish_ob"
    )
    in_bearish_ob = any(
        b["price_low"] <= current_price <= b["price_high"]
        for b in blocks if b["type"] == "bearish_ob"
    )
    df["in_bullish_ob"] = float(in_bullish_ob)
    df["in_bearish_ob"] = float(in_bearish_ob)

    return df


def get_order_flow_signal(df: pd.DataFrame) -> dict:
    """
    Return a flat signal dict for the orchestrator.
    signal: +1 = bullish order flow, -1 = bearish, 0 = neutral
    """
    if df is None or len(df) < 15:
        return {"signal": 0, "confidence": 0.0, "reason": "insufficient data"}

    df = compute_order_flow_features(df)
    last = df.iloc[-1]

    bull_score = 0.0
    bear_score = 0.0

    pr = float(last.get("pressure_ratio", 0.5))
    if pr > 0.6:
        bull_score += (pr - 0.5) * 2
    elif pr < 0.4:
        bear_score += (0.5 - pr) * 2

    delta_ratio = float(last.get("volume_delta_ratio", 0.0))
    if delta_ratio > 0.1:
        bull_score += delta_ratio
    elif delta_ratio < -0.1:
        bear_score += abs(delta_ratio)

    vwap_dev = float(last.get("vwap_dev", 0.0))
    if vwap_dev > 0.001:
        bull_score += 0.2
    elif vwap_dev < -0.001:
        bear_score += 0.2

    bull_score += float(last.get("in_bullish_ob", 0.0)) * 0.3
    bear_score += float(last.get("in_bearish_ob", 0.0)) * 0.3

    total = bull_score + bear_score + 1e-8
    if bull_score > bear_score * 1.2:
        signal = 1
        confidence = float(np.clip(bull_score / total, 0.5, 0.9))
        reason = f"buy pressure={pr:.2f} delta={delta_ratio:+.2f}"
    elif bear_score > bull_score * 1.2:
        signal = -1
        confidence = float(np.clip(bear_score / total, 0.5, 0.9))
        reason = f"sell pressure={1-pr:.2f} delta={delta_ratio:+.2f}"
    else:
        signal = 0
        confidence = 0.0
        reason = "order flow neutral"

    return {"signal": signal, "confidence": round(confidence, 3), "reason": reason}
