"""
Technical indicator computation — 50+ indicators using pandas-ta.
Falls back to manual computation if TA-Lib unavailable.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import pandas_ta as ta
    _HAS_TA = True
except ImportError:
    _HAS_TA = False
    logger.warning("pandas-ta not installed; using manual indicator computation")


# ─────────────────────────────────────────────────────────────────────────────
# Manual fallback implementations
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    k_pct = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d_pct = k_pct.rolling(d).mean()
    return k_pct, d_pct


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _atr(high, low, close, period)
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mean_dev + 1e-10)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = pd.Series(np.where(tp > tp.shift(1), mf, 0.0), index=close.index)
    neg_mf = pd.Series(np.where(tp < tp.shift(1), mf, 0.0), index=close.index)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfr = pos_sum / (neg_sum + 1e-10)
    return 100 - 100 / (1 + mfr)


# ─────────────────────────────────────────────────────────────────────────────
# Main feature builder
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame, pair: str = "EURUSD") -> pd.DataFrame:
    """
    Compute 50+ technical indicators.
    Input: DataFrame with columns [open, high, low, close, volume].
    Returns: same DataFrame with all indicator columns added.
    """
    df = df.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]
    open_ = df["open"]
    pip = 0.0001 if "JPY" not in pair.upper() else 0.01

    # ── Moving averages ──────────────────────────────────────────────────────
    for period in [5, 10, 20, 50, 100, 200]:
        df[f"sma_{period}"] = close.rolling(period).mean()
        df[f"ema_{period}"] = _ema(close, period)

    # EMA crosses
    df["ema_cross_5_20"] = (df["ema_5"] > df["ema_20"]).astype(float)
    df["ema_cross_20_50"] = (df["ema_20"] > df["ema_50"]).astype(float)
    df["ema_cross_50_200"] = (df["ema_50"] > df["ema_200"]).astype(float)

    # Price vs MAs
    df["close_vs_sma20"] = (close - df["sma_20"]) / (df["sma_20"] + 1e-10)
    df["close_vs_sma50"] = (close - df["sma_50"]) / (df["sma_50"] + 1e-10)
    df["close_vs_sma200"] = (close - df["sma_200"]) / (df["sma_200"] + 1e-10)

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["rsi_14"] = _rsi(close, 14)
    df["rsi_7"] = _rsi(close, 7)
    df["rsi_21"] = _rsi(close, 21)
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(float)
    df["rsi_oversold"] = (df["rsi_14"] < 30).astype(float)

    macd, macd_signal, macd_hist = _macd(close)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["macd_cross"] = (macd > macd_signal).astype(float)

    df["cci_20"] = _cci(high, low, close, 20)
    df["williams_r"] = _williams_r(high, low, close, 14)
    df["mfi_14"] = _mfi(high, low, close, volume, 14)
    df["roc_10"] = close.pct_change(10)
    df["roc_20"] = close.pct_change(20)

    stoch_k, stoch_d = _stochastic(high, low, close)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d
    df["stoch_cross"] = (stoch_k > stoch_d).astype(float)

    # ── Volatility ────────────────────────────────────────────────────────────
    df["atr_14"] = _atr(high, low, close, 14)
    df["atr_7"] = _atr(high, low, close, 7)
    df["atr_pct"] = df["atr_14"] / (close + 1e-10)

    bb_upper, bb_mid, bb_lower = _bollinger(close, 20, 2.0)
    df["bb_upper"] = bb_upper
    df["bb_mid"] = bb_mid
    df["bb_lower"] = bb_lower
    df["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    df["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(20).mean()).astype(float)

    df["historical_vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    df["historical_vol_5"] = close.pct_change().rolling(5).std() * np.sqrt(252)
    df["vol_regime"] = (df["historical_vol_20"] > df["historical_vol_20"].rolling(50).mean()).astype(float)

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["adx_14"] = _adx(high, low, close, 14)
    df["strong_trend"] = (df["adx_14"] > 25).astype(float)
    df["obv"] = _obv(close, volume)
    df["obv_sma"] = df["obv"].rolling(20).mean()
    df["obv_trend"] = (df["obv"] > df["obv_sma"]).astype(float)

    # ── Price action ──────────────────────────────────────────────────────────
    df["pip_range"] = (high - low) / pip
    df["body_size"] = (close - open_).abs() / pip
    df["upper_wick"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / pip
    df["lower_wick"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / pip
    df["is_bullish"] = (close > open_).astype(float)
    df["gap"] = (open_ - close.shift(1)) / pip

    # ── Returns ───────────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_{lag}"] = close.pct_change(lag)

    df["future_return_1"] = close.shift(-1).pct_change()
    df["direction"] = (df["future_return_1"] > 0).astype(int)

    # ── Session (UTC) ─────────────────────────────────────────────────────────
    if hasattr(df.index, "hour"):
        hour = df.index.hour
    elif isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
    else:
        hour = pd.to_datetime(df.index).hour

    hour_series = pd.Series(hour, index=df.index)
    df["session_asian"]   = ((hour_series >= 0) & (hour_series < 8)).astype(float)
    df["session_london"]  = ((hour_series >= 8) & (hour_series < 16)).astype(float)
    df["session_ny"]      = ((hour_series >= 13) & (hour_series < 21)).astype(float)
    df["session_overlap"] = ((hour_series >= 13) & (hour_series < 16)).astype(float)

    # ── Volume ────────────────────────────────────────────────────────────────
    df["volume_sma20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / (df["volume_sma20"] + 1e-10)
    df["volume_spike"] = (df["volume_ratio"] > 2.0).astype(float)

    # ── Support / Resistance ──────────────────────────────────────────────────
    df["resistance_20"] = high.rolling(20).max()
    df["support_20"] = low.rolling(20).min()
    df["near_resistance"] = ((df["resistance_20"] - close) / (df["atr_14"] + 1e-10)).clip(0, 5)
    df["near_support"] = ((close - df["support_20"]) / (df["atr_14"] + 1e-10)).clip(0, 5)

    # ── Day of week / hour ────────────────────────────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df["hour_sin"] = np.sin(2 * np.pi * hour_series / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour_series / 24)

    df = df.replace([np.inf, -np.inf], np.nan)
    logger.debug("Computed %d features for %s", len(df.columns), pair)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (exclude raw OHLCV and target)."""
    exclude = {"open", "high", "low", "close", "volume", "direction", "future_return_1"}
    return [c for c in df.columns if c not in exclude]
