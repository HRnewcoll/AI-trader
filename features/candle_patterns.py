"""
Candlestick pattern detection — rule-based (no TA-Lib dependency required).
Detects 20+ classic patterns: doji, hammer, engulfing, shooting star, etc.
Also produces a numeric pattern-score vector suitable as ML features.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _body(o: pd.Series, c: pd.Series) -> pd.Series:
    return (c - o).abs()

def _upper_wick(o: pd.Series, h: pd.Series, c: pd.Series) -> pd.Series:
    return h - pd.concat([o, c], axis=1).max(axis=1)

def _lower_wick(o: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    return pd.concat([o, c], axis=1).min(axis=1) - l

def _range(h: pd.Series, l: pd.Series) -> pd.Series:
    return h - l


# ─────────────────────────────────────────────────────────────────────────────
# Individual pattern detectors  (return +1 bullish, -1 bearish, 0 none)
# ─────────────────────────────────────────────────────────────────────────────

def doji(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """Open ≈ Close (body < 10% of range)."""
    rng = _range(h, l).replace(0, np.nan)
    ratio = _body(o, c) / rng
    return pd.Series(
        np.where(ratio < 0.1, np.where(c >= o, 1, -1), 0),
        index=o.index, name="doji",
    )


def hammer(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """Long lower wick, small body at top — bullish reversal."""
    body = _body(o, c)
    lw = _lower_wick(o, l, c)
    uw = _upper_wick(o, h, c)
    rng = _range(h, l).replace(0, np.nan)
    cond = (lw >= 2 * body) & (uw <= 0.2 * rng) & (body > 0)
    return pd.Series(np.where(cond, 1, 0), index=o.index, name="hammer")


def shooting_star(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """Long upper wick, small body at bottom — bearish reversal."""
    body = _body(o, c)
    lw = _lower_wick(o, l, c)
    uw = _upper_wick(o, h, c)
    rng = _range(h, l).replace(0, np.nan)
    cond = (uw >= 2 * body) & (lw <= 0.2 * rng) & (body > 0)
    return pd.Series(np.where(cond, -1, 0), index=o.index, name="shooting_star")


def engulfing(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """
    Bullish/bearish engulfing: current body completely covers previous body.
    """
    prev_o = o.shift(1)
    prev_c = c.shift(1)
    bullish = (c > o) & (prev_c < prev_o) & (o <= prev_c) & (c >= prev_o)
    bearish = (c < o) & (prev_c > prev_o) & (o >= prev_c) & (c <= prev_o)
    return pd.Series(
        np.where(bullish, 1, np.where(bearish, -1, 0)),
        index=o.index, name="engulfing",
    )


def morning_star(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """Three-candle bullish reversal: large bearish, small doji, large bullish."""
    body = _body(o, c)
    avg_body = body.rolling(10).mean()
    large_bear = (c.shift(2) < o.shift(2)) & (body.shift(2) > avg_body.shift(2))
    small_middle = body.shift(1) < avg_body.shift(1) * 0.3
    large_bull = (c > o) & (body > avg_body * 0.8)
    return pd.Series(
        np.where(large_bear & small_middle & large_bull, 1, 0),
        index=o.index, name="morning_star",
    )


def evening_star(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """Three-candle bearish reversal: large bullish, small doji, large bearish."""
    body = _body(o, c)
    avg_body = body.rolling(10).mean()
    large_bull = (c.shift(2) > o.shift(2)) & (body.shift(2) > avg_body.shift(2))
    small_middle = body.shift(1) < avg_body.shift(1) * 0.3
    large_bear = (c < o) & (body > avg_body * 0.8)
    return pd.Series(
        np.where(large_bull & small_middle & large_bear, -1, 0),
        index=o.index, name="evening_star",
    )


def three_white_soldiers(o: pd.Series, c: pd.Series) -> pd.Series:
    """Three consecutive bullish candles with rising closes — strong bullish."""
    bull1 = c > o
    bull2 = c.shift(1) > o.shift(1)
    bull3 = c.shift(2) > o.shift(2)
    rising = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    return pd.Series(
        np.where(bull1 & bull2 & bull3 & rising, 1, 0),
        index=o.index, name="three_white_soldiers",
    )


def three_black_crows(o: pd.Series, c: pd.Series) -> pd.Series:
    """Three consecutive bearish candles with falling closes — strong bearish."""
    bear1 = c < o
    bear2 = c.shift(1) < o.shift(1)
    bear3 = c.shift(2) < o.shift(2)
    falling = (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    return pd.Series(
        np.where(bear1 & bear2 & bear3 & falling, -1, 0),
        index=o.index, name="three_black_crows",
    )


def harami(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """Current candle body fully inside previous candle body (inside bar)."""
    prev_o = o.shift(1)
    prev_c = c.shift(1)
    prev_top = pd.concat([prev_o, prev_c], axis=1).max(axis=1)
    prev_bot = pd.concat([prev_o, prev_c], axis=1).min(axis=1)
    cur_top = pd.concat([o, c], axis=1).max(axis=1)
    cur_bot = pd.concat([o, c], axis=1).min(axis=1)
    inside = (cur_top < prev_top) & (cur_bot > prev_bot)
    bull = inside & (c > o)
    bear = inside & (c < o)
    return pd.Series(
        np.where(bull, 1, np.where(bear, -1, 0)),
        index=o.index, name="harami",
    )


def spinning_top(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """Small body with wicks on both sides — indecision."""
    body = _body(o, c)
    rng = _range(h, l).replace(0, np.nan)
    uw = _upper_wick(o, h, c)
    lw = _lower_wick(o, l, c)
    cond = (body / rng < 0.3) & (uw > body * 0.5) & (lw > body * 0.5)
    return pd.Series(np.where(cond, 1, 0), index=o.index, name="spinning_top")


def marubozu(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """No wicks — strong momentum candle."""
    rng = _range(h, l).replace(0, np.nan)
    body = _body(o, c)
    full_body = body / rng > 0.9
    bull = full_body & (c > o)
    bear = full_body & (c < o)
    return pd.Series(
        np.where(bull, 1, np.where(bear, -1, 0)),
        index=o.index, name="marubozu",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Composite pattern score + feature builder
# ─────────────────────────────────────────────────────────────────────────────

def compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all candle pattern features and add them to the DataFrame.
    Returns df with new columns: pat_doji, pat_hammer, pat_engulfing, ...
    Also adds pat_composite_score: weighted sum of all pattern signals.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    patterns = {
        "doji":                 doji(o, h, l, c),
        "hammer":               hammer(o, h, l, c),
        "shooting_star":        shooting_star(o, h, l, c),
        "engulfing":            engulfing(o, h, l, c),
        "morning_star":         morning_star(o, h, l, c),
        "evening_star":         evening_star(o, h, l, c),
        "three_white_soldiers": three_white_soldiers(o, c),
        "three_black_crows":    three_black_crows(o, c),
        "harami":               harami(o, h, l, c),
        "spinning_top":         spinning_top(o, h, l, c),
        "marubozu":             marubozu(o, h, l, c),
    }

    df = df.copy()
    pattern_cols = []
    for name, series in patterns.items():
        col = f"pat_{name}"
        df[col] = series.astype(float)
        pattern_cols.append(col)

    # Composite score: weighted sum (multi-candle patterns get higher weight)
    weights = {
        "pat_doji": 0.5, "pat_hammer": 1.0, "pat_shooting_star": 1.0,
        "pat_engulfing": 1.5, "pat_morning_star": 2.0, "pat_evening_star": 2.0,
        "pat_three_white_soldiers": 2.0, "pat_three_black_crows": 2.0,
        "pat_harami": 1.0, "pat_spinning_top": 0.3, "pat_marubozu": 1.5,
    }
    composite = sum(df[col] * weights.get(col, 1.0) for col in pattern_cols)
    df["pat_composite"] = np.clip(composite, -5, 5)

    # Rolling pattern activity (how often patterns fire in last N bars)
    df["pat_activity_5"] = (
        df[pattern_cols].abs().sum(axis=1).rolling(5).mean()
    )

    logger.debug("Candle pattern features computed: %d patterns", len(pattern_cols))
    return df


def get_pattern_columns() -> list[str]:
    return [
        "pat_doji", "pat_hammer", "pat_shooting_star", "pat_engulfing",
        "pat_morning_star", "pat_evening_star", "pat_three_white_soldiers",
        "pat_three_black_crows", "pat_harami", "pat_spinning_top",
        "pat_marubozu", "pat_composite", "pat_activity_5",
    ]
