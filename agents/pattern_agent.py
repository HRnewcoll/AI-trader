"""
Classical chart pattern detection agent.

Detects the following patterns from OHLCV data:
  - Head & Shoulders (bearish reversal)
  - Inverse Head & Shoulders (bullish reversal)
  - Double Top (bearish reversal)
  - Double Bottom (bullish reversal)
  - Ascending Triangle (bullish continuation)
  - Descending Triangle (bearish continuation)
  - Symmetrical Triangle (neutral / breakout pending)
  - Rising Wedge (bearish reversal)
  - Falling Wedge (bullish reversal)
  - Cup & Handle (bullish continuation)

Each pattern returns a signal dict:
  {"pattern": str, "signal": 1|-1|0, "confidence": 0-1, "target": float, "stop": float}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatternSignal:
    pattern: str
    signal: int           # +1 bullish, -1 bearish, 0 neutral
    confidence: float     # 0-1
    target: float         # price target (0.0 = unknown)
    stop: float           # suggested stop-loss (0.0 = unknown)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "signal": self.signal,
            "confidence": round(self.confidence, 3),
            "target": round(self.target, 5),
            "stop": round(self.stop, 5),
            "description": self.description,
        }


@dataclass
class PatternAgentResult:
    pair: str
    patterns: list[PatternSignal] = field(default_factory=list)

    @property
    def dominant_signal(self) -> int:
        """Net signal from all detected patterns (+1, -1, 0)."""
        if not self.patterns:
            return 0
        bull = sum(p.confidence for p in self.patterns if p.signal == 1)
        bear = sum(p.confidence for p in self.patterns if p.signal == -1)
        if bull > bear * 1.1:
            return 1
        if bear > bull * 1.1:
            return -1
        return 0

    @property
    def composite_confidence(self) -> float:
        if not self.patterns:
            return 0.0
        return float(np.clip(
            sum(p.confidence for p in self.patterns) / max(len(self.patterns), 1),
            0.0, 1.0,
        ))

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "dominant_signal": self.dominant_signal,
            "composite_confidence": round(self.composite_confidence, 3),
            "patterns": [p.to_dict() for p in self.patterns],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_local_maxima(arr: np.ndarray, order: int = 5) -> np.ndarray:
    return argrelextrema(arr, np.greater_equal, order=order)[0]


def _find_local_minima(arr: np.ndarray, order: int = 5) -> np.ndarray:
    return argrelextrema(arr, np.less_equal, order=order)[0]


def _pct_diff(a: float, b: float) -> float:
    return abs(a - b) / (max(abs(a), abs(b)) + 1e-10)


def _linear_regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return slope of OLS fit."""
    if len(x) < 2:
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Individual pattern detectors
# ─────────────────────────────────────────────────────────────────────────────

def detect_head_and_shoulders(
    df: pd.DataFrame,
    order: int = 5,
    tolerance: float = 0.03,
) -> Optional[PatternSignal]:
    """Detect bearish Head & Shoulders from the last 60 bars."""
    close = df["close"].values[-60:]
    highs = df["high"].values[-60:]
    if len(close) < 30:
        return None

    peaks = _find_local_maxima(highs, order=order)
    if len(peaks) < 3:
        return None

    # Use last 3 peaks
    lp, mp, rp = peaks[-3], peaks[-2], peaks[-1]
    lh, mh, rh = highs[lp], highs[mp], highs[rp]

    # Head higher than both shoulders
    if not (mh > lh * (1 + tolerance * 0.5) and mh > rh * (1 + tolerance * 0.5)):
        return None

    # Shoulders roughly equal
    if _pct_diff(lh, rh) > tolerance:
        return None

    # Neckline
    troughs = _find_local_minima(close, order=order)
    between_troughs = [t for t in troughs if lp < t < rp]
    if len(between_troughs) < 1:
        return None
    neckline = np.mean(close[between_troughs])

    # Confirmation: price should be near or below neckline
    current_close = close[-1]
    if current_close > neckline * 1.02:
        return None

    # Target = neckline - (head - neckline)
    head_height = mh - neckline
    target = neckline - head_height
    stop = max(float(mh), float(close[-1])) * 1.001
    shoulder_symmetry = 1 - _pct_diff(lh, rh)
    confidence = float(np.clip(shoulder_symmetry * 0.7 + 0.3, 0.4, 0.9))

    return PatternSignal(
        pattern="head_and_shoulders",
        signal=-1,
        confidence=confidence,
        target=target,
        stop=stop,
        description=f"H&S neckline={neckline:.5f}, target={target:.5f}",
    )


def detect_inverse_head_and_shoulders(
    df: pd.DataFrame,
    order: int = 5,
    tolerance: float = 0.03,
) -> Optional[PatternSignal]:
    """Detect bullish Inverse Head & Shoulders."""
    close = df["close"].values[-60:]
    lows = df["low"].values[-60:]
    if len(close) < 30:
        return None

    troughs = _find_local_minima(lows, order=order)
    if len(troughs) < 3:
        return None

    lt, mt, rt = troughs[-3], troughs[-2], troughs[-1]
    ll, ml, rl = lows[lt], lows[mt], lows[rt]

    # Head (middle) lower than both shoulders
    if not (ml < ll * (1 - tolerance * 0.5) and ml < rl * (1 - tolerance * 0.5)):
        return None
    if _pct_diff(ll, rl) > tolerance:
        return None

    peaks = _find_local_maxima(close, order=order)
    between_peaks = [p for p in peaks if lt < p < rt]
    if not between_peaks:
        return None
    neckline = np.mean(close[between_peaks])
    current_close = close[-1]
    if current_close < neckline * 0.98:
        return None

    target = neckline + (neckline - ml)
    stop = min(float(ml), float(close[-1])) * 0.999
    shoulder_symmetry = 1 - _pct_diff(ll, rl)
    confidence = float(np.clip(shoulder_symmetry * 0.7 + 0.3, 0.4, 0.9))

    return PatternSignal(
        pattern="inverse_head_and_shoulders",
        signal=1,
        confidence=confidence,
        target=target,
        stop=stop,
        description=f"IH&S neckline={neckline:.5f}, target={target:.5f}",
    )


def detect_double_top(
    df: pd.DataFrame,
    order: int = 5,
    tolerance: float = 0.02,
) -> Optional[PatternSignal]:
    """Detect bearish Double Top pattern."""
    close = df["close"].values[-60:]
    highs = df["high"].values[-60:]
    if len(close) < 20:
        return None

    peaks = _find_local_maxima(highs, order=order)
    if len(peaks) < 2:
        return None

    p1, p2 = peaks[-2], peaks[-1]
    h1, h2 = highs[p1], highs[p2]
    if _pct_diff(h1, h2) > tolerance:
        return None

    troughs = _find_local_minima(close, order=order)
    valley = [t for t in troughs if p1 < t < p2]
    if not valley:
        return None
    valley_price = np.min(close[valley])
    neckline = valley_price
    current = close[-1]
    if current > neckline * 1.01:
        return None

    top_height = np.mean([h1, h2]) - neckline
    target = neckline - top_height
    stop = max(h1, h2) * 1.005
    top_equality = 1 - _pct_diff(h1, h2) / tolerance
    confidence = float(np.clip(top_equality * 0.6 + 0.3, 0.4, 0.85))

    return PatternSignal(
        pattern="double_top",
        signal=-1,
        confidence=confidence,
        target=target,
        stop=stop,
        description=f"Double Top tops={h1:.5f}/{h2:.5f}, target={target:.5f}",
    )


def detect_double_bottom(
    df: pd.DataFrame,
    order: int = 5,
    tolerance: float = 0.02,
) -> Optional[PatternSignal]:
    """Detect bullish Double Bottom pattern."""
    close = df["close"].values[-60:]
    lows = df["low"].values[-60:]
    if len(close) < 20:
        return None

    troughs = _find_local_minima(lows, order=order)
    if len(troughs) < 2:
        return None

    t1, t2 = troughs[-2], troughs[-1]
    l1, l2 = lows[t1], lows[t2]
    if _pct_diff(l1, l2) > tolerance:
        return None

    peaks = _find_local_maxima(close, order=order)
    peak_between = [p for p in peaks if t1 < p < t2]
    if not peak_between:
        return None
    neckline = np.max(close[peak_between])
    current = close[-1]
    if current < neckline * 0.99:
        return None

    bottom_height = neckline - np.mean([l1, l2])
    target = neckline + bottom_height
    stop = min(l1, l2) * 0.995
    bottom_equality = 1 - _pct_diff(l1, l2) / tolerance
    confidence = float(np.clip(bottom_equality * 0.6 + 0.3, 0.4, 0.85))

    return PatternSignal(
        pattern="double_bottom",
        signal=1,
        confidence=confidence,
        target=target,
        stop=stop,
        description=f"Double Bottom bottoms={l1:.5f}/{l2:.5f}, target={target:.5f}",
    )


def detect_triangle(
    df: pd.DataFrame,
    lookback: int = 40,
    order: int = 4,
    slope_threshold: float = 0.0002,
) -> Optional[PatternSignal]:
    """
    Detect ascending / descending / symmetrical triangle.
    Returns the pattern with the highest confidence.
    """
    close = df["close"].values[-lookback:]
    highs = df["high"].values[-lookback:]
    lows = df["low"].values[-lookback:]
    if len(close) < 20:
        return None

    peak_idx = _find_local_maxima(highs, order=order)
    trough_idx = _find_local_minima(lows, order=order)
    if len(peak_idx) < 2 or len(trough_idx) < 2:
        return None

    x_peaks = peak_idx.astype(float)
    y_peaks = highs[peak_idx]
    x_troughs = trough_idx.astype(float)
    y_troughs = lows[trough_idx]

    peak_slope = _linear_regression_slope(x_peaks, y_peaks)
    trough_slope = _linear_regression_slope(x_troughs, y_troughs)

    atr_approx = float(np.mean(highs - lows))
    norm = atr_approx + 1e-8

    peak_flat = abs(peak_slope) / norm < slope_threshold
    trough_flat = abs(trough_slope) / norm < slope_threshold
    peak_rising = peak_slope / norm > slope_threshold
    peak_falling = peak_slope / norm < -slope_threshold
    trough_rising = trough_slope / norm > slope_threshold
    trough_falling = trough_slope / norm < -slope_threshold

    current = close[-1]

    # Ascending triangle: flat top, rising bottom
    if peak_flat and trough_rising:
        resistance = float(np.mean(y_peaks[-2:]))
        target = resistance + (resistance - float(np.mean(y_troughs[-2:])))
        stop = float(np.mean(y_troughs[-2:])) * 0.998
        return PatternSignal(
            pattern="ascending_triangle",
            signal=1,
            confidence=0.72,
            target=target,
            stop=stop,
            description="Ascending triangle: flat resistance, rising support",
        )

    # Descending triangle: flat bottom, falling top
    if trough_flat and peak_falling:
        support = float(np.mean(y_troughs[-2:]))
        target = support - (float(np.mean(y_peaks[-2:])) - support)
        stop = float(np.mean(y_peaks[-2:])) * 1.002
        return PatternSignal(
            pattern="descending_triangle",
            signal=-1,
            confidence=0.72,
            target=target,
            stop=stop,
            description="Descending triangle: flat support, falling resistance",
        )

    # Symmetrical: converging (both sides sloping)
    if peak_falling and trough_rising:
        midpoint = (float(np.mean(y_peaks[-2:])) + float(np.mean(y_troughs[-2:]))) / 2
        atr = atr_approx
        return PatternSignal(
            pattern="symmetrical_triangle",
            signal=0,
            confidence=0.55,
            target=current + atr * 2,
            stop=current - atr * 1.5,
            description="Symmetrical triangle: breakout direction unknown",
        )

    return None


def detect_wedge(
    df: pd.DataFrame,
    lookback: int = 40,
    order: int = 4,
) -> Optional[PatternSignal]:
    """Detect rising wedge (bearish) or falling wedge (bullish)."""
    highs = df["high"].values[-lookback:]
    lows = df["low"].values[-lookback:]
    if len(highs) < 20:
        return None

    peak_idx = _find_local_maxima(highs, order=order)
    trough_idx = _find_local_minima(lows, order=order)
    if len(peak_idx) < 2 or len(trough_idx) < 2:
        return None

    x_p = peak_idx.astype(float)
    x_t = trough_idx.astype(float)
    peak_slope = _linear_regression_slope(x_p, highs[peak_idx])
    trough_slope = _linear_regression_slope(x_t, lows[trough_idx])

    atr = float(np.mean(highs - lows)) + 1e-8

    peak_rising = peak_slope / atr > 0.0001
    peak_falling = peak_slope / atr < -0.0001
    trough_rising = trough_slope / atr > 0.0001
    trough_falling = trough_slope / atr < -0.0001

    # Rising wedge: both lines rising but converging (trough rising faster)
    if peak_rising and trough_rising and trough_slope > peak_slope:
        stop = float(highs[peak_idx[-1]]) * 1.003
        target = float(lows[trough_idx[-1]]) * 0.99
        return PatternSignal(
            pattern="rising_wedge",
            signal=-1,
            confidence=0.68,
            target=target,
            stop=stop,
            description="Rising wedge: bearish reversal signal",
        )

    # Falling wedge: both lines falling but converging (trough falling faster)
    if peak_falling and trough_falling and peak_slope < trough_slope:
        stop = float(lows[trough_idx[-1]]) * 0.997
        target = float(highs[peak_idx[-1]]) * 1.01
        return PatternSignal(
            pattern="falling_wedge",
            signal=1,
            confidence=0.68,
            target=target,
            stop=stop,
            description="Falling wedge: bullish reversal signal",
        )

    return None


def detect_cup_and_handle(
    df: pd.DataFrame,
    min_cup_bars: int = 20,
    max_cup_bars: int = 50,
    handle_pct: float = 0.5,
) -> Optional[PatternSignal]:
    """Detect bullish Cup & Handle pattern."""
    close = df["close"].values
    n = len(close)
    if n < min_cup_bars + 5:
        return None

    # Look for cup: high → bottom → high
    window = close[-max_cup_bars:]
    left_rim = window[0]
    bottom = np.min(window[5:-5])
    right_rim = window[-1]

    # Both rims roughly equal
    if _pct_diff(left_rim, right_rim) > 0.05:
        return None
    if bottom >= left_rim * 0.97:  # must have a meaningful bowl
        return None

    depth = left_rim - bottom
    if depth / left_rim < 0.05:
        return None

    # Handle: small downward pullback in last 5–10 bars
    handle = close[-8:]
    handle_drop = np.max(handle) - np.min(handle)
    if handle_drop > depth * handle_pct:
        return None  # handle too deep

    current = close[-1]
    if current < right_rim * 0.995:
        return None  # not yet at breakout

    target = right_rim + depth
    stop = current * (1 - (handle_drop / current) * 1.2)
    confidence = float(np.clip(
        (1 - _pct_diff(left_rim, right_rim)) * 0.5 + 0.4, 0.45, 0.80
    ))

    return PatternSignal(
        pattern="cup_and_handle",
        signal=1,
        confidence=confidence,
        target=target,
        stop=stop,
        description=f"Cup&Handle: breakout at {right_rim:.5f}, target={target:.5f}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main agent class
# ─────────────────────────────────────────────────────────────────────────────

class PatternAgent:
    """
    Runs all pattern detectors on a OHLCV DataFrame and aggregates results.
    Used as an additional signal layer in the orchestrator.
    """

    def __init__(
        self,
        order: int = 5,
        tolerance: float = 0.025,
        min_confidence: float = 0.50,
    ):
        self.order = order
        self.tolerance = tolerance
        self.min_confidence = min_confidence

    def detect(self, df: pd.DataFrame, pair: str = "") -> PatternAgentResult:
        """
        Run all pattern detectors on df.
        Returns PatternAgentResult with all detected patterns.
        """
        if df is None or len(df) < 30:
            return PatternAgentResult(pair=pair)

        result = PatternAgentResult(pair=pair)
        detectors = [
            lambda: detect_head_and_shoulders(df, order=self.order, tolerance=self.tolerance),
            lambda: detect_inverse_head_and_shoulders(df, order=self.order, tolerance=self.tolerance),
            lambda: detect_double_top(df, order=self.order, tolerance=self.tolerance),
            lambda: detect_double_bottom(df, order=self.order, tolerance=self.tolerance),
            lambda: detect_triangle(df, order=self.order),
            lambda: detect_wedge(df, order=self.order),
            lambda: detect_cup_and_handle(df),
        ]

        for detector in detectors:
            try:
                signal = detector()
                if signal and signal.confidence >= self.min_confidence:
                    result.patterns.append(signal)
            except Exception as e:
                logger.debug("Pattern detector error: %s", e)

        if result.patterns:
            logger.info(
                "PatternAgent %s: %d patterns — dominant=%+d conf=%.2f",
                pair,
                len(result.patterns),
                result.dominant_signal,
                result.composite_confidence,
            )

        return result

    def get_signal(self, df: pd.DataFrame, pair: str = "") -> dict:
        """
        Convenience method — returns a flat signal dict compatible
        with the orchestrator's signal format.
        """
        r = self.detect(df, pair)
        return {
            "signal": r.dominant_signal,
            "confidence": r.composite_confidence,
            "patterns": [p.pattern for p in r.patterns],
            "details": r.to_dict(),
        }
