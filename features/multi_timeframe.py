"""
Multi-timeframe (MTF) confluence analysis.
Fetches H1, H4, D1 data for a pair and checks whether all timeframes
agree on direction before issuing a signal.
Higher confluence score = stronger signal.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MTFResult:
    """Multi-timeframe confluence result for a single pair."""

    pair: str
    # Per-timeframe directional bias: +1 bullish, -1 bearish, 0 neutral
    bias: dict[str, int]        # e.g. {"M15": 1, "H1": 1, "H4": 1, "D1": -1}
    # Trend strength per timeframe (ADX normalised 0-1)
    strength: dict[str, float]

    # Composite
    confluence_score: float     # -1 (strong bear) → +1 (strong bull)
    agreed_direction: int       # majority direction after weighing
    n_aligned: int              # number of timeframes that agree with majority
    n_total: int                # total timeframes analysed

    @property
    def confidence(self) -> float:
        """Fraction of timeframes aligned with majority direction."""
        return self.n_aligned / max(self.n_total, 1)

    @property
    def is_strong_confluence(self) -> bool:
        """True if ≥75% of timeframes agree."""
        return self.confidence >= 0.75 and abs(self.confluence_score) > 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Bias detector for a single timeframe
# ─────────────────────────────────────────────────────────────────────────────

def _detect_bias(df: pd.DataFrame) -> tuple[int, float]:
    """
    Detect directional bias from a pre-computed indicator DataFrame.
    Returns (direction: -1/0/1, strength: 0-1).
    """
    if df is None or len(df) < 20:
        return 0, 0.0

    last = df.iloc[-1]

    bull_votes = 0.0
    bear_votes = 0.0
    total_weight = 0.0

    # ── EMA trend ──────────────────────────────────────────────────────────
    ema20 = last.get("ema_20", np.nan)
    ema50 = last.get("ema_50", np.nan)
    ema200 = last.get("ema_200", np.nan)
    close = last.get("close", np.nan)

    if not np.isnan(ema20) and not np.isnan(ema50):
        w = 1.5
        if ema20 > ema50:
            bull_votes += w
        else:
            bear_votes += w
        total_weight += w

    if not np.isnan(ema50) and not np.isnan(ema200):
        w = 2.0
        if ema50 > ema200:
            bull_votes += w
        else:
            bear_votes += w
        total_weight += w

    if not np.isnan(close) and not np.isnan(ema200):
        w = 1.0
        if close > ema200:
            bull_votes += w
        else:
            bear_votes += w
        total_weight += w

    # ── RSI ─────────────────────────────────────────────────────────────────
    rsi = last.get("rsi_14", np.nan)
    if not np.isnan(rsi):
        w = 1.0
        if rsi > 55:
            bull_votes += w
        elif rsi < 45:
            bear_votes += w
        total_weight += w

    # ── MACD ────────────────────────────────────────────────────────────────
    macd = last.get("macd", np.nan)
    macd_signal = last.get("macd_signal", np.nan)
    if not np.isnan(macd) and not np.isnan(macd_signal):
        w = 1.0
        if macd > macd_signal and macd > 0:
            bull_votes += w
        elif macd < macd_signal and macd < 0:
            bear_votes += w
        total_weight += w

    # ── ADX strength ────────────────────────────────────────────────────────
    adx = last.get("adx_14", np.nan)
    if np.isnan(adx):
        adx = last.get("adx", np.nan)
    trend_strength = float(np.clip(adx / 50.0, 0.0, 1.0)) if not np.isnan(adx) else 0.5

    if total_weight == 0:
        return 0, 0.0

    net = (bull_votes - bear_votes) / (total_weight + 1e-8)  # -1 to +1
    direction = 1 if net > 0.1 else (-1 if net < -0.1 else 0)
    return direction, float(np.clip(abs(net) * trend_strength, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timeframe analysis
# ─────────────────────────────────────────────────────────────────────────────

# Timeframes from shortest to longest, with their directional weight
_DEFAULT_TIMEFRAMES = ["M15", "H1", "H4", "D1"]
_TF_WEIGHTS = {"M15": 0.5, "H1": 1.0, "H4": 2.0, "D1": 3.0}


class MultiTimeframeAnalyser:
    """
    Fetches OHLCV at multiple timeframes and computes directional confluence.
    Uses a lazy cache to avoid redundant fetches within the same cycle.
    """

    def __init__(
        self,
        timeframes: list[str] | None = None,
        bars_per_tf: int = 200,
        cache_ttl_seconds: int = 300,
    ):
        self.timeframes = timeframes or _DEFAULT_TIMEFRAMES
        self.bars = bars_per_tf
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}  # (timestamp, df)

    # ─────────────────────────────────────────────────────────────────────
    # Data access (with cache)
    # ─────────────────────────────────────────────────────────────────────

    def _get_df(self, pair: str, timeframe: str, cfg: dict) -> pd.DataFrame:
        key = f"{pair}_{timeframe}"
        ts, cached_df = self._cache.get(key, (0.0, pd.DataFrame()))
        if time.time() - ts < self.cache_ttl and not cached_df.empty:
            return cached_df

        try:
            from data_pipeline.market_data import fetch_ohlcv
            from features.technical_indicators import compute_all_indicators
            df = fetch_ohlcv(pair, timeframe=timeframe, bars=self.bars, cfg=cfg)
            if not df.empty:
                df = compute_all_indicators(df, pair)
                df = df.dropna(subset=["close"])
        except Exception as e:
            logger.debug("MTF fetch %s %s: %s", pair, timeframe, e)
            df = pd.DataFrame()

        self._cache[key] = (time.time(), df)
        return df

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def analyse(
        self,
        pair: str,
        cfg: dict | None = None,
        preloaded: dict[str, pd.DataFrame] | None = None,
    ) -> MTFResult:
        """
        Run multi-timeframe analysis for a pair.

        Parameters
        ----------
        pair        : Currency pair e.g. "EURUSD".
        cfg         : Config dict (passed to fetch_ohlcv).
        preloaded   : Optional dict of {timeframe: df} — skips fetching.
        """
        cfg = cfg or {}
        bias: dict[str, int] = {}
        strength: dict[str, float] = {}

        for tf in self.timeframes:
            if preloaded and tf in preloaded:
                df = preloaded[tf]
            else:
                df = self._get_df(pair, tf, cfg)

            direction, stren = _detect_bias(df)
            bias[tf] = direction
            strength[tf] = stren

        # ── Weighted confluence score ──────────────────────────────────────
        weighted_sum = 0.0
        weight_total = 0.0
        for tf in self.timeframes:
            w = _TF_WEIGHTS.get(tf, 1.0)
            weighted_sum += bias.get(tf, 0) * strength.get(tf, 0.5) * w
            weight_total += w

        confluence_score = float(
            np.clip(weighted_sum / (weight_total + 1e-8), -1.0, 1.0)
        )
        agreed_direction = 1 if confluence_score > 0.1 else (-1 if confluence_score < -0.1 else 0)

        # Count how many agree with the majority
        n_aligned = sum(
            1 for d in bias.values() if d == agreed_direction and d != 0
        )
        n_total = len([d for d in bias.values() if d != 0]) or 1

        return MTFResult(
            pair=pair,
            bias=bias,
            strength=strength,
            confluence_score=round(confluence_score, 4),
            agreed_direction=agreed_direction,
            n_aligned=n_aligned,
            n_total=n_total,
        )

    def analyse_batch(
        self,
        pairs: list[str],
        cfg: dict | None = None,
    ) -> dict[str, MTFResult]:
        """Analyse all pairs and return a dict of results."""
        return {p: self.analyse(p, cfg=cfg) for p in pairs}

    def filter_signal(
        self,
        pair: str,
        proposed_direction: int,
        mtf_result: MTFResult,
        min_confidence: float = 0.5,
    ) -> tuple[bool, str]:
        """
        Return (allowed, reason).
        Blocks a trade if higher timeframes disagree.
        """
        if mtf_result.n_total == 0:
            return True, "MTF: no data — not blocking"

        if mtf_result.agreed_direction == 0:
            # Consolidating — reduce to allowing only if short-term aligns
            return True, "MTF: consolidating — short-term signal allowed"

        if mtf_result.agreed_direction != proposed_direction:
            return False, (
                f"MTF blocked: {pair} confluence={mtf_result.confluence_score:.2f} "
                f"({mtf_result.n_aligned}/{mtf_result.n_total} TFs disagree)"
            )

        if mtf_result.confidence < min_confidence:
            return True, f"MTF weak ({mtf_result.confidence:.0%}) — signal allowed"

        return True, (
            f"MTF confirmed: {mtf_result.confluence_score:+.2f} "
            f"({mtf_result.n_aligned}/{mtf_result.n_total} TFs aligned)"
        )
