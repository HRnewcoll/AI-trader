"""
Correlation Agent — tracks currency pair correlations and prevents
over-exposure to correlated positions.
Also provides a correlation-weighted signal boost/veto.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Known static correlation groups (same base/quote currency)
# Pairs in the same group tend to move together
CORRELATION_GROUPS: list[list[str]] = [
    ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],  # USD strength / risk-on
    ["USDJPY", "USDCHF"],                        # Safe-haven USD
    ["USDCAD"],                                  # Oil-linked
    ["EURUSD", "EURGBP", "EURJPY"],              # EUR crosses
]

# Static baseline correlation matrix (approximate, real-time updated when data available)
BASELINE_CORRELATIONS: dict[tuple[str, str], float] = {
    ("EURUSD", "GBPUSD"): 0.82,
    ("EURUSD", "AUDUSD"): 0.75,
    ("EURUSD", "NZDUSD"): 0.70,
    ("GBPUSD", "AUDUSD"): 0.72,
    ("USDJPY", "USDCHF"): 0.78,
    ("EURUSD", "USDJPY"): -0.65,
    ("GBPUSD", "USDJPY"): -0.60,
    ("AUDUSD", "USDCAD"): -0.55,
}


class CorrelationAgent:
    """
    Manages currency correlation to:
    1. Prevent holding too many correlated positions simultaneously.
    2. Boost confidence when non-correlated pairs show the same signal.
    3. Provide a dynamic correlation matrix updated from live OHLCV data.
    """

    def __init__(
        self,
        max_correlation: float = 0.75,
        max_correlated_open: int = 2,
        lookback_bars: int = 60,
    ):
        self.max_correlation = max_correlation
        self.max_correlated_open = max_correlated_open
        self.lookback_bars = lookback_bars
        self._live_corr: pd.DataFrame | None = None  # updated from live data
        self._open_positions: dict[str, int] = {}    # pair → direction

    # ── Live correlation matrix ────────────────────────────────────────────

    def update_correlation_matrix(self, price_data: dict[str, pd.DataFrame]) -> None:
        """
        Recompute correlation matrix from recent close prices.
        price_data: {pair: DataFrame with 'close' column}
        """
        closes = {}
        for pair, df in price_data.items():
            if "close" in df.columns and len(df) >= self.lookback_bars:
                closes[pair] = df["close"].iloc[-self.lookback_bars:].pct_change().dropna()

        if len(closes) < 2:
            return

        close_df = pd.DataFrame(closes).dropna()
        if len(close_df) < 10:
            return

        self._live_corr = close_df.corr()
        logger.debug("Correlation matrix updated for %d pairs", len(closes))

    def get_correlation(self, pair1: str, pair2: str) -> float:
        """Get correlation between two pairs (live if available, static fallback)."""
        if self._live_corr is not None:
            if pair1 in self._live_corr.index and pair2 in self._live_corr.columns:
                return float(self._live_corr.loc[pair1, pair2])

        key = (pair1, pair2)
        rev = (pair2, pair1)
        return BASELINE_CORRELATIONS.get(key, BASELINE_CORRELATIONS.get(rev, 0.0))

    # ── Position management ────────────────────────────────────────────────

    def register_open(self, pair: str, direction: int) -> None:
        self._open_positions[pair] = direction

    def register_close(self, pair: str) -> None:
        self._open_positions.pop(pair, None)

    def can_open_position(self, pair: str, direction: int) -> tuple[bool, str]:
        """
        Check if opening this position would violate correlation limits.
        Returns (allowed, reason).
        """
        correlated_open = 0
        conflicting = []

        for open_pair, open_dir in self._open_positions.items():
            corr = self.get_correlation(pair, open_pair)

            # High positive correlation + same direction = over-exposure
            if abs(corr) > self.max_correlation:
                effective_dir = open_dir * (1 if corr > 0 else -1)
                if effective_dir == direction:
                    correlated_open += 1
                    conflicting.append(f"{open_pair} (corr={corr:.2f})")

        if correlated_open >= self.max_correlated_open:
            return False, f"Too many correlated positions: {', '.join(conflicting)}"

        return True, "OK"

    def get_diversification_score(self, signals: dict[str, int]) -> float:
        """
        Score 0–1 for how diversified a set of simultaneous signals is.
        1.0 = all uncorrelated, 0.0 = all same direction on correlated pairs.
        """
        pairs = list(signals.keys())
        if len(pairs) <= 1:
            return 1.0

        conflict_count = 0
        total_pairs = 0
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                corr = self.get_correlation(pairs[i], pairs[j])
                if abs(corr) > self.max_correlation:
                    total_pairs += 1
                    if signals[pairs[i]] == signals[pairs[j]] * (1 if corr > 0 else -1):
                        conflict_count += 1

        if total_pairs == 0:
            return 1.0
        return 1.0 - conflict_count / total_pairs

    def get_correlation_summary(self, pairs: list[str]) -> dict:
        """Return a summary of correlations for the given pair list."""
        matrix = {}
        for p1 in pairs:
            for p2 in pairs:
                if p1 != p2:
                    matrix[f"{p1}-{p2}"] = round(self.get_correlation(p1, p2), 3)
        return {
            "matrix": matrix,
            "open_positions": dict(self._open_positions),
            "live_data_available": self._live_corr is not None,
        }
