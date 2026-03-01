"""
Technical Analysis Agent — dedicated TA signal generator.
Analyses price, indicators, candle patterns, and regime to produce
a structured signal dict without any ML model inference.
Used by the orchestrator as one voting member of the ensemble.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from features.candle_patterns import compute_candle_features
from features.regime_detector import detect_regime, Regime

logger = logging.getLogger(__name__)


class TechnicalAgent:
    """
    Rule-based technical analysis agent.
    Combines: trend, momentum, volatility, candle patterns, and regime.
    Returns a signal: +1 buy, -1 sell, 0 hold.
    """

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        adx_trend_threshold: float = 25.0,
        min_confidence: float = 0.55,
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.adx_trend_threshold = adx_trend_threshold
        self.min_confidence = min_confidence

    def analyse(self, df: pd.DataFrame) -> dict:
        """
        Analyse the latest bar and return a structured signal.

        Returns:
            {
              "signal":      +1 | -1 | 0,
              "confidence":  float 0–1,
              "reasons":     list[str],
              "regime":      str,
            }
        """
        if len(df) < 50:
            return {"signal": 0, "confidence": 0.0, "reasons": ["Insufficient data"], "regime": "unknown"}

        # Compute candle patterns if not already present
        if "pat_composite" not in df.columns:
            df = compute_candle_features(df)

        row = df.iloc[-1]
        reasons: list[str] = []
        bullish_score = 0.0
        bearish_score = 0.0

        # ── Trend (EMA crossover + ADX) ──────────────────────────────────
        ema_20 = row.get("ema_20", np.nan)
        ema_50 = row.get("ema_50", np.nan)
        ema_200 = row.get("ema_200", np.nan)
        adx = row.get("adx_14", 0.0)
        close = row.get("close", 0.0)

        if not (np.isnan(ema_20) or np.isnan(ema_50)):
            if ema_20 > ema_50 and adx > self.adx_trend_threshold:
                bullish_score += 1.5
                reasons.append(f"EMA20 ({ema_20:.5f}) above EMA50 ({ema_50:.5f}), ADX={adx:.1f}")
            elif ema_20 < ema_50 and adx > self.adx_trend_threshold:
                bearish_score += 1.5
                reasons.append(f"EMA20 ({ema_20:.5f}) below EMA50 ({ema_50:.5f}), downtrend (ADX={adx:.1f})")

        if not np.isnan(ema_200):
            if close > ema_200:
                bullish_score += 0.5
                reasons.append("Price above EMA200 (long-term bull)")
            else:
                bearish_score += 0.5
                reasons.append("Price below EMA200 (long-term bear)")

        # ── RSI momentum ─────────────────────────────────────────────────
        rsi = row.get("rsi_14", 50.0)
        if rsi < self.rsi_oversold:
            bullish_score += 1.2
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > self.rsi_overbought:
            bearish_score += 1.2
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif 45 < rsi < 60:
            bullish_score += 0.3  # neutral-bullish zone

        # ── MACD ─────────────────────────────────────────────────────────
        macd = row.get("macd", np.nan)
        macd_signal = row.get("macd_signal", np.nan)
        if not (np.isnan(macd) or np.isnan(macd_signal)):
            if macd > macd_signal and macd > 0:
                bullish_score += 1.0
                reasons.append("MACD bullish crossover above zero")
            elif macd < macd_signal and macd < 0:
                bearish_score += 1.0
                reasons.append("MACD bearish crossover below zero")

        # ── Bollinger Bands ───────────────────────────────────────────────
        bb_pct = row.get("bb_pct", 0.5)
        bb_squeeze = row.get("bb_squeeze", 0.0)
        if bb_pct < 0.1:
            bullish_score += 0.8
            reasons.append(f"Price near lower Bollinger Band (bb_pct={bb_pct:.2f})")
        elif bb_pct > 0.9:
            bearish_score += 0.8
            reasons.append(f"Price near upper Bollinger Band (bb_pct={bb_pct:.2f})")
        if bb_squeeze:
            reasons.append("Bollinger squeeze — breakout imminent")

        # ── Candle patterns ───────────────────────────────────────────────
        pat_composite = row.get("pat_composite", 0.0)
        if pat_composite > 1.5:
            bullish_score += min(pat_composite * 0.4, 1.5)
            reasons.append(f"Bullish candle patterns (composite={pat_composite:.1f})")
        elif pat_composite < -1.5:
            bearish_score += min(-pat_composite * 0.4, 1.5)
            reasons.append(f"Bearish candle patterns (composite={pat_composite:.1f})")

        # ── Volume confirmation ───────────────────────────────────────────
        vol_ratio = row.get("volume_ratio", 1.0)
        if vol_ratio > 1.5:
            if bullish_score > bearish_score:
                bullish_score += 0.5
                reasons.append(f"High volume confirms bullish move (ratio={vol_ratio:.1f})")
            else:
                bearish_score += 0.5
                reasons.append(f"High volume confirms bearish move (ratio={vol_ratio:.1f})")

        # ── Stochastic ────────────────────────────────────────────────────
        stoch_k = row.get("stoch_k", 50.0)
        stoch_cross = row.get("stoch_cross", 0.0)
        if stoch_k < 20 and stoch_cross:
            bullish_score += 0.6
            reasons.append(f"Stochastic bullish cross in oversold ({stoch_k:.1f})")
        elif stoch_k > 80 and not stoch_cross:
            bearish_score += 0.6
            reasons.append(f"Stochastic in overbought zone ({stoch_k:.1f})")

        # ── Regime adjustment ─────────────────────────────────────────────
        regime_state = detect_regime(df)
        regime = regime_state.regime

        if regime == Regime.VOLATILE:
            bullish_score *= 0.5
            bearish_score *= 0.5
            reasons.append("Regime: volatile — reducing signal strength")
        elif regime == Regime.QUIET:
            bullish_score *= 0.7
            bearish_score *= 0.7
            reasons.append("Regime: quiet/low-vol — reduced confidence")
        elif regime == Regime.TRENDING_UP:
            bullish_score *= 1.2
        elif regime == Regime.TRENDING_DOWN:
            bearish_score *= 1.2

        # ── Decision ─────────────────────────────────────────────────────
        total = bullish_score + bearish_score + 1e-8
        if bullish_score > bearish_score:
            signal = 1
            confidence = min(bullish_score / total + 0.1, 0.95)
        elif bearish_score > bullish_score:
            signal = -1
            confidence = min(bearish_score / total + 0.1, 0.95)
        else:
            signal = 0
            confidence = 0.0

        if confidence < self.min_confidence:
            signal = 0

        return {
            "signal": signal,
            "confidence": round(confidence, 3),
            "reasons": reasons[:6],   # top 6 reasons
            "regime": regime.value,
            "bullish_score": round(bullish_score, 2),
            "bearish_score": round(bearish_score, 2),
        }
