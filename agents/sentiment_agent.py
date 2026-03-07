"""
SENTIENCE sentiment engine.
FinBERT (loaded locally from HuggingFace) + VADER + LLM summarizer.
No external API required — all models run locally.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    _HAS_VADER = True
except ImportError:
    _HAS_VADER = False
    logger.warning("vaderSentiment not installed")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    logger.warning("transformers not installed — FinBERT disabled")


CURRENCY_KEYWORDS = {
    "USD": ["usd", "dollar", "fed", "federal reserve", "fomc", "powell"],
    "EUR": ["eur", "euro", "ecb", "european central bank", "lagarde"],
    "GBP": ["gbp", "pound", "sterling", "boe", "bank of england", "bailey"],
    "JPY": ["jpy", "yen", "boj", "bank of japan", "ueda"],
    "AUD": ["aud", "aussie", "rba", "reserve bank australia"],
    "CAD": ["cad", "loonie", "boc", "bank of canada"],
    "NZD": ["nzd", "kiwi", "rbnz"],
    "CHF": ["chf", "franc", "snb", "swiss national bank"],
}

PAIR_CURRENCIES = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"), "AUDUSD": ("AUD", "USD"),
    "USDCAD": ("USD", "CAD"), "NZDUSD": ("NZD", "USD"),
    "USDCHF": ("USD", "CHF"),
}


def _vader_score(text: str) -> float:
    """VADER compound score: -1 to +1."""
    if not _HAS_VADER:
        return 0.0
    scores = _vader.polarity_scores(text)
    return float(scores["compound"])


def _currency_relevance(text: str) -> dict[str, float]:
    """Score how relevant each currency is in the text (0 to 1)."""
    text_lower = text.lower()
    relevance = {}
    for currency, keywords in CURRENCY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        relevance[currency] = min(hits / 2.0, 1.0)
    return relevance


class SentimentAgent:
    """
    Multi-source sentiment aggregator.
    Uses FinBERT (local) + VADER for scoring,
    aggregates per currency pair.
    """

    def __init__(
        self,
        finbert_model: str = "ProsusAI/finbert",
        use_finbert: bool = True,
        aggregate_window_hours: int = 4,
    ):
        self.aggregate_window_hours = aggregate_window_hours
        self.scores_history: list[dict] = []
        self._finbert = None

        if use_finbert and _HAS_TRANSFORMERS:
            try:
                logger.info("Loading FinBERT model: %s (first run downloads ~400MB)", finbert_model)
                self._finbert = pipeline(
                    "text-classification",
                    model=finbert_model,
                    tokenizer=finbert_model,
                    device=-1,  # CPU; set to 0 for GPU
                    truncation=True,
                    max_length=512,
                )
                logger.info("FinBERT loaded successfully")
            except Exception as e:
                logger.warning("FinBERT load failed: %s — using VADER only", e)

    def _finbert_score(self, text: str) -> float:
        """FinBERT: returns score in [-1, +1]."""
        if self._finbert is None:
            return 0.0
        try:
            result = self._finbert(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]
            if label == "positive":
                return score
            elif label == "negative":
                return -score
            return 0.0
        except Exception:
            return 0.0

    def score_text(self, text: str) -> float:
        """
        Combined score: FinBERT (0.7 weight) + VADER (0.3 weight).
        Returns [-1, +1].
        """
        finbert = self._finbert_score(text)
        vader = _vader_score(text)

        if self._finbert is not None:
            combined = 0.7 * finbert + 0.3 * vader
        else:
            combined = vader

        return float(np.clip(combined, -1.0, 1.0))

    def process_articles(self, articles: list[dict]) -> list[dict]:
        """Score a list of news/tweet dicts."""
        scored = []
        for art in articles:
            text = f"{art.get('title', '')} {art.get('text', art.get('summary', ''))}"
            score = self.score_text(text)
            relevance = _currency_relevance(text)
            scored.append({
                **art,
                "sentiment_score": score,
                "currency_relevance": relevance,
                "scored_at": datetime.now(tz=timezone.utc).isoformat(),
            })
        return scored

    def aggregate_pair_sentiment(
        self,
        scored_articles: list[dict],
        pair: str,
    ) -> float:
        """
        Aggregate sentiment for a specific pair over recent window.
        Returns [-1, +1] where +1 = strongly bullish, -1 = strongly bearish.
        """
        if pair not in PAIR_CURRENCIES:
            return 0.0

        base_ccy, quote_ccy = PAIR_CURRENCIES[pair]
        cutoff = time.time() - self.aggregate_window_hours * 3600

        relevant = [
            a for a in scored_articles
            if a.get("timestamp", 0) > cutoff
        ]

        if not relevant:
            return 0.0

        scores = []
        weights = []
        for art in relevant:
            rel = art.get("currency_relevance", {})
            base_rel = rel.get(base_ccy, 0.1)
            quote_rel = rel.get(quote_ccy, 0.1)
            relevance_weight = max(base_rel, quote_rel) + 0.1

            raw_score = art.get("sentiment_score", 0.0)
            # For USDJPY: USD positive = pair up (USD stronger)
            # USD is always quote for EUR/GBP/AUD/NZD — so positive USD sentiment = pair down
            if quote_ccy == "USD":
                adjusted_score = raw_score * base_rel - raw_score * quote_rel
            else:
                adjusted_score = raw_score * quote_rel - raw_score * base_rel

            scores.append(adjusted_score)
            weights.append(relevance_weight)

        if not weights:
            return 0.0

        weighted_avg = np.average(scores, weights=weights)
        return float(np.clip(weighted_avg, -1.0, 1.0))

    def get_signal(self, sentiment_score: float, threshold: float = 0.3) -> int:
        """Convert score to signal: +1 bullish, -1 bearish, 0 neutral."""
        if sentiment_score > threshold:
            return 1
        elif sentiment_score < -threshold:
            return -1
        return 0

    def get_all_pair_sentiments(
        self,
        scored_articles: list[dict],
        pairs: list[str],
    ) -> dict[str, float]:
        """Return sentiment scores for all pairs."""
        return {pair: self.aggregate_pair_sentiment(scored_articles, pair) for pair in pairs}
