"""
Feature store — Redis-based with in-memory fallback.
Stores computed features keyed by (pair, timeframe, timestamp).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Lightweight feature store.
    Backend: Redis (production) or in-memory dict (fallback).
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl_seconds: int = 3600):
        self._ttl = ttl_seconds
        self._mem: dict[str, Any] = {}
        self._redis = None

        try:
            import redis
            r = redis.from_url(redis_url, socket_connect_timeout=2)
            r.ping()
            self._redis = r
            logger.info("FeatureStore: Redis connected at %s", redis_url)
        except Exception:
            logger.warning("FeatureStore: Redis unavailable — using in-memory store")

    def _key(self, pair: str, timeframe: str, timestamp: str) -> str:
        return f"features:{pair}:{timeframe}:{timestamp}"

    def store(
        self,
        pair: str,
        timeframe: str,
        timestamp: str,
        features: dict[str, float],
    ) -> None:
        key = self._key(pair, timeframe, timestamp)
        data = json.dumps(features)
        if self._redis:
            try:
                self._redis.setex(key, self._ttl, data)
                return
            except Exception as e:
                logger.debug("Redis store error: %s", e)
        self._mem[key] = (data, time.time() + self._ttl)

    def retrieve(
        self,
        pair: str,
        timeframe: str,
        timestamp: str,
    ) -> dict[str, float] | None:
        key = self._key(pair, timeframe, timestamp)
        if self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    return json.loads(data)
            except Exception:
                pass
        # In-memory fallback
        entry = self._mem.get(key)
        if entry:
            data, expiry = entry
            if time.time() < expiry:
                return json.loads(data)
            del self._mem[key]
        return None

    def store_dataframe(self, pair: str, timeframe: str, df: pd.DataFrame) -> None:
        """Store each row of a feature DataFrame."""
        for ts, row in df.iterrows():
            features = {k: float(v) for k, v in row.items() if pd.notna(v)}
            self.store(pair, timeframe, str(ts), features)

    def health_check(self) -> dict[str, Any]:
        return {
            "backend": "redis" if self._redis else "memory",
            "memory_keys": len(self._mem),
        }
