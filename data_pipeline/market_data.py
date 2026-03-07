"""
Market data ingestion — OANDA / MT5 / yfinance with automatic fallback.
Provides both historical OHLCV and real-time tick streaming.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

TIMEFRAME_MAP = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1d", "W1": "1wk",
}

OANDA_GRAN = {
    "M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30",
    "H1": "H1", "H4": "H4", "D1": "D", "W1": "W",
}


def _pair_to_yf(pair: str) -> str:
    """EURUSD → EURUSD=X"""
    return f"{pair}=X"


def _pair_to_oanda(pair: str) -> str:
    """EURUSD → EUR_USD"""
    return f"{pair[:3]}_{pair[3:]}"


# ─────────────────────────────────────────────────────────────────────────────
# yfinance (free, no API key)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yfinance(pair: str, timeframe: str = "H4", bars: int = 500) -> pd.DataFrame:
    """Fetch OHLCV via yfinance — free, no API key required."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed")
        return pd.DataFrame()

    yf_period_map = {
        "M1": ("7d", "1m"), "M5": ("60d", "5m"), "M15": ("60d", "15m"),
        "M30": ("60d", "30m"), "H1": ("730d", "1h"), "H4": ("2y", "1h"),
        "D1": ("5y", "1d"), "W1": ("10y", "1wk"),
    }
    period, interval = yf_period_map.get(timeframe, ("2y", "1d"))

    ticker = yf.Ticker(_pair_to_yf(pair))
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        logger.warning("yfinance returned empty for %s %s", pair, timeframe)
        return df

    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "datetime"
    return df.tail(bars)


# ─────────────────────────────────────────────────────────────────────────────
# OANDA v20
# ─────────────────────────────────────────────────────────────────────────────

def fetch_oanda(
    pair: str,
    timeframe: str = "H4",
    bars: int = 500,
    account_id: str = "",
    token: str = "",
    environment: str = "practice",
) -> pd.DataFrame:
    """Fetch OHLCV from OANDA v20 REST API."""
    try:
        import oandapyV20
        import oandapyV20.endpoints.instruments as instruments
    except ImportError:
        logger.warning("oandapyV20 not installed — falling back to yfinance")
        return fetch_yfinance(pair, timeframe, bars)

    if not token:
        logger.warning("No OANDA token — falling back to yfinance")
        return fetch_yfinance(pair, timeframe, bars)

    client = oandapyV20.API(access_token=token, environment=environment)
    instrument = _pair_to_oanda(pair)
    gran = OANDA_GRAN.get(timeframe, "H4")

    rows = []
    try:
        params = {"count": min(bars, 5000), "granularity": gran, "price": "M"}
        r = instruments.InstrumentsCandles(instrument, params=params)
        client.request(r)
        for candle in r.response["candles"]:
            if candle["complete"]:
                mid = candle["mid"]
                rows.append({
                    "datetime": pd.Timestamp(candle["time"]),
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low":  float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": int(candle["volume"]),
                })
    except Exception as e:
        logger.error("OANDA fetch error: %s — falling back to yfinance", e)
        return fetch_yfinance(pair, timeframe, bars)

    df = pd.DataFrame(rows).set_index("datetime")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MT5
# ─────────────────────────────────────────────────────────────────────────────

def fetch_mt5(
    pair: str,
    timeframe: str = "H4",
    bars: int = 500,
    login: int = 0,
    password: str = "",
    server: str = "",
) -> pd.DataFrame:
    """Fetch from MetaTrader5 (Windows only)."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.warning("MetaTrader5 not available — falling back to yfinance")
        return fetch_yfinance(pair, timeframe, bars)

    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
    }
    if not mt5.initialize(login=login, password=password, server=server):
        logger.error("MT5 init failed: %s", mt5.last_error())
        return fetch_yfinance(pair, timeframe, bars)

    tf = tf_map.get(timeframe, mt5.TIMEFRAME_H4)
    rates = mt5.copy_rates_from_pos(pair, tf, 0, bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        return fetch_yfinance(pair, timeframe, bars)

    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("datetime").rename(
        columns={"open": "open", "high": "high", "low": "low",
                 "close": "close", "tick_volume": "volume"}
    )[["open", "high", "low", "close", "volume"]]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Unified fetcher with fallback chain
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    pair: str,
    timeframe: str = "H4",
    bars: int = 500,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Unified OHLCV fetch with automatic fallback:
    OANDA → MT5 → yfinance
    """
    cfg = cfg or {}
    broker = cfg.get("execution", {}).get("broker_api", "yfinance")

    df = pd.DataFrame()

    if broker == "oanda":
        account_id = cfg.get("execution", {}).get("oanda_account", "")
        token = cfg.get("execution", {}).get("oanda_token", "")
        df = fetch_oanda(pair, timeframe, bars, account_id, token)

    elif broker == "mt5":
        login = int(cfg.get("execution", {}).get("mt5_login", 0) or 0)
        password = cfg.get("execution", {}).get("mt5_password", "")
        server = cfg.get("execution", {}).get("mt5_server", "")
        df = fetch_mt5(pair, timeframe, bars, login, password, server)

    if df.empty:
        df = fetch_yfinance(pair, timeframe, bars)

    if not df.empty:
        df = df.sort_index()
        # Remove duplicate indices
        df = df[~df.index.duplicated(keep="last")]

    logger.info("Fetched %d bars for %s %s", len(df), pair, timeframe)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Streaming (Redis-based tick publisher)
# ─────────────────────────────────────────────────────────────────────────────

class TickStreamer:
    """Publishes simulated or live ticks to Redis stream."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            import redis
            self._redis = redis.from_url(redis_url)
        except Exception:
            self._redis = None
            logger.warning("Redis unavailable — tick streaming disabled")

    def publish(self, pair: str, bid: float, ask: float) -> None:
        if self._redis is None:
            return
        try:
            self._redis.xadd(
                f"ticks:{pair}",
                {"bid": bid, "ask": ask, "ts": time.time()},
                maxlen=10000,
            )
        except Exception as e:
            logger.debug("Redis publish error: %s", e)

    def simulate_ticks(
        self,
        pair: str,
        df: pd.DataFrame,
        spread_pips: float = 1.0,
        callback: Callable | None = None,
    ) -> None:
        """Replay historical OHLCV as tick stream (for backtesting)."""
        pip = 0.0001 if "JPY" not in pair else 0.01
        for _, row in df.iterrows():
            mid = (row["open"] + row["close"]) / 2
            spread = spread_pips * pip
            bid = mid - spread / 2
            ask = mid + spread / 2
            self.publish(pair, bid, ask)
            if callback:
                callback(pair, bid, ask)
