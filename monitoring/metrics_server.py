"""
Prometheus metrics server + drift detection.
Exposes equity, sharpe, trade counts, system health via /metrics.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, start_http_server
    )
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False
    logger.warning("prometheus-client not installed — metrics disabled")


class MetricsServer:
    """
    Prometheus metrics for the Forex trader.
    Starts an HTTP server on configured port.
    """

    def __init__(self, port: int = 9090):
        self.port = port
        self._started = False

        if not _HAS_PROMETHEUS:
            return

        # Define metrics
        self.equity = Gauge("forex_equity", "Current account equity", ["pair"])
        self.drawdown = Gauge("forex_drawdown_pct", "Current drawdown percentage")
        self.sharpe = Gauge("forex_sharpe_ratio", "Rolling Sharpe ratio", ["pair"])
        self.trades_total = Counter("forex_trades_total", "Total trades executed", ["pair", "direction"])
        self.wins_total = Counter("forex_wins_total", "Total winning trades", ["pair"])
        self.losses_total = Counter("forex_losses_total", "Total losing trades", ["pair"])
        self.sentiment_score = Gauge("forex_sentiment_score", "Current sentiment", ["pair"])
        self.model_confidence = Gauge("forex_model_confidence", "Model prediction confidence", ["model", "pair"])
        self.signal_latency = Histogram("forex_signal_latency_ms",
                                        "Signal generation latency in ms",
                                        buckets=[10, 25, 50, 100, 200, 500, 1000])
        self.circuit_breaker = Gauge("forex_circuit_breaker", "Circuit breaker status (1=triggered)")
        self.survival_mode = Gauge("forex_survival_mode", "Survival mode status (1=active)")
        self.system_cpu = Gauge("forex_system_cpu_pct", "System CPU usage")
        self.system_mem = Gauge("forex_system_mem_pct", "System memory usage")
        self.drift_detected = Gauge("forex_drift_detected", "Feature drift detected", ["pair"])
        self.model_accuracy = Gauge("forex_model_accuracy", "Recent model accuracy", ["model", "pair"])

    def start(self) -> None:
        if not _HAS_PROMETHEUS or self._started:
            return
        try:
            start_http_server(self.port)
            self._started = True
            logger.info("Prometheus metrics server started on port %d", self.port)
        except Exception as e:
            logger.warning("Prometheus server failed to start: %s", e)

    def update_equity(self, equity: float, pair: str = "portfolio") -> None:
        if _HAS_PROMETHEUS and self._started:
            self.equity.labels(pair=pair).set(equity)

    def update_trade(self, pair: str, direction: str, pnl: float) -> None:
        if _HAS_PROMETHEUS and self._started:
            self.trades_total.labels(pair=pair, direction=direction).inc()
            if pnl > 0:
                self.wins_total.labels(pair=pair).inc()
            else:
                self.losses_total.labels(pair=pair).inc()

    def update_sentiment(self, pair: str, score: float) -> None:
        if _HAS_PROMETHEUS and self._started:
            self.sentiment_score.labels(pair=pair).set(score)

    def update_model_confidence(self, model: str, pair: str, confidence: float) -> None:
        if _HAS_PROMETHEUS and self._started:
            self.model_confidence.labels(model=model, pair=pair).set(confidence)

    def record_latency(self, latency_ms: float) -> None:
        if _HAS_PROMETHEUS and self._started:
            self.signal_latency.observe(latency_ms)

    def update_system_health(self, cpu_pct: float, mem_pct: float) -> None:
        if _HAS_PROMETHEUS and self._started:
            self.system_cpu.set(cpu_pct)
            self.system_mem.set(mem_pct)

    def set_circuit_breaker(self, triggered: bool) -> None:
        if _HAS_PROMETHEUS and self._started:
            self.circuit_breaker.set(1.0 if triggered else 0.0)

    def set_survival_mode(self, active: bool) -> None:
        if _HAS_PROMETHEUS and self._started:
            self.survival_mode.set(1.0 if active else 0.0)
