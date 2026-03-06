"""
Alert manager — event-driven alert system for the trader.

Monitors trading conditions and fires alerts when thresholds are breached:
  - Drawdown threshold (warning / critical)
  - Consecutive losses
  - Large single-trade loss
  - Model confidence drop (potential drift)
  - Correlation spike between open positions
  - Daily loss limit approaching
  - Circuit breaker / survival mode activation

Each alert is:
  - Logged to logs/alerts.jsonl (append-only)
  - Sent via the notifier (Telegram / Slack / email)
  - Accessible from the dashboard via the log file

Usage:
    manager = AlertManager(notifier=notifier)
    manager.check_all(equity=9500, peak_equity=10000, recent_trades=[...])
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Alert levels & types
# ─────────────────────────────────────────────────────────────────────────────

class AlertLevel(str, Enum):
    INFO    = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    DRAWDOWN           = "drawdown"
    CONSECUTIVE_LOSS   = "consecutive_loss"
    LARGE_LOSS         = "large_loss"
    LOW_CONFIDENCE     = "low_confidence"
    CORRELATION_SPIKE  = "correlation_spike"
    DAILY_LOSS_LIMIT   = "daily_loss_limit"
    CIRCUIT_BREAKER    = "circuit_breaker"
    SURVIVAL_MODE      = "survival_mode"
    MODEL_DRIFT        = "model_drift"
    TRADE_RATE_HIGH    = "trade_rate_high"
    SYSTEM_HEALTH      = "system_health"


@dataclass
class Alert:
    alert_type: str
    level: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    value: float = 0.0
    threshold: float = 0.0
    pair: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def emoji(self) -> str:
        return {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(self.level, "📢")

    def format_message(self) -> str:
        return (
            f"{self.emoji()} [{self.level.upper()}] {self.alert_type}\n"
            f"{self.message}\n"
            f"Value: {self.value:.4f}  |  Threshold: {self.threshold:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Alert thresholds config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlertConfig:
    # Drawdown
    drawdown_warning_pct: float = 0.03      # 3% → warning
    drawdown_critical_pct: float = 0.06     # 6% → critical

    # Consecutive losses
    consecutive_loss_warning: int = 3
    consecutive_loss_critical: int = 5

    # Single trade loss (as fraction of equity)
    large_loss_warning_pct: float = 0.02
    large_loss_critical_pct: float = 0.04

    # Model confidence
    low_confidence_warning: float = 0.50
    low_confidence_critical: float = 0.40

    # Daily loss
    daily_loss_warning_pct: float = 0.02
    daily_loss_critical_pct: float = 0.03

    # Correlation spike
    correlation_spike_threshold: float = 0.90

    # Trade rate (per hour — excessive trading detection)
    trade_rate_warning: int = 10
    trade_rate_critical: int = 20

    # Cooldown: don't re-fire same alert type within N seconds
    cooldown_seconds: int = 300


# ─────────────────────────────────────────────────────────────────────────────
# Alert Manager
# ─────────────────────────────────────────────────────────────────────────────

class AlertManager:
    """
    Central alert manager. Checks conditions and fires alerts.
    """

    def __init__(
        self,
        notifier=None,
        cfg: AlertConfig | None = None,
        log_dir: str = "logs",
    ):
        self.notifier = notifier
        self.cfg = cfg or AlertConfig()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.log_dir / "alerts.jsonl"
        self._last_fired: dict[str, float] = {}  # alert_type → timestamp
        self._alert_history: list[Alert] = []

        logger.info("AlertManager initialised. Log: %s", self._log_path)

    # ─────────────────────────────────────────────────────────────────────
    # Main check methods
    # ─────────────────────────────────────────────────────────────────────

    def check_drawdown(self, equity: float, peak_equity: float) -> Optional[Alert]:
        """Fire if current drawdown exceeds warning or critical thresholds."""
        if peak_equity <= 0:
            return None
        dd_pct = (peak_equity - equity) / peak_equity

        if dd_pct >= self.cfg.drawdown_critical_pct:
            return self._maybe_fire(Alert(
                alert_type=AlertType.DRAWDOWN,
                level=AlertLevel.CRITICAL,
                message=f"Drawdown {dd_pct:.1%} exceeds critical threshold {self.cfg.drawdown_critical_pct:.1%}",
                value=dd_pct,
                threshold=self.cfg.drawdown_critical_pct,
            ))
        elif dd_pct >= self.cfg.drawdown_warning_pct:
            return self._maybe_fire(Alert(
                alert_type=AlertType.DRAWDOWN,
                level=AlertLevel.WARNING,
                message=f"Drawdown {dd_pct:.1%} exceeds warning threshold {self.cfg.drawdown_warning_pct:.1%}",
                value=dd_pct,
                threshold=self.cfg.drawdown_warning_pct,
            ))
        return None

    def check_consecutive_losses(self, recent_pnls: list[float]) -> Optional[Alert]:
        """Fire if consecutive losing trades exceed threshold."""
        if not recent_pnls:
            return None
        count = 0
        for pnl in reversed(recent_pnls):
            if pnl < 0:
                count += 1
            else:
                break

        if count >= self.cfg.consecutive_loss_critical:
            return self._maybe_fire(Alert(
                alert_type=AlertType.CONSECUTIVE_LOSS,
                level=AlertLevel.CRITICAL,
                message=f"{count} consecutive losses — strategy may be failing",
                value=float(count),
                threshold=float(self.cfg.consecutive_loss_critical),
            ))
        elif count >= self.cfg.consecutive_loss_warning:
            return self._maybe_fire(Alert(
                alert_type=AlertType.CONSECUTIVE_LOSS,
                level=AlertLevel.WARNING,
                message=f"{count} consecutive losses",
                value=float(count),
                threshold=float(self.cfg.consecutive_loss_warning),
            ))
        return None

    def check_large_loss(self, trade_pnl: float, equity: float, pair: str = "") -> Optional[Alert]:
        """Fire if a single trade loss is abnormally large."""
        if trade_pnl >= 0 or equity <= 0:
            return None
        loss_pct = abs(trade_pnl) / equity

        if loss_pct >= self.cfg.large_loss_critical_pct:
            return self._maybe_fire(Alert(
                alert_type=AlertType.LARGE_LOSS,
                level=AlertLevel.CRITICAL,
                message=f"Single trade loss {loss_pct:.1%} on {pair or 'unknown'}",
                value=loss_pct,
                threshold=self.cfg.large_loss_critical_pct,
                pair=pair,
            ))
        elif loss_pct >= self.cfg.large_loss_warning_pct:
            return self._maybe_fire(Alert(
                alert_type=AlertType.LARGE_LOSS,
                level=AlertLevel.WARNING,
                message=f"Single trade loss {loss_pct:.1%} on {pair or 'unknown'}",
                value=loss_pct,
                threshold=self.cfg.large_loss_warning_pct,
                pair=pair,
            ))
        return None

    def check_model_confidence(self, confidence: float, pair: str = "") -> Optional[Alert]:
        """Fire if model confidence drops, suggesting possible drift."""
        if confidence <= self.cfg.low_confidence_critical:
            return self._maybe_fire(Alert(
                alert_type=AlertType.LOW_CONFIDENCE,
                level=AlertLevel.CRITICAL,
                message=f"Model confidence {confidence:.2f} critically low for {pair} — possible drift",
                value=confidence,
                threshold=self.cfg.low_confidence_critical,
                pair=pair,
            ))
        elif confidence <= self.cfg.low_confidence_warning:
            return self._maybe_fire(Alert(
                alert_type=AlertType.LOW_CONFIDENCE,
                level=AlertLevel.WARNING,
                message=f"Model confidence {confidence:.2f} below normal for {pair}",
                value=confidence,
                threshold=self.cfg.low_confidence_warning,
                pair=pair,
            ))
        return None

    def check_daily_loss(self, daily_pnl: float, equity: float) -> Optional[Alert]:
        """Fire if daily loss approaches limit."""
        if daily_pnl >= 0 or equity <= 0:
            return None
        daily_loss_pct = abs(daily_pnl) / equity

        if daily_loss_pct >= self.cfg.daily_loss_critical_pct:
            return self._maybe_fire(Alert(
                alert_type=AlertType.DAILY_LOSS_LIMIT,
                level=AlertLevel.CRITICAL,
                message=f"Daily loss {daily_loss_pct:.1%} at/above critical limit",
                value=daily_loss_pct,
                threshold=self.cfg.daily_loss_critical_pct,
            ))
        elif daily_loss_pct >= self.cfg.daily_loss_warning_pct:
            return self._maybe_fire(Alert(
                alert_type=AlertType.DAILY_LOSS_LIMIT,
                level=AlertLevel.WARNING,
                message=f"Daily loss {daily_loss_pct:.1%} approaching limit",
                value=daily_loss_pct,
                threshold=self.cfg.daily_loss_warning_pct,
            ))
        return None

    def check_circuit_breaker(self, is_triggered: bool) -> Optional[Alert]:
        if is_triggered:
            return self._maybe_fire(Alert(
                alert_type=AlertType.CIRCUIT_BREAKER,
                level=AlertLevel.CRITICAL,
                message="Circuit breaker triggered — trading halted",
                value=1.0,
                threshold=1.0,
            ))
        return None

    def check_survival_mode(self, is_active: bool) -> Optional[Alert]:
        if is_active:
            return self._maybe_fire(Alert(
                alert_type=AlertType.SURVIVAL_MODE,
                level=AlertLevel.WARNING,
                message="Survival mode active — position sizes reduced",
                value=1.0,
                threshold=1.0,
            ))
        return None

    def check_correlation_spike(self, max_correlation: float) -> Optional[Alert]:
        if max_correlation >= self.cfg.correlation_spike_threshold:
            return self._maybe_fire(Alert(
                alert_type=AlertType.CORRELATION_SPIKE,
                level=AlertLevel.WARNING,
                message=f"Open position correlation={max_correlation:.2f} — concentrated risk",
                value=max_correlation,
                threshold=self.cfg.correlation_spike_threshold,
            ))
        return None

    def check_all(
        self,
        equity: float = 0.0,
        peak_equity: float = 0.0,
        recent_pnls: list[float] | None = None,
        daily_pnl: float = 0.0,
        model_confidence: float = 1.0,
        is_circuit_broken: bool = False,
        is_survival_mode: bool = False,
        max_correlation: float = 0.0,
        pair: str = "",
    ) -> list[Alert]:
        """Run all checks and return list of fired alerts."""
        fired = []
        checks = [
            self.check_drawdown(equity, peak_equity),
            self.check_consecutive_losses(recent_pnls or []),
            self.check_daily_loss(daily_pnl, equity),
            self.check_model_confidence(model_confidence, pair),
            self.check_circuit_breaker(is_circuit_broken),
            self.check_survival_mode(is_survival_mode),
            self.check_correlation_spike(max_correlation),
        ]
        for alert in checks:
            if alert:
                fired.append(alert)
        return fired

    def fire(self, alert: Alert) -> None:
        """
        Explicitly fire a custom alert (bypasses cooldown check).
        Use this for one-off events.
        """
        self._log_alert(alert)
        self._send_alert(alert)
        self._alert_history.append(alert)

    # ─────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────

    def _maybe_fire(self, alert: Alert) -> Optional[Alert]:
        """Fire alert if not in cooldown."""
        import time
        key = f"{alert.alert_type}_{alert.level}"
        last = self._last_fired.get(key, 0.0)
        if time.time() - last < self.cfg.cooldown_seconds:
            return None  # still in cooldown

        self._last_fired[key] = time.time()
        self._log_alert(alert)
        self._send_alert(alert)
        self._alert_history.append(alert)
        logger.warning("ALERT [%s] %s: %s", alert.level.upper(), alert.alert_type, alert.message)
        return alert

    def _log_alert(self, alert: Alert) -> None:
        """Append alert to JSONL log file."""
        try:
            with self._log_path.open("a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except Exception as e:
            logger.debug("Alert log write error: %s", e)

    def _send_alert(self, alert: Alert) -> None:
        """Send alert via notifier if available."""
        if self.notifier is None:
            return
        try:
            self.notifier.send(
                subject=f"ForexAI Alert: {alert.alert_type}",
                message=alert.format_message(),
                level=alert.level,
            )
        except Exception as e:
            logger.debug("Alert send error: %s", e)

    # ─────────────────────────────────────────────────────────────────────
    # History / dashboard
    # ─────────────────────────────────────────────────────────────────────

    def get_recent_alerts(self, n: int = 20) -> list[dict]:
        """Return last N alerts as dicts (for dashboard)."""
        return [a.to_dict() for a in self._alert_history[-n:]]

    def get_alerts_from_log(self, n: int = 50) -> list[dict]:
        """Read last N alerts from the JSONL log file."""
        if not self._log_path.exists():
            return []
        try:
            lines = self._log_path.read_text().strip().splitlines()
            tail = lines[-n:]
            return [json.loads(l) for l in tail]
        except Exception:
            return []

    def save_summary_json(self) -> Path:
        """Write summary JSON for dashboard consumption."""
        out = self.log_dir / "alert_summary.json"
        try:
            recent = self.get_alerts_from_log(n=20)
            counts: dict[str, int] = {}
            for a in self.get_alerts_from_log(n=200):
                counts[a.get("alert_type", "unknown")] = counts.get(a.get("alert_type", "unknown"), 0) + 1
            out.write_text(json.dumps({
                "recent_alerts": recent,
                "alert_counts": counts,
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            }, indent=2))
        except Exception as e:
            logger.debug("Alert summary save error: %s", e)
        return out
