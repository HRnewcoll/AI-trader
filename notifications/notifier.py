"""
Notification hub — Telegram + Slack + Email.
All are optional; system runs without any keys configured.
"""
from __future__ import annotations

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class Notifier:
    """
    Multi-channel notifier. Send via any or all of:
    Telegram (bot), Slack (webhook), Email (SMTP).
    All require credentials — system works without any.
    """

    def __init__(
        self,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        slack_webhook: str = "",
        email_cfg: dict | None = None,
    ):
        self.tg_token = telegram_token
        self.tg_chat_id = telegram_chat_id
        self.slack_webhook = slack_webhook
        self.email_cfg = email_cfg or {}
        self._enabled_channels: list[str] = []

        if telegram_token and telegram_chat_id:
            self._enabled_channels.append("telegram")
        if slack_webhook:
            self._enabled_channels.append("slack")
        if email_cfg and email_cfg.get("user"):
            self._enabled_channels.append("email")

        if not self._enabled_channels:
            logger.info("Notifier: no channels configured — logging-only mode")

    def send(self, title: str, message: str, level: str = "info") -> dict[str, bool]:
        """Send notification to all configured channels."""
        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "🚨", "trade": "💹", "heal": "🔄"}.get(level, "📢")
        full_message = f"{emoji} *{title}*\n{message}"
        results = {}

        if "telegram" in self._enabled_channels:
            results["telegram"] = self._send_telegram(full_message)
        if "slack" in self._enabled_channels:
            results["slack"] = self._send_slack(full_message)
        if "email" in self._enabled_channels:
            results["email"] = self._send_email(title, message)

        # Always log
        log_fn = {"warning": logger.warning, "error": logger.error}.get(level, logger.info)
        log_fn("Notification [%s]: %s — %s", level.upper(), title, message[:100])

        return results

    def send_trade(self, trade: dict, explanation: str = "") -> None:
        """Send trade notification with SHAP explanation."""
        pair = trade.get("pair", "")
        direction = trade.get("direction", "")
        price = trade.get("entry_price", 0)
        size = trade.get("size_lots", 0)
        confidence = trade.get("confidence", 0)
        pnl = trade.get("pnl")

        msg = (
            f"Pair: {pair} | {direction.upper()}\n"
            f"Price: {price:.5f} | Size: {size:.2f} lots\n"
            f"Confidence: {confidence:.0%}\n"
        )
        if pnl is not None:
            msg += f"PnL: {'+' if pnl >= 0 else ''}{pnl:.2f}\n"
        if explanation:
            msg += f"\n📊 {explanation[:300]}"

        self.send("Trade Signal", msg, level="trade")

    def send_healing_event(self, event_type: str, details: str) -> None:
        self.send(f"Self-Healing: {event_type}", details, level="heal")

    def send_daily_summary(self, metrics: dict) -> None:
        msg = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
        self.send("Daily Summary", msg, level="info")

    def _send_telegram(self, message: str) -> bool:
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        try:
            resp = requests.post(url, json={
                "chat_id": self.tg_chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.debug("Telegram send error: %s", e)
            return False

    def _send_slack(self, message: str) -> bool:
        try:
            resp = requests.post(self.slack_webhook, json={"text": message}, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.debug("Slack send error: %s", e)
            return False

    def _send_email(self, subject: str, body: str) -> bool:
        cfg = self.email_cfg
        try:
            msg = MIMEMultipart()
            msg["From"] = cfg.get("user", "")
            msg["To"] = cfg.get("to", "")
            msg["Subject"] = f"[ForexAI] {subject}"
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(cfg.get("smtp_host", "smtp.gmail.com"),
                               int(cfg.get("smtp_port", 587))) as smtp:
                smtp.starttls()
                smtp.login(cfg.get("user", ""), cfg.get("password", ""))
                smtp.sendmail(cfg.get("user", ""), cfg.get("to", ""), msg.as_string())
            return True
        except Exception as e:
            logger.debug("Email send error: %s", e)
            return False


def build_notifier_from_cfg(cfg: dict) -> Notifier:
    """Build Notifier from settings.yaml config."""
    n_cfg = cfg.get("notifications", {})
    tg = n_cfg.get("telegram", {})
    slack = n_cfg.get("slack", {})
    email = n_cfg.get("email", {})

    return Notifier(
        telegram_token=tg.get("bot_token", "") if tg.get("enabled") else "",
        telegram_chat_id=tg.get("chat_id", "") if tg.get("enabled") else "",
        slack_webhook=slack.get("webhook_url", "") if slack.get("enabled") else "",
        email_cfg=email if email.get("enabled") else None,
    )
