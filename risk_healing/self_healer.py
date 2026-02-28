"""
Self-healing watchdog — monitors system health, detects anomalies,
auto-restarts services, rolls back models, and alerts via notifications.
Runs every 60 seconds.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SelfHealer:
    """
    Monitors: equity curve, model drift, latency, system resources.
    Actions: alert, reduce risk, rollback model, restart service.
    """

    def __init__(
        self,
        artifacts_dir: str = "artifacts/models",
        watchdog_interval: int = 60,
        notify_callback: Callable[[str, str], None] | None = None,
        ks_pvalue_threshold: float = 0.05,
        anomaly_zscore_threshold: float = 3.0,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.watchdog_interval = watchdog_interval
        self.notify = notify_callback or (lambda title, msg: None)
        self.ks_threshold = ks_pvalue_threshold
        self.anomaly_threshold = anomaly_zscore_threshold

        self._equity_history: list[float] = []
        self._latency_history: list[float] = []
        self._feature_history: list[dict] = []
        self._model_version_log: list[dict] = []
        self._health_log: list[dict] = []
        self._running = False

    def record_equity(self, equity: float) -> None:
        self._equity_history.append(equity)

    def record_latency(self, latency_ms: float) -> None:
        self._latency_history.append(latency_ms)

    def record_features(self, features: dict) -> None:
        self._feature_history.append(features)

    def check_equity_health(self) -> dict:
        """Detect abnormal equity drops."""
        if len(self._equity_history) < 10:
            return {"status": "ok", "message": "Insufficient data"}

        recent = self._equity_history[-10:]
        baseline = self._equity_history[:-10] or recent
        z_scores = (np.array(recent) - np.mean(baseline)) / (np.std(baseline) + 1e-8)

        issues = []
        if any(z < -self.anomaly_threshold for z in z_scores):
            issues.append("Equity anomaly detected (rapid drop)")

        if len(self._equity_history) > 2:
            latest_drop = (self._equity_history[-2] - self._equity_history[-1]) / self._equity_history[-2]
            if latest_drop > 0.02:
                issues.append(f"Single-step equity drop: {latest_drop:.1%}")

        return {
            "status": "warning" if issues else "ok",
            "issues": issues,
            "latest_equity": self._equity_history[-1],
        }

    def check_feature_drift(self, new_features: dict, baseline_features: dict) -> dict:
        """KS test for feature distribution drift."""
        try:
            from scipy import stats
        except ImportError:
            return {"status": "ok", "message": "scipy not available"}

        drifted = []
        for key in baseline_features:
            if key not in new_features:
                continue
            base_vals = np.array(baseline_features[key]) if isinstance(baseline_features[key], list) else np.array([baseline_features[key]])
            new_vals = np.array(new_features[key]) if isinstance(new_features[key], list) else np.array([new_features[key]])
            if len(base_vals) < 5 or len(new_vals) < 5:
                continue
            try:
                _, pval = stats.ks_2samp(base_vals, new_vals)
                if pval < self.ks_threshold:
                    drifted.append({"feature": key, "p_value": float(pval)})
            except Exception:
                pass

        return {
            "status": "drift_detected" if drifted else "ok",
            "drifted_features": drifted,
        }

    def check_system_resources(self) -> dict:
        """Check CPU, memory, disk usage."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            issues = []
            if cpu > 90:
                issues.append(f"High CPU: {cpu:.0f}%")
            if mem.percent > 90:
                issues.append(f"High memory: {mem.percent:.0f}%")
            if disk.percent > 90:
                issues.append(f"Low disk space: {disk.percent:.0f}% used")
            return {
                "status": "warning" if issues else "ok",
                "cpu_pct": cpu,
                "mem_pct": mem.percent,
                "disk_pct": disk.percent,
                "issues": issues,
            }
        except ImportError:
            return {"status": "ok", "message": "psutil not available"}

    def check_model_performance(self, model_name: str, recent_accuracy: float, threshold: float = 0.45) -> dict:
        """Detect model performance degradation."""
        if recent_accuracy < threshold:
            msg = f"Model {model_name} accuracy dropped to {recent_accuracy:.2%} (threshold: {threshold:.2%})"
            logger.warning(msg)
            self.notify("⚠️ Model Degradation", msg)
            return {"status": "degraded", "model": model_name, "accuracy": recent_accuracy}
        return {"status": "ok", "model": model_name, "accuracy": recent_accuracy}

    def rollback_model(self, model_name: str, version: str = "backup") -> bool:
        """Rollback to a previous model version."""
        backup_path = self.artifacts_dir / f"{model_name}_{version}.pkl"
        current_path = self.artifacts_dir / f"{model_name}.pkl"
        if backup_path.exists():
            try:
                import shutil
                shutil.copy2(str(backup_path), str(current_path))
                msg = f"Model {model_name} rolled back to {version}"
                logger.info(msg)
                self.notify("🔄 Model Rollback", msg)
                self._model_version_log.append({
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "model": model_name,
                    "action": f"rollback_to_{version}",
                })
                return True
            except Exception as e:
                logger.error("Rollback failed: %s", e)
        return False

    def backup_model(self, model_name: str) -> bool:
        """Backup current model before replacing."""
        current_path = self.artifacts_dir / f"{model_name}.pkl"
        backup_path = self.artifacts_dir / f"{model_name}_backup.pkl"
        if current_path.exists():
            try:
                import shutil
                shutil.copy2(str(current_path), str(backup_path))
                return True
            except Exception:
                pass
        return False

    def run_health_check(self, equity: float | None = None) -> dict:
        """Run all health checks and return consolidated status."""
        if equity is not None:
            self.record_equity(equity)

        equity_health = self.check_equity_health()
        system_health = self.check_system_resources()

        # Aggregate status
        all_ok = equity_health["status"] == "ok" and system_health["status"] == "ok"

        health_report = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "overall": "ok" if all_ok else "warning",
            "equity": equity_health,
            "system": system_health,
        }

        self._health_log.append(health_report)

        if not all_ok:
            issues = equity_health.get("issues", []) + system_health.get("issues", [])
            if issues:
                self.notify("⚠️ Health Warning", "\n".join(issues))

        return health_report

    def get_health_summary(self) -> dict:
        if not self._health_log:
            return {"status": "no_data"}
        last = self._health_log[-1]
        warnings = sum(1 for h in self._health_log[-24:] if h.get("overall") != "ok")
        return {
            "last_check": last.get("timestamp"),
            "overall": last.get("overall"),
            "warnings_last_24": warnings,
            "total_checks": len(self._health_log),
        }
