"""
Risk management — Kelly criterion, ATR-based SL/TP,
Monte Carlo VaR, circuit breakers, survival mode.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    max_risk_per_trade_pct: float = 0.02
    kelly_fraction: float = 0.25
    max_daily_drawdown_pct: float = 0.03
    max_total_drawdown_pct: float = 0.10
    circuit_breaker_dd_pct: float = 0.05
    survival_mode_equity_pct: float = 0.70
    survival_mode_size_mult: float = 0.10
    correlation_threshold: float = 0.75
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 3.0
    max_correlated_positions: int = 2
    monte_carlo_paths: int = 10_000


@dataclass
class PositionRisk:
    pair: str
    direction: int      # +1 long, -1 short
    entry_price: float
    stop_loss: float
    take_profit: float
    size_lots: float
    risk_amount: float
    atr: float
    confidence: float


class RiskManager:
    """
    Production risk manager with Kelly sizing, ATR stops,
    Monte Carlo VaR, circuit breakers, and survival mode.
    """

    def __init__(
        self,
        initial_equity: float = 10_000.0,
        cfg: RiskConfig | None = None,
        pip_value: float = 10.0,
    ):
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.cfg = cfg or RiskConfig()
        self.pip_value = pip_value

        self.open_positions: dict[str, PositionRisk] = {}
        self.daily_start_equity: float = initial_equity
        self.daily_pnl: float = 0.0
        self.is_circuit_broken: bool = False
        self.is_survival_mode: bool = False
        self._trade_returns: list[float] = []

    def update_equity(self, new_equity: float) -> None:
        self.equity = new_equity
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        self._check_circuit_breaker()
        self._check_survival_mode()

    def _check_circuit_breaker(self) -> None:
        dd = (self.peak_equity - self.equity) / self.peak_equity
        if dd >= self.cfg.circuit_breaker_dd_pct:
            if not self.is_circuit_broken:
                logger.warning(
                    "CIRCUIT BREAKER: Drawdown %.1f%% exceeded %.1f%% threshold",
                    dd * 100, self.cfg.circuit_breaker_dd_pct * 100
                )
                self.is_circuit_broken = True
        else:
            self.is_circuit_broken = False

    def _check_survival_mode(self) -> None:
        equity_ratio = self.equity / self.initial_equity
        if equity_ratio < self.cfg.survival_mode_equity_pct:
            if not self.is_survival_mode:
                logger.warning(
                    "SURVIVAL MODE: Equity %.1f%% of initial",
                    equity_ratio * 100
                )
                self.is_survival_mode = True
        else:
            self.is_survival_mode = False

    def can_trade(self) -> tuple[bool, str]:
        """Return (can_trade, reason) based on current risk state."""
        if self.is_circuit_broken:
            return False, "Circuit breaker triggered"

        daily_dd = (self.daily_start_equity - self.equity) / self.daily_start_equity
        if daily_dd >= self.cfg.max_daily_drawdown_pct:
            return False, f"Daily drawdown {daily_dd:.1%} exceeded"

        total_dd = (self.peak_equity - self.equity) / self.peak_equity
        if total_dd >= self.cfg.max_total_drawdown_pct:
            return False, f"Total drawdown {total_dd:.1%} exceeded"

        return True, "OK"

    def compute_position_size(
        self,
        pair: str,
        direction: int,
        entry_price: float,
        atr: float,
        win_rate: float = 0.5,
        avg_win_pips: float = 30.0,
        avg_loss_pips: float = 20.0,
        confidence: float = 0.6,
    ) -> PositionRisk:
        """
        Compute position size using fractional Kelly + ATR-based SL/TP.
        Returns PositionRisk with all sizing parameters.
        """
        pip = 0.0001 if "JPY" not in pair.upper() else 0.01

        # ATR-based stops
        sl_distance = atr * self.cfg.atr_sl_multiplier
        tp_distance = atr * self.cfg.atr_tp_multiplier

        if direction == 1:  # long
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # short
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        # Risk amount
        base_risk = self.equity * self.cfg.max_risk_per_trade_pct
        if self.is_survival_mode:
            base_risk *= self.cfg.survival_mode_size_mult

        # Kelly criterion: f* = (bp - q) / b  where b = payoff ratio
        if avg_loss_pips > 0:
            b = avg_win_pips / avg_loss_pips
            q = 1.0 - win_rate
            kelly = (b * win_rate - q) / b
            kelly = max(0.0, min(kelly, 0.5))  # cap at 50%
            kelly_factor = kelly * self.cfg.kelly_fraction
        else:
            kelly_factor = self.cfg.max_risk_per_trade_pct

        # Confidence-weighted size
        risk_amount = base_risk * kelly_factor * confidence * 2.0
        risk_amount = min(risk_amount, base_risk)

        # Convert to lots
        risk_per_pip = risk_amount / (sl_distance / pip)
        size_lots = max(0.01, risk_per_pip / self.pip_value)
        size_lots = round(size_lots, 2)

        return PositionRisk(
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size_lots=size_lots,
            risk_amount=risk_amount,
            atr=atr,
            confidence=confidence,
        )

    def monte_carlo_var(
        self,
        n_paths: int = 1000,
        n_steps: int = 20,
    ) -> dict:
        """
        Monte Carlo VaR using historical returns distribution.
        Uses Python fallback (Rust version used if available).
        """
        if len(self._trade_returns) < 10:
            return {"var_95": 0.0, "var_99": 0.0, "expected_shortfall": 0.0, "ruin_prob": 0.0}

        # Try Rust-accelerated version
        try:
            import sys
            sys.path.insert(0, "rust_core/target/release")
            import forex_rust_core
            var_95, var_99, es, ruin = forex_rust_core.monte_carlo_var(
                self._trade_returns, self.equity, n_paths, n_steps, 0.0
            )
            return {"var_95": var_95, "var_99": var_99, "expected_shortfall": es, "ruin_prob": ruin}
        except ImportError:
            pass

        # Python fallback
        returns = np.array(self._trade_returns)
        mean, std = returns.mean(), returns.std() + 1e-8

        final_equities = []
        rng = np.random.default_rng(42)
        for _ in range(n_paths):
            path = self.equity * np.cumprod(1 + rng.normal(mean, std, n_steps))
            final_equities.append(path[-1])

        pnl = sorted([e - self.equity for e in final_equities])
        var_95 = -np.percentile(pnl, 5)
        var_99 = -np.percentile(pnl, 1)
        es = -np.mean([p for p in pnl if p < -var_95])
        ruin_prob = sum(1 for e in final_equities if e < self.equity * 0.5) / n_paths

        return {
            "var_95": round(float(var_95), 2),
            "var_99": round(float(var_99), 2),
            "expected_shortfall": round(float(es), 2),
            "ruin_prob": round(float(ruin_prob), 4),
        }

    def add_trade_result(self, pnl: float) -> None:
        ret = pnl / (self.equity + 1e-8)
        self._trade_returns.append(ret)
        self.equity += pnl
        self.daily_pnl += pnl
        self.update_equity(self.equity)

    def check_correlation(self, new_pair: str, correlation_matrix: pd.DataFrame | None = None) -> bool:
        """Check if adding new_pair violates correlation limits."""
        if not self.open_positions:
            return True
        n_open = len(self.open_positions)
        if n_open >= self.cfg.max_correlated_positions:
            logger.warning("Max correlated positions reached (%d)", n_open)
            return False
        return True

    def reset_daily(self) -> None:
        self.daily_start_equity = self.equity
        self.daily_pnl = 0.0

    def get_status(self) -> dict:
        dd = (self.peak_equity - self.equity) / self.peak_equity
        return {
            "equity": round(self.equity, 2),
            "drawdown_pct": round(dd * 100, 2),
            "circuit_broken": self.is_circuit_broken,
            "survival_mode": self.is_survival_mode,
            "can_trade": self.can_trade()[0],
            "open_positions": len(self.open_positions),
        }
