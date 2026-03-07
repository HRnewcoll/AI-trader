"""
Forex Gym environment for RL training (PPO / DQN).
State: technical features + sentiment + position + equity + session.
Action: 0=hold, 1=buy, 2=sell, 3=close.
Reward: Sharpe-based + cost penalty + session bonus + survival bonus.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

logger = logging.getLogger(__name__)


class ForexTradingEnv(gym.Env):
    """
    Custom Forex gymnasium environment for RL training.
    Supports discrete actions: hold / buy / sell / close.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        initial_equity: float = 10_000.0,
        spread_pips: float = 1.5,
        commission_pct: float = 0.0001,
        pip_value: float = 10.0,
        max_position_size: float = 1.0,
        risk_per_trade: float = 0.02,
        sharpe_weight: float = 1.0,
        cost_penalty: float = 0.001,
        loss_aversion: float = 2.0,
        session_bonus: float = 0.1,
        survival_bonus: float = 0.05,
        pip: float = 0.0001,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = [c for c in feature_cols if c in df.columns]
        self.n_features = len(self.feature_cols)
        self.initial_equity = initial_equity
        self.spread_pips = spread_pips
        self.commission_pct = commission_pct
        self.pip_value = pip_value
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        self.sharpe_weight = sharpe_weight
        self.cost_penalty = cost_penalty
        self.loss_aversion = loss_aversion
        self.session_bonus = session_bonus
        self.survival_bonus = survival_bonus
        self.pip = pip

        # Spaces
        self.action_space = spaces.Discrete(4)  # hold, buy, sell, close
        obs_size = self.n_features + 4  # features + [position, equity_pct, drawdown, unrealized_pnl]
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_size,), dtype=np.float32
        )

        # Pre-compute feature matrix for fast step() access (avoids per-row dict lookup)
        valid_cols = [c for c in feature_cols if c in df.columns]
        self._feature_matrix = (
            df[valid_cols].ffill().fillna(0.0).values.astype(np.float32)
        )

        self._reset_state()

    def _reset_state(self) -> None:
        self.current_step = 0
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.position = 0.0   # +1 long, -1 short, 0 flat
        self.entry_price = 0.0
        self.position_size = 0.0
        self.returns: list[float] = []
        self.trades: list[dict] = []
        self.done = False

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        self.current_step = 0
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        # Use pre-computed matrix — O(1) lookup vs per-row dict scan
        features_arr = np.clip(self._feature_matrix[self.current_step], -10, 10)

        # Portfolio state features (use df row for current price)
        current_price = float(self.df.at[self.current_step, "close"]) if "close" in self.df.columns else 0.0
        equity_pct = (self.equity / self.initial_equity) - 1.0
        drawdown = (self.peak_equity - self.equity) / (self.peak_equity + 1e-8)
        unrealized_pnl = 0.0
        if self.position != 0.0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position / (current_price + 1e-8)

        portfolio_state = np.array([
            self.position,
            np.clip(equity_pct, -1.0, 5.0),
            np.clip(drawdown, 0.0, 1.0),
            np.clip(unrealized_pnl, -1.0, 1.0),
        ], dtype=np.float32)

        return np.concatenate([features_arr, portfolio_state])

    def _compute_reward(self, pnl: float, action: int) -> float:
        """Sharpe-based reward with multiple bonus terms."""
        # Base return
        ret = pnl / (self.equity + 1e-8)
        self.returns.append(ret)

        reward = ret

        # Sharpe penalty/bonus
        if len(self.returns) > 10:
            ret_arr = np.array(self.returns[-20:])
            std = ret_arr.std() + 1e-8
            sharpe_contrib = ret_arr.mean() / std
            reward += self.sharpe_weight * sharpe_contrib

        # Transaction cost penalty
        if action in (1, 2):
            cost = self.spread_pips * self.pip * self.pip_value + self.equity * self.commission_pct
            reward -= self.cost_penalty * cost / (self.equity + 1e-8)

        # Loss aversion (penalise losses more)
        if pnl < 0:
            reward *= self.loss_aversion

        # Session bonus (reward trading in high-vol sessions)
        row = self.df.iloc[self.current_step]
        if row.get("session_overlap", 0) == 1.0 or row.get("session_london", 0) == 1.0:
            if pnl > 0:
                reward += self.session_bonus * abs(ret)

        # Survival bonus (equity above initial)
        if self.equity > self.initial_equity:
            reward += self.survival_bonus * (self.equity / self.initial_equity - 1.0)

        # Drawdown penalty
        drawdown = (self.peak_equity - self.equity) / (self.peak_equity + 1e-8)
        if drawdown > 0.05:
            reward -= drawdown * 2.0

        return float(np.clip(reward, -10.0, 10.0))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            obs = self._get_observation()
            return obs, 0.0, True, False, {}

        row = self.df.iloc[self.current_step]
        current_price = float(row.get("close", 0.0))
        spread = self.spread_pips * self.pip

        pnl = 0.0

        # Close existing position if action=close or direction reversal
        if self.position != 0.0 and (action == 3 or
           (action == 1 and self.position < 0) or
           (action == 2 and self.position > 0)):
            exit_price = current_price + spread * (-self.position)
            trade_pnl = (exit_price - self.entry_price) * self.position * self.position_size * self.pip_value
            commission = self.equity * self.commission_pct
            pnl = trade_pnl - commission
            self.equity += pnl
            self.trades.append({
                "step": self.current_step,
                "pnl": pnl,
                "exit_price": exit_price,
            })
            self.position = 0.0
            self.entry_price = 0.0
            self.position_size = 0.0

        # Open new position
        if action in (1, 2) and self.position == 0.0:
            direction = 1.0 if action == 1 else -1.0
            # Size by risk
            atr = float(row.get("atr_14", spread * 10))
            if atr > 0:
                risk_amount = self.equity * self.risk_per_trade
                self.position_size = min(risk_amount / (atr * self.pip_value), self.max_position_size)
            else:
                self.position_size = 0.1
            self.entry_price = current_price + spread * direction
            self.position = direction

        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        reward = self._compute_reward(pnl, action)

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        # Kill if equity drops below 50%
        truncated = self.equity < self.initial_equity * 0.5
        self.done = terminated or truncated

        obs = self._get_observation() if not self.done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"equity": self.equity, "pnl": pnl, "position": self.position}
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        logger.info("Step %d | Equity: %.2f | Position: %s",
                    self.current_step, self.equity, self.position)

    def get_metrics(self) -> dict:
        """Compute final performance metrics."""
        total_return = (self.equity - self.initial_equity) / self.initial_equity
        max_dd = (self.peak_equity - min(self.equity, self.initial_equity)) / self.peak_equity

        ret_arr = np.array(self.returns) if self.returns else np.array([0.0])
        sharpe = 0.0
        if ret_arr.std() > 0:
            sharpe = float(ret_arr.mean() / ret_arr.std() * np.sqrt(252 * 24))

        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        win_rate = len(wins) / max(len(self.trades), 1)

        return {
            "total_return": round(total_return, 4),
            "final_equity": round(self.equity, 2),
            "max_drawdown": round(max_dd, 4),
            "sharpe": round(sharpe, 3),
            "n_trades": len(self.trades),
            "win_rate": round(win_rate, 3),
        }
