"""
RL agent — PPO + DQN ensemble using Stable-Baselines3.
Trains locally, saves model artifacts, provides predict() interface.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False
    logger.error("stable-baselines3 not installed — RL disabled")

from models.rl_environment import ForexTradingEnv
from features.technical_indicators import get_feature_columns


class RLForexAgent:
    """
    Ensemble of PPO + DQN agents for Forex trading.
    Both are trained locally — no external API.
    Prediction is by majority vote weighted by recent performance.
    """

    def __init__(
        self,
        algorithms: list[str] | None = None,
        total_timesteps: int = 200_000,
        artifacts_dir: str = "artifacts/models",
        device: str = "auto",
    ):
        if not _HAS_SB3:
            raise ImportError("stable-baselines3 required for RL")
        self.algorithms = algorithms or ["ppo", "dqn"]
        self.total_timesteps = total_timesteps
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.models: dict[str, object] = {}
        self.weights: dict[str, float] = {}
        self.is_trained = False
        self.train_metrics: dict = {}
        self.feature_cols: list[str] = []

    def _make_env(self, df: pd.DataFrame, feature_cols: list[str]) -> Monitor:
        env = ForexTradingEnv(df, feature_cols)
        return Monitor(env)

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        pair: str = "EURUSD",
        val_split: float = 0.2,
    ) -> dict:
        """Train PPO and/or DQN on Forex environment."""
        self.feature_cols = feature_cols
        split = int(len(df) * (1 - val_split))
        train_df = df.iloc[:split].reset_index(drop=True)
        val_df = df.iloc[split:].reset_index(drop=True)

        logger.info("Training RL agents for %s (%d train bars)", pair, len(train_df))

        algo_map = {"ppo": PPO, "dqn": DQN}

        for algo_name in self.algorithms:
            if algo_name not in algo_map:
                continue
            AlgoClass = algo_map[algo_name]
            logger.info("Training %s...", algo_name.upper())

            train_env = DummyVecEnv([lambda: self._make_env(train_df, feature_cols)])

            policy_kwargs = dict(net_arch=[256, 256, 128])
            if algo_name == "ppo":
                model = PPO(
                    "MlpPolicy", train_env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    ent_coef=0.01,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    device=self.device,
                )
            else:  # dqn
                model = DQN(
                    "MlpPolicy", train_env,
                    learning_rate=1e-4,
                    buffer_size=50_000,
                    batch_size=64,
                    gamma=0.99,
                    exploration_fraction=0.2,
                    exploration_final_eps=0.05,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    device=self.device,
                )

            model.learn(total_timesteps=self.total_timesteps, progress_bar=False)
            self.models[algo_name] = model
            self.weights[algo_name] = 1.0  # equal initial weight

            # Evaluate on validation set
            val_env = self._make_env(val_df, feature_cols)
            metrics = self._evaluate(model, val_env)
            self.train_metrics[algo_name] = metrics
            logger.info("%s metrics: %s", algo_name.upper(), metrics)

            # Save
            model_path = self.artifacts_dir / f"rl_{algo_name}_{pair}"
            model.save(str(model_path))
            logger.info("%s saved to %s", algo_name, model_path)

        # Normalise weights by Sharpe
        self._update_weights_from_metrics()
        self.is_trained = True
        return self.train_metrics

    def _evaluate(self, model, env: ForexTradingEnv) -> dict:
        """Run one episode and return performance metrics."""
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        return env.env.get_metrics() if hasattr(env, "env") else {}

    def _update_weights_from_metrics(self) -> None:
        """Update ensemble weights based on Sharpe ratio."""
        sharpes = {
            name: max(self.train_metrics.get(name, {}).get("sharpe", 0.0), 0.01)
            for name in self.models
        }
        total = sum(sharpes.values()) + 1e-8
        self.weights = {name: s / total for name, s in sharpes.items()}
        logger.info("Ensemble weights: %s", self.weights)

    def predict(self, observation: np.ndarray) -> Tuple[int, float]:
        """
        Ensemble vote: weighted majority over all trained agents.
        Returns (action: 0-3, confidence: 0-1).
        """
        if not self.models:
            return 0, 0.5  # hold

        vote_counts = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        for name, model in self.models.items():
            action, _ = model.predict(observation, deterministic=True)
            vote_counts[int(action)] += self.weights.get(name, 1.0)

        best_action = max(vote_counts, key=vote_counts.get)
        total_votes = sum(vote_counts.values()) + 1e-8
        confidence = vote_counts[best_action] / total_votes
        return best_action, float(confidence)

    def update_weights(self, recent_pnl: dict[str, float]) -> None:
        """Live weight update based on recent PnL (exponential decay)."""
        alpha = 0.1  # learning rate for weight update
        for name, pnl in recent_pnl.items():
            if name in self.weights:
                self.weights[name] = max(0.05, self.weights[name] + alpha * np.sign(pnl))
        # Normalise
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    @classmethod
    def load(cls, pair: str, algorithms: list[str], artifacts_dir: str = "artifacts/models") -> "RLForexAgent":
        obj = cls(algorithms=algorithms, artifacts_dir=artifacts_dir)
        for algo in algorithms:
            AlgoClass = {"ppo": PPO, "dqn": DQN}.get(algo)
            if AlgoClass is None:
                continue
            path = Path(artifacts_dir) / f"rl_{algo}_{pair}.zip"
            if path.exists():
                obj.models[algo] = AlgoClass.load(str(path))
                obj.weights[algo] = 1.0
                logger.info("Loaded %s from %s", algo, path)
        obj.is_trained = bool(obj.models)
        return obj
