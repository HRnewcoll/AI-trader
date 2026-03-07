"""
Hyperparameter optimisation using Optuna (inspired by freqtrade/hyperopt).

Supports tuning:
  - XGBoost model parameters
  - DLinear / PatchTST Transformer parameters
  - RL reward shaping weights
  - Risk management parameters (ATR multipliers, Kelly fraction)
  - Feature engineering parameters (indicator periods)
  - Signal thresholds (RSI, sentiment, confidence)

The optimiser uses TPE (Tree-structured Parzen Estimators) by default
and falls back to random search if Optuna is unavailable.

All trials are saved to SQLite so optimisation can be interrupted
and resumed.  Best params are exported as YAML/JSON.

Usage:
    ho = HyperOptimiser(objective="sharpe", n_trials=100)
    best = ho.run(df, feature_cols)
    ho.save_best(best, "artifacts/hyperopt/best_params.json")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Log-scale a float param when its max/min ratio exceeds this threshold
# (e.g., learning_rate spans 1e-4→1e-2, ratio=100 → log scale is more efficient)
_LOG_SCALE_RATIO_THRESHOLD = 10

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False
    logger.warning("optuna not installed — falling back to random search")


# ─────────────────────────────────────────────────────────────────────────────
# Parameter search space definitions
# ─────────────────────────────────────────────────────────────────────────────

XGBOOST_SPACE: dict[str, tuple] = {
    "max_depth":        ("int",   3,    10,   1),
    "learning_rate":    ("float", 0.01, 0.30, None),
    "n_estimators":     ("int",   50,   500,  10),
    "subsample":        ("float", 0.5,  1.0,  None),
    "colsample_bytree": ("float", 0.5,  1.0,  None),
    "min_child_weight": ("int",   1,    10,   1),
    "reg_alpha":        ("float", 1e-4, 1.0,  None),
    "reg_lambda":       ("float", 1e-4, 1.0,  None),
}

TRANSFORMER_SPACE: dict[str, tuple] = {
    "seq_len":          ("int",   20,   120,  10),
    "d_model":          ("int",   32,   256,  32),
    "n_heads":          ("int",   2,    8,    2),
    "n_layers":         ("int",   1,    4,    1),
    "dropout":          ("float", 0.0,  0.5,  None),
    "learning_rate":    ("float", 1e-4, 1e-2, None),
    "batch_size":       ("int",   16,   128,  16),
}

RL_SPACE: dict[str, tuple] = {
    "learning_rate":    ("float", 1e-4, 1e-2, None),
    "gamma":            ("float", 0.90, 0.999, None),
    "sharpe_weight":    ("float", 0.5,  3.0,  None),
    "cost_penalty":     ("float", 1e-4, 0.01, None),
    "loss_aversion":    ("float", 1.0,  5.0,  None),
    "session_bonus":    ("float", 0.0,  0.5,  None),
}

RISK_SPACE: dict[str, tuple] = {
    "atr_sl_multiplier": ("float", 1.0, 4.0,  None),
    "atr_tp_multiplier": ("float", 1.5, 6.0,  None),
    "kelly_fraction":    ("float", 0.1, 0.5,  None),
    "sentiment_threshold": ("float", 0.1, 0.6, None),
    "min_confidence":    ("float", 0.50, 0.80, None),
}

INDICATOR_SPACE: dict[str, tuple] = {
    "rsi_period":       ("int",   7,    21,   1),
    "atr_period":       ("int",   7,    28,   1),
    "bb_period":        ("int",   10,   30,   2),
    "ema_fast":         ("int",   5,    20,   1),
    "ema_slow":         ("int",   20,   100,  5),
}

ALL_SPACES = {
    "xgboost": XGBOOST_SPACE,
    "transformer": TRANSFORMER_SPACE,
    "rl": RL_SPACE,
    "risk": RISK_SPACE,
    "indicators": INDICATOR_SPACE,
}


# ─────────────────────────────────────────────────────────────────────────────
# Trial result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    trial_number: int
    params: dict[str, Any]
    objective_value: float
    objective_name: str
    extra_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HyperOptResult:
    best_params: dict[str, Any]
    best_value: float
    objective_name: str
    n_trials: int
    all_trials: list[TrialResult] = field(default_factory=list)
    spaces_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params,
            "best_value": round(self.best_value, 5),
            "objective_name": self.objective_name,
            "n_trials": self.n_trials,
            "spaces_used": self.spaces_used,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Objective functions
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_params(
    df: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    objective: str = "sharpe",
) -> tuple[float, dict]:
    """
    Evaluate a parameter set using a quick vectorised backtest.
    Returns (objective_value, extra_metrics).
    """
    from models.xgboost_model import XGBoostForexModel
    from backtesting.walk_forward import vectorised_backtest

    # Extract XGBoost params (only pass known kwargs)
    xgb_params = {
        k: params[k]
        for k in ["max_depth", "learning_rate", "n_estimators",
                  "subsample", "colsample_bytree", "min_child_weight"]
        if k in params
    }

    try:
        model = XGBoostForexModel(**xgb_params)
        split = int(len(df) * 0.80)
        train_df, test_df = df.iloc[:split], df.iloc[split:]

        if len(train_df) < 50 or len(test_df) < 20:
            return -99.0, {}

        model.fit(train_df, feature_cols)
        signals, _ = model.predict_all(test_df, feature_cols)
        signals_series = pd.Series(signals, index=test_df.index)

        result = vectorised_backtest(
            test_df,
            signals_series,
            initial_equity=10_000.0,
            spread_pips=params.get("spread_pips", 1.5),
        )

        extra = {
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "calmar": result.calmar,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "n_trades": result.n_trades,
        }

        value = {
            "sharpe":   result.sharpe,
            "sortino":  result.sortino,
            "calmar":   result.calmar,
            "combined": result.sharpe * 0.4 + result.sortino * 0.3 + result.calmar * 0.3,
        }.get(objective, result.sharpe)

        return float(np.nan_to_num(value, nan=-99.0)), extra

    except Exception as e:
        logger.debug("Evaluation error: %s", e)
        return -99.0, {}


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _make_optuna_objective(
    df: pd.DataFrame,
    feature_cols: list[str],
    spaces: list[str],
    objective: str,
    custom_fn: Optional[Callable] = None,
):
    """Return an Optuna objective function."""
    def objective_fn(trial):
        params: dict[str, Any] = {}
        for space_name in spaces:
            space = ALL_SPACES.get(space_name, {})
            for name, spec in space.items():
                kind = spec[0]
                lo, hi = spec[1], spec[2]
                step = spec[3]
                if kind == "int":
                    params[name] = trial.suggest_int(name, lo, hi, step=step or 1)
                elif kind == "float":
                    params[name] = trial.suggest_float(
                        name, lo, hi,
                        log=(lo > 0 and hi / lo > _LOG_SCALE_RATIO_THRESHOLD),
                    )

        if custom_fn is not None:
            val, _ = custom_fn(df, feature_cols, params)
        else:
            val, _ = _evaluate_params(df, feature_cols, params, objective)
        return val

    return objective_fn


# ─────────────────────────────────────────────────────────────────────────────
# Random search fallback
# ─────────────────────────────────────────────────────────────────────────────

def _random_sample(spaces: list[str]) -> dict[str, Any]:
    """Sample random parameters from specified spaces."""
    params: dict[str, Any] = {}
    for space_name in spaces:
        space = ALL_SPACES.get(space_name, {})
        for name, spec in space.items():
            kind, lo, hi = spec[0], spec[1], spec[2]
            if kind == "int":
                step = spec[3] or 1
                params[name] = int(np.random.randint(lo, hi + 1) // step * step)
            else:
                params[name] = float(np.random.uniform(lo, hi))
    return params


def _random_search(
    df: pd.DataFrame,
    feature_cols: list[str],
    spaces: list[str],
    objective: str,
    n_trials: int,
    custom_fn: Optional[Callable] = None,
) -> HyperOptResult:
    """Random search fallback when Optuna is unavailable."""
    best_value = -np.inf
    best_params: dict = {}
    trials = []

    for i in range(n_trials):
        params = _random_sample(spaces)
        if custom_fn is not None:
            val, extra = custom_fn(df, feature_cols, params)
        else:
            val, extra = _evaluate_params(df, feature_cols, params, objective)

        trials.append(TrialResult(i, params, val, objective, extra))
        if val > best_value:
            best_value = val
            best_params = params.copy()

    return HyperOptResult(
        best_params=best_params,
        best_value=best_value,
        objective_name=objective,
        n_trials=n_trials,
        all_trials=trials,
        spaces_used=spaces,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class HyperOptimiser:
    """
    Hyperparameter optimiser for the Forex trader.

    Uses Optuna TPE (or random search fallback) to find the best parameter
    combination for a given objective metric.

    Parameters
    ----------
    objective : "sharpe" | "sortino" | "calmar" | "combined"
    spaces    : list of spaces to optimise (e.g. ["xgboost", "risk"])
    n_trials  : total number of trials to run
    n_jobs    : parallel trials (1 = sequential, -1 = all CPUs)
    storage   : path to SQLite DB for persistence (None = in-memory)
    """

    def __init__(
        self,
        objective: str = "sharpe",
        spaces: list[str] | None = None,
        n_trials: int = 50,
        n_jobs: int = 1,
        storage: Optional[str] = None,
        study_name: str = "forex_hyperopt",
    ):
        self.objective = objective
        self.spaces = spaces or ["xgboost", "risk"]
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.storage = storage
        self.study_name = study_name

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        custom_fn: Optional[Callable] = None,
    ) -> HyperOptResult:
        """
        Run hyperparameter optimisation.

        Parameters
        ----------
        df           : feature DataFrame (must include all columns)
        feature_cols : list of feature column names
        custom_fn    : optional custom evaluation function
                       signature: (df, feature_cols, params) → (float, dict)

        Returns
        -------
        HyperOptResult
        """
        logger.info(
            "HyperOpt: objective=%s spaces=%s n_trials=%d",
            self.objective, self.spaces, self.n_trials,
        )

        if not _HAS_OPTUNA:
            logger.warning("Optuna not available — using random search")
            return _random_search(
                df, feature_cols, self.spaces, self.objective,
                self.n_trials, custom_fn,
            )

        try:
            storage_url = f"sqlite:///{self.storage}" if self.storage else None
            study = optuna.create_study(
                direction="maximize",
                study_name=self.study_name,
                storage=storage_url,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
            )

            obj_fn = _make_optuna_objective(df, feature_cols, self.spaces, self.objective, custom_fn)
            study.optimize(obj_fn, n_trials=self.n_trials, n_jobs=self.n_jobs, show_progress_bar=False)

            best_trial = study.best_trial
            all_trials = [
                TrialResult(
                    trial_number=t.number,
                    params=t.params,
                    objective_value=t.value if t.value is not None else -99.0,
                    objective_name=self.objective,
                )
                for t in study.trials
                if t.value is not None
            ]

            result = HyperOptResult(
                best_params=best_trial.params,
                best_value=best_trial.value,
                objective_name=self.objective,
                n_trials=len(study.trials),
                all_trials=all_trials,
                spaces_used=self.spaces,
            )
            logger.info("HyperOpt done. Best %s=%.4f", self.objective, result.best_value)
            return result

        except Exception as e:
            logger.warning("Optuna failed (%s) — falling back to random search", e)
            return _random_search(
                df, feature_cols, self.spaces, self.objective,
                self.n_trials, custom_fn,
            )

    def save_best(self, result: HyperOptResult, path: str = "artifacts/hyperopt/best_params.json") -> Path:
        """Save best parameters to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info("Best params saved to %s", out)
        return out

    def load_best(self, path: str = "artifacts/hyperopt/best_params.json") -> dict:
        """Load best parameters from JSON."""
        p = Path(path)
        if not p.exists():
            return {}
        return json.loads(p.read_text()).get("best_params", {})

    @staticmethod
    def list_spaces() -> list[str]:
        """Return available search space names."""
        return list(ALL_SPACES.keys())

    @staticmethod
    def get_space(name: str) -> dict:
        """Return a specific search space definition."""
        return ALL_SPACES.get(name, {})
