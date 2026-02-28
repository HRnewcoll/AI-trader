"""
Genetic optimizer — DEAP multi-objective.
Evolves: hyperparameters, feature subsets, reward weights.
Objectives: Sharpe, Sortino, Calmar, -MaxDrawdown.
Runs on weekends + loss-triggered.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from deap import base, creator, tools, algorithms
    _HAS_DEAP = True
except ImportError:
    _HAS_DEAP = False
    logger.warning("deap not installed — genetic optimizer disabled")


@dataclass
class GeneticConfig:
    population_size: int = 50
    generations: int = 30
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    tournament_size: int = 3
    objectives: list[str] = field(default_factory=lambda: ["sharpe", "sortino", "calmar", "neg_drawdown"])


# Parameter search space
PARAM_SPACE = {
    # XGBoost
    "xgb_max_depth":       (3, 10, int),
    "xgb_learning_rate":   (0.01, 0.3, float),
    "xgb_n_estimators":    (100, 500, int),
    "xgb_subsample":       (0.6, 1.0, float),
    # RL reward weights
    "rl_sharpe_weight":    (0.5, 3.0, float),
    "rl_cost_penalty":     (0.0001, 0.01, float),
    "rl_loss_aversion":    (1.0, 5.0, float),
    "rl_session_bonus":    (0.0, 0.5, float),
    # Risk
    "kelly_fraction":      (0.1, 0.5, float),
    "atr_sl_multiplier":   (1.0, 4.0, float),
    "atr_tp_multiplier":   (1.5, 6.0, float),
    # Sentiment threshold
    "sentiment_threshold": (0.1, 0.6, float),
}

PARAM_NAMES = list(PARAM_SPACE.keys())
N_PARAMS = len(PARAM_NAMES)


def _random_individual() -> list:
    """Create a random individual (parameter vector)."""
    ind = []
    for name in PARAM_NAMES:
        lo, hi, dtype = PARAM_SPACE[name]
        val = random.uniform(lo, hi)
        if dtype == int:
            val = int(round(val))
        ind.append(val)
    return ind


def _decode_individual(ind: list) -> dict:
    """Convert flat list to named parameter dict."""
    return {
        name: (int(round(val)) if PARAM_SPACE[name][2] == int else float(val))
        for name, val in zip(PARAM_NAMES, ind)
    }


def _clamp_individual(ind: list) -> list:
    """Keep all genes within valid bounds."""
    for i, name in enumerate(PARAM_NAMES):
        lo, hi, dtype = PARAM_SPACE[name]
        ind[i] = max(lo, min(hi, ind[i]))
        if dtype == int:
            ind[i] = int(round(ind[i]))
    return ind


class GeneticOptimizer:
    """
    Multi-objective genetic algorithm for hyperparameter optimization.
    Uses NSGA-II (non-dominated sorting) for Pareto front optimization.
    """

    def __init__(
        self,
        fitness_fn: Callable[[dict], tuple],
        cfg: GeneticConfig | None = None,
    ):
        """
        Args:
            fitness_fn: Function(params_dict) -> tuple of objective scores
                       (sharpe, sortino, calmar, neg_drawdown).
                       Higher is better for all objectives.
        """
        if not _HAS_DEAP:
            raise ImportError("deap is required for genetic optimizer")

        self.fitness_fn = fitness_fn
        self.cfg = cfg or GeneticConfig()
        self._setup_deap()
        self.pareto_front: list[dict] = []
        self.best_params: dict = {}
        self.generation_log: list[dict] = []

    def _setup_deap(self) -> None:
        # Multi-objective: maximise all 4 objectives
        n_obj = len(self.cfg.objectives)
        weights = tuple([1.0] * n_obj)

        # Re-create if weights changed (e.g. different objective count)
        if hasattr(creator, "FitnessMulti"):
            if getattr(creator.FitnessMulti, "weights", None) != weights:
                del creator.FitnessMulti
                if hasattr(creator, "Individual"):
                    del creator.Individual
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=weights)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, _random_individual)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_wrapper)
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                              eta=20, low=[PARAM_SPACE[n][0] for n in PARAM_NAMES],
                              up=[PARAM_SPACE[n][1] for n in PARAM_NAMES])
        self.toolbox.register("mutate", tools.mutPolynomialBounded,
                              eta=20, low=[PARAM_SPACE[n][0] for n in PARAM_NAMES],
                              up=[PARAM_SPACE[n][1] for n in PARAM_NAMES],
                              indpb=self.cfg.mutation_rate)
        self.toolbox.register("select", tools.selNSGA2)

    def _evaluate_wrapper(self, individual: list) -> tuple:
        params = _decode_individual(_clamp_individual(individual))
        try:
            result = self.fitness_fn(params)
            if not isinstance(result, tuple):
                result = (float(result), float(result), float(result), -0.05)
            # Replace NaN/inf
            result = tuple(0.0 if (np.isnan(v) or np.isinf(v)) else float(v) for v in result)
            return result
        except Exception as e:
            logger.debug("Fitness eval error: %s", e)
            return (0.0,) * len(self.cfg.objectives)

    def evolve(self) -> dict:
        """Run NSGA-II evolution. Returns best parameters."""
        logger.info("Starting genetic optimization: %d generations × %d individuals",
                    self.cfg.generations, self.cfg.population_size)

        pop = self.toolbox.population(n=self.cfg.population_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: [round(float(np.mean([v[i] for v in x])), 4)
                                          for i in range(len(self.cfg.objectives))])
        stats.register("max", lambda x: [round(float(np.max([v[i] for v in x])), 4)
                                          for i in range(len(self.cfg.objectives))])

        hof = tools.ParetoFront()

        pop, logbook = algorithms.eaMuPlusLambda(
            pop, self.toolbox,
            mu=self.cfg.population_size,
            lambda_=self.cfg.population_size,
            cxpb=self.cfg.crossover_rate,
            mutpb=self.cfg.mutation_rate,
            ngen=self.cfg.generations,
            stats=stats,
            halloffame=hof,
            verbose=False,
        )

        self.generation_log = logbook

        # Extract Pareto front
        self.pareto_front = [_decode_individual(list(ind)) for ind in hof]

        # Best = highest sum of normalised objectives (lexicographic)
        if self.pareto_front:
            best_score = -float("inf")
            for params in self.pareto_front:
                ind = [params.get(n, 0) for n in PARAM_NAMES]
                fitness = self.toolbox.evaluate(ind)
                score = sum(fitness)
                if score > best_score:
                    best_score = score
                    self.best_params = params

        logger.info("Genetic optimization complete. Best params: %s", self.best_params)
        return self.best_params

    def get_pareto_front_summary(self) -> list[dict]:
        return self.pareto_front[:10]  # top 10
