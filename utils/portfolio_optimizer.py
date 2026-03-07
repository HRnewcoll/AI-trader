"""
Portfolio optimizer — multi-pair position sizing.

Implements three complementary approaches inspired by PyPortfolioOpt:

1. **Hierarchical Risk Parity (HRP)**  
   Clusters assets by return correlation, allocates capital inversely
   proportional to cluster risk. No mean estimates needed — robust to
   estimation error.  (López de Prado 2016)

2. **Mean-Variance Efficient Frontier (MVO)**  
   Maximum Sharpe ratio portfolio from expected returns + covariance.
   Falls back gracefully when optimisation fails.

3. **Risk Parity (Equal Risk Contribution)**  
   Each asset contributes equally to portfolio volatility.  Simpler
   than HRP but effective as a baseline.

Usage:
    optimizer = PortfolioOptimizer(method="hrp")
    weights = optimizer.optimise(returns_df)
    # weights: {pair: fraction_of_capital}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioWeights:
    weights: dict[str, float]          # {pair: weight_fraction}
    method: str
    diversification_ratio: float = 0.0
    expected_annual_return: float = 0.0
    expected_annual_vol: float = 0.0
    expected_sharpe: float = 0.0

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "diversification_ratio": round(self.diversification_ratio, 4),
            "expected_annual_return": round(self.expected_annual_return, 4),
            "expected_annual_vol": round(self.expected_annual_vol, 4),
            "expected_sharpe": round(self.expected_sharpe, 4),
        }

    def max_weight(self) -> float:
        return max(self.weights.values()) if self.weights else 0.0

    def min_weight(self) -> float:
        return min(self.weights.values()) if self.weights else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# HRP helpers (López de Prado)
# ─────────────────────────────────────────────────────────────────────────────

def _corr_to_dist(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix (0=identical, 1=opposite)."""
    return np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))


def _quasi_diagonalise(link: np.ndarray) -> list[int]:
    """Re-order items from a hierarchical clustering to quasi-diagonal form."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    # Column 3 of scipy's linkage matrix = count of original observations in the merged cluster
    num_items = link[-1, 3]  # number of original (leaf) items

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # make even indices
        df0 = sort_ix[sort_ix >= num_items]  # find clusters
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


def _get_cluster_var(cov: np.ndarray, cluster_items: list[int]) -> float:
    """Variance of a portfolio with equal weights within a cluster."""
    cov_slice = cov[np.ix_(cluster_items, cluster_items)]
    w = np.full(len(cluster_items), 1.0 / len(cluster_items))
    return float(w @ cov_slice @ w)


def _get_recursive_bisection_weights(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
    """Recursive bisection allocation across the quasi-diagonal covariance matrix."""
    weights = pd.Series(1.0, index=sort_ix)
    cluster_items = [sort_ix]

    while len(cluster_items) > 0:
        cluster_items = [
            i[j:k]
            for i in cluster_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(cluster_items), 2):
            if i + 1 >= len(cluster_items):
                break
            cluster_0 = cluster_items[i]
            cluster_1 = cluster_items[i + 1]
            alpha = _get_cluster_var(cov, cluster_0)
            alpha /= (alpha + _get_cluster_var(cov, cluster_1) + 1e-10)
            weights[cluster_0] *= 1 - alpha
            weights[cluster_1] *= alpha

    return weights.values


# ─────────────────────────────────────────────────────────────────────────────
# Equal Risk Contribution
# ─────────────────────────────────────────────────────────────────────────────

def _risk_parity_weights(cov: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """
    Equal risk contribution (ERC) weights via gradient descent.
    Each asset contributes σ_i * w_i equally to total portfolio vol.
    """
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        sigma = np.sqrt(np.dot(w, np.dot(cov, w)) + 1e-12)
        mrc = np.dot(cov, w) / sigma          # marginal risk contribution
        rc = w * mrc                           # risk contribution
        target = sigma / n                     # equal target
        w = w * (target / (rc + 1e-12))
        w = np.clip(w, 1e-6, None)
        w /= w.sum()

    return w


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioOptimizer:
    """
    Multi-pair portfolio optimizer.

    Parameters
    ----------
    method : "hrp" | "mvo" | "risk_parity"
    min_weight : minimum weight per asset (prevents near-zero allocations)
    max_weight : maximum weight per asset (concentration limit)
    risk_free_rate : annual risk-free rate for Sharpe computation
    trading_periods_per_year : 252 for daily, 1460 for H4
    """

    def __init__(
        self,
        method: str = "hrp",
        min_weight: float = 0.02,
        max_weight: float = 0.40,
        risk_free_rate: float = 0.0,
        trading_periods_per_year: int = 1460,
    ):
        self.method = method.lower()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.risk_free_rate = risk_free_rate
        self.trading_periods = trading_periods_per_year

    def optimise(
        self,
        returns_df: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None,
    ) -> PortfolioWeights:
        """
        Compute optimal portfolio weights from returns DataFrame.

        Parameters
        ----------
        returns_df : DataFrame with columns = pairs, rows = time periods.
                     Values = per-period returns (not cumulative).
        expected_returns : optional pd.Series with expected returns per pair.
                          Used only for MVO. If None, uses historical mean.

        Returns
        -------
        PortfolioWeights
        """
        # Clean: drop all-NaN columns, forward-fill sparse NaNs
        df = returns_df.copy()
        df = df.dropna(axis=1, how="all")
        df = df.ffill().fillna(0.0)

        if df.shape[1] < 2:
            logger.warning("PortfolioOptimizer: fewer than 2 assets — returning equal weights")
            pair = df.columns[0] if df.shape[1] == 1 else "unknown"
            return PortfolioWeights(weights={pair: 1.0}, method=self.method)

        if df.shape[0] < 10:
            logger.warning("PortfolioOptimizer: insufficient history — returning equal weights")
            n = df.shape[1]
            return PortfolioWeights(
                weights={c: 1.0 / n for c in df.columns},
                method=self.method,
            )

        cov = df.cov().values
        corr = df.corr().values
        pairs = list(df.columns)
        n = len(pairs)

        if self.method == "hrp":
            weights_arr = self._hrp(cov, corr)
        elif self.method == "mvo":
            mu = (
                expected_returns.values
                if expected_returns is not None
                else df.mean().values
            )
            weights_arr = self._mvo(mu, cov)
        elif self.method == "risk_parity":
            weights_arr = _risk_parity_weights(cov)
        else:
            logger.warning("Unknown method '%s' — defaulting to HRP", self.method)
            weights_arr = self._hrp(cov, corr)

        # Apply min/max constraints and renormalise
        weights_arr = np.clip(weights_arr, self.min_weight, self.max_weight)
        weights_arr /= weights_arr.sum()

        weights_dict = {p: float(w) for p, w in zip(pairs, weights_arr)}

        # Compute portfolio statistics
        ann = self.trading_periods
        port_ret = float(df.mean().dot(weights_arr) * ann)
        port_vol = float(np.sqrt(weights_arr @ cov @ weights_arr * ann))
        sharpe = (port_ret - self.risk_free_rate) / (port_vol + 1e-8)

        # Diversification ratio = weighted avg vol / portfolio vol
        asset_vols = np.sqrt(np.diag(cov) * ann)
        div_ratio = float(weights_arr.dot(asset_vols) / (port_vol + 1e-8))

        return PortfolioWeights(
            weights=weights_dict,
            method=self.method,
            diversification_ratio=div_ratio,
            expected_annual_return=port_ret,
            expected_annual_vol=port_vol,
            expected_sharpe=sharpe,
        )

    # ─────────────────────────────────────────────────────────────────────
    # HRP
    # ─────────────────────────────────────────────────────────────────────

    def _hrp(self, cov: np.ndarray, corr: np.ndarray) -> np.ndarray:
        """Hierarchical Risk Parity allocation."""
        dist = _corr_to_dist(corr)
        dist_condensed = squareform(dist, checks=False)
        link = linkage(dist_condensed, method="single")
        sort_ix = _quasi_diagonalise(link)
        weights_arr = _get_recursive_bisection_weights(cov, sort_ix)
        return np.asarray(weights_arr, dtype=float)

    # ─────────────────────────────────────────────────────────────────────
    # MVO — Maximum Sharpe
    # ─────────────────────────────────────────────────────────────────────

    def _mvo(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Maximum Sharpe portfolio via scipy optimisation."""
        n = len(mu)
        w0 = np.full(n, 1.0 / n)

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(w @ mu)
            port_vol = float(np.sqrt(w @ cov @ w) + 1e-10)
            return -(port_ret - self.risk_free_rate) / port_vol

        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(self.min_weight, self.max_weight)] * n

        try:
            res = minimize(
                neg_sharpe,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if res.success:
                return np.clip(res.x, 0, None)
        except Exception as e:
            logger.debug("MVO optimisation failed: %s", e)

        # Fallback to equal weights
        return w0

    # ─────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────

    def compute_lot_sizes(
        self,
        weights: PortfolioWeights,
        total_equity: float,
        pip_value: float = 10.0,
        lot_multiplier: float = 100_000.0,
    ) -> dict[str, float]:
        """
        Convert portfolio weights to lot sizes.

        lot_size = (equity * weight) / (pip_value * lot_multiplier)
        """
        return {
            pair: round(total_equity * w / (pip_value * lot_multiplier), 3)
            for pair, w in weights.weights.items()
        }

    @staticmethod
    def rolling_hrp(
        returns_df: pd.DataFrame,
        window: int = 252,
        rebalance_every: int = 20,
    ) -> pd.DataFrame:
        """
        Compute rolling HRP weights, rebalanced every N periods.
        Returns DataFrame: rows = time, cols = pairs.
        """
        optimizer = PortfolioOptimizer(method="hrp")
        weights_history = []

        for i in range(window, len(returns_df) + 1, rebalance_every):
            window_df = returns_df.iloc[i - window:i]
            result = optimizer.optimise(window_df)
            weights_history.append({
                "date": returns_df.index[i - 1] if hasattr(returns_df.index[i - 1], "date") else i - 1,
                **result.weights,
            })

        if not weights_history:
            return pd.DataFrame()

        return pd.DataFrame(weights_history).set_index("date")
