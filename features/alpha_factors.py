"""
Alpha factor research module (inspired by microsoft/qlib).

Computes, evaluates, and combines alpha signals:

1. **Raw factor computation** — momentum, mean-reversion, vol, carry, etc.
2. **IC / ICIR** — Information Coefficient (rank correlation factor → forward return)
3. **Factor neutralisation** — orthogonalise factor against a benchmark factor
4. **Factor combination** — weighted composite alpha from ranked ICs
5. **Turnover analysis** — factor autocorrelation (stability)

IC > 0.05 is generally tradeable.
ICIR > 0.3 (IC / IC_std) indicates consistent predictive power.

Usage:
    calculator = AlphaFactorCalculator()
    df = calculator.compute_all(ohlcv_df)
    evaluator = FactorEvaluator(forward_returns_col="fwd_1")
    scores = evaluator.evaluate_all(df, calculator.factor_cols)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Factor evaluation result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FactorScore:
    name: str
    ic: float           # mean Information Coefficient
    icir: float         # IC / IC_std  (IC information ratio)
    rank_ic: float      # rank (Spearman) IC
    turnover: float     # 1 - autocorrelation (lower = more stable)
    n_obs: int

    def is_tradeable(self, min_ic: float = 0.04, min_icir: float = 0.3) -> bool:
        return abs(self.ic) >= min_ic and abs(self.icir) >= min_icir

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ic": round(self.ic, 4),
            "icir": round(self.icir, 4),
            "rank_ic": round(self.rank_ic, 4),
            "turnover": round(self.turnover, 4),
            "n_obs": self.n_obs,
            "tradeable": self.is_tradeable(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factor computations
# ─────────────────────────────────────────────────────────────────────────────

class AlphaFactorCalculator:
    """
    Compute a library of alpha factors from OHLCV data.

    All factors are cross-sectionally z-scored (mean=0, std=1) after
    computation so they can be combined on a common scale.
    """

    def __init__(self, zscore: bool = True):
        self.zscore = zscore
        self.factor_cols: list[str] = []

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all factors and append to df. Returns augmented DataFrame."""
        df = df.copy()
        fns = [
            self._momentum,
            self._mean_reversion,
            self._volatility_factor,
            self._volume_factor,
            self._trend_strength,
            self._carry_proxy,
            self._skew_factor,
            self._range_factor,
            self._efficiency_ratio,
            self._acf_factor,
        ]
        new_cols = []
        for fn in fns:
            try:
                df, cols = fn(df)
                new_cols.extend(cols)
            except Exception as e:
                logger.debug("Factor computation error in %s: %s", fn.__name__, e)

        if self.zscore:
            df = self._zscore_factors(df, new_cols)

        self.factor_cols = new_cols
        return df

    # ─────────────────────────────────────────────────────────────────────
    # Individual factors
    # ─────────────────────────────────────────────────────────────────────

    def _momentum(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Cross-sectional momentum: returns over N periods."""
        cols = []
        for p in [5, 10, 20, 60]:
            col = f"alpha_mom_{p}"
            df[col] = df["close"].pct_change(p)
            cols.append(col)
        return df, cols

    def _mean_reversion(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Deviation from moving average — higher value → overextended → reversal."""
        cols = []
        for p in [10, 20, 50]:
            col = f"alpha_mr_{p}"
            ma = df["close"].rolling(p).mean()
            # Negative sign: when price > MA (overextended up), factor is negative (bearish reversion signal)
            df[col] = -(df["close"] - ma) / (ma + 1e-8)
            cols.append(col)
        return df, cols

    def _volatility_factor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Inverse volatility — low-vol assets tend to outperform (low-vol anomaly)."""
        cols = []
        for p in [10, 20]:
            col = f"alpha_invvol_{p}"
            vol = df["close"].pct_change().rolling(p).std() + 1e-8
            df[col] = -vol  # negative: lower vol → higher alpha
            cols.append(col)
        return df, cols

    def _volume_factor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Volume surprise — abnormal volume often precedes moves."""
        col = "alpha_vol_surprise"
        vol_ma = df["volume"].rolling(20).mean() + 1e-8
        df[col] = (df["volume"] - vol_ma) / vol_ma
        return df, [col]

    def _trend_strength(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """ADX-like trend strength — strong trends tend to continue."""
        col = "alpha_trend_strength"
        if "adx" in df.columns:
            df[col] = df["adx"] / 100.0
        else:
            # Simple proxy: |close - open| / (high - low + 1e-8)
            df[col] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-8)
        return df, [col]

    def _carry_proxy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        FX carry proxy: overnight gap as proxy for interest differential.
        (Proper carry requires swap rates — this is an approximation.)
        """
        col = "alpha_carry"
        # Use log close-to-open gap (overnight return as carry proxy)
        df[col] = np.log(df["open"] / df["close"].shift(1) + 1e-8)
        return df, [col]

    def _skew_factor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Rolling return skewness — negative skew tends to mean-revert."""
        col = "alpha_skew"
        rets = df["close"].pct_change()
        df[col] = -rets.rolling(20).apply(lambda x: float(stats.skew(x)) if len(x) > 3 else 0.0, raw=True)
        return df, [col]

    def _range_factor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Normalised range expansion — breakout signal."""
        col = "alpha_range"
        bar_range = (df["high"] - df["low"])
        avg_range = bar_range.rolling(20).mean() + 1e-8
        df[col] = bar_range / avg_range
        return df, [col]

    def _efficiency_ratio(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Kaufman Efficiency Ratio: directional movement / total path.
        1 = perfectly trending, 0 = random walk.
        """
        col = "alpha_efficiency"
        n = 10
        direction = df["close"].diff(n).abs()
        noise = df["close"].diff().abs().rolling(n).sum() + 1e-8
        df[col] = direction / noise
        return df, [col]

    def _acf_factor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Lag-1 autocorrelation of returns — positive = momentum, negative = mean reversion."""
        col = "alpha_acf1"
        rets = df["close"].pct_change()
        df[col] = rets.rolling(20).apply(
            lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) > 5 else 0.0,
            raw=False,
        )
        return df, [col]

    def _zscore_factors(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Rolling z-score normalisation for each factor column."""
        for col in cols:
            if col not in df.columns:
                continue
            mu = df[col].rolling(60, min_periods=5).mean()
            sigma = df[col].rolling(60, min_periods=5).std() + 1e-8
            df[col] = ((df[col] - mu) / sigma).clip(-3, 3)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Factor evaluation
# ─────────────────────────────────────────────────────────────────────────────

class FactorEvaluator:
    """
    Evaluates alpha factors using IC / ICIR metrics.

    IC (Information Coefficient) = Spearman rank correlation between
    factor values today and forward returns N periods ahead.
    """

    def __init__(
        self,
        forward_periods: list[int] | None = None,
        primary_period: int = 1,
    ):
        self.forward_periods = forward_periods or [1, 5, 10]
        self.primary_period = primary_period

    def compute_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add forward return columns to df."""
        df = df.copy()
        for p in self.forward_periods:
            df[f"fwd_{p}"] = df["close"].pct_change(p).shift(-p)
        return df

    def ic(self, factor: pd.Series, forward_ret: pd.Series) -> float:
        """Pearson IC between factor and forward return."""
        valid = pd.concat([factor, forward_ret], axis=1).dropna()
        if len(valid) < 10:
            return 0.0
        return float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))

    def rank_ic(self, factor: pd.Series, forward_ret: pd.Series) -> float:
        """Spearman rank IC."""
        valid = pd.concat([factor, forward_ret], axis=1).dropna()
        if len(valid) < 10:
            return 0.0
        corr, _ = stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
        return float(corr) if not np.isnan(corr) else 0.0

    def rolling_ic(
        self,
        factor: pd.Series,
        forward_ret: pd.Series,
        window: int = 60,
    ) -> pd.Series:
        """Rolling IC over a window."""
        ic_series = []
        for i in range(window, len(factor)):
            f_slice = factor.iloc[i - window:i]
            r_slice = forward_ret.iloc[i - window:i]
            ic_series.append(self.ic(f_slice, r_slice))
        idx = factor.index[window:]
        return pd.Series(ic_series, index=idx, name=f"IC_{factor.name}")

    def evaluate_factor(
        self,
        df: pd.DataFrame,
        factor_col: str,
        fwd_col: str | None = None,
    ) -> FactorScore:
        """Evaluate a single factor."""
        if fwd_col is None:
            fwd_col = f"fwd_{self.primary_period}"
        if fwd_col not in df.columns:
            df = self.compute_forward_returns(df)

        factor = df[factor_col].dropna()
        fwd = df[fwd_col].reindex(factor.index).dropna()
        both = pd.concat([factor, fwd], axis=1).dropna()
        if len(both) < 20:
            return FactorScore(name=factor_col, ic=0.0, icir=0.0, rank_ic=0.0, turnover=1.0, n_obs=0)

        f_vals = both.iloc[:, 0]
        r_vals = both.iloc[:, 1]

        # Rolling IC
        window = min(60, len(both) // 2)
        ic_series = []
        for i in range(window, len(both)):
            fs = f_vals.iloc[i - window:i]
            rs = r_vals.iloc[i - window:i]
            ic_series.append(self.ic(fs, rs))

        ic_arr = np.array(ic_series)
        mean_ic = float(np.nanmean(ic_arr)) if len(ic_arr) > 0 else 0.0
        std_ic = float(np.nanstd(ic_arr)) + 1e-8
        icir = mean_ic / std_ic

        # Rank IC
        r_ic = self.rank_ic(f_vals, r_vals)

        # Turnover = 1 - autocorrelation of ranked factor (stability)
        ranked = f_vals.rank()
        ac1 = float(ranked.autocorr(lag=1)) if len(ranked) > 2 else 0.0
        turnover = 1.0 - abs(ac1) if not np.isnan(ac1) else 1.0

        return FactorScore(
            name=factor_col,
            ic=mean_ic,
            icir=icir,
            rank_ic=r_ic,
            turnover=turnover,
            n_obs=len(both),
        )

    def evaluate_all(
        self,
        df: pd.DataFrame,
        factor_cols: list[str],
    ) -> list[FactorScore]:
        """Evaluate all factors and return sorted by |ICIR|."""
        if f"fwd_{self.primary_period}" not in df.columns:
            df = self.compute_forward_returns(df)

        scores = []
        for col in factor_cols:
            if col not in df.columns:
                continue
            score = self.evaluate_factor(df, col)
            scores.append(score)

        return sorted(scores, key=lambda s: abs(s.icir), reverse=True)

    def get_best_factors(
        self,
        scores: list[FactorScore],
        n: int = 10,
        min_ic: float = 0.04,
        min_icir: float = 0.3,
    ) -> list[str]:
        """Return names of top N tradeable factors."""
        tradeable = [s for s in scores if s.is_tradeable(min_ic, min_icir)]
        return [s.name for s in tradeable[:n]]


# ─────────────────────────────────────────────────────────────────────────────
# Factor combination
# ─────────────────────────────────────────────────────────────────────────────

def combine_factors(
    df: pd.DataFrame,
    factor_cols: list[str],
    weights: Optional[dict[str, float]] = None,
    method: str = "ic_weighted",
    scores: Optional[list[FactorScore]] = None,
) -> pd.Series:
    """
    Combine multiple factors into a single composite alpha signal.

    Parameters
    ----------
    df          : DataFrame with factor columns
    factor_cols : list of factor column names to combine
    weights     : optional manual weights dict {col: weight}
    method      : "equal" | "ic_weighted" | "manual"
    scores      : list of FactorScore (needed for ic_weighted)

    Returns
    -------
    pd.Series   : composite alpha signal (z-scored, range ≈ -3 to +3)
    """
    available = [c for c in factor_cols if c in df.columns]
    if not available:
        return pd.Series(0.0, index=df.index)

    if method == "equal" or weights is None:
        w = {c: 1.0 / len(available) for c in available}
    elif method == "ic_weighted" and scores is not None:
        ic_map = {s.name: abs(s.icir) for s in scores}
        total = sum(ic_map.get(c, 0.0) for c in available) + 1e-8
        w = {c: ic_map.get(c, 0.0) / total for c in available}
    elif method == "manual" and weights is not None:
        total = sum(weights.get(c, 0.0) for c in available) + 1e-8
        w = {c: weights.get(c, 0.0) / total for c in available}
    else:
        w = {c: 1.0 / len(available) for c in available}

    composite = sum(df[c].fillna(0) * w[c] for c in available)

    # Z-score the composite
    mu = composite.rolling(60, min_periods=10).mean()
    sigma = composite.rolling(60, min_periods=10).std() + 1e-8
    return ((composite - mu) / sigma).clip(-3, 3).rename("composite_alpha")


def neutralise_factor(
    factor: pd.Series,
    benchmark: pd.Series,
) -> pd.Series:
    """
    Remove linear exposure to a benchmark factor via OLS regression.
    Returns the residual (orthogonalised) factor.
    """
    df = pd.concat([factor, benchmark], axis=1).dropna()
    if len(df) < 10:
        return factor

    x = df.iloc[:, 1].values
    y = df.iloc[:, 0].values
    # OLS: y = a + b*x + e
    b, a = np.polyfit(x, y, 1)
    residual = y - (a + b * x)
    result = pd.Series(residual, index=df.index)
    return result.reindex(factor.index)
