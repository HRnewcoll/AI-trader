"""
QuantStats-inspired performance report generator.

Produces a comprehensive performance report in JSON and HTML formats
with all standard quantitative trading metrics, equity curves, and
drawdown analysis — inspired by ranaroussi/quantstats.

Metrics included:
  - Return statistics (CAGR, total return, best/worst period)
  - Risk metrics (Sharpe, Sortino, Calmar, Omega, MAR)
  - Drawdown analysis (max DD, avg DD, DD duration, recovery time)
  - Trade statistics (win rate, profit factor, expectancy, streaks)
  - Tail risk (VaR 95/99, CVaR, skewness, kurtosis, max loss)
  - Rolling metrics (rolling Sharpe, rolling volatility)

Usage:
    reporter = QuantStatsReporter()
    report = reporter.generate(equity_curve, trades, pair="EURUSD")
    reporter.save_json(report, "reports/EURUSD.json")
    reporter.save_html(report, "reports/EURUSD.html")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Report data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceReport:
    pair: str
    generated_at: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    # Overview
    start_date: str = ""
    end_date: str = ""
    n_trading_periods: int = 0
    initial_equity: float = 10_000.0
    final_equity: float = 10_000.0

    # Returns
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    best_period_pct: float = 0.0
    worst_period_pct: float = 0.0
    avg_period_return_pct: float = 0.0
    positive_periods_pct: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    mar_ratio: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0
    recovery_periods: int = 0
    ulcer_index: float = 0.0

    # Trade stats
    n_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Tail risk
    var_95_pct: float = 0.0
    var_99_pct: float = 0.0
    cvar_95_pct: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    max_loss_single_pct: float = 0.0
    gain_to_pain: float = 0.0

    # Rolling series (stored as lists for JSON serialisation)
    rolling_sharpe_30: list[float] = field(default_factory=list)
    rolling_vol_30: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    drawdown_curve: list[float] = field(default_factory=list)
    monthly_returns: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Round all floats
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(rets: np.ndarray, periods_per_year: int = 252) -> float:
    if len(rets) < 2:
        return 0.0
    std = np.std(rets) + 1e-10
    return float(np.mean(rets) / std * np.sqrt(periods_per_year))


def _sortino(rets: np.ndarray, periods_per_year: int = 252) -> float:
    if len(rets) < 2:
        return 0.0
    downside = rets[rets < 0]
    if len(downside) == 0:
        return np.inf
    downside_std = np.std(downside) + 1e-10
    return float(np.mean(rets) / downside_std * np.sqrt(periods_per_year))


def _calmar(rets: np.ndarray, periods_per_year: int = 252) -> float:
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / (peak + 1e-10)
    max_dd = abs(dd.min())
    if max_dd == 0:
        return np.inf
    total_return = cum[-1] - 1
    n_years = len(rets) / periods_per_year
    cagr = (1 + total_return) ** (1 / max(n_years, 1e-3)) - 1
    return float(cagr / max_dd)


def _omega(rets: np.ndarray, threshold: float = 0.0) -> float:
    gains = rets[rets > threshold] - threshold
    losses = threshold - rets[rets <= threshold]
    total_gain = gains.sum()
    total_loss = losses.sum()
    if total_loss == 0:
        return np.inf
    return float(total_gain / total_loss)


def _compute_drawdowns(equity: np.ndarray) -> tuple[float, float, int, int, float]:
    """Returns (max_dd, avg_dd, max_duration, recovery_periods, ulcer_index)."""
    if len(equity) == 0:
        return 0.0, 0.0, 0, 0, 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-10)

    # Max drawdown
    max_dd = float(abs(dd.min()))

    # Average drawdown (only drawdown periods)
    in_dd = dd < 0
    avg_dd = float(abs(dd[in_dd].mean())) if in_dd.any() else 0.0

    # Max drawdown duration
    max_dur, cur_dur = 0, 0
    for d in dd:
        if d < 0:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0

    # Recovery periods (count of times equity returned to new high)
    recovery = 0
    prev_in_dd = False
    for i, d in enumerate(dd):
        if prev_in_dd and d >= 0:
            recovery += 1
        prev_in_dd = (d < 0)

    # Ulcer Index = sqrt(mean(dd^2))
    ulcer = float(np.sqrt(np.mean(dd ** 2)))

    return max_dd, avg_dd, max_dur, recovery, ulcer


def _monthly_returns(equity: np.ndarray, dates: Optional[pd.Index] = None) -> dict[str, float]:
    """Aggregate returns by month."""
    if dates is None or len(equity) != len(dates):
        return {}
    try:
        s = pd.Series(equity, index=pd.to_datetime(dates))
        monthly = s.resample("ME").last().pct_change()
        return {str(d.date())[:7]: round(float(r), 4) for d, r in monthly.items() if not np.isnan(r)}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Reporter
# ─────────────────────────────────────────────────────────────────────────────

class QuantStatsReporter:
    """
    Generate comprehensive performance reports from equity curves and trades.
    """

    def __init__(self, periods_per_year: int = 252, risk_free_rate: float = 0.0):
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

    def generate(
        self,
        equity_curve: list[float] | np.ndarray | pd.Series,
        trades: list[dict] | None = None,
        pair: str = "",
        dates: Optional[pd.Index] = None,
    ) -> PerformanceReport:
        """
        Generate a full performance report.

        Parameters
        ----------
        equity_curve : time series of equity values
        trades       : list of trade dicts with 'pnl' key (optional)
        pair         : currency pair name
        dates        : DatetimeIndex for the equity curve (optional)

        Returns
        -------
        PerformanceReport
        """
        equity = np.asarray(equity_curve, dtype=float)
        if len(equity) < 2:
            return PerformanceReport(pair=pair)

        report = PerformanceReport(pair=pair)
        report.initial_equity = float(equity[0])
        report.final_equity = float(equity[-1])
        report.n_trading_periods = len(equity)

        if dates is not None and len(dates) == len(equity):
            try:
                report.start_date = str(dates[0])[:10]
                report.end_date = str(dates[-1])[:10]
            except Exception:
                pass

        # Per-period returns
        rets = np.diff(equity) / (equity[:-1] + 1e-10)

        # Return stats
        report.total_return_pct = round(float((equity[-1] / equity[0] - 1) * 100), 4)
        n_years = len(rets) / self.periods_per_year
        report.cagr_pct = round(float(((equity[-1] / equity[0]) ** (1 / max(n_years, 1e-3)) - 1) * 100), 4)
        report.best_period_pct = round(float(rets.max() * 100), 4)
        report.worst_period_pct = round(float(rets.min() * 100), 4)
        report.avg_period_return_pct = round(float(rets.mean() * 100), 6)
        report.positive_periods_pct = round(float((rets > 0).mean() * 100), 2)

        # Risk-adjusted
        report.sharpe_ratio = round(_sharpe(rets, self.periods_per_year), 4)
        report.sortino_ratio = round(_sortino(rets, self.periods_per_year), 4)
        report.calmar_ratio = round(_calmar(rets, self.periods_per_year), 4)
        report.omega_ratio = round(_omega(rets), 4)
        total_return = equity[-1] / equity[0] - 1
        cagr = (1 + total_return) ** (1 / max(n_years, 1e-3)) - 1
        max_dd, avg_dd, max_dur, recovery, ulcer = _compute_drawdowns(equity)
        report.mar_ratio = round(float(cagr / (max_dd + 1e-10)), 4)

        # Drawdown
        report.max_drawdown_pct = round(max_dd * 100, 4)
        report.avg_drawdown_pct = round(avg_dd * 100, 4)
        report.max_drawdown_duration = max_dur
        report.recovery_periods = recovery
        report.ulcer_index = round(ulcer * 100, 4)

        # Tail risk
        sorted_rets = np.sort(rets)
        var95_idx = max(0, int(len(sorted_rets) * 0.05))
        var99_idx = max(0, int(len(sorted_rets) * 0.01))
        report.var_95_pct = round(float(-sorted_rets[var95_idx] * 100), 4)
        report.var_99_pct = round(float(-sorted_rets[var99_idx] * 100), 4)
        report.cvar_95_pct = round(float(-sorted_rets[:var95_idx].mean() * 100) if var95_idx > 0 else 0.0, 4)
        try:
            from scipy import stats as scipy_stats
            report.skewness = round(float(scipy_stats.skew(rets)), 4)
            report.kurtosis = round(float(scipy_stats.kurtosis(rets)), 4)
        except Exception:
            report.skewness = round(float(pd.Series(rets).skew()), 4)
            report.kurtosis = round(float(pd.Series(rets).kurtosis()), 4)

        report.max_loss_single_pct = round(float(rets.min() * 100), 4)
        report.gain_to_pain = round(float(rets.sum() / (abs(rets[rets < 0]).sum() + 1e-10)), 4)

        # Rolling metrics
        w = min(30, len(rets) // 2)
        if w >= 5:
            rolling_sharpe = [
                _sharpe(rets[i - w:i], self.periods_per_year)
                for i in range(w, len(rets) + 1)
            ]
            rolling_vol = [
                float(np.std(rets[i - w:i]) * np.sqrt(self.periods_per_year) * 100)
                for i in range(w, len(rets) + 1)
            ]
            report.rolling_sharpe_30 = [round(x, 3) for x in rolling_sharpe]
            report.rolling_vol_30 = [round(x, 3) for x in rolling_vol]

        # Curves (downsample to max 500 points for dashboard)
        step = max(1, len(equity) // 500)
        report.equity_curve = [round(float(e), 4) for e in equity[::step]]
        peak = np.maximum.accumulate(equity)
        dd_curve = (equity - peak) / (peak + 1e-10) * 100
        report.drawdown_curve = [round(float(d), 4) for d in dd_curve[::step]]

        # Monthly returns
        if dates is not None:
            report.monthly_returns = _monthly_returns(equity, dates)

        # Trade stats
        if trades:
            self._add_trade_stats(report, trades)

        return report

    def _add_trade_stats(self, report: PerformanceReport, trades: list[dict]) -> None:
        """Add trade-level statistics to report."""
        pnls = [float(t.get("pnl", t.get("profit", 0.0))) for t in trades if "pnl" in t or "profit" in t]
        if not pnls:
            return

        pnls_arr = np.array(pnls)
        wins = pnls_arr[pnls_arr > 0]
        losses = pnls_arr[pnls_arr < 0]

        report.n_trades = len(pnls)
        report.win_rate_pct = round(float(len(wins) / len(pnls) * 100), 2)
        report.avg_win = round(float(wins.mean()) if len(wins) > 0 else 0.0, 4)
        report.avg_loss = round(float(losses.mean()) if len(losses) > 0 else 0.0, 4)
        report.profit_factor = round(
            float(wins.sum() / (abs(losses.sum()) + 1e-10)) if len(losses) > 0 else np.inf, 4
        )
        report.expectancy = round(float(pnls_arr.mean()), 4)
        report.avg_rr = round(
            float(abs(wins.mean() / losses.mean())) if len(wins) > 0 and len(losses) > 0 else 0.0, 4
        )

        # Streaks
        max_win, cur_win = 0, 0
        max_loss, cur_loss = 0, 0
        for p in pnls:
            if p > 0:
                cur_win += 1
                cur_loss = 0
                max_win = max(max_win, cur_win)
            else:
                cur_loss += 1
                cur_win = 0
                max_loss = max(max_loss, cur_loss)
        report.max_consecutive_wins = max_win
        report.max_consecutive_losses = max_loss

    def save_json(self, report: PerformanceReport, path: str) -> Path:
        """Save report as JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("Report saved to %s", out)
        return out

    def save_html(self, report: PerformanceReport, path: str) -> Path:
        """Save report as a self-contained HTML file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        html = self._render_html(report)
        out.write_text(html)
        logger.info("HTML report saved to %s", out)
        return out

    def _render_html(self, r: PerformanceReport) -> str:
        """Render a minimal but complete HTML performance report."""
        ec = r.equity_curve
        dd = r.drawdown_curve
        x_labels = list(range(len(ec)))

        ec_js = json.dumps(ec)
        dd_js = json.dumps(dd)
        x_js = json.dumps(x_labels)

        # Monthly returns table
        monthly_rows = ""
        for ym, ret in list(r.monthly_returns.items())[-24:]:
            color = "#27ae60" if ret >= 0 else "#e74c3c"
            monthly_rows += f"<tr><td>{ym}</td><td style='color:{color}'>{ret:+.2%}</td></tr>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ForexAI Report — {r.pair}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1 {{ color: #00d4ff; }} h2 {{ color: #aaa; border-bottom: 1px solid #333; padding-bottom: 5px; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
  .card {{ background: #1e2130; border-radius: 8px; padding: 16px; }}
  .card .label {{ font-size: 11px; color: #888; text-transform: uppercase; }}
  .card .value {{ font-size: 22px; font-weight: bold; margin-top: 4px; }}
  .positive {{ color: #27ae60; }} .negative {{ color: #e74c3c; }} .neutral {{ color: #f39c12; }}
  canvas {{ max-height: 300px; }}
  table {{ width: 100%; border-collapse: collapse; }} td, th {{ padding: 6px 10px; border: 1px solid #333; font-size: 12px; }}
  th {{ background: #1e2130; color: #888; }}
</style>
</head>
<body>
<h1>📈 Performance Report — {r.pair}</h1>
<p style="color:#888">{r.start_date} → {r.end_date} | Generated {r.generated_at[:19]} UTC</p>

<div class="grid">
  <div class="card"><div class="label">Total Return</div>
    <div class="value {'positive' if r.total_return_pct >= 0 else 'negative'}">{r.total_return_pct:+.2f}%</div></div>
  <div class="card"><div class="label">CAGR</div>
    <div class="value {'positive' if r.cagr_pct >= 0 else 'negative'}">{r.cagr_pct:+.2f}%</div></div>
  <div class="card"><div class="label">Sharpe</div>
    <div class="value {'positive' if r.sharpe_ratio >= 1 else ('neutral' if r.sharpe_ratio >= 0 else 'negative')}">{r.sharpe_ratio:.3f}</div></div>
  <div class="card"><div class="label">Sortino</div>
    <div class="value {'positive' if r.sortino_ratio >= 1 else 'neutral'}">{r.sortino_ratio:.3f}</div></div>
  <div class="card"><div class="label">Calmar</div>
    <div class="value">{r.calmar_ratio:.3f}</div></div>
  <div class="card"><div class="label">Max Drawdown</div>
    <div class="value negative">-{r.max_drawdown_pct:.2f}%</div></div>
  <div class="card"><div class="label">Win Rate</div>
    <div class="value {'positive' if r.win_rate_pct >= 50 else 'negative'}">{r.win_rate_pct:.1f}%</div></div>
  <div class="card"><div class="label">Profit Factor</div>
    <div class="value {'positive' if r.profit_factor >= 1.5 else ('neutral' if r.profit_factor >= 1 else 'negative')}">{r.profit_factor:.2f}</div></div>
</div>

<h2>Equity Curve</h2>
<canvas id="eqChart"></canvas>

<h2>Drawdown</h2>
<canvas id="ddChart"></canvas>

<h2>Monthly Returns</h2>
<table><tr><th>Month</th><th>Return</th></tr>{monthly_rows}</table>

<h2>Risk Metrics</h2>
<table>
  <tr><th>Metric</th><th>Value</th><th>Metric</th><th>Value</th></tr>
  <tr><td>VaR 95%</td><td>{r.var_95_pct:.3f}%</td><td>CVaR 95%</td><td>{r.cvar_95_pct:.3f}%</td></tr>
  <tr><td>VaR 99%</td><td>{r.var_99_pct:.3f}%</td><td>Skewness</td><td>{r.skewness:.3f}</td></tr>
  <tr><td>Kurtosis</td><td>{r.kurtosis:.3f}</td><td>Omega Ratio</td><td>{r.omega_ratio:.3f}</td></tr>
  <tr><td>MAR Ratio</td><td>{r.mar_ratio:.3f}</td><td>Ulcer Index</td><td>{r.ulcer_index:.3f}%</td></tr>
  <tr><td>Gain-to-Pain</td><td>{r.gain_to_pain:.3f}</td><td>Max Loss</td><td>{r.max_loss_single_pct:.3f}%</td></tr>
</table>

<h2>Trade Statistics</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total Trades</td><td>{r.n_trades}</td></tr>
  <tr><td>Win Rate</td><td>{r.win_rate_pct:.1f}%</td></tr>
  <tr><td>Avg Win</td><td>{r.avg_win:.4f}</td></tr>
  <tr><td>Avg Loss</td><td>{r.avg_loss:.4f}</td></tr>
  <tr><td>Profit Factor</td><td>{r.profit_factor:.3f}</td></tr>
  <tr><td>Expectancy</td><td>{r.expectancy:.4f}</td></tr>
  <tr><td>Avg R:R</td><td>{r.avg_rr:.2f}</td></tr>
  <tr><td>Max Consec. Wins</td><td>{r.max_consecutive_wins}</td></tr>
  <tr><td>Max Consec. Losses</td><td>{r.max_consecutive_losses}</td></tr>
</table>

<script>
const ctx1 = document.getElementById('eqChart').getContext('2d');
new Chart(ctx1, {{ type: 'line', data: {{ labels: {x_js}, datasets: [{{ label: 'Equity', data: {ec_js}, borderColor: '#00d4ff', borderWidth: 1.5, pointRadius: 0, fill: true, backgroundColor: 'rgba(0,212,255,0.05)' }}] }}, options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: '#aaa' }} }} }}, scales: {{ x: {{ ticks: {{ color: '#666' }}, grid: {{ color: '#222' }} }}, y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }} }} }} }});
const ctx2 = document.getElementById('ddChart').getContext('2d');
new Chart(ctx2, {{ type: 'line', data: {{ labels: {x_js}, datasets: [{{ label: 'Drawdown %', data: {dd_js}, borderColor: '#e74c3c', borderWidth: 1.5, pointRadius: 0, fill: true, backgroundColor: 'rgba(231,76,60,0.15)' }}] }}, options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: '#aaa' }} }} }}, scales: {{ x: {{ ticks: {{ color: '#666' }}, grid: {{ color: '#222' }} }}, y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }} }} }} }});
</script>
</body>
</html>"""
