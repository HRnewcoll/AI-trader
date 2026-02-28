use pyo3::prelude::*;
use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::{Normal, Distribution};

/// Monte Carlo simulation — 10,000 paths in parallel
/// Returns (var_95, var_99, expected_shortfall, ruin_probability)
#[pyfunction]
fn monte_carlo_var(
    returns: Vec<f64>,
    initial_equity: f64,
    n_paths: usize,
    n_steps: usize,
    risk_free_rate: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    let n = returns.len();
    if n == 0 {
        return Ok((0.0, 0.0, 0.0, 0.0));
    }

    let mean: f64 = returns.iter().sum::<f64>() / n as f64;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    let final_equities: Vec<f64> = (0..n_paths)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let normal = Normal::new(mean, std_dev).unwrap();
            let mut equity = initial_equity;
            for _ in 0..n_steps {
                let r: f64 = normal.sample(&mut rng);
                equity *= 1.0 + r;
                if equity <= 0.0 {
                    return 0.0;
                }
            }
            equity
        })
        .collect();

    let mut pnl: Vec<f64> = final_equities.iter().map(|e| e - initial_equity).collect();
    pnl.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let var_95_idx = (n_paths as f64 * 0.05) as usize;
    let var_99_idx = (n_paths as f64 * 0.01) as usize;
    let var_95 = -pnl[var_95_idx];
    let var_99 = -pnl[var_99_idx];

    let es: f64 = pnl[..var_95_idx].iter().sum::<f64>() / var_95_idx as f64;
    let expected_shortfall = -es;

    let ruin_count = final_equities.iter().filter(|&&e| e <= initial_equity * 0.5).count();
    let ruin_probability = ruin_count as f64 / n_paths as f64;

    Ok((var_95, var_99, expected_shortfall, ruin_probability))
}

/// Rust-accelerated tick-level backtester
/// ohlcv: list of (open, high, low, close, volume)
/// signals: list of (timestamp_idx, direction, size) — direction: 1=buy, -1=sell, 0=close
/// Returns (total_return, sharpe, max_drawdown, win_rate, trades_count)
#[pyfunction]
fn backtest_tick(
    ohlcv: Vec<(f64, f64, f64, f64, f64)>,
    signals: Vec<(usize, i32, f64)>,
    spread_pips: f64,
    commission_pct: f64,
    slippage_pips: f64,
    pip_value: f64,
    initial_equity: f64,
) -> PyResult<(f64, f64, f64, f64, usize)> {
    let mut equity = initial_equity;
    let mut peak_equity = initial_equity;
    let mut max_drawdown: f64 = 0.0;
    let mut position: f64 = 0.0;
    let mut entry_price: f64 = 0.0;
    let mut returns: Vec<f64> = Vec::new();
    let mut wins: usize = 0;
    let mut losses: usize = 0;

    let total_cost_pips = spread_pips + slippage_pips;

    for (idx, direction, size) in &signals {
        if *idx >= ohlcv.len() {
            continue;
        }
        let (open, _high, _low, close, _vol) = ohlcv[*idx];

        // Close existing position
        if position != 0.0 {
            let raw_pnl = (close - entry_price) * position * size * pip_value;
            let cost = total_cost_pips * pip_value * size.abs();
            let commission = equity * commission_pct;
            let trade_pnl = raw_pnl - cost - commission;
            let ret = trade_pnl / equity;
            returns.push(ret);
            equity += trade_pnl;
            if trade_pnl > 0.0 { wins += 1; } else { losses += 1; }
            position = 0.0;
        }

        // Open new position
        if *direction != 0 {
            position = *direction as f64;
            entry_price = open + (total_cost_pips * pip_value * *direction as f64);
        }

        // Update drawdown
        if equity > peak_equity {
            peak_equity = equity;
        }
        let dd = (peak_equity - equity) / peak_equity;
        if dd > max_drawdown {
            max_drawdown = dd;
        }
    }

    let total_return = (equity - initial_equity) / initial_equity;
    let trades = wins + losses;
    let win_rate = if trades > 0 { wins as f64 / trades as f64 } else { 0.0 };

    // Sharpe ratio
    let sharpe = if returns.len() > 1 {
        let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let var: f64 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
        let std = var.sqrt();
        if std > 0.0 { mean_ret / std * (252f64).sqrt() } else { 0.0 }
    } else {
        0.0
    };

    Ok((total_return, sharpe, max_drawdown, win_rate, trades))
}

/// Genetic optimizer fitness evaluation — batch Sharpe calculation
#[pyfunction]
fn evaluate_population_fitness(
    returns_matrix: Vec<Vec<f64>>,
    drawdowns: Vec<f64>,
) -> PyResult<Vec<(f64, f64, f64)>> {
    let results: Vec<(f64, f64, f64)> = returns_matrix
        .par_iter()
        .zip(drawdowns.par_iter())
        .map(|(returns, &max_dd)| {
            let n = returns.len();
            if n < 2 {
                return (0.0, 0.0, 0.0);
            }
            let mean: f64 = returns.iter().sum::<f64>() / n as f64;
            let var: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
            let std = var.sqrt();
            let sharpe = if std > 0.0 { mean / std * (252f64).sqrt() } else { 0.0 };

            // Sortino
            let neg_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
            let sortino = if neg_returns.len() > 0 {
                let downside_var: f64 = neg_returns.iter().map(|r| r.powi(2)).sum::<f64>() / neg_returns.len() as f64;
                let downside_std = downside_var.sqrt();
                if downside_std > 0.0 { mean / downside_std * (252f64).sqrt() } else { 0.0 }
            } else { sharpe };

            // Calmar
            let total_return: f64 = returns.iter().product::<f64>() - 1.0;
            let calmar = if max_dd > 0.0 { total_return / max_dd } else { 0.0 };

            (sharpe, sortino, calmar)
        })
        .collect();

    Ok(results)
}

#[pymodule]
fn forex_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(monte_carlo_var, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_tick, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_population_fitness, m)?)?;
    Ok(())
}
