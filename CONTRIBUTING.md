# Contributing to ForexAI

Thank you for your interest in contributing! This is an open-source algorithmic trading
framework and all contributions — bug fixes, new strategies, new data sources, tests,
or documentation improvements — are welcome.

---

## Quick Setup

```bash
git clone https://github.com/HRnewcoll/AI-trader.git
cd AI-trader
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
pytest tests/test_core.py -v       # all 94 tests should pass
```

---

## Project Layout

| Directory | Purpose |
|---|---|
| `agents/` | Trading agents (technical, sentiment, correlation, pattern, LLM) |
| `data_pipeline/` | Market data fetchers, news/RSS scrapers, feature store |
| `features/` | Technical indicators, candle patterns, order flow, alpha factors |
| `models/` | XGBoost, Transformer, RL, ensemble stacker, hyperopt |
| `execution/` | Broker bridges (MT5, OANDA, paper) |
| `backtesting/` | Walk-forward backtesting engine |
| `risk_healing/` | Risk manager, self-healer, circuit breaker |
| `utils/` | Portfolio optimizer, alert manager, QuantStats reporter, logger |
| `dashboard/` | Streamlit real-time dashboard |
| `tests/` | Offline pytest test suite |

---

## How to Contribute

### 1. Bug Fixes

1. Open an issue describing the bug and steps to reproduce
2. Fork the repo and create a branch: `git checkout -b fix/short-description`
3. Add a test that reproduces the bug (see `tests/test_core.py` for examples)
4. Fix the bug
5. Confirm all tests pass: `pytest tests/test_core.py -v`
6. Submit a pull request referencing the issue

### 2. New Features

Good candidates for new features (inspired by the reference repos in the issue):

**New strategies / signals**
- New technical indicators or candle patterns (`features/`)
- New alpha factors with IC evaluation (`features/alpha_factors.py`)
- Additional chart pattern detection (`agents/pattern_agent.py`)

**New data sources**
- New broker API adapters (`execution/`)
- New news/social data scrapers (`data_pipeline/`)
- Alternative market data providers

**New models**
- Additional ML architectures (`models/`)
- New RL reward functions (`models/rl_environment.py`)
- New ensemble combination methods (`models/ensemble_stacker.py`)

**Infrastructure**
- Portfolio analytics improvements (`utils/portfolio_optimizer.py`)
- New notification channels (`notifications/notifier.py`)
- Backtesting improvements (`backtesting/walk_forward.py`)

### 3. Tests

All tests live in `tests/test_core.py`. Each new module should have at least:

- One "happy path" test that covers normal operation
- One edge-case test (empty data, single element, invalid input)

Tests must:
- Run fully **offline** (no external APIs, no trained models, no MT5)
- Use `pytest` fixtures for shared test data
- Complete in under 30 seconds total

Example test structure:
```python
def test_my_new_feature(sample_ohlcv):
    from features.my_module import MyNewFeature
    result = MyNewFeature().compute(sample_ohlcv)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_ohlcv)
```

### 4. Documentation

- Update `README.md` when adding new modules (architecture table + features table)
- Add docstrings to all public classes and functions
- Update `config/settings.yaml` when adding configurable parameters

---

## Code Style

- **Python 3.10+** with `from __future__ import annotations`
- **Black** for formatting (line length 100): `black . --line-length 100`
- **isort** for import ordering: `isort .`
- **flake8** for linting: `flake8 . --max-line-length 100 --ignore E203,W503`
- Type hints on all public function signatures
- No external API calls in production code without a fallback / mock

---

## Pull Request Checklist

Before submitting a PR, please confirm:

- [ ] `pytest tests/test_core.py -v` — all tests pass (including any new tests you added)
- [ ] New module has at least 2 tests
- [ ] `README.md` architecture/features table updated (if applicable)
- [ ] `config/settings.yaml` updated with new configurable parameters
- [ ] No hardcoded API keys, credentials, or secrets
- [ ] All new functions have docstrings
- [ ] Code formatted with Black (`black . --line-length 100`)

---

## Adding a New Strategy

The simplest way to add a new trading strategy signal is through the `TechnicalAgent`:

```python
# agents/technical_agent.py — add a new rule in analyse()
def analyse(self, df: pd.DataFrame) -> int:
    # ... existing rules ...

    # Your new strategy: London Breakout
    if self._london_breakout(df):
        signals.append(1)  # buy signal

def _london_breakout(self, df: pd.DataFrame) -> bool:
    """Detect London session breakout above Asian range high."""
    # implementation ...
```

For a standalone strategy agent, follow the pattern in `agents/technical_agent.py`:
- Class with `analyse(df) -> int` returning `+1` (buy), `-1` (sell), or `0` (hold)
- Registered in `agents/orchestrator.py` vote weighting

---

## Adding a New Data Source

1. Create `data_pipeline/my_source.py` with a function returning `List[str]` (text snippets)
2. Register in `agents/orchestrator.py` in `_fetch_sentiment()`
3. Add config keys to `config/settings.yaml`
4. Add tests that mock HTTP calls

---

## Adding a New Alpha Factor

```python
# features/alpha_factors.py — add a new _factor method to AlphaFactorCalculator
def _my_factor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Description of what this factor captures."""
    col = "alpha_my_factor"
    # compute the factor...
    df[col] = ...
    return df, [col]
```

Then register it in `compute_all()`:
```python
fns = [
    ...
    self._my_factor,  # add here
]
```

The `FactorEvaluator` will automatically compute IC/ICIR for it.

---

## Questions?

Open an issue with the label `question` — we're happy to help!
