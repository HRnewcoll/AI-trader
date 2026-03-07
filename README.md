# ForexAI — Self-Healing Autonomous Forex Trader

[![Tests](https://img.shields.io/badge/tests-94%20passing-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#requirements)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

A production-grade, multi-agent algorithmic trading system for FX markets.  
Combines ensemble ML models, RL agents, NLP sentiment, explainable AI, alpha factor research, portfolio optimisation, hyperparameter search, and self-healing risk management — **all in one codebase, zero paid APIs required**.

---

## ⚡ One-Click Start

```bash
git clone https://github.com/HRnewcoll/AI-trader.git
cd AI-trader
python run.py
```

That's it. The launcher automatically:
1. Checks and installs all dependencies
2. Copies `.env.example` → `.env` (edit it to add credentials)
3. Trains all models on first launch (XGBoost + DLinear + PatchTST + PPO/DQN)
4. Starts the Streamlit dashboard at [http://localhost:8501](http://localhost:8501)
5. Runs the trader in **paper mode** — no real money, no broker needed

### All CLI modes

```bash
python run.py                    # paper trading + dashboard (default)
python run.py --mode live        # live trading (add MT5/OANDA creds to .env)
python run.py --train-only       # train all models then exit
python run.py --backtest         # walk-forward backtest then exit
python run.py --hyperopt         # Optuna hyperparameter search then exit
python run.py --generate-report  # produce HTML+JSON QuantStats report then exit
python run.py --force-retrain    # retrain even if artifacts exist
python run.py --no-dashboard     # trader only, no Streamlit
python run.py --dashboard-only   # Streamlit only (requires prior training)
```

---

## Requirements

- **Python 3.10+** (3.11 recommended)
- No paid APIs required — works entirely on free data sources
- MT5 or OANDA credentials only needed for **live trading**
- GPU optional — all models train on CPU (slower but fully functional)

---

## Architecture

```
run.py                           ← ONE-CLICK ENTRY POINT
main_orchestrator.py             ← 5-min trading loop + scheduled tasks
│
├── agents/
│   ├── orchestrator.py          ← Multi-agent coordinator (concurrent pairs)
│   ├── technical_agent.py       ← TA rule engine (EMA, RSI, MACD, BB, Stoch)
│   ├── sentiment_agent.py       ← FinBERT + VADER NLP sentiment scorer
│   ├── correlation_agent.py     ← Correlation matrix, position diversification
│   ├── pattern_agent.py         ← 9 chart patterns (H&S, triangles, wedges…)
│   └── llm_orchestrator.py      ← Weighted agent voting + Ollama/rule fallback
│
├── data_pipeline/
│   ├── market_data.py           ← OHLCV fetcher (CCXT / yfinance / MetaApi)
│   ├── news_scraper.py          ← RSS feed ingestion (no API key)
│   ├── reddit_scraper.py        ← Reddit HTML scraper (no API key)
│   ├── twitter_scraper.py       ← Nitter scraper (no API key)
│   ├── economic_calendar.py     ← Macro event calendar (scraped)
│   └── feature_store.py         ← Parquet-backed feature cache
│
├── features/
│   ├── technical_indicators.py  ← 60+ TA features (RSI, MACD, ATR, BB…)
│   ├── candle_patterns.py       ← 11 candlestick patterns + composite score
│   ├── order_flow.py            ← VWAP bands, volume delta, order blocks
│   ├── multi_timeframe.py       ← MTF confluence (M15/H1/H4/D1)
│   ├── regime_detector.py       ← trending/ranging/volatile/quiet regime
│   └── alpha_factors.py         ← 10 alpha factors + IC/ICIR evaluation
│
├── models/
│   ├── xgboost_model.py         ← XGBoost classifier (Optuna HPO, ONNX export)
│   ├── transformer_model.py     ← DLinear & PatchTST time-series transformers
│   ├── rl_agent.py              ← PPO + DQN RL agent (Stable-Baselines3)
│   ├── rl_environment.py        ← Custom Gymnasium FX trading environment
│   ├── ensemble_stacker.py      ← OOF logistic meta-learner over all models
│   ├── hyperopt.py              ← Optuna TPE hyperparameter optimiser
│   ├── genetic_optimizer.py     ← NSGA-II multi-objective genetic optimizer
│   └── training_pipeline.py     ← Full training pipeline (all models)
│
├── execution/
│   ├── mt5_bridge.py            ← MetaTrader5 bridge + PaperBroker fallback
│   └── mt5_ea.mq5               ← MQL5 Expert Advisor (JSON signal bridge)
│
├── backtesting/
│   └── walk_forward.py          ← Vectorised + walk-forward backtesting engine
│
├── risk_healing/
│   ├── risk_manager.py          ← ATR sizing, circuit breaker, Monte Carlo VaR
│   └── self_healer.py           ← Autonomous model retraining + health watchdog
│
├── memory/
│   └── vector_memory.py         ← ChromaDB / in-memory vector store
│
├── xai/
│   └── explainer.py             ← SHAP + LIME + narrative explanations
│
├── notifications/
│   └── notifier.py              ← Telegram / Discord / Slack / email
│
├── monitoring/
│   └── metrics_server.py        ← Prometheus metrics endpoint
│
├── dashboard/
│   └── streamlit_app.py         ← Real-time Streamlit monitoring dashboard
│
├── utils/
│   ├── config_loader.py         ← YAML config with env-var overrides
│   ├── logger.py                ← Structured JSON logging
│   ├── alert_manager.py         ← Event-driven alerts (drawdown, loss, circuit break)
│   ├── portfolio_optimizer.py   ← HRP / MVO / Risk Parity position sizing
│   ├── quantstats_reporter.py   ← HTML + JSON performance reports
│   ├── performance_analytics.py ← Sharpe, Sortino, Calmar, VaR, CVAR…
│   └── trade_journal.py         ← Persistent per-trade logging
│
├── tests/
│   └── test_core.py             ← 94 offline tests (no external APIs needed)
│
├── rust_core/                   ← Rust extensions (high-freq backtesting)
├── config/settings.yaml         ← Main configuration file
├── docker-compose.yml           ← Full stack (trader + Grafana + Prometheus)
├── Dockerfile
├── Makefile                     ← Shortcuts: make test, make docker, make lint
└── CONTRIBUTING.md              ← Contribution guide
```

---

## Key Features

| Component | Details |
|---|---|
| **One-Click** | `python run.py` — auto-installs deps, trains, starts everything |
| **Ensemble Models** | XGBoost + DLinear + PatchTST + PPO/DQN + OOF meta-stacker |
| **Alpha Factors** | 10 research-grade factors: momentum, mean-reversion, carry, skew, efficiency… |
| **Portfolio Optimizer** | HRP (Hierarchical Risk Parity), MVO, Risk Parity — weights all pairs |
| **Hyperparameter Search** | Optuna TPE across 5 spaces (XGB, Transformer, RL, Risk, Indicators) |
| **Candle Patterns** | 11 candlestick patterns + composite candle score |
| **Chart Patterns** | 9 classical chart patterns (H&S, triangles, wedges, cup & handle…) |
| **Order Flow** | VWAP bands, volume delta, order blocks, absorption detection |
| **MTF Confluence** | M15/H1/H4/D1 multi-timeframe signal alignment |
| **Regime Detection** | Trending/ranging/volatile/quiet — auto-adjusts signal strength |
| **Sentiment** | FinBERT + VADER on RSS, Reddit, Twitter (zero API keys needed) |
| **XAI** | SHAP + LIME + natural-language trade explanations |
| **Risk** | ATR sizing, circuit breaker, Monte Carlo VaR, survival mode |
| **Alerts** | Drawdown, consecutive loss, circuit break, daily limit — Telegram/Slack/email |
| **Self-Healing** | Auto model retraining when performance degrades |
| **QuantStats Reports** | HTML + JSON reports: CAGR, Sharpe, VaR, drawdown, monthly returns |
| **Performance Reporter** | `--generate-report` produces a self-contained HTML performance report |
| **Execution** | MetaTrader5 live + PaperBroker fallback + MQL5 EA bridge |
| **Memory** | ChromaDB vector store for similar-trade recall |
| **Concurrent** | All pairs processed in parallel (ThreadPoolExecutor) |
| **Scheduling** | Weekly retraining, genetic optimisation, hourly weight updates |
| **Monitoring** | Prometheus metrics + Grafana dashboard via docker-compose |
| **Trade Journal** | Persistent per-trade log with Parquet export |

---

## Configuration

Copy `.env.example` → `.env` and fill in credentials (**all optional** — system works without any):

```bash
cp .env.example .env
# then edit .env to add Telegram/MT5/OANDA credentials
```

Key settings in `config/settings.yaml` (edit to change pairs, risk, or models):

```yaml
execution:
  mode: paper          # "paper" or "live"

risk:
  max_risk_per_trade_pct: 0.02   # 2% per trade
  circuit_breaker_dd_pct: 0.05   # halt trading at 5% drawdown

portfolio:
  method: hrp          # hrp | mvo | risk_parity
  min_weight: 0.02     # minimum 2% allocation per pair
  max_weight: 0.40     # maximum 40% allocation per pair
  rebalance_cycles: 20 # rebalance every N trading cycles

pairs:
  majors: [EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD]
```

---

## Testing

```bash
# all tests (94 offline, no external APIs, no trained models, no MT5)
pytest tests/test_core.py -v

# or with make:
make test
```

---

## Makefile shortcuts

```bash
make test        # run pytest
make lint        # run flake8
make format      # run black + isort
make docker      # docker-compose up (trader + Grafana + Prometheus)
make report      # generate performance report
make hyperopt    # run hyperparameter optimisation
make clean       # remove artifacts, __pycache__, logs
```

---

## Docker (full stack with Grafana + Prometheus)

```bash
docker-compose up
# Streamlit dashboard → http://localhost:8501
# Grafana            → http://localhost:3000  (admin / forex_admin)
# Prometheus         → http://localhost:9090
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add new features, strategies, or data sources.

---

## Disclaimer

This software is for **educational and research purposes only**. Forex trading involves substantial risk of loss. Past performance is not indicative of future results. Always paper-trade for several months before risking real capital. The authors accept no liability for any trading losses.

---

## License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

