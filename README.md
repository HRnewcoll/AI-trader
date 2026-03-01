# ForexAI — Self-Healing Autonomous Forex Trader

A production-grade, multi-agent algorithmic trading system for FX markets. Combines ensemble ML models, RL agents, NLP sentiment analysis, explainable AI, candle pattern recognition, regime detection, and self-healing risk management — all in one codebase.

---

## ⚡ One-Click Start

```bash
python run.py
```

That's it. The launcher automatically:
1. Checks and installs all dependencies
2. Copies `.env.example` → `.env` (edit it to add credentials)
3. Trains all models on first launch (XGBoost + DLinear + PatchTST + PPO/DQN)
4. Starts the Streamlit dashboard at [http://localhost:8501](http://localhost:8501)
5. Runs the trader in **paper mode** (no real money)

Other modes:
```bash
python run.py --mode live          # live trading (needs MT5 credentials in .env)
python run.py --train-only         # train then exit
python run.py --backtest           # run walk-forward backtest then exit
python run.py --force-retrain      # retrain even if artifacts exist
python run.py --no-dashboard       # trader only, no Streamlit
python run.py --dashboard-only     # Streamlit only
```

---

## Architecture

```
run.py                        ← ONE-CLICK ENTRY POINT
main_orchestrator.py          ← Trader loop (5-min cycle, scheduled tasks)
│
├── agents/
│   ├── orchestrator.py       ← Multi-agent coordinator (concurrent pair processing)
│   ├── technical_agent.py    ← Rule-based TA signal agent (NEW)
│   ├── sentiment_agent.py    ← FinBERT / VADER NLP sentiment scorer
│   ├── correlation_agent.py  ← Currency correlation & position diversification (NEW)
│   └── llm_orchestrator.py   ← LLM reasoning layer (Ollama/rule-based fallback) (NEW)
│
├── data_pipeline/
│   ├── market_data.py        ← OHLCV fetcher (CCXT / yfinance / MetaApi)
│   ├── news_scraper.py       ← RSS feed ingestion (no API key)
│   ├── reddit_scraper.py     ← Reddit post scraper (no API key)
│   ├── twitter_scraper.py    ← Twitter / Nitter scraper (no API key)
│   ├── economic_calendar.py  ← Macro event calendar (scraped)
│   └── feature_store.py      ← Parquet-backed feature cache
│
├── features/
│   ├── technical_indicators.py  ← 60+ TA features (RSI, MACD, ATR, Bollinger…)
│   ├── candle_patterns.py    ← 11 candlestick patterns + composite score (NEW)
│   └── regime_detector.py    ← Market regime: trending/ranging/volatile/quiet (NEW)
│
├── models/
│   ├── xgboost_model.py      ← XGBoost classifier (Optuna HPO, ONNX export)
│   ├── transformer_model.py  ← DLinear & PatchTST time-series transformers
│   ├── rl_agent.py           ← PPO + DQN ensemble RL agent (Stable-Baselines3)
│   ├── rl_environment.py     ← Custom Gymnasium FX trading environment (optimised)
│   ├── genetic_optimizer.py  ← NSGA-II multi-objective genetic optimizer
│   └── training_pipeline.py  ← Orchestrated full training pipeline
│
├── execution/
│   ├── mt5_bridge.py         ← MetaTrader5 bridge + PaperBroker fallback
│   └── mt5_ea.mq5            ← MQL5 Expert Advisor (reads Python signals via JSON)
│
├── backtesting/
│   └── walk_forward.py       ← Vectorised + walk-forward backtesting engine
│
├── risk_healing/
│   ├── risk_manager.py       ← ATR sizing, circuit breaker, Monte Carlo VaR, survival mode
│   └── self_healer.py        ← Autonomous model retraining + health watchdog
│
├── memory/
│   └── vector_memory.py      ← ChromaDB / in-memory vector store for trade memory
│
├── xai/
│   └── explainer.py          ← SHAP + LIME explainability with narrative generation
│
├── notifications/
│   └── notifier.py           ← Telegram / Discord / Slack / email notifications
│
├── monitoring/
│   └── metrics_server.py     ← Prometheus metrics endpoint
│
├── dashboard/
│   └── streamlit_app.py      ← Real-time Streamlit monitoring dashboard
│
├── utils/
│   ├── config_loader.py      ← YAML config loader with env-var overrides
│   └── logger.py             ← Structured JSON logging
│
├── tests/
│   └── test_core.py          ← Pytest test suite (23 tests, no external APIs)
│
├── rust_core/                ← Rust extensions for high-performance backtesting
├── config/settings.yaml      ← Main configuration file
├── docker-compose.yml        ← Full stack (trader + Grafana + Prometheus)
└── Dockerfile
```

---

## Key Features

| Component | Details |
|---|---|
| **One-Click** | `python run.py` — installs deps, trains, starts everything |
| **Models** | XGBoost + DLinear + PatchTST + PPO/DQN ensemble (all trained locally) |
| **Candle Patterns** | 11 patterns (doji, engulfing, hammer, morning/evening star…) + composite score |
| **Regime Detection** | Trending/ranging/volatile/quiet — adjusts signal strength automatically |
| **Technical Agent** | Dedicated TA rule engine (EMA, RSI, MACD, BB, Stochastic, patterns, regime) |
| **Correlation Agent** | Prevents over-exposure on correlated pairs; live correlation matrix |
| **LLM Orchestrator** | Weighted voting across all agents; Ollama/Llama 3 if running, rule-based fallback |
| **Sentiment** | FinBERT NLP on RSS, Reddit, Twitter (zero API keys needed) |
| **Risk** | ATR-based sizing, circuit breaker, Monte Carlo VaR, survival mode |
| **XAI** | SHAP + LIME explanations with natural language narratives |
| **Self-Healing** | Automatic model retraining when performance degrades |
| **Execution** | MetaTrader5 live + PaperBroker fallback + MQL5 EA bridge |
| **Memory** | ChromaDB vector store for similar-trade recall |
| **Concurrent** | All pairs processed in parallel (ThreadPoolExecutor) |
| **Scheduling** | Weekly retraining, genetic optimization, hourly weight updates |

---

## Configuration

Copy `.env.example` → `.env` and fill in credentials (all optional — system works without any):

```bash
cp .env.example .env
```

Key settings in `config/settings.yaml`:

```yaml
execution:
  mode: paper          # "paper" or "live"
  mt5_login: 0         # MT5 account number (live mode only)

risk:
  max_risk_per_trade: 0.01    # 1% per trade
  circuit_breaker_dd_pct: 0.10

pairs:
  majors: [EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD]
```

---

## Testing

```bash
pytest tests/test_core.py -v
```

All 23 tests run fully offline — no external APIs, no trained models, no MT5 required.

---

## Docker (full stack with Grafana + Prometheus)

```bash
docker-compose up
```

---

## Disclaimer

This software is for educational and research purposes. Forex trading involves substantial risk of loss. Past performance is not indicative of future results. Use at your own risk. Always paper-trade for several months before using real capital.

