# ForexAI — Self-Healing Autonomous Forex Trader

A production-grade, multi-agent algorithmic trading system for FX markets. Combines ensemble ML models, RL agents, NLP sentiment analysis, explainable AI, and self-healing risk management.

---

## Architecture

```
main_orchestrator.py          ← Entry point: starts all services, runs signal loop
│
├── agents/
│   ├── orchestrator.py       ← Multi-agent coordinator (Technical + Sentiment + RL + Risk)
│   └── sentiment_agent.py    ← FinBERT / VADER NLP sentiment scorer
│
├── data_pipeline/
│   ├── market_data.py        ← OHLCV fetcher (CCXT / yfinance / MetaApi)
│   ├── news_scraper.py       ← RSS feed ingestion
│   ├── reddit_scraper.py     ← Reddit post scraper (no API key)
│   ├── twitter_scraper.py    ← Twitter / Nitter scraper
│   ├── economic_calendar.py  ← Macro event calendar
│   └── feature_store.py      ← Parquet-backed feature cache
│
├── features/
│   └── technical_indicators.py  ← 60+ TA features (RSI, MACD, ATR, Bollinger, sessions…)
│
├── models/
│   ├── xgboost_model.py      ← XGBoost classifier with Optuna HPO + SHAP
│   ├── transformer_model.py  ← DLinear & PatchTST time-series transformers
│   ├── rl_agent.py           ← PPO + DQN ensemble RL agent (Stable-Baselines3)
│   ├── rl_environment.py     ← Custom Gymnasium FX trading environment
│   ├── genetic_optimizer.py  ← NSGA-II multi-objective genetic optimizer
│   └── training_pipeline.py  ← Orchestrated full training pipeline
│
├── execution/
│   ├── mt5_bridge.py         ← MetaTrader5 bridge + PaperBroker fallback
│   └── mt5_ea.mq5            ← MQL5 Expert Advisor (reads Python signals via JSON file)
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
│   └── test_core.py          ← Pytest test suite (no external APIs required)
│
├── rust_core/                ← Rust extensions for high-performance indicators
├── config/settings.yaml      ← Main configuration file
├── docker-compose.yml        ← Full stack (trader + Grafana + Prometheus)
└── Dockerfile
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
Copy `.env.example` to `.env` and fill in any optional credentials (MT5, Telegram, etc.):
```bash
cp .env.example .env
```

Edit `config/settings.yaml` to set your trading pairs, risk limits, and execution mode.

### 3. Run (paper trading — no real money)
```bash
python main_orchestrator.py
```

The system will automatically train models on first launch if no artifacts exist.

### 4. Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

### 5. Docker (full stack with Grafana + Prometheus)
```bash
docker-compose up
```

---

## Key Features

| Component | Details |
|---|---|
| **Models** | XGBoost + DLinear + PatchTST + PPO/DQN ensemble |
| **Sentiment** | FinBERT NLP on RSS, Reddit, Twitter (no API keys needed) |
| **Risk** | ATR-based sizing, circuit breaker, Monte Carlo VaR, survival mode |
| **XAI** | SHAP + LIME explanations with natural language narratives |
| **Self-Healing** | Automatic model retraining when performance degrades |
| **Execution** | MetaTrader5 live + PaperBroker fallback + MQL5 EA bridge |
| **Memory** | ChromaDB vector store for similar-trade recall |
| **Scheduling** | Weekly retraining, genetic optimization, hourly weight updates |

---

## Testing
```bash
pytest tests/test_core.py -v
```

Tests run fully offline — no external APIs or trained models required.

---

## Configuration

Key settings in `config/settings.yaml`:

```yaml
execution:
  mode: paper          # "paper" or "live"
  mt5_login: 0         # MT5 account (live mode only)

risk:
  max_risk_per_trade: 0.01   # 1% per trade
  circuit_breaker_dd_pct: 0.10

pairs:
  majors: [EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD]
```

---

## Disclaimer

This software is for educational and research purposes. Forex trading involves substantial risk of loss. Past performance is not indicative of future results. Use at your own risk.
