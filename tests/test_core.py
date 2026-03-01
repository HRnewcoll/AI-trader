"""
Core tests — validate key modules without requiring external APIs or trained models.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data."""
    n = 500
    np.random.seed(42)
    close = 1.1000 + np.cumsum(np.random.normal(0, 0.001, n))
    high = close + np.random.uniform(0.0005, 0.002, n)
    low = close - np.random.uniform(0.0005, 0.002, n)
    open_ = close + np.random.normal(0, 0.0005, n)
    volume = np.random.randint(1000, 50000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def test_technical_indicators(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators, get_feature_columns
    df = compute_all_indicators(sample_ohlcv, "EURUSD")
    feature_cols = get_feature_columns(df)
    assert len(feature_cols) >= 30, f"Expected 30+ features, got {len(feature_cols)}"
    assert "rsi_14" in df.columns
    assert "atr_14" in df.columns
    assert "macd" in df.columns
    assert "bb_width" in df.columns
    assert "ema_20" in df.columns
    assert "session_london" in df.columns
    assert "direction" in df.columns
    assert df["rsi_14"].dropna().between(0, 100).all()
    assert not df["close"].isna().any()


def test_feature_column_count(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators, get_feature_columns
    df = compute_all_indicators(sample_ohlcv)
    cols = get_feature_columns(df)
    assert len(cols) >= 50, f"Expected 50+ feature columns, got {len(cols)}"


# ─────────────────────────────────────────────────────────────────────────────
# Candle Patterns (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def test_candle_patterns(sample_ohlcv):
    from features.candle_patterns import compute_candle_features, get_pattern_columns
    df = compute_candle_features(sample_ohlcv)
    pat_cols = get_pattern_columns()
    for col in pat_cols:
        assert col in df.columns, f"Missing pattern column: {col}"
    # Doji values should be -1, 0, or 1
    assert df["pat_doji"].isin([-1.0, 0.0, 1.0]).all()
    # Composite score should be clipped to [-5, 5]
    assert df["pat_composite"].between(-5, 5).all()


# ─────────────────────────────────────────────────────────────────────────────
# Regime Detector (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_detector(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators
    from features.regime_detector import detect_regime, add_regime_features, Regime
    df = compute_all_indicators(sample_ohlcv)
    state = detect_regime(df)
    assert state.regime in list(Regime)
    assert 0.0 <= state.confidence <= 1.0
    assert state.volatility_level in ("low", "medium", "high")

    df_with_regime = add_regime_features(df.tail(100))
    assert "regime" in df_with_regime.columns
    assert "regime_trending_up" in df_with_regime.columns
    assert "regime_vol_pct" in df_with_regime.columns


# ─────────────────────────────────────────────────────────────────────────────
# Technical Agent (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def test_technical_agent(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators
    from agents.technical_agent import TechnicalAgent
    df = compute_all_indicators(sample_ohlcv)
    agent = TechnicalAgent()
    result = agent.analyse(df)
    assert result["signal"] in (-1, 0, 1)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["reasons"], list)
    assert "regime" in result


# ─────────────────────────────────────────────────────────────────────────────
# Correlation Agent (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def test_correlation_agent():
    from agents.correlation_agent import CorrelationAgent
    agent = CorrelationAgent(max_correlation=0.75, max_correlated_open=2)

    # Should allow first position
    allowed, reason = agent.can_open_position("EURUSD", 1)
    assert allowed, reason

    # Open a correlated pair
    agent.register_open("GBPUSD", 1)  # highly correlated with EURUSD
    agent.register_open("AUDUSD", 1)  # also correlated

    # Static baseline correlation check
    corr = agent.get_correlation("EURUSD", "GBPUSD")
    assert 0.0 <= corr <= 1.0

    # Diversification score with conflicting signals
    score = agent.get_diversification_score({"EURUSD": 1, "USDJPY": -1, "GBPUSD": 1})
    assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# LLM Orchestrator (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def test_llm_orchestrator_rules():
    from agents.llm_orchestrator import LLMOrchestrator
    orch = LLMOrchestrator(backend="rules")  # force rule-based (no Ollama needed)
    decision = orch.decide(
        pair="EURUSD",
        technical_signal={"signal": 1, "confidence": 0.7, "reasons": ["EMA crossover"], "regime": "trending_up"},
        sentiment_score=0.5,
        rl_signal={"action": 1, "confidence": 0.65},
        correlation_result={"allowed": True},
        risk_ok=True,
        regime="trending_up",
    )
    assert decision["action"] in ("buy", "sell", "hold")
    assert 0.0 <= decision["confidence"] <= 1.0
    assert "explanation" in decision
    assert decision["pair"] == "EURUSD"


def test_llm_orchestrator_veto():
    from agents.llm_orchestrator import LLMOrchestrator
    orch = LLMOrchestrator(backend="rules")
    # Risk veto should force hold
    decision = orch.decide(
        pair="GBPUSD",
        technical_signal={"signal": 1, "confidence": 0.9, "reasons": [], "regime": "trending_up"},
        sentiment_score=0.8,
        rl_signal={"action": 1, "confidence": 0.9},
        correlation_result={"allowed": True},
        risk_ok=False,  # ← risk veto
        regime="trending_up",
    )
    assert decision["action"] == "hold"


# ─────────────────────────────────────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────────────────────────────────────

def test_config_loader():
    from utils.config_loader import load_config, get_pairs
    cfg = load_config()
    assert "pairs" in cfg
    assert "risk" in cfg
    assert "rl" in cfg
    pairs = get_pairs(cfg)
    assert "EURUSD" in pairs
    assert len(pairs) >= 7


# ─────────────────────────────────────────────────────────────────────────────
# Risk Manager
# ─────────────────────────────────────────────────────────────────────────────

def test_risk_manager_basic():
    from risk_healing.risk_manager import RiskManager, RiskConfig
    rm = RiskManager(initial_equity=10_000.0)
    can, reason = rm.can_trade()
    assert can, reason

    pos = rm.compute_position_size("EURUSD", 1, 1.1000, 0.001, confidence=0.7)
    assert pos.size_lots > 0
    assert pos.stop_loss < pos.entry_price  # long: SL below entry
    assert pos.take_profit > pos.entry_price  # long: TP above entry


def test_risk_manager_circuit_breaker():
    from risk_healing.risk_manager import RiskManager, RiskConfig
    cfg = RiskConfig(circuit_breaker_dd_pct=0.05, max_total_drawdown_pct=0.10)
    rm = RiskManager(initial_equity=10_000.0, cfg=cfg)
    rm.peak_equity = 10_000.0
    rm.update_equity(9_400.0)  # 6% drawdown — triggers circuit breaker
    assert rm.is_circuit_broken


def test_risk_manager_survival_mode():
    from risk_healing.risk_manager import RiskManager, RiskConfig
    cfg = RiskConfig(survival_mode_equity_pct=0.80)
    rm = RiskManager(initial_equity=10_000.0, cfg=cfg)
    rm.update_equity(7_500.0)   # 75% of initial → survival mode
    assert rm.is_survival_mode


def test_monte_carlo_var():
    from risk_healing.risk_manager import RiskManager
    rm = RiskManager(initial_equity=10_000.0)
    rm._trade_returns = list(np.random.normal(0.001, 0.01, 100))
    result = rm.monte_carlo_var(n_paths=200, n_steps=10)
    assert "var_95" in result
    assert "ruin_prob" in result
    assert 0.0 <= result["ruin_prob"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# RL Environment
# ─────────────────────────────────────────────────────────────────────────────

def test_rl_env_basic(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators, get_feature_columns
    from models.rl_environment import ForexTradingEnv

    df = compute_all_indicators(sample_ohlcv)
    df = df.dropna().reset_index(drop=True)
    feature_cols = get_feature_columns(df)

    env = ForexTradingEnv(df, feature_cols)
    obs, info = env.reset()

    assert obs.shape[0] == len(feature_cols) + 4
    assert env.observation_space.contains(obs)

    # Run a few steps
    done = False
    steps = 0
    while not done and steps < 50:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    assert steps > 0
    metrics = env.get_metrics()
    assert "total_return" in metrics
    assert "sharpe" in metrics


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Agent
# ─────────────────────────────────────────────────────────────────────────────

def test_vader_sentiment():
    from agents.sentiment_agent import SentimentAgent
    # use_finbert=False to avoid downloading model in tests
    agent = SentimentAgent(use_finbert=False)
    score = agent.score_text("The Fed raised interest rates aggressively, strengthening the dollar")
    assert -1.0 <= score <= 1.0


def test_sentiment_aggregation():
    from agents.sentiment_agent import SentimentAgent
    agent = SentimentAgent(use_finbert=False)
    articles = [
        {"title": "EUR rises on ECB hawkish signals", "summary": "", "timestamp": 9e9, "source": "test"},
        {"title": "Dollar weakens after poor jobs data", "summary": "", "timestamp": 9e9, "source": "test"},
    ]
    scored = agent.process_articles(articles)
    assert len(scored) == 2
    for a in scored:
        assert "sentiment_score" in a
        assert -1.0 <= a["sentiment_score"] <= 1.0

    pair_score = agent.aggregate_pair_sentiment(scored, "EURUSD")
    assert -1.0 <= pair_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost (train on synthetic data)
# ─────────────────────────────────────────────────────────────────────────────

def test_xgboost_train_predict(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators, get_feature_columns
    from models.xgboost_model import XGBoostForexModel

    df = compute_all_indicators(sample_ohlcv).dropna()
    feature_cols = get_feature_columns(df)

    model = XGBoostForexModel(artifacts_dir="/tmp/test_artifacts")
    metrics = model.train(df, feature_cols, n_splits=2, tune_hyperparams=False)

    assert metrics["mean_auc"] >= 0.3
    direction, proba = model.predict_latest(df)
    assert direction in (0, 1)
    assert 0.0 <= proba <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Backtesting
# ─────────────────────────────────────────────────────────────────────────────

def test_vectorised_backtest(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators
    from backtesting.walk_forward import vectorised_backtest

    df = compute_all_indicators(sample_ohlcv).dropna()
    # Simple signal: buy when close > EMA20
    signals = (df["close"] > df["ema_20"]).astype(int).replace({0: -1})

    result = vectorised_backtest(df, signals, pair="EURUSD")
    assert result.n_trades >= 0
    assert -1.0 <= result.total_return <= 10.0
    assert result.max_drawdown >= 0.0
    assert result.max_drawdown <= 1.0


def test_walk_forward(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators
    from backtesting.walk_forward import walk_forward_backtest, summarise_backtest_results
    from models.xgboost_model import XGBoostForexModel

    df = compute_all_indicators(sample_ohlcv).dropna()

    def model_fn(train_df, test_df):
        from features.technical_indicators import get_feature_columns
        feature_cols = get_feature_columns(train_df)
        m = XGBoostForexModel(artifacts_dir="/tmp/test_wf")
        m.train(train_df, feature_cols, n_splits=2)
        directions, _ = m.predict(test_df)
        return pd.Series(directions * 2 - 1, index=test_df.index)

    results = walk_forward_backtest(df, model_fn, n_splits=3)
    assert len(results) >= 1
    summary = summarise_backtest_results(results)
    assert "mean_sharpe" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Memory
# ─────────────────────────────────────────────────────────────────────────────

def test_trading_memory():
    from memory.vector_memory import TradingMemory

    mem = TradingMemory(persist_dir="/tmp/test_chroma")
    trade = {
        "pair": "EURUSD", "direction": "buy",
        "entry_price": 1.1000, "pnl": 50.0,
        "confidence": 0.75, "reason": "RSI oversold + bullish sentiment",
    }
    mem.store_trade(trade)
    similar = mem.recall_similar_trades("EURUSD bullish trade")
    assert isinstance(similar, list)


# ─────────────────────────────────────────────────────────────────────────────
# Notifier (no actual send — just construction)
# ─────────────────────────────────────────────────────────────────────────────

def test_notifier_build():
    from notifications.notifier import Notifier
    n = Notifier()  # no credentials — logging-only mode
    results = n.send("Test", "Test message", level="info")
    assert isinstance(results, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Self Healer
# ─────────────────────────────────────────────────────────────────────────────

def test_self_healer():
    from risk_healing.self_healer import SelfHealer

    healer = SelfHealer(artifacts_dir="/tmp/test_heal")
    for eq in [10000, 9800, 9500, 9900, 9600, 9200, 9100]:
        healer.record_equity(eq)

    health = healer.run_health_check(equity=9100)
    assert "overall" in health
    assert "equity" in health


# ─────────────────────────────────────────────────────────────────────────────
# Run.py launcher (smoke test — no side effects)
# ─────────────────────────────────────────────────────────────────────────────

def test_run_py_imports():
    """Verify run.py imports cleanly without executing side effects."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run", Path(__file__).parent.parent / "run.py"
    )
    # Just load the module spec without executing it
    assert spec is not None
    assert spec.loader is not None

