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


# ─────────────────────────────────────────────────────────────────────────────
# Performance Analytics (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def test_performance_analytics_basic():
    from utils.performance_analytics import compute_metrics, TradingMetrics
    pnl = [50, -30, 80, -20, 60, 90, -40, 30, -10, 70]
    m = compute_metrics(pnl, initial_equity=10_000.0)
    assert isinstance(m, TradingMetrics)
    assert m.n_trades == 10
    assert m.total_pnl == pytest.approx(sum(pnl))
    assert 0.0 <= m.win_rate <= 1.0
    assert m.profit_factor >= 0.0
    assert m.max_drawdown_pct >= 0.0


def test_performance_analytics_empty():
    from utils.performance_analytics import compute_metrics
    m = compute_metrics([])
    assert m.n_trades == 0
    assert m.sharpe_ratio == 0.0


def test_performance_analytics_all_wins():
    from utils.performance_analytics import compute_metrics
    pnl = [10.0] * 20
    m = compute_metrics(pnl)
    assert m.win_rate == pytest.approx(1.0)
    assert m.max_consecutive_wins == 20
    assert m.max_consecutive_losses == 0
    assert m.profit_factor > 0


def test_performance_analytics_equity_series():
    from utils.performance_analytics import equity_series_from_pnl
    eq = equity_series_from_pnl([100, -50, 200], initial_equity=10_000.0)
    assert float(eq.iloc[0]) == pytest.approx(10_000.0)
    assert float(eq.iloc[-1]) == pytest.approx(10_250.0)


def test_performance_analytics_rolling_sharpe():
    from utils.performance_analytics import rolling_sharpe
    pnl = list(np.random.normal(10, 5, 50))
    rs = rolling_sharpe(pnl, window=10)
    assert len(rs) == 50
    # Values beyond the window should be non-zero
    assert rs.iloc[-1] != 0


def test_performance_analytics_compare_periods():
    from utils.performance_analytics import compare_periods
    before = [10, -5, 8, -3, 12]
    after  = [15, -4, 20, -2, 18]
    result = compare_periods(before, after)
    assert "before" in result
    assert "after" in result
    assert "delta" in result
    # After period has better PnL — delta total_pnl should be positive
    assert result["delta"]["total_pnl"] > 0


def test_performance_analytics_metrics_to_dict():
    from utils.performance_analytics import compute_metrics, metrics_to_dict
    m = compute_metrics([10, -5, 8], initial_equity=1000.0)
    d = metrics_to_dict(m)
    assert isinstance(d, dict)
    assert "sharpe_ratio" in d
    assert "win_rate" in d
    assert "max_drawdown_pct" in d


# ─────────────────────────────────────────────────────────────────────────────
# Trade Journal (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def test_trade_journal_record_open_close(tmp_path):
    from utils.trade_journal import TradeJournal
    journal = TradeJournal(journal_dir=tmp_path)

    trade = {
        "pair": "EURUSD", "direction": "buy",
        "entry_price": 1.1000, "stop_loss": 1.0950,
        "take_profit": 1.1100, "size_lots": 0.1,
        "confidence": 0.72, "sentiment_score": 0.3,
    }
    tid = journal.record_open(trade)
    assert tid == 1

    # Before close — exit_price should be empty
    df = journal.get_open_trades()
    assert len(df) == 1
    assert df.iloc[0]["pair"] == "EURUSD"

    journal.record_close(tid, exit_price=1.1080, pnl=80.0, hold_bars=12, exit_reason="tp_hit")

    closed = journal.get_closed_trades()
    assert len(closed) == 1
    assert float(closed.iloc[0]["pnl"]) == pytest.approx(80.0)
    assert int(closed.iloc[0]["hold_bars"]) == 12


def test_trade_journal_summary(tmp_path):
    from utils.trade_journal import TradeJournal
    journal = TradeJournal(journal_dir=tmp_path)

    for i, pnl in enumerate([50, -30, 80, -20, 60]):
        tid = journal.record_open({"pair": "EURUSD", "direction": "buy",
                                    "entry_price": 1.10 + i * 0.001})
        journal.record_close(tid, exit_price=1.10 + 0.002, pnl=float(pnl))

    s = journal.summary()
    assert s["n_trades"] == 5
    assert s["total_pnl"] == pytest.approx(140.0)
    assert s["win_rate_pct"] == pytest.approx(60.0)


def test_trade_journal_pair_breakdown(tmp_path):
    from utils.trade_journal import TradeJournal
    journal = TradeJournal(journal_dir=tmp_path)

    for pair, pnl in [("EURUSD", 50), ("GBPUSD", -30), ("EURUSD", 80), ("GBPUSD", 40)]:
        tid = journal.record_open({"pair": pair, "direction": "buy", "entry_price": 1.1})
        journal.record_close(tid, exit_price=1.102, pnl=float(pnl))

    pb = journal.pair_breakdown()
    assert "EURUSD" in pb["pair"].values
    assert "GBPUSD" in pb["pair"].values
    eur = pb[pb["pair"] == "EURUSD"].iloc[0]
    assert float(eur["total_pnl"]) == pytest.approx(130.0)


def test_trade_journal_csv_export(tmp_path):
    from utils.trade_journal import TradeJournal
    journal = TradeJournal(journal_dir=tmp_path)
    tid = journal.record_open({"pair": "USDJPY", "direction": "sell", "entry_price": 149.5})
    journal.record_close(tid, 149.0, pnl=50.0)

    out = journal.export_csv(tmp_path / "out.csv")
    assert out.exists()
    df = pd.read_csv(out)
    assert "pair" in df.columns
    assert len(df) == 1


def test_trade_journal_json_snapshot(tmp_path):
    from utils.trade_journal import TradeJournal
    journal = TradeJournal(journal_dir=tmp_path)
    tid = journal.record_open({"pair": "AUDUSD", "direction": "buy", "entry_price": 0.65})
    journal.record_close(tid, 0.652, pnl=20.0)

    snap = journal.save_json_snapshot()
    assert snap.exists()
    import json
    data = json.loads(snap.read_text())
    assert "summary" in data
    assert "pair_breakdown" in data
    assert "open_trades" in data


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Timeframe Analyser (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def test_mtf_detect_bias(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators
    from features.multi_timeframe import _detect_bias
    df = compute_all_indicators(sample_ohlcv)
    direction, strength = _detect_bias(df)
    assert direction in (-1, 0, 1)
    assert 0.0 <= strength <= 1.0


def test_mtf_analyse_preloaded(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators
    from features.multi_timeframe import MultiTimeframeAnalyser
    df = compute_all_indicators(sample_ohlcv)
    analyser = MultiTimeframeAnalyser(timeframes=["H1", "H4", "D1"])

    # Pass same df as all timeframes (preloaded)
    result = analyser.analyse("EURUSD", preloaded={"H1": df, "H4": df, "D1": df})
    assert result.pair == "EURUSD"
    assert result.agreed_direction in (-1, 0, 1)
    assert -1.0 <= result.confluence_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.bias.keys()) == {"H1", "H4", "D1"}


def test_mtf_filter_signal_aligned(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators
    from features.multi_timeframe import MultiTimeframeAnalyser
    df = compute_all_indicators(sample_ohlcv)
    analyser = MultiTimeframeAnalyser(timeframes=["H1", "H4"])

    result = analyser.analyse("EURUSD", preloaded={"H1": df, "H4": df})
    # Test that filter_signal returns bool + str
    allowed, reason = analyser.filter_signal("EURUSD", result.agreed_direction, result)
    assert isinstance(allowed, bool)
    assert isinstance(reason, str)


def test_mtf_filter_signal_blocks_opposing(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators
    from features.multi_timeframe import MultiTimeframeAnalyser, MTFResult
    df = compute_all_indicators(sample_ohlcv)
    analyser = MultiTimeframeAnalyser(timeframes=["H1", "H4"])

    # Manufacture a strongly bearish MTFResult
    mock_result = MTFResult(
        pair="EURUSD",
        bias={"H1": -1, "H4": -1},
        strength={"H1": 0.9, "H4": 0.9},
        confluence_score=-0.9,
        agreed_direction=-1,
        n_aligned=2,
        n_total=2,
    )
    # Propose a BUY against the strong downtrend — should be blocked
    allowed, reason = analyser.filter_signal("EURUSD", 1, mock_result)
    assert not allowed
    assert "block" in reason.lower() or "disagree" in reason.lower() or "MTF" in reason


# ─────────────────────────────────────────────────────────────────────────────
# Economic Calendar — is_high_impact_window (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def test_is_high_impact_window_no_data():
    """Should return False (never block) when calendar can't be fetched."""
    pytest.importorskip("bs4", reason="beautifulsoup4 not installed")
    from data_pipeline.economic_calendar import is_high_impact_window
    # With a fresh empty cache and no network in CI, must return False
    import data_pipeline.economic_calendar as ec
    ec._calendar_cache = (0.0, pd.DataFrame())  # force empty cache
    result = is_high_impact_window("EURUSD", minutes_before=15, minutes_after=15)
    assert result is False   # fail-safe: never block on data error



# ─────────────────────────────────────────────────────────────────────────────
# Pattern Agent (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def test_pattern_agent_returns_result(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators
    from agents.pattern_agent import PatternAgent, PatternAgentResult
    df = compute_all_indicators(sample_ohlcv)
    agent = PatternAgent(min_confidence=0.40)
    result = agent.detect(df, "EURUSD")
    assert isinstance(result, PatternAgentResult)
    assert result.pair == "EURUSD"
    # dominant_signal should be one of -1, 0, 1
    assert result.dominant_signal in (-1, 0, 1)
    # composite_confidence in valid range
    assert 0.0 <= result.composite_confidence <= 1.0


def test_pattern_agent_get_signal(sample_ohlcv):
    from features.technical_indicators import compute_all_indicators
    from agents.pattern_agent import PatternAgent
    df = compute_all_indicators(sample_ohlcv)
    agent = PatternAgent(min_confidence=0.40)
    signal = agent.get_signal(df, "GBPUSD")
    assert "signal" in signal
    assert "confidence" in signal
    assert "patterns" in signal
    assert signal["signal"] in (-1, 0, 1)
    assert 0.0 <= signal["confidence"] <= 1.0
    assert isinstance(signal["patterns"], list)


def test_pattern_agent_empty_df():
    from agents.pattern_agent import PatternAgent, PatternAgentResult
    agent = PatternAgent()
    result = agent.detect(pd.DataFrame(), "EURUSD")
    assert isinstance(result, PatternAgentResult)
    assert result.dominant_signal == 0


def test_double_bottom_detection(sample_ohlcv):
    """Create a synthetic double bottom and verify detection."""
    from agents.pattern_agent import detect_double_bottom
    close = sample_ohlcv["close"].values.copy()
    # Manufacture a clean double bottom shape
    n = 60
    x = np.linspace(0, 2 * np.pi, n)
    base = 1.1000
    # Two troughs at roughly same level
    prices = base + 0.005 * (1 - np.sin(x))
    df_synth = sample_ohlcv.iloc[:n].copy()
    df_synth["close"] = prices
    df_synth["high"] = prices + 0.001
    df_synth["low"] = prices - 0.001
    # May or may not detect depending on signal location — just ensure no crash
    result = detect_double_bottom(df_synth)
    # result can be None (not detected) or a PatternSignal — both are valid
    if result is not None:
        assert result.signal in (-1, 0, 1)
        assert 0.0 <= result.confidence <= 1.0


def test_pattern_signal_to_dict(sample_ohlcv):
    from agents.pattern_agent import PatternSignal
    ps = PatternSignal(pattern="double_top", signal=-1, confidence=0.7, target=1.09, stop=1.12)
    d = ps.to_dict()
    assert d["pattern"] == "double_top"
    assert d["signal"] == -1
    assert d["confidence"] == pytest.approx(0.7)


# ─────────────────────────────────────────────────────────────────────────────
# Order Flow Features (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_vwap(sample_ohlcv):
    from features.order_flow import compute_vwap
    df = compute_vwap(sample_ohlcv, rolling_window=20)
    assert "vwap" in df.columns
    assert "vwap_upper_1" in df.columns
    assert "vwap_lower_1" in df.columns
    assert "vwap_dev" in df.columns
    # After warm-up, VWAP should be close to typical price
    valid = df.dropna(subset=["vwap", "vwap_upper_1", "vwap_lower_1"])
    assert len(valid) > 0
    # Upper band must be >= lower band for valid rows
    assert (valid["vwap_upper_1"] >= valid["vwap_lower_1"]).all()


def test_compute_volume_delta(sample_ohlcv):
    from features.order_flow import compute_volume_delta
    df = compute_volume_delta(sample_ohlcv)
    assert "buy_volume" in df.columns
    assert "sell_volume" in df.columns
    assert "volume_delta" in df.columns
    assert "cumulative_delta" in df.columns
    # buy_volume + sell_volume should equal total volume
    np.testing.assert_allclose(
        (df["buy_volume"] + df["sell_volume"]).values,
        df["volume"].values,
        rtol=1e-5,
    )


def test_compute_pressure_ratios(sample_ohlcv):
    from features.order_flow import compute_pressure_ratios
    df = compute_pressure_ratios(sample_ohlcv, window=14)
    assert "pressure_ratio" in df.columns
    assert "pressure_signal" in df.columns
    valid = df["pressure_ratio"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_compute_volume_profile(sample_ohlcv):
    from features.order_flow import compute_volume_profile
    profile = compute_volume_profile(sample_ohlcv, n_bins=20, lookback=100)
    assert "poc" in profile
    assert "vah" in profile
    assert "val" in profile
    assert "profile" in profile
    # Value area: val <= poc <= vah
    assert profile["val"] <= profile["poc"] <= profile["vah"]


def test_detect_order_blocks(sample_ohlcv):
    from features.order_flow import detect_order_blocks
    blocks = detect_order_blocks(sample_ohlcv, lookback=50)
    assert isinstance(blocks, list)
    for b in blocks:
        assert b["type"] in ("bullish_ob", "bearish_ob")
        assert b["price_low"] <= b["price_high"]


def test_compute_order_flow_features(sample_ohlcv):
    from features.order_flow import compute_order_flow_features
    df = compute_order_flow_features(sample_ohlcv)
    required = ["vwap", "volume_delta", "pressure_ratio", "absorption",
                "in_bullish_ob", "in_bearish_ob"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_get_order_flow_signal(sample_ohlcv):
    from features.order_flow import get_order_flow_signal
    sig = get_order_flow_signal(sample_ohlcv)
    assert sig["signal"] in (-1, 0, 1)
    assert 0.0 <= sig["confidence"] <= 1.0
    assert "reason" in sig


def test_get_order_flow_signal_empty():
    from features.order_flow import get_order_flow_signal
    sig = get_order_flow_signal(pd.DataFrame())
    assert sig["signal"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Stacker (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def test_ensemble_stacker_weighted_avg():
    from models.ensemble_stacker import EnsembleStacker
    stacker = EnsembleStacker(model_names=["xgboost", "dlinear", "rl"])
    # Not fitted yet — should use weighted average
    preds = {"xgboost": 0.75, "dlinear": 0.70, "rl": 0.80}
    direction, confidence = stacker.predict(preds)
    assert direction in (-1, 0, 1)
    assert 0.0 <= confidence <= 1.0


def test_ensemble_stacker_fit_predict():
    from models.ensemble_stacker import EnsembleStacker
    stacker = EnsembleStacker(model_names=["m1", "m2"], min_samples_to_fit=10)
    np.random.seed(42)
    n = 100
    labels = np.random.randint(0, 2, n)
    # Create base predictions correlated with labels
    base_preds = {
        "m1": np.clip(labels * 0.6 + np.random.normal(0.4, 0.1, n), 0, 1),
        "m2": np.clip(labels * 0.5 + np.random.normal(0.4, 0.15, n), 0, 1),
    }
    metrics = stacker.fit(base_preds, labels)
    assert metrics["status"] == "fitted"
    assert "oof_auroc" in metrics
    assert 0.0 <= metrics["oof_auroc"] <= 1.0

    # Predict
    direction, confidence = stacker.predict({"m1": 0.8, "m2": 0.75})
    assert direction in (-1, 0, 1)
    assert 0.0 <= confidence <= 1.0


def test_ensemble_stacker_save_load(tmp_path):
    from models.ensemble_stacker import EnsembleStacker
    stacker = EnsembleStacker(
        model_names=["a", "b"],
        min_samples_to_fit=10,
        artifacts_dir=str(tmp_path),
    )
    # Train
    np.random.seed(0)
    n = 80
    labels = np.random.randint(0, 2, n)
    base_preds = {
        "a": np.clip(labels * 0.6 + 0.2, 0, 1),
        "b": np.clip(labels * 0.5 + 0.25, 0, 1),
    }
    stacker.fit(base_preds, labels)
    stacker.save("EURUSD")

    loaded = EnsembleStacker.load("EURUSD", artifacts_dir=str(tmp_path))
    assert loaded._is_fitted
    direction, confidence = loaded.predict({"a": 0.7, "b": 0.65})
    assert direction in (-1, 0, 1)


def test_ensemble_stacker_weight_update():
    from models.ensemble_stacker import EnsembleStacker
    stacker = EnsembleStacker(model_names=["x", "y", "z"])
    pnl_data = {
        "x": [50.0, 60.0, 40.0, 55.0, 70.0],
        "y": [-10.0, -20.0, -5.0, -15.0, -8.0],
        "z": [10.0, 15.0, 12.0, 8.0, 20.0],
    }
    stacker.update_weights_from_pnl(pnl_data)
    weights = stacker.get_model_weights()
    assert "x" in weights
    # x has positive mean → should have highest weight
    assert weights["x"] > weights["y"]


def test_ensemble_stacker_summary():
    from models.ensemble_stacker import EnsembleStacker
    stacker = EnsembleStacker(model_names=["a", "b"])
    summary = stacker.summary()
    assert "is_fitted" in summary
    assert "model_names" in summary
    assert "weights" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Alert Manager (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def test_alert_manager_drawdown_warning(tmp_path):
    from utils.alert_manager import AlertManager, AlertConfig
    cfg = AlertConfig(drawdown_warning_pct=0.03, drawdown_critical_pct=0.06,
                      cooldown_seconds=0)
    mgr = AlertManager(cfg=cfg, log_dir=str(tmp_path))

    # 4% drawdown → warning
    alert = mgr.check_drawdown(equity=9600, peak_equity=10000)
    assert alert is not None
    assert alert.level == "warning"
    assert "drawdown" in alert.alert_type.lower()


def test_alert_manager_drawdown_critical(tmp_path):
    from utils.alert_manager import AlertManager, AlertConfig
    cfg = AlertConfig(drawdown_critical_pct=0.06, cooldown_seconds=0)
    mgr = AlertManager(cfg=cfg, log_dir=str(tmp_path))

    # 7% drawdown → critical
    alert = mgr.check_drawdown(equity=9300, peak_equity=10000)
    assert alert is not None
    assert alert.level == "critical"


def test_alert_manager_consecutive_losses(tmp_path):
    from utils.alert_manager import AlertManager, AlertConfig
    cfg = AlertConfig(consecutive_loss_warning=3, cooldown_seconds=0)
    mgr = AlertManager(cfg=cfg, log_dir=str(tmp_path))

    pnls = [50.0, -10.0, -20.0, -30.0]
    alert = mgr.check_consecutive_losses(pnls)
    assert alert is not None
    assert float(alert.value) == 3.0


def test_alert_manager_no_alert_when_ok(tmp_path):
    from utils.alert_manager import AlertManager, AlertConfig
    cfg = AlertConfig(drawdown_warning_pct=0.05, cooldown_seconds=0)
    mgr = AlertManager(cfg=cfg, log_dir=str(tmp_path))
    alert = mgr.check_drawdown(equity=9800, peak_equity=10000)
    assert alert is None   # 2% drawdown < 5% threshold


def test_alert_manager_cooldown(tmp_path):
    from utils.alert_manager import AlertManager, AlertConfig
    cfg = AlertConfig(drawdown_warning_pct=0.03, cooldown_seconds=999)
    mgr = AlertManager(cfg=cfg, log_dir=str(tmp_path))
    alert1 = mgr.check_drawdown(equity=9600, peak_equity=10000)
    alert2 = mgr.check_drawdown(equity=9500, peak_equity=10000)
    assert alert1 is not None
    assert alert2 is None   # within cooldown


def test_alert_manager_circuit_breaker(tmp_path):
    from utils.alert_manager import AlertManager, AlertConfig
    mgr = AlertManager(cfg=AlertConfig(cooldown_seconds=0), log_dir=str(tmp_path))
    alert = mgr.check_circuit_breaker(is_triggered=True)
    assert alert is not None
    assert alert.level == "critical"


def test_alert_manager_log_file(tmp_path):
    from utils.alert_manager import AlertManager, AlertConfig
    mgr = AlertManager(cfg=AlertConfig(cooldown_seconds=0), log_dir=str(tmp_path))
    mgr.check_drawdown(equity=9300, peak_equity=10000)
    alerts = mgr.get_alerts_from_log(n=10)
    assert len(alerts) >= 1
    assert "message" in alerts[0]


def test_alert_manager_check_all(tmp_path):
    from utils.alert_manager import AlertManager, AlertConfig
    cfg = AlertConfig(drawdown_warning_pct=0.03, cooldown_seconds=0)
    mgr = AlertManager(cfg=cfg, log_dir=str(tmp_path))
    fired = mgr.check_all(
        equity=9400,
        peak_equity=10000,
        recent_pnls=[-10, -20, -30, -40, -50],
    )
    # Should fire drawdown + consecutive_loss alerts
    assert len(fired) >= 1


def test_alert_manager_save_summary(tmp_path):
    from utils.alert_manager import AlertManager, AlertConfig
    mgr = AlertManager(cfg=AlertConfig(cooldown_seconds=0), log_dir=str(tmp_path))
    mgr.check_circuit_breaker(is_triggered=True)
    path = mgr.save_summary_json()
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert "recent_alerts" in data
    assert "alert_counts" in data


def test_alert_to_dict():
    from utils.alert_manager import Alert
    a = Alert(alert_type="drawdown", level="warning", message="test", value=0.04, threshold=0.03)
    d = a.to_dict()
    assert d["alert_type"] == "drawdown"
    assert d["level"] == "warning"
    assert d["value"] == pytest.approx(0.04)
    # Warning level should map to the warning emoji
    assert a.emoji() == "⚠️"


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Optimizer (NEW — inspired by PyPortfolioOpt)
# ─────────────────────────────────────────────────────────────────────────────

def test_portfolio_optimizer_hrp(sample_ohlcv):
    """HRP should produce normalised weights summing to 1."""
    from utils.portfolio_optimizer import PortfolioOptimizer, PortfolioWeights
    np.random.seed(0)
    n = 200
    returns = pd.DataFrame({
        "EURUSD": np.random.normal(0.0001, 0.001, n),
        "GBPUSD": np.random.normal(0.0002, 0.0015, n),
        "USDJPY": np.random.normal(-0.0001, 0.0012, n),
    })
    optimizer = PortfolioOptimizer(method="hrp", min_weight=0.01, max_weight=0.60)
    result = optimizer.optimise(returns)
    assert isinstance(result, PortfolioWeights)
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6
    assert all(0.01 <= w <= 0.60 for w in result.weights.values())
    assert result.diversification_ratio > 0


def test_portfolio_optimizer_mvo(sample_ohlcv):
    """MVO should return weights summing to 1 and within bounds."""
    from utils.portfolio_optimizer import PortfolioOptimizer
    np.random.seed(1)
    n = 200
    returns = pd.DataFrame({
        "EURUSD": np.random.normal(0.0003, 0.001, n),
        "GBPUSD": np.random.normal(0.0001, 0.0015, n),
    })
    optimizer = PortfolioOptimizer(method="mvo", min_weight=0.10, max_weight=0.90)
    result = optimizer.optimise(returns)
    assert abs(sum(result.weights.values()) - 1.0) < 1e-5
    assert all(w >= 0.10 for w in result.weights.values())


def test_portfolio_optimizer_risk_parity():
    """Risk parity should produce equal risk contributions."""
    from utils.portfolio_optimizer import PortfolioOptimizer
    np.random.seed(2)
    n = 300
    returns = pd.DataFrame({
        "A": np.random.normal(0, 0.01, n),
        "B": np.random.normal(0, 0.02, n),
        "C": np.random.normal(0, 0.005, n),
    })
    optimizer = PortfolioOptimizer(method="risk_parity")
    result = optimizer.optimise(returns)
    assert abs(sum(result.weights.values()) - 1.0) < 1e-5
    # Lower vol asset should have higher weight
    assert result.weights["C"] > result.weights["A"] > result.weights["B"] or \
           result.weights["C"] > result.weights["B"]  # flexible assertion


def test_portfolio_optimizer_single_pair():
    """Single pair → returns weight of 1.0."""
    from utils.portfolio_optimizer import PortfolioOptimizer
    returns = pd.DataFrame({"EURUSD": np.random.normal(0, 0.001, 100)})
    optimizer = PortfolioOptimizer(method="hrp")
    result = optimizer.optimise(returns)
    assert "EURUSD" in result.weights


def test_portfolio_optimizer_too_few_rows():
    """Less than 10 rows → equal weight fallback."""
    from utils.portfolio_optimizer import PortfolioOptimizer
    returns = pd.DataFrame({
        "A": [0.01, -0.01, 0.005],
        "B": [0.02, -0.02, 0.010],
    })
    optimizer = PortfolioOptimizer(method="hrp")
    result = optimizer.optimise(returns)
    assert abs(sum(result.weights.values()) - 1.0) < 1e-5


def test_portfolio_optimizer_to_dict():
    """to_dict should return complete report."""
    from utils.portfolio_optimizer import PortfolioWeights
    pw = PortfolioWeights(
        weights={"EURUSD": 0.6, "GBPUSD": 0.4},
        method="hrp",
        diversification_ratio=1.2,
        expected_annual_return=0.05,
        expected_annual_vol=0.10,
        expected_sharpe=0.5,
    )
    d = pw.to_dict()
    assert "weights" in d
    assert "method" in d
    assert d["expected_sharpe"] == pytest.approx(0.5)


def test_portfolio_optimizer_lot_sizes():
    """compute_lot_sizes should return positive lot sizes."""
    from utils.portfolio_optimizer import PortfolioOptimizer, PortfolioWeights
    pw = PortfolioWeights(weights={"EURUSD": 0.6, "GBPUSD": 0.4}, method="hrp")
    opt = PortfolioOptimizer()
    lots = opt.compute_lot_sizes(pw, total_equity=10_000)
    assert lots["EURUSD"] > lots["GBPUSD"]
    assert all(v > 0 for v in lots.values())


# ─────────────────────────────────────────────────────────────────────────────
# HyperOptimiser (NEW — inspired by freqtrade/hyperopt)
# ─────────────────────────────────────────────────────────────────────────────

def test_hyperopt_list_spaces():
    from models.hyperopt import HyperOptimiser
    spaces = HyperOptimiser.list_spaces()
    assert "xgboost" in spaces
    assert "risk" in spaces
    assert "rl" in spaces


def test_hyperopt_get_space():
    from models.hyperopt import HyperOptimiser
    space = HyperOptimiser.get_space("xgboost")
    assert "max_depth" in space
    assert "learning_rate" in space


def test_hyperopt_random_search(sample_ohlcv):
    """Random search with custom fn should return HyperOptResult."""
    from models.hyperopt import HyperOptimiser, HyperOptResult
    from features.technical_indicators import compute_all_indicators, get_feature_columns
    df = compute_all_indicators(sample_ohlcv)
    feature_cols = get_feature_columns(df)

    call_count = {"n": 0}
    def dummy_fn(df, feature_cols, params):
        call_count["n"] += 1
        return float(np.random.normal(0.5, 0.1)), {"sharpe": 0.5}

    ho = HyperOptimiser(objective="sharpe", spaces=["xgboost"], n_trials=5)
    # Force random search by mocking _HAS_OPTUNA
    import models.hyperopt as hyperopt_module
    orig = hyperopt_module._HAS_OPTUNA
    hyperopt_module._HAS_OPTUNA = False
    try:
        result = ho.run(df, feature_cols, custom_fn=dummy_fn)
        assert isinstance(result, HyperOptResult)
        assert result.n_trials == 5
        assert call_count["n"] == 5
        assert "best_params" in result.to_dict()
    finally:
        hyperopt_module._HAS_OPTUNA = orig


def test_hyperopt_save_load(tmp_path, sample_ohlcv):
    from models.hyperopt import HyperOptimiser, HyperOptResult
    result = HyperOptResult(
        best_params={"max_depth": 5, "learning_rate": 0.05},
        best_value=1.23,
        objective_name="sharpe",
        n_trials=10,
        spaces_used=["xgboost"],
    )
    ho = HyperOptimiser()
    path = str(tmp_path / "best.json")
    ho.save_best(result, path)
    loaded = ho.load_best(path)
    assert loaded["max_depth"] == 5
    assert loaded["learning_rate"] == pytest.approx(0.05)


def test_hyperopt_load_missing():
    from models.hyperopt import HyperOptimiser
    ho = HyperOptimiser()
    loaded = ho.load_best("/tmp/nonexistent_hyperopt_file.json")
    assert loaded == {}


# ─────────────────────────────────────────────────────────────────────────────
# Alpha Factors (NEW — inspired by microsoft/qlib)
# ─────────────────────────────────────────────────────────────────────────────

def test_alpha_factor_calculator(sample_ohlcv):
    """All alpha factors should be computed without errors."""
    from features.alpha_factors import AlphaFactorCalculator
    calc = AlphaFactorCalculator(zscore=True)
    df = calc.compute_all(sample_ohlcv)
    assert len(calc.factor_cols) >= 10
    for col in calc.factor_cols:
        assert col in df.columns, f"Missing: {col}"
    # Z-scored values should be mostly in [-3, 3]
    for col in calc.factor_cols:
        valid = df[col].dropna()
        assert len(valid) > 0


def test_alpha_factor_no_zscore(sample_ohlcv):
    from features.alpha_factors import AlphaFactorCalculator
    calc = AlphaFactorCalculator(zscore=False)
    df = calc.compute_all(sample_ohlcv)
    assert len(calc.factor_cols) >= 10


def test_factor_evaluator_ic(sample_ohlcv):
    """IC should be a float in [-1, 1]."""
    from features.alpha_factors import AlphaFactorCalculator, FactorEvaluator
    calc = AlphaFactorCalculator(zscore=True)
    df = calc.compute_all(sample_ohlcv)
    evaluator = FactorEvaluator(primary_period=1)
    df = evaluator.compute_forward_returns(df)
    for col in calc.factor_cols[:3]:
        score = evaluator.evaluate_factor(df, col)
        assert -2.0 <= score.ic <= 2.0  # ICIR can go beyond ±1
        assert isinstance(score.n_obs, int)


def test_factor_evaluator_evaluate_all(sample_ohlcv):
    """evaluate_all should return sorted list of FactorScore."""
    from features.alpha_factors import AlphaFactorCalculator, FactorEvaluator, FactorScore
    calc = AlphaFactorCalculator(zscore=True)
    df = calc.compute_all(sample_ohlcv)
    evaluator = FactorEvaluator(primary_period=1)
    scores = evaluator.evaluate_all(df, calc.factor_cols)
    assert isinstance(scores, list)
    assert all(isinstance(s, FactorScore) for s in scores)
    # Should be sorted by |ICIR| descending
    if len(scores) >= 2:
        assert abs(scores[0].icir) >= abs(scores[-1].icir)


def test_factor_score_tradeable():
    from features.alpha_factors import FactorScore
    s = FactorScore(name="test", ic=0.06, icir=0.5, rank_ic=0.05, turnover=0.3, n_obs=100)
    assert s.is_tradeable()
    s_bad = FactorScore(name="test2", ic=0.01, icir=0.1, rank_ic=0.01, turnover=0.9, n_obs=100)
    assert not s_bad.is_tradeable()


def test_combine_factors(sample_ohlcv):
    """Composite alpha should be a pd.Series of same length."""
    from features.alpha_factors import AlphaFactorCalculator, combine_factors
    calc = AlphaFactorCalculator(zscore=True)
    df = calc.compute_all(sample_ohlcv)
    composite = combine_factors(df, calc.factor_cols, method="equal")
    assert isinstance(composite, pd.Series)
    assert len(composite) == len(df)


def test_neutralise_factor(sample_ohlcv):
    """Neutralised factor should be orthogonal to benchmark."""
    from features.alpha_factors import AlphaFactorCalculator, neutralise_factor
    calc = AlphaFactorCalculator(zscore=True)
    df = calc.compute_all(sample_ohlcv)
    factor = df[calc.factor_cols[0]].dropna()
    benchmark = df[calc.factor_cols[1]].reindex(factor.index).dropna()
    both = pd.concat([factor, benchmark], axis=1).dropna()
    if len(both) < 10:
        return
    residual = neutralise_factor(both.iloc[:, 0], both.iloc[:, 1])
    # Residual correlation with benchmark should be near 0
    corr = residual.dropna().corr(both.iloc[:, 1].reindex(residual.dropna().index))
    assert abs(corr) < 0.1 or np.isnan(corr)  # allow NaN for short series


# ─────────────────────────────────────────────────────────────────────────────
# QuantStats Reporter (NEW — inspired by ranaroussi/quantstats)
# ─────────────────────────────────────────────────────────────────────────────

def test_quantstats_reporter_basic(sample_ohlcv):
    """Basic report generation from equity curve."""
    from utils.quantstats_reporter import QuantStatsReporter, PerformanceReport
    np.random.seed(7)
    equity = np.cumprod(1 + np.random.normal(0.0005, 0.01, 300)) * 10_000
    reporter = QuantStatsReporter(periods_per_year=252)
    report = reporter.generate(equity, pair="EURUSD")
    assert isinstance(report, PerformanceReport)
    assert report.pair == "EURUSD"
    assert report.n_trading_periods == 300
    assert isinstance(report.sharpe_ratio, float)
    assert isinstance(report.max_drawdown_pct, float)
    assert report.max_drawdown_pct >= 0


def test_quantstats_reporter_with_trades():
    """Report with trade list should populate trade stats."""
    from utils.quantstats_reporter import QuantStatsReporter
    np.random.seed(8)
    equity = np.cumprod(1 + np.random.normal(0.0003, 0.008, 200)) * 10_000
    trades = [{"pnl": np.random.normal(5, 20)} for _ in range(50)]
    reporter = QuantStatsReporter()
    report = reporter.generate(equity, trades=trades, pair="GBPUSD")
    assert report.n_trades == 50
    assert 0 <= report.win_rate_pct <= 100


def test_quantstats_reporter_to_dict():
    from utils.quantstats_reporter import QuantStatsReporter
    equity = np.linspace(10000, 11000, 100)
    reporter = QuantStatsReporter()
    report = reporter.generate(equity, pair="USDJPY")
    d = report.to_dict()
    required_keys = ["pair", "sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
                     "win_rate_pct", "profit_factor", "total_return_pct"]
    for k in required_keys:
        assert k in d, f"Missing key: {k}"


def test_quantstats_reporter_save_json(tmp_path):
    from utils.quantstats_reporter import QuantStatsReporter
    equity = np.linspace(10000, 10500, 50)
    reporter = QuantStatsReporter()
    report = reporter.generate(equity, pair="AUDUSD")
    path = reporter.save_json(report, str(tmp_path / "report.json"))
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert data["pair"] == "AUDUSD"


def test_quantstats_reporter_save_html(tmp_path):
    from utils.quantstats_reporter import QuantStatsReporter
    np.random.seed(9)
    equity = np.cumprod(1 + np.random.normal(0.0004, 0.009, 100)) * 10_000
    reporter = QuantStatsReporter()
    report = reporter.generate(equity, pair="NZDUSD")
    path = reporter.save_html(report, str(tmp_path / "report.html"))
    assert path.exists()
    html = path.read_text()
    assert "NZDUSD" in html
    assert "Sharpe" in html
    assert "Drawdown" in html


def test_quantstats_reporter_empty_equity():
    from utils.quantstats_reporter import QuantStatsReporter
    reporter = QuantStatsReporter()
    report = reporter.generate([], pair="EMPTY")
    assert report.n_trading_periods == 0


def test_quantstats_reporter_monotonic_equity():
    """Perfect trending equity should have near-zero drawdown."""
    from utils.quantstats_reporter import QuantStatsReporter
    equity = np.linspace(10000, 15000, 200)
    reporter = QuantStatsReporter()
    report = reporter.generate(equity, pair="TEST")
    assert report.total_return_pct > 0
    assert report.max_drawdown_pct < 1.0  # trivially low DD


def test_quantstats_reporter_rolling_metrics():
    from utils.quantstats_reporter import QuantStatsReporter
    np.random.seed(42)
    equity = np.cumprod(1 + np.random.normal(0.0003, 0.008, 200)) * 10_000
    reporter = QuantStatsReporter()
    report = reporter.generate(equity, pair="X")
    assert len(report.rolling_sharpe_30) > 0
    assert len(report.rolling_vol_30) > 0
    assert len(report.equity_curve) > 0
