"""
Central multi-agent orchestrator.
Coordinates: Technical, Sentiment, Correlation, RL agents + LLM Orchestrator.
Produces final trade signal as structured JSON with XAI explanation.
Sends signal to MT5 bridge + notifications.
Processes all pairs concurrently for speed.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_pipeline.market_data import fetch_ohlcv
from data_pipeline.news_scraper import fetch_rss_feeds
from data_pipeline.reddit_scraper import fetch_reddit_posts
from data_pipeline.twitter_scraper import fetch_tweets
from data_pipeline.economic_calendar import is_high_impact_window
from features.technical_indicators import compute_all_indicators, get_feature_columns
from features.candle_patterns import compute_candle_features
from features.regime_detector import detect_regime
from features.multi_timeframe import MultiTimeframeAnalyser
from features.order_flow import compute_order_flow_features, get_order_flow_signal
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from agents.correlation_agent import CorrelationAgent
from agents.llm_orchestrator import LLMOrchestrator
from agents.pattern_agent import PatternAgent
from risk_healing.risk_manager import RiskManager, RiskConfig
from risk_healing.self_healer import SelfHealer
from notifications.notifier import Notifier, build_notifier_from_cfg
from monitoring.metrics_server import MetricsServer
from memory.vector_memory import TradingMemory
from models.ensemble_stacker import EnsembleStacker
from utils.trade_journal import TradeJournal
from utils.alert_manager import AlertManager, AlertConfig
from xai.explainer import ForexExplainer

logger = logging.getLogger(__name__)

ACTION_MAP = {0: "hold", 1: "buy", 2: "sell", 3: "close"}


class ForexOrchestrator:
    """
    Master orchestrator — runs the full signal pipeline:
    Data → Features → Sentiment → Ensemble Prediction → Risk Check
    → XAI Explanation → Execute → Log → Notify → Self-Heal.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pairs = (
            cfg.get("pairs", {}).get("majors", ["EURUSD"])
            + cfg.get("pairs", {}).get("emerging", [])
        )
        self.timeframe = cfg.get("timeframes", {}).get("primary", "H4")
        self.mode = cfg.get("execution", {}).get("mode", "paper")

        # Components — loaded lazily to avoid import errors if deps missing
        self._models: dict[str, dict] = {}
        self._rl_agents: dict[str, object] = {}
        self._ensemble_weights: dict[str, dict] = {}
        self._feature_cols: list[str] = []
        self._df_cache: dict[str, pd.DataFrame] = {}
        self._sentiment_cache: list[dict] = []
        self._last_sentiment_fetch = 0.0
        self._trade_count = 0
        self._recent_pnls: list[float] = []

        # Build components
        self.sentiment_agent = SentimentAgent(
            finbert_model=cfg.get("sentiment", {}).get("finbert_model", "ProsusAI/finbert"),
            aggregate_window_hours=cfg.get("sentiment", {}).get("aggregate_window_hours", 4),
        )
        self.technical_agent = TechnicalAgent(
            rsi_oversold=cfg.get("technical_agent", {}).get("rsi_oversold", 30.0),
            rsi_overbought=cfg.get("technical_agent", {}).get("rsi_overbought", 70.0),
            adx_trend_threshold=cfg.get("technical_agent", {}).get("adx_trend_threshold", 25.0),
        )
        self.correlation_agent = CorrelationAgent(
            max_correlation=cfg.get("risk", {}).get("max_correlation", 0.75),
        )
        self.llm_orchestrator = LLMOrchestrator(
            backend=cfg.get("llm", {}).get("backend", "auto"),
            ollama_host=cfg.get("llm", {}).get("ollama_host", "http://localhost:11434"),
            sentiment_threshold=cfg.get("sentiment", {}).get("signal_threshold", 0.3),
        )
        self.risk_manager = RiskManager(
            initial_equity=10_000.0,
            cfg=RiskConfig(**{
                k: cfg.get("risk", {}).get(k, v)
                for k, v in RiskConfig.__dataclass_fields__.items()
                if k in cfg.get("risk", {})
            }) if cfg.get("risk") else None,
        )
        self.memory = TradingMemory(
            persist_dir=cfg.get("memory", {}).get("chroma_persist_dir", "artifacts/chromadb"),
        )
        self.journal = TradeJournal(
            journal_dir=cfg.get("memory", {}).get("journal_dir", "artifacts/trade_journal"),
        )
        self.mtf_analyser = MultiTimeframeAnalyser(
            timeframes=cfg.get("timeframes", {}).get("mtf", ["M15", "H1", "H4", "D1"]),
        )
        self.pattern_agent = PatternAgent(
            order=cfg.get("pattern_agent", {}).get("order", 5),
            tolerance=cfg.get("pattern_agent", {}).get("tolerance", 0.025),
            min_confidence=cfg.get("pattern_agent", {}).get("min_confidence", 0.50),
        )
        self.alert_manager = AlertManager(
            notifier=self.notifier if hasattr(self, "notifier") else None,
            cfg=AlertConfig(**cfg.get("alerts", {})) if cfg.get("alerts") else None,
        )
        self._stackers: dict[str, EnsembleStacker] = {}
        self.healer = SelfHealer(
            artifacts_dir=cfg.get("models", {}).get("models_dir", "artifacts/models"),
            watchdog_interval=cfg.get("monitoring", {}).get("watchdog_interval_seconds", 60),
        )
        self.notifier = build_notifier_from_cfg(cfg)
        self.metrics = MetricsServer(
            port=cfg.get("monitoring", {}).get("prometheus_port", 9090)
        )
        self.metrics.start()

        # Execution bridge
        self._init_broker()

        # Load pre-trained models
        self._load_models()

        logger.info("ForexOrchestrator ready. Mode: %s, Pairs: %s", self.mode, self.pairs)

    def _init_broker(self) -> None:
        from execution.mt5_bridge import MT5Bridge
        ex_cfg = self.cfg.get("execution", {})
        self.broker = MT5Bridge(
            login=int(ex_cfg.get("mt5_login", 0) or 0),
            password=ex_cfg.get("mt5_password", ""),
            server=ex_cfg.get("mt5_server", ""),
            mode=self.mode,
            initial_equity=10_000.0,
        )

    def _load_models(self) -> None:
        """Load all trained model artifacts."""
        models_dir = Path(self.cfg.get("models", {}).get("models_dir", "artifacts/models"))
        if not models_dir.exists():
            logger.info("No artifacts directory — run training pipeline first")
            return

        for pair in self.pairs:
            self._models[pair] = {}

            # XGBoost
            xgb_path = models_dir / f"xgboost_{pair}.pkl"
            if xgb_path.exists():
                try:
                    from models.xgboost_model import XGBoostForexModel
                    self._models[pair]["xgboost"] = XGBoostForexModel.load(xgb_path)
                    logger.info("Loaded XGBoost for %s", pair)
                except Exception as e:
                    logger.warning("XGBoost load failed %s: %s", pair, e)

            # DLinear
            dlinear_path = models_dir / f"dlinear_{pair}.pt"
            if dlinear_path.exists():
                try:
                    from models.transformer_model import TransformerTrainer
                    self._models[pair]["dlinear"] = TransformerTrainer.load(dlinear_path)
                    logger.info("Loaded DLinear for %s", pair)
                except Exception as e:
                    logger.warning("DLinear load failed %s: %s", pair, e)

            # PatchTST
            patchtst_path = models_dir / f"patchtst_{pair}.pt"
            if patchtst_path.exists():
                try:
                    from models.transformer_model import TransformerTrainer
                    self._models[pair]["patchtst"] = TransformerTrainer.load(patchtst_path)
                    logger.info("Loaded PatchTST for %s", pair)
                except Exception as e:
                    logger.warning("PatchTST load failed %s: %s", pair, e)

            # RL Agent
            try:
                from models.rl_agent import RLForexAgent
                rl_agent = RLForexAgent.load(
                    pair,
                    algorithms=self.cfg.get("rl", {}).get("algorithms", ["ppo", "dqn"]),
                    artifacts_dir=str(models_dir),
                )
                if rl_agent.is_trained:
                    self._rl_agents[pair] = rl_agent
                    logger.info("Loaded RL agent for %s", pair)
            except Exception as e:
                logger.debug("RL agent load %s: %s", pair, e)

            # Equal initial ensemble weights
            n_models = len(self._models[pair]) + (1 if pair in self._rl_agents else 0)
            w = 1.0 / max(n_models, 1)
            self._ensemble_weights[pair] = {name: w for name in self._models[pair]}
            if pair in self._rl_agents:
                self._ensemble_weights[pair]["rl"] = w

    def _fetch_sentiment(self) -> list[dict]:
        """Fetch + score news/reddit/tweets (cached for 1 hour)."""
        now = time.time()
        if now - self._last_sentiment_fetch < 3600 and self._sentiment_cache:
            return self._sentiment_cache

        articles = []
        # RSS news (no API key)
        try:
            articles.extend(fetch_rss_feeds())
        except Exception as e:
            logger.warning("RSS fetch error: %s", e)

        # Reddit (no API key)
        try:
            posts = fetch_reddit_posts()
            for p in posts:
                articles.append({"title": p.get("title", ""), "summary": p.get("text", ""),
                                  "timestamp": p.get("timestamp", now), "source": "reddit"})
        except Exception as e:
            logger.warning("Reddit fetch error: %s", e)

        # Tweets (nitter, no API key)
        try:
            tweets = fetch_tweets(bearer_token=self.cfg.get("execution", {}).get("twitter_bearer", ""))
            for tw in tweets:
                articles.append({"title": tw.get("text", ""), "summary": "",
                                  "timestamp": tw.get("timestamp", now), "source": "twitter"})
        except Exception as e:
            logger.warning("Twitter fetch error: %s", e)

        self._sentiment_cache = self.sentiment_agent.process_articles(articles)
        self._last_sentiment_fetch = now
        logger.info("Sentiment cache: %d items", len(self._sentiment_cache))
        return self._sentiment_cache

    def _get_ensemble_signal(self, pair: str, df: pd.DataFrame) -> tuple[int, float]:
        """
        Run all models + RL for pair, return (direction, confidence).
        direction: 1=buy, -1=sell, 0=hold.
        """
        votes: dict[int, float] = {1: 0.0, -1: 0.0, 0: 0.0}
        models = self._models.get(pair, {})
        weights = self._ensemble_weights.get(pair, {})

        for model_name, model in models.items():
            weight = weights.get(model_name, 1.0)
            try:
                direction, confidence = model.predict_latest(df)
                direction_mapped = 1 if direction == 1 else -1
                votes[direction_mapped] += weight * float(confidence)
                self.metrics.update_model_confidence(model_name, pair, float(confidence))
            except Exception as e:
                logger.debug("Model %s predict error: %s", model_name, e)

        # RL agent
        rl_agent = self._rl_agents.get(pair)
        if rl_agent:
            try:
                feature_cols = get_feature_columns(df)
                obs_size = self.cfg.get("rl", {}).get("observation_size", 50)
                obs_row = df[feature_cols].iloc[-1].fillna(0).values[:obs_size]
                obs = np.array(obs_row, dtype=np.float32)
                obs = np.pad(obs, (0, max(0, obs_size - len(obs))))
                action, confidence = rl_agent.predict(obs)
                if action == 1:
                    votes[1] += weights.get("rl", 1.0) * confidence
                elif action == 2:
                    votes[-1] += weights.get("rl", 1.0) * confidence
                self.metrics.update_model_confidence("rl", pair, confidence)
            except Exception as e:
                logger.debug("RL predict error: %s", e)

        # Majority vote
        best_dir = max(votes, key=votes.__getitem__)
        total = sum(votes.values()) + 1e-8
        confidence = votes[best_dir] / total
        return best_dir, float(confidence)

    def process_pair(self, pair: str) -> dict | None:
        """
        Full signal pipeline for one pair.
        Returns trade dict if signal generated, else None.
        """
        t0 = time.time()

        # 1. Check risk
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.info("Risk block for %s: %s", pair, reason)
            return None

        # 1b. Economic calendar — skip trading 15 min before/after high-impact events
        try:
            if is_high_impact_window(pair, minutes_before=15, minutes_after=15):
                logger.info("%s: High-impact news window — skipping trade", pair)
                return None
        except Exception as e:
            logger.debug("Economic calendar check error: %s", e)

        # 2. Fetch + compute features
        try:
            df = fetch_ohlcv(pair, self.timeframe, bars=500, cfg=self.cfg)
            if df.empty or len(df) < 100:
                return None
            df = compute_all_indicators(df, pair)
            df = compute_candle_features(df)      # candle patterns
            df = compute_order_flow_features(df)  # VWAP, delta, pressure
            self._df_cache[pair] = df
            feature_cols = get_feature_columns(df)
            if not self._feature_cols:
                self._feature_cols = feature_cols
        except Exception as e:
            logger.error("Data fetch error %s: %s", pair, e)
            return None

        # 3. Sentiment
        articles = self._fetch_sentiment()
        sentiment_score = self.sentiment_agent.aggregate_pair_sentiment(articles, pair)
        sentiment_signal = self.sentiment_agent.get_signal(
            sentiment_score,
            self.cfg.get("sentiment", {}).get("signal_threshold", 0.3),
        )
        self.metrics.update_sentiment(pair, sentiment_score)

        # 4. Ensemble prediction
        if not self._models.get(pair) and pair not in self._rl_agents:
            logger.debug("No trained models for %s — run training first", pair)
            return None

        direction, confidence = self._get_ensemble_signal(pair, df)

        # 4b. Technical agent signal
        ta_signal = self.technical_agent.analyse(df)

        # 4c. Correlation check
        corr_allowed, corr_reason = self.correlation_agent.can_open_position(
            pair, direction
        )
        corr_result = {"allowed": corr_allowed, "reason": corr_reason}

        # 4d. Regime
        regime_state = detect_regime(df)

        # 4e. Multi-timeframe confluence
        mtf_result = self.mtf_analyser.analyse(pair, cfg=self.cfg)
        mtf_allowed, mtf_reason = self.mtf_analyser.filter_signal(
            pair, direction, mtf_result
        )
        if not mtf_allowed:
            logger.info("%s: MTF block — %s", pair, mtf_reason)
            return None
        logger.debug("%s MTF: %s", pair, mtf_reason)

        # 4f. Pattern agent
        pattern_signal = self.pattern_agent.get_signal(df, pair)
        if pattern_signal["signal"] != 0 and pattern_signal["signal"] != direction:
            logger.info(
                "%s: Chart pattern opposes signal (pattern=%+d, dir=%+d patterns=%s) — skipping",
                pair, pattern_signal["signal"], direction, pattern_signal["patterns"],
            )
            return None

        # 4g. Order flow signal
        of_signal = get_order_flow_signal(df)
        if of_signal["signal"] != 0 and of_signal["signal"] != direction:
            logger.debug("%s: Order flow disagrees (%s) — weakening confidence", pair, of_signal["reason"])
            confidence *= 0.85  # reduce but don't block

        # 4h. Ensemble stacker — combine all model predictions
        stacker = self._stackers.get(pair)
        if stacker and self._models.get(pair):
            try:
                base_preds: dict[str, float] = {}
                for model_name, model in self._models[pair].items():
                    _, prob = model.predict_latest(df)
                    base_preds[model_name] = float(prob)
                stacked_dir, stacked_conf = stacker.predict(base_preds)
                if stacked_conf > confidence * 1.05:
                    direction = stacked_dir
                    confidence = stacked_conf
            except Exception as e:
                logger.debug("Stacker predict error %s: %s", pair, e)

        # 4e. LLM orchestrator — final reasoning
        rl_signal = {"action": 1 if direction == 1 else (2 if direction == -1 else 0),
                     "confidence": confidence}
        llm_decision = self.llm_orchestrator.decide(
            pair=pair,
            technical_signal=ta_signal,
            sentiment_score=sentiment_score,
            rl_signal=rl_signal,
            correlation_result=corr_result,
            risk_ok=True,
            regime=regime_state.regime.value,
        )
        # Let LLM override direction if confidence higher
        if llm_decision["confidence"] > confidence * 1.05:
            direction = llm_decision["direction"]
            confidence = llm_decision["confidence"]

        # 5. Confidence filter
        min_conf = self.cfg.get("models", {}).get("min_confidence", 0.65)
        if confidence < min_conf:
            logger.debug("%s: confidence %.2f < threshold %.2f", pair, confidence, min_conf)
            return None

        # 6. Sentiment alignment filter (only trade when TA + sentiment agree)
        if sentiment_signal != 0 and sentiment_signal != direction:
            logger.info("%s: Sentiment disagrees (sent=%d, dir=%d) — skipping", pair, sentiment_signal, direction)
            return None

        # 7. Risk sizing
        latest = df.iloc[-1]
        atr = float(latest.get("atr_14", 0.001))
        entry_price = float(latest["close"])
        pos_risk = self.risk_manager.compute_position_size(
            pair, direction, entry_price, atr, confidence=confidence
        )

        # 8. Monte Carlo pre-trade check
        var_stats = self.risk_manager.monte_carlo_var(n_paths=500)
        ruin_prob = var_stats.get("ruin_prob", 0.0)
        if ruin_prob > 0.1:
            logger.warning("%s: ruin probability %.1f%% too high — skipping", pair, ruin_prob * 100)
            return None

        # 9. Generate XAI explanation
        explanation_text = ""
        models = self._models.get(pair, {})
        if "xgboost" in models:
            try:
                model = models["xgboost"]
                X = model.scaler.transform(
                    df[model.feature_names].fillna(0).tail(1).values
                )
                explainer = ForexExplainer(
                    model.model, model.feature_names, "xgboost"
                )
                expl = explainer.explain_prediction(X, direction, confidence, pair)
                explanation_text = expl.get("narrative", "")
            except Exception as e:
                logger.debug("XAI error: %s", e)

        # 10. Build trade dict
        trade = {
            "pair": pair,
            "direction": ACTION_MAP.get(1 if direction == 1 else 2, "hold"),
            "entry_price": round(entry_price, 5),
            "stop_loss": round(pos_risk.stop_loss, 5),
            "take_profit": round(pos_risk.take_profit, 5),
            "size_lots": pos_risk.size_lots,
            "confidence": round(confidence, 3),
            "sentiment_score": round(sentiment_score, 3),
            "atr": round(atr, 5),
            "explanation": explanation_text,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        # 11. Execute
        try:
            order = self.broker.place_order(
                pair=pair,
                direction=trade["direction"],
                size_lots=pos_risk.size_lots,
                price=entry_price,
                stop_loss=pos_risk.stop_loss,
                take_profit=pos_risk.take_profit,
            )
            trade["order_id"] = order.order_id
            trade["fill_price"] = order.price
        except Exception as e:
            logger.error("Order placement error: %s", e)
            return None

        # 12. Memory + journal + metrics + notify
        self.memory.store_trade(trade)
        self.journal.record_open(trade)
        self.metrics.update_trade(pair, trade["direction"], 0.0)
        self.notifier.send_trade(trade, explanation_text)
        self._trade_count += 1

        latency_ms = (time.time() - t0) * 1000
        self.metrics.record_latency(latency_ms)
        logger.info("Trade %s %s @ %.5f conf=%.2f | %.0fms",
                    pair, trade["direction"], entry_price, confidence, latency_ms)

        # Store signal file for MT5 EA
        self._write_mt5_signal(trade)

        return trade

    def _write_mt5_signal(self, trade: dict) -> None:
        """Write signal JSON for MT5 EA to pick up."""
        signal = {
            "pair": trade["pair"],
            "direction": 1 if trade["direction"] == "buy" else -1,
            "size": trade["size_lots"],
            "stop_loss": trade["stop_loss"],
            "take_profit": trade["take_profit"],
            "confidence": trade["confidence"],
        }
        try:
            signal_path = Path("forex_ai_signal.json")
            signal_path.write_text(json.dumps(signal))
        except Exception:
            pass

    def run_cycle(self) -> list[dict]:
        """Run one complete signal cycle for all pairs — concurrently."""
        trades = []
        health = self.healer.run_health_check(equity=self.broker.get_account_equity())
        self.metrics.set_circuit_breaker(self.risk_manager.is_circuit_broken)
        self.metrics.set_survival_mode(self.risk_manager.is_survival_mode)

        # Update correlation matrix from cached price data
        if self._df_cache:
            self.correlation_agent.update_correlation_matrix(self._df_cache)

        max_workers = min(len(self.pairs), 4)   # cap threads to avoid API rate limits
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_pair = {
                pool.submit(self.process_pair, pair): pair
                for pair in self.pairs
            }
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    trade = future.result()
                    if trade:
                        trades.append(trade)
                except Exception as e:
                    logger.error("Pair %s cycle error: %s", pair, e)

        # Persist MTF status and journal snapshot for dashboard
        self._save_mtf_status()
        self._save_journal_snapshot()

        # Run alert checks
        self._run_alert_checks()

        return trades

    def _run_alert_checks(self) -> None:
        """Run all alert checks after each cycle and persist summary."""
        try:
            equity = self.broker.get_account_equity()
            self.alert_manager.check_all(
                equity=equity,
                peak_equity=self.risk_manager.peak_equity,
                recent_pnls=self._recent_pnls,
                daily_pnl=sum(self._recent_pnls[-20:]) if self._recent_pnls else 0.0,
                is_circuit_broken=self.risk_manager.is_circuit_broken,
                is_survival_mode=self.risk_manager.is_survival_mode,
            )
            self.alert_manager.save_summary_json()
        except Exception as e:
            logger.debug("Alert checks error: %s", e)

    def _save_mtf_status(self) -> None:
        """Write per-pair MTF results to logs/mtf_status.json for the dashboard."""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            results = {}
            for pair in self.pairs:
                try:
                    r = self.mtf_analyser.analyse(pair, cfg=self.cfg)
                    results[pair] = {
                        "confluence_score": r.confluence_score,
                        "agreed_direction": r.agreed_direction,
                        "bias": r.bias,
                        "strength": r.strength,
                        "n_aligned": r.n_aligned,
                        "n_total": r.n_total,
                    }
                except Exception:
                    pass
            (log_dir / "mtf_status.json").write_text(json.dumps(results))
        except Exception as e:
            logger.debug("MTF status save error: %s", e)

    def _save_journal_snapshot(self) -> None:
        """Save a JSON snapshot of the trade journal for the dashboard."""
        try:
            self.journal.save_json_snapshot()
        except Exception as e:
            logger.debug("Journal snapshot error: %s", e)

    def update_ensemble_weights(self) -> None:
        """Update model weights based on recent PnL (called hourly)."""
        for pair in self.pairs:
            rl_agent = self._rl_agents.get(pair)
            if rl_agent and self._recent_pnls:
                recent = {"ppo": np.mean(self._recent_pnls), "dqn": np.mean(self._recent_pnls)}
                rl_agent.update_weights(recent)
