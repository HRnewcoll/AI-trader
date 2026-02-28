"""
Vector DB memory — ChromaDB with in-memory fallback.
Stores: trade history, regime embeddings, past contexts.
MemGPT-style persistent long-term memory.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False
    logger.warning("chromadb not installed — using in-memory vector store")

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False
    logger.warning("sentence-transformers not installed — using hash-based embeddings")


def _simple_embedding(text: str, dim: int = 64) -> list[float]:
    """Simple deterministic embedding fallback (no ML model needed)."""
    np.random.seed(hash(text) % (2**32))
    return np.random.normal(0, 1, dim).tolist()


class TradingMemory:
    """
    Persistent trading memory using ChromaDB vector store.
    Stores trade records, regime states, and context embeddings.
    Never forgets — all history persisted to disk.
    """

    def __init__(
        self,
        persist_dir: str = "artifacts/chromadb",
        embedding_model: str = "all-MiniLM-L6-v2",
        context_window: int = 100,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.context_window = context_window
        self._encoder = None
        self._client = None
        self._trade_collection = None
        self._regime_collection = None
        self._context_collection = None

        # Load sentence encoder
        if _HAS_SBERT:
            try:
                self._encoder = SentenceTransformer(embedding_model)
                logger.info("Sentence encoder loaded: %s", embedding_model)
            except Exception as e:
                logger.warning("Sentence encoder failed: %s", e)

        # Init ChromaDB
        if _HAS_CHROMA:
            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=Settings(anonymized_telemetry=False),
                )
                self._trade_collection = self._client.get_or_create_collection("trades")
                self._regime_collection = self._client.get_or_create_collection("regimes")
                self._context_collection = self._client.get_or_create_collection("contexts")
                logger.info("ChromaDB initialised at %s", self.persist_dir)
            except Exception as e:
                logger.warning("ChromaDB init failed: %s — using fallback", e)
                self._client = None

        # In-memory fallback
        self._mem_trades: list[dict] = []
        self._mem_regimes: list[dict] = []
        self._mem_contexts: list[dict] = []

    def _embed(self, text: str) -> list[float]:
        if self._encoder:
            try:
                return self._encoder.encode(text).tolist()
            except Exception:
                pass
        return _simple_embedding(text)

    def store_trade(self, trade: dict) -> None:
        """Store a completed trade record."""
        trade_id = f"trade_{int(time.time() * 1000)}_{trade.get('pair', 'UNK')}"
        trade["stored_at"] = datetime.now(tz=timezone.utc).isoformat()

        text = (
            f"Trade {trade.get('pair','')} {trade.get('direction','')} "
            f"at {trade.get('entry_price', 0):.5f} "
            f"PnL: {trade.get('pnl', 0):.2f} "
            f"Reason: {trade.get('reason', '')}"
        )

        if self._trade_collection:
            try:
                self._trade_collection.upsert(
                    ids=[trade_id],
                    documents=[text],
                    embeddings=[self._embed(text)],
                    metadatas=[{k: str(v) for k, v in trade.items()}],
                )
                return
            except Exception as e:
                logger.debug("ChromaDB store error: %s", e)

        self._mem_trades.append({"id": trade_id, "text": text, "data": trade})

    def recall_similar_trades(self, context: str, n: int = 5) -> list[dict]:
        """Retrieve trades similar to current context."""
        embedding = self._embed(context)

        if self._trade_collection:
            try:
                results = self._trade_collection.query(
                    query_embeddings=[embedding],
                    n_results=min(n, self._trade_collection.count() or 1),
                )
                trades = []
                for meta in (results.get("metadatas") or [[]])[0]:
                    trades.append({k: v for k, v in meta.items()})
                return trades
            except Exception as e:
                logger.debug("ChromaDB query error: %s", e)

        # In-memory fallback: return recent trades
        return [t["data"] for t in self._mem_trades[-n:]]

    def store_regime(self, regime: dict) -> None:
        """Store a market regime snapshot."""
        regime_id = f"regime_{int(time.time())}"
        text = (
            f"Regime at {regime.get('timestamp','')} "
            f"volatility: {regime.get('volatility_level','')} "
            f"trend: {regime.get('trend_direction','')} "
            f"session: {regime.get('session', '')}"
        )
        if self._regime_collection:
            try:
                self._regime_collection.upsert(
                    ids=[regime_id],
                    documents=[text],
                    embeddings=[self._embed(text)],
                    metadatas=[{k: str(v) for k, v in regime.items()}],
                )
                return
            except Exception as e:
                logger.debug("ChromaDB regime error: %s", e)
        self._mem_regimes.append({"id": regime_id, "data": regime})

    def get_recent_context(self, pair: str, n: int | None = None) -> list[dict]:
        """Get recent trade history for a pair (for LLM context)."""
        n = n or self.context_window
        if self._trade_collection:
            try:
                count = self._trade_collection.count()
                if count > 0:
                    results = self._trade_collection.query(
                        query_embeddings=[self._embed(f"trade {pair}")],
                        n_results=min(n, count),
                        where={"pair": pair},
                    )
                    return (results.get("metadatas") or [[]])[0]
            except Exception:
                pass
        return [t["data"] for t in self._mem_trades if t["data"].get("pair") == pair][-n:]

    def get_stats(self) -> dict:
        trade_count = (
            self._trade_collection.count() if self._trade_collection else len(self._mem_trades)
        )
        regime_count = (
            self._regime_collection.count() if self._regime_collection else len(self._mem_regimes)
        )
        return {
            "trade_count": trade_count,
            "regime_count": regime_count,
            "backend": "chromadb" if self._client else "memory",
        }
