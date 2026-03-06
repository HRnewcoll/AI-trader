"""
Trade journal — persistent CSV-backed record of every trade.
Provides statistics, export to Excel/CSV, and per-pair summaries.
Uses performance_analytics for metrics.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_JOURNAL_COLUMNS = [
    "id", "timestamp", "pair", "direction",
    "entry_price", "exit_price", "stop_loss", "take_profit",
    "size_lots", "pnl", "pnl_pips", "confidence",
    "sentiment_score", "regime", "hold_bars", "exit_reason",
    "explanation",
]

_DEFAULT_DIR = Path("artifacts/trade_journal")


class TradeJournal:
    """
    Persistent CSV-backed trade journal.
    Records every filled order and provides aggregated statistics.
    """

    def __init__(self, journal_dir: str | Path = _DEFAULT_DIR):
        self.dir = Path(journal_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.dir / "trades.csv"
        self._ensure_csv()
        logger.info("TradeJournal initialised at %s", self._csv_path)

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_csv(self) -> None:
        if not self._csv_path.exists():
            with self._csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_JOURNAL_COLUMNS)
                writer.writeheader()

    def _next_id(self) -> int:
        df = self._load_df()
        return int(df["id"].max() + 1) if len(df) > 0 else 1

    def _load_df(self) -> pd.DataFrame:
        try:
            # keep_default_na=False prevents empty strings from becoming NaN
            # which would cause pandas to infer float64 dtype for string columns
            df = pd.read_csv(self._csv_path, dtype=str, keep_default_na=False)
            if "id" in df.columns:
                df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)
            return df
        except Exception:
            return pd.DataFrame(columns=_JOURNAL_COLUMNS)

    # ─────────────────────────────────────────────────────────────────────
    # Write
    # ─────────────────────────────────────────────────────────────────────

    def record_open(self, trade: dict) -> int:
        """
        Record a newly opened trade.
        Returns the assigned trade id.
        """
        trade_id = self._next_id()
        row = {col: "" for col in _JOURNAL_COLUMNS}
        row["id"] = trade_id
        row["timestamp"] = trade.get("timestamp", datetime.now(tz=timezone.utc).isoformat())
        row["pair"] = trade.get("pair", "")
        row["direction"] = trade.get("direction", "")
        row["entry_price"] = trade.get("entry_price", 0.0)
        row["exit_price"] = ""
        row["stop_loss"] = trade.get("stop_loss", 0.0)
        row["take_profit"] = trade.get("take_profit", 0.0)
        row["size_lots"] = trade.get("size_lots", 0.0)
        row["pnl"] = ""
        row["pnl_pips"] = ""
        row["confidence"] = trade.get("confidence", 0.0)
        row["sentiment_score"] = trade.get("sentiment_score", 0.0)
        row["regime"] = trade.get("regime", "")
        row["hold_bars"] = ""
        row["exit_reason"] = ""
        row["explanation"] = trade.get("explanation", "")[:200]  # truncate

        with self._csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_JOURNAL_COLUMNS)
            writer.writerow(row)

        return trade_id

    def record_close(
        self,
        trade_id: int,
        exit_price: float,
        pnl: float,
        hold_bars: int = 0,
        exit_reason: str = "",
    ) -> None:
        """Update an open trade row with its close data."""
        df = self._load_df()
        mask = df["id"] == trade_id
        if not mask.any():
            logger.warning("TradeJournal: trade id %d not found", trade_id)
            return

        entry = float(df.loc[mask, "entry_price"].iloc[0]) or 0.0
        direction = str(df.loc[mask, "direction"].iloc[0])
        pip = 0.0001 if "JPY" not in str(df.loc[mask, "pair"].iloc[0]) else 0.01
        pnl_pips = ((exit_price - entry) / pip) * (1 if direction == "buy" else -1)

        df.loc[mask, "exit_price"] = str(exit_price)
        df.loc[mask, "pnl"] = str(round(pnl, 4))
        df.loc[mask, "pnl_pips"] = str(round(pnl_pips, 1))
        df.loc[mask, "hold_bars"] = str(hold_bars)
        df.loc[mask, "exit_reason"] = str(exit_reason)
        df.to_csv(self._csv_path, index=False)

    def log_paper_trade(self, trade: dict) -> int:
        """
        Shortcut: record open + immediate close for paper trades where
        entry/exit are both known at signal time (backtested, simulated).
        """
        trade_id = self.record_open(trade)
        if "exit_price" in trade and "pnl" in trade:
            self.record_close(
                trade_id,
                exit_price=float(trade["exit_price"]),
                pnl=float(trade["pnl"]),
                hold_bars=int(trade.get("hold_bars", 0)),
                exit_reason=str(trade.get("exit_reason", "signal")),
            )
        return trade_id

    # ─────────────────────────────────────────────────────────────────────
    # Read / statistics
    # ─────────────────────────────────────────────────────────────────────

    def get_all_trades(self) -> pd.DataFrame:
        return self._load_df()

    def get_closed_trades(self) -> pd.DataFrame:
        df = self._load_df()
        return df[df["exit_price"].astype(str).str.strip() != ""].copy()

    def get_open_trades(self) -> pd.DataFrame:
        df = self._load_df()
        return df[df["exit_price"].astype(str).str.strip() == ""].copy()

    def get_pair_trades(self, pair: str) -> pd.DataFrame:
        df = self._load_df()
        return df[df["pair"] == pair].copy()

    def summary(self, pair: str | None = None) -> dict:
        """
        Return a summary dict with key performance statistics.
        If pair is given, filters to that pair only.
        """
        df = self.get_closed_trades()
        if pair:
            df = df[df["pair"] == pair]
        if df.empty:
            return {"n_trades": 0, "message": "No closed trades yet"}

        # Coerce pnl to numeric
        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
        pnl = df["pnl"].values

        from utils.performance_analytics import compute_metrics
        m = compute_metrics(pnl)

        return {
            "n_trades": m.n_trades,
            "total_pnl": round(m.total_pnl, 2),
            "total_return_pct": round(m.total_return_pct, 2),
            "win_rate_pct": round(m.win_rate * 100, 1),
            "profit_factor": round(m.profit_factor, 2),
            "expectancy": round(m.expectancy, 2),
            "sharpe_ratio": round(m.sharpe_ratio, 2),
            "max_drawdown_pct": round(m.max_drawdown_pct, 1),
            "avg_win": round(m.avg_win, 2),
            "avg_loss": round(m.avg_loss, 2),
            "max_consecutive_losses": m.max_consecutive_losses,
        }

    def pair_breakdown(self) -> pd.DataFrame:
        """Return per-pair statistics as a DataFrame."""
        df = self.get_closed_trades()
        if df.empty:
            return pd.DataFrame()
        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)

        rows = []
        for pair, grp in df.groupby("pair"):
            pnl_arr = grp["pnl"].values
            wins = pnl_arr[pnl_arr > 0]
            losses = pnl_arr[pnl_arr < 0]
            rows.append({
                "pair": pair,
                "n_trades": len(grp),
                "total_pnl": round(float(pnl_arr.sum()), 2),
                "win_rate_%": round(len(wins) / len(pnl_arr) * 100, 1),
                "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
                "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
                "profit_factor": round(
                    float(wins.sum()) / (abs(float(losses.sum())) + 1e-8), 2
                ),
            })
        return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)

    # ─────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────

    def export_csv(self, output_path: str | Path | None = None) -> Path:
        """Export full journal to a CSV file. Returns the path."""
        path = Path(output_path) if output_path else self.dir / "journal_export.csv"
        df = self._load_df()
        df.to_csv(path, index=False)
        logger.info("Journal exported to %s (%d rows)", path, len(df))
        return path

    def export_excel(self, output_path: str | Path | None = None) -> Path | None:
        """Export full journal + pair summary to an Excel file with two sheets."""
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            logger.warning("openpyxl not installed — skipping Excel export")
            return None

        path = Path(output_path) if output_path else self.dir / "journal_export.xlsx"
        df_trades = self._load_df()
        df_summary = self.pair_breakdown()

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_trades.to_excel(writer, sheet_name="All Trades", index=False)
            df_summary.to_excel(writer, sheet_name="Pair Summary", index=False)

        logger.info("Journal exported to %s", path)
        return path

    def save_json_snapshot(self, output_path: str | Path | None = None) -> Path:
        """Save a JSON snapshot of the summary for use by the dashboard."""
        path = Path(output_path) if output_path else self.dir / "journal_snapshot.json"
        try:
            data = {
                "summary": self.summary(),
                "pair_breakdown": self.pair_breakdown().to_dict(orient="records"),
                "open_trades": self.get_open_trades().to_dict(orient="records"),
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning("Journal snapshot save failed: %s", e)
        return path
