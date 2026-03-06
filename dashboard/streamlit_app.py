"""
Streamlit monitoring dashboard.
Shows equity curve, SHAP explanations, sentiment scores,
model weights, risk status, system health, open positions,
performance analytics, and multi-timeframe confluence.
Run: streamlit run dashboard/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

from utils.config_loader import load_config
from memory.vector_memory import TradingMemory


def _load_json(path: Path, default=None):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def main():
    if not _HAS_STREAMLIT:
        print("streamlit not installed. Run: pip install streamlit")
        return

    st.set_page_config(
        page_title="ForexAI Dashboard",
        page_icon="💹",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Auto-refresh ───────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Configuration")
    refresh_interval = st.sidebar.selectbox(
        "Auto-refresh (seconds)", [0, 10, 30, 60, 120], index=2
    )

    try:
        cfg = load_config()
        pairs = cfg.get("pairs", {}).get("majors", ["EURUSD"])
        mode = cfg.get("execution", {}).get("mode", "paper")
    except Exception:
        cfg = {}
        pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
        mode = "paper"

    st.sidebar.metric("Trading Mode", mode.upper())
    selected_pair = st.sidebar.selectbox("Select Pair", pairs)

    # ── Load trade journal snapshot ────────────────────────────────────────
    journal_dir = Path(cfg.get("memory", {}).get("journal_dir", "artifacts/trade_journal"))
    journal_snapshot = _load_json(journal_dir / "journal_snapshot.json", {})
    summary = journal_snapshot.get("summary", {})
    pair_breakdown_raw = journal_snapshot.get("pair_breakdown", [])
    open_trades_raw = journal_snapshot.get("open_trades", [])

    # ── Load trade memory ──────────────────────────────────────────────────
    try:
        memory = TradingMemory(
            persist_dir=cfg.get("memory", {}).get("chroma_persist_dir", "artifacts/chromadb")
        )
        stats = memory.get_stats()
        trades = memory.get_recent_context(selected_pair, n=50)
    except Exception:
        stats = {"trade_count": 0, "backend": "unavailable"}
        trades = []

    # ══════════════════════════════════════════════════════════════════════
    # TITLE & TOP METRICS
    # ══════════════════════════════════════════════════════════════════════
    st.title("💹 ForexAI — Self-Healing Autonomous Trader")
    st.caption(f"Real-time monitoring dashboard · Mode: **{mode.upper()}** · Pair: **{selected_pair}**")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Trades", summary.get("n_trades", stats.get("trade_count", 0)))
    col2.metric("Total P&L", f"${summary.get('total_pnl', 0):+,.2f}")
    col3.metric("Win Rate", f"{summary.get('win_rate_pct', 0):.1f}%")
    col4.metric("Sharpe", f"{summary.get('sharpe_ratio', 0):.2f}")
    col5.metric("Max DD", f"{summary.get('max_drawdown_pct', 0):.1f}%")
    col6.metric("Profit Factor", f"{summary.get('profit_factor', 0):.2f}")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # ROW 1 — Equity Curve + Open Positions
    # ══════════════════════════════════════════════════════════════════════
    eq_col, pos_col = st.columns([2, 1])

    with eq_col:
        st.subheader(f"📈 Equity Curve")
        if trades:
            equity_vals = [float(t.get("equity", 10000)) for t in trades if "equity" in t]
        else:
            equity_vals = []

        # Also try to build from journal PnL
        if not equity_vals and pair_breakdown_raw:
            try:
                from utils.trade_journal import TradeJournal
                tj = TradeJournal(journal_dir)
                closed = tj.get_closed_trades()
                if not closed.empty:
                    closed["pnl"] = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0)
                    equity_vals = (10000 + closed["pnl"].cumsum()).tolist()
            except Exception:
                pass

        if equity_vals:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=equity_vals, mode="lines",
                name="Equity", line=dict(color="#00ff88", width=2),
                fill="tozeroy", fillcolor="rgba(0,255,136,0.08)",
            ))
            # Add drawdown shading
            eq_arr = np.array(equity_vals, dtype=float)
            peak = np.maximum.accumulate(eq_arr)
            dd_pct = (peak - eq_arr) / (peak + 1e-8) * 100
            fig.add_trace(go.Scatter(
                y=-dd_pct, mode="lines",
                name="Drawdown %", line=dict(color="#ff4444", width=1),
                yaxis="y2",
            ))
            fig.update_layout(
                template="plotly_dark", height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis2=dict(overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No equity data yet — start the trader to see live data")

    with pos_col:
        st.subheader("📂 Open Positions")
        if open_trades_raw:
            df_open = pd.DataFrame(open_trades_raw)
            display = ["pair", "direction", "entry_price", "size_lots", "confidence"]
            display = [c for c in display if c in df_open.columns]
            st.dataframe(df_open[display], use_container_width=True, height=250)
        else:
            st.info("No open positions")

    # ══════════════════════════════════════════════════════════════════════
    # ROW 2 — Performance Analytics + Pair Comparison
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("📊 Performance Analytics")
    perf_col, pair_col = st.columns([1, 1])

    with perf_col:
        if summary.get("n_trades", 0) > 0:
            metric_rows = [
                ("Expectancy / trade", f"${summary.get('expectancy', 0):.2f}"),
                ("Avg Win", f"${summary.get('avg_win', 0):.2f}"),
                ("Avg Loss", f"${summary.get('avg_loss', 0):.2f}"),
                ("Max Consec. Losses", str(summary.get("max_consecutive_losses", 0))),
            ]
            for label, val in metric_rows:
                st.metric(label, val)
        else:
            st.info("Performance metrics will appear after the first closed trades")

    with pair_col:
        if pair_breakdown_raw:
            df_pb = pd.DataFrame(pair_breakdown_raw)
            fig_pb = px.bar(
                df_pb, x="pair", y="total_pnl",
                color="win_rate_%",
                color_continuous_scale=["red", "yellow", "green"],
                title="P&L by Pair",
            )
            fig_pb.update_layout(template="plotly_dark", height=220,
                                  margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_pb, use_container_width=True)
        else:
            st.info("Pair breakdown will appear after first closed trades")

    # ══════════════════════════════════════════════════════════════════════
    # ROW 3 — Recent Trades Table
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("📋 Recent Trades")
    try:
        from utils.trade_journal import TradeJournal
        tj = TradeJournal(journal_dir)
        recent = tj.get_closed_trades().tail(20)
        if not recent.empty:
            show_cols = [c for c in ["timestamp", "pair", "direction", "entry_price",
                                      "exit_price", "pnl", "pnl_pips", "confidence",
                                      "regime", "exit_reason"] if c in recent.columns]
            st.dataframe(recent[show_cols], use_container_width=True)
            # Export button
            if st.button("⬇️ Export Journal to CSV"):
                csv_path = tj.export_csv()
                st.success(f"Exported to {csv_path}")
        else:
            st.info("No closed trades yet — journal will fill as the trader runs")
    except Exception:
        # Fall back to memory trades
        if trades:
            display_cols = ["timestamp", "pair", "direction", "entry_price", "confidence", "explanation"]
            rows = [{k: t.get(k, "") for k in display_cols} for t in trades[:20]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No trades logged yet")

    # ══════════════════════════════════════════════════════════════════════
    # ROW 4 — Multi-Timeframe + Sentiment
    # ══════════════════════════════════════════════════════════════════════
    mtf_col, sent_col = st.columns([1, 1])

    with mtf_col:
        st.subheader("🕐 Multi-Timeframe Confluence")
        mtf_log = Path("logs/mtf_status.json")
        mtf_data = _load_json(mtf_log, {})
        if mtf_data:
            pair_mtf = mtf_data.get(selected_pair, {})
            score = pair_mtf.get("confluence_score", 0)
            bias = pair_mtf.get("bias", {})
            direction_label = "🟢 BULLISH" if score > 0.1 else ("🔴 BEARISH" if score < -0.1 else "⚪ NEUTRAL")
            st.metric("Confluence", direction_label, delta=f"{score:+.2f}")
            if bias:
                rows = [{"Timeframe": tf, "Bias": "↑ Bull" if b == 1 else ("↓ Bear" if b == -1 else "→ Flat")}
                        for tf, b in bias.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("MTF data will appear once the trader has run at least one cycle")

    with sent_col:
        st.subheader("🧠 Sentiment Scores")
        sentiment_data = {}
        log_dir = Path("logs")
        sentiment_log = log_dir / "sentiment_cache.json"
        if sentiment_log.exists():
            try:
                sentiment_data = json.loads(sentiment_log.read_text())
            except Exception:
                pass

        if sentiment_data:
            pairs_sent = [(p, float(sentiment_data.get(p, 0.0))) for p in pairs[:8]]
            fig_sent = go.Figure(go.Bar(
                x=[p[0] for p in pairs_sent],
                y=[p[1] for p in pairs_sent],
                marker_color=["green" if s > 0 else "red" for _, s in pairs_sent],
            ))
            fig_sent.update_layout(
                template="plotly_dark", height=220,
                margin=dict(l=0, r=0, t=10, b=0),
                title="Pair Sentiment Score",
            )
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            for i, p in enumerate(pairs[:6]):
                score = sentiment_data.get(p, 0.0)
                st.metric(p, f"{score:+.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # ROW 5 — XAI Explanation
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("🔍 XAI — Latest Explanation")
    latest_explanation = ""
    expl_log = Path("logs/last_explanation.json")
    if expl_log.exists():
        try:
            expl = json.loads(expl_log.read_text())
            latest_explanation = expl.get("narrative", "")
            shap_vals = expl.get("shap_values", {})
            if shap_vals:
                # Top 15 features by absolute value
                top = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
                fig_shap = px.bar(
                    x=[v for _, v in top],
                    y=[k for k, _ in top],
                    orientation="h",
                    title="SHAP Feature Importance (top 15)",
                    color=[v for _, v in top],
                    color_continuous_scale=["red", "gray", "green"],
                )
                fig_shap.update_layout(template="plotly_dark", height=320,
                                        margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_shap, use_container_width=True)
        except Exception:
            pass
    if latest_explanation:
        st.info(latest_explanation)
    else:
        st.info("Run the trader to see SHAP explanations here")

    # ══════════════════════════════════════════════════════════════════════
    # ROW 6 — Risk Status + Health Log
    # ══════════════════════════════════════════════════════════════════════
    risk_col1, risk_col2, health_col = st.columns([1, 1, 2])

    with risk_col1:
        st.subheader("🛡️ Risk Status")
        risk_log = Path("logs/risk_status.json")
        if risk_log.exists():
            try:
                risk_status = json.loads(risk_log.read_text())
                st.metric("Equity", f"${risk_status.get('equity', 0):,.2f}")
                st.metric("Drawdown", f"{risk_status.get('drawdown_pct', 0):.2f}%")
            except Exception:
                pass
        else:
            st.info("Risk data will appear once the trader is running")

    with risk_col2:
        st.subheader("🔴 Circuit Breakers")
        risk_log = Path("logs/risk_status.json")
        if risk_log.exists():
            try:
                risk_status = json.loads(risk_log.read_text())
                st.metric("Circuit Breaker",
                          "🔴 ACTIVE" if risk_status.get("circuit_broken") else "🟢 OK")
                st.metric("Survival Mode",
                          "🟡 ACTIVE" if risk_status.get("survival_mode") else "🟢 Normal")
            except Exception:
                pass

    with health_col:
        st.subheader("🔧 Self-Healing Log")
        health_log = Path("logs/health_log.json")
        if health_log.exists():
            try:
                health_data = json.loads(health_log.read_text())
                if isinstance(health_data, list):
                    df_health = pd.DataFrame(health_data[-10:])
                    st.dataframe(df_health, use_container_width=True, height=200)
            except Exception:
                pass
        else:
            st.info("Health log will appear once the trader is running")

    # ══════════════════════════════════════════════════════════════════════
    # ROW 7 — Alerts Panel
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("🚨 Alerts")
    alert_summary = _load_json(Path("logs/alert_summary.json"), {})
    recent_alerts = alert_summary.get("recent_alerts", [])
    alert_counts = alert_summary.get("alert_counts", {})

    if recent_alerts:
        level_colors = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
        df_alerts = pd.DataFrame(recent_alerts)[["timestamp", "level", "alert_type", "message", "value"]]
        df_alerts["level"] = df_alerts["level"].map(level_colors).fillna("⚪") + " " + df_alerts["level"]
        st.dataframe(df_alerts, use_container_width=True, height=200)

        if alert_counts:
            fig_ac = px.bar(
                x=list(alert_counts.keys()),
                y=list(alert_counts.values()),
                title="Alert Frequency by Type",
            )
            fig_ac.update_layout(template="plotly_dark", height=200,
                                  margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_ac, use_container_width=True)
    else:
        st.info("No alerts fired yet — system is healthy")

    # ── Footer ─────────────────────────────────────────────────────────────
    st.divider()
    updated = journal_snapshot.get("updated_at", "—")
    st.caption(f"ForexAI 2026 — Self-Healing Autonomous Trader | Last journal update: {updated}")

    # ── Auto-refresh ───────────────────────────────────────────────────────
    if refresh_interval > 0:
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore[import]
            st_autorefresh(interval=refresh_interval * 1000, key="dashboard_autorefresh")
        except ImportError:
            # Graceful fallback — user can manually click rerun
            st.caption(
                f"⏱ Auto-refresh every {refresh_interval}s "
                "(install `streamlit-autorefresh` for non-blocking refresh)"
            )
            if st.button("🔄 Refresh now"):
                st.rerun()


if __name__ == "__main__":
    main()
