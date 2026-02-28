"""
Streamlit monitoring dashboard.
Shows equity curve, SHAP explanations, sentiment scores,
model weights, risk status, system health.
Run: streamlit run dashboard/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

from utils.config_loader import load_config
from memory.vector_memory import TradingMemory


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

    st.title("💹 ForexAI — Self-Healing Autonomous Trader")
    st.caption("Real-time monitoring dashboard")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    try:
        cfg = load_config()
        pairs = cfg.get("pairs", {}).get("majors", ["EURUSD"])
        mode = cfg.get("execution", {}).get("mode", "paper")
    except Exception:
        cfg = {}
        pairs = ["EURUSD", "GBPUSD", "USDJPY"]
        mode = "paper"

    st.sidebar.metric("Trading Mode", mode.upper())
    selected_pair = st.sidebar.selectbox("Select Pair", pairs)

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)

    # Load trade memory
    try:
        memory = TradingMemory(
            persist_dir=cfg.get("memory", {}).get("chroma_persist_dir", "artifacts/chromadb")
        )
        stats = memory.get_stats()
        trades = memory.get_recent_context(selected_pair, n=50)
    except Exception:
        stats = {"trade_count": 0, "backend": "unavailable"}
        trades = []

    col1.metric("Total Trades", stats.get("trade_count", 0))
    col2.metric("Memory Backend", stats.get("backend", "N/A"))

    # Equity curve (from trade log)
    st.subheader(f"📈 Equity Curve — {selected_pair}")
    if trades:
        try:
            equity_vals = [float(t.get("equity", 10000)) for t in trades if "equity" in t]
            if equity_vals:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=equity_vals, mode="lines+markers",
                    name="Equity", line=dict(color="#00ff88", width=2)
                ))
                fig.update_layout(
                    template="plotly_dark", height=300,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No equity data yet — start the trader to see live data")
        except Exception:
            st.info("Equity chart unavailable")
    else:
        st.info("No trade history — run main_orchestrator.py to start trading")

    # Recent trades table
    st.subheader("📋 Recent Trades")
    if trades:
        display_cols = ["timestamp", "pair", "direction", "entry_price", "confidence", "explanation"]
        rows = [{k: t.get(k, "") for k in display_cols} for t in trades[:20]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No trades logged yet")

    # Sentiment panel
    st.subheader("🧠 SENTIENCE — Sentiment Scores")
    col5, col6, col7 = st.columns(3)

    sentiment_data = {}
    log_dir = Path("logs")
    sentiment_log = log_dir / "sentiment_cache.json"
    if sentiment_log.exists():
        try:
            sentiment_data = json.loads(sentiment_log.read_text())
        except Exception:
            pass

    for i, p in enumerate(pairs[:6]):
        score = sentiment_data.get(p, 0.0)
        col = [col5, col6, col7][i % 3]
        col.metric(p, f"{score:+.3f}", delta_color="normal")

    # SHAP explanation panel
    st.subheader("🔍 XAI — Latest Explanation")
    latest_explanation = ""
    expl_log = Path("logs/last_explanation.json")
    if expl_log.exists():
        try:
            expl = json.loads(expl_log.read_text())
            latest_explanation = expl.get("narrative", "")
            shap_vals = expl.get("shap_values", {})
            if shap_vals:
                fig = px.bar(
                    x=list(shap_vals.values()),
                    y=list(shap_vals.keys()),
                    orientation="h",
                    title="SHAP Feature Importance",
                    color=list(shap_vals.values()),
                    color_continuous_scale=["red", "gray", "green"],
                )
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
    if latest_explanation:
        st.info(latest_explanation)
    else:
        st.info("Run the trader to see SHAP explanations here")

    # Risk status
    st.subheader("🛡️ Risk & System Status")
    risk_col1, risk_col2 = st.columns(2)

    risk_log = Path("logs/risk_status.json")
    if risk_log.exists():
        try:
            risk_status = json.loads(risk_log.read_text())
            risk_col1.metric("Equity", f"${risk_status.get('equity', 0):,.2f}")
            risk_col1.metric("Drawdown", f"{risk_status.get('drawdown_pct', 0):.2f}%")
            risk_col2.metric("Circuit Breaker", "🔴 ACTIVE" if risk_status.get("circuit_broken") else "🟢 OK")
            risk_col2.metric("Survival Mode", "🟡 ACTIVE" if risk_status.get("survival_mode") else "🟢 Normal")
        except Exception:
            pass
    else:
        risk_col1.info("Risk status will appear once the trader is running")

    # Health log
    st.subheader("🔧 Self-Healing Log")
    health_log = Path("logs/health_log.json")
    if health_log.exists():
        try:
            health_data = json.loads(health_log.read_text())
            if isinstance(health_data, list):
                df_health = pd.DataFrame(health_data[-20:])
                st.dataframe(df_health, use_container_width=True)
        except Exception:
            pass
    else:
        st.info("Health log will appear once the trader is running")

    st.markdown("---")
    st.caption("ForexAI 2026 — Self-Healing Autonomous Trader | Refresh to update")


if __name__ == "__main__":
    main()
