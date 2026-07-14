"""CLV Engine — Streamlit Dashboard

Entry point. Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure app/ is importable for components and data_loader
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="CLV Engine — Customer Lifetime Value",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation ──────────────────────────────────────────────
portfolio = st.Page("pages/portfolio_view.py", title="Portfolio View", icon="📊", default=True)
lookup = st.Page("pages/customer_lookup.py", title="Customer Lookup", icon="🔍")
simulator = st.Page("pages/what_if_simulator.py", title="What-If Simulator", icon="🎛️")

pg = st.navigation([portfolio, lookup, simulator])

# ── Sidebar branding ───────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 CLV Engine")
    st.caption("Customer Lifetime Value Prediction & Segmentation")
    st.divider()
    st.caption("Built by Kajol Dave")
    st.caption("[GitHub](https://github.com/Kajoldave173/clv-engine) · [Model Card](https://github.com/Kajoldave173/clv-engine/blob/main/reports/model_card.md)")

pg.run()