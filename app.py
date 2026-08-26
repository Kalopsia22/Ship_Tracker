"""
AI-Powered Maritime Intelligence Dashboard — main entry point.

Run with: streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from data.database import init_db
from utils.configuration import config
from utils.logging import get_logger
from utils.pipeline import initialize_state, run_pipeline
from utils.security import Role, SimulatedUser
from utils.ui_components import data_source_badge, render_kpi_cards

logger = get_logger(__name__)

st.set_page_config(
    page_title=config.app_name,
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
initialize_state()

# --- Simulated auth/role selector (see utils/security.py for the real architecture) ---
with st.sidebar:
    st.markdown("### 👤 Session")
    username = st.text_input("User", value="demo.analyst")
    role_name = st.selectbox("Role", [r.value for r in Role], index=1)
    st.session_state.user = SimulatedUser(username=username, role=Role(role_name))
    st.caption("Simulated auth for prototype purposes — see utils/security.py")

st.title("🚢 Maritime Intelligence Dashboard")
st.caption(
    "Global commercial vessel tracking — Panamax · Aframax · Suezmax · VLCC · ULCC"
)

col_a, col_b, col_c = st.columns([2, 2, 6])
with col_a:
    data_source_badge(st.session_state.provider.source_label)
with col_b:
    st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S UTC')}")
with col_c:
    refresh_col, auto_col = st.columns([1, 3])
    with refresh_col:
        if st.button("🔄 Refresh now"):
            run_pipeline(advance_minutes=5)
            st.rerun()
    with auto_col:
        st.session_state.auto_refresh = st.toggle(
            "Auto-refresh", value=st.session_state.get("auto_refresh", False)
        )

if config.is_demo_mode:
    st.info(
        "**DEMO DATA** — this dashboard is running on a synthetic AIS simulator. "
        "See the sidebar and the *Responsible AI / Model Governance* page for details "
        "on connecting a live AIS provider.",
        icon="ℹ️",
    )

st.divider()
st.subheader("Fleet Snapshot")
render_kpi_cards(st.session_state.kpis)

st.divider()
st.markdown(
    """
    #### 👋 Welcome

    Use the pages in the left sidebar to explore:

    - **Global Fleet Overview** — high-level fleet composition & stats
    - **Live Vessel Map** — interactive global tracking map with filters
    - **Vessel Intelligence** — deep dive on a single vessel with AI explanations
    - **Voyage Analytics** — movement analytics across the fleet
    - **Port Intelligence** — predictive port congestion
    - **AI Alerts** — configurable real-time alert feed
    - **Predictive Analytics** — ETA prediction & fuel-efficiency intelligence
    - **Historical Analysis** — trajectory & performance history
    - **Business Intelligence** — how AI outputs map to business decisions
    - **Responsible AI / Model Governance** — fairness, transparency & oversight
    """
)

if st.session_state.get("auto_refresh"):
    import time

    time.sleep(config.refresh_interval_seconds)
    run_pipeline(advance_minutes=5)
    st.rerun()
