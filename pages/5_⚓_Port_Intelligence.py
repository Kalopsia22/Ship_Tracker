"""Port Intelligence: predictive congestion & waiting time estimates."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from utils.pipeline import initialize_state
from utils.ui_components import data_source_badge

st.set_page_config(page_title="Port Intelligence", page_icon="⚓", layout="wide")
initialize_state()

st.title("⚓ Port Intelligence")
data_source_badge(st.session_state.provider.source_label)
st.caption(
    "Predictive congestion estimate based on vessels currently converging on or "
    "waiting at each port. Prototype uses a simple queueing heuristic — a "
    "production system would incorporate historical berth-occupancy data."
)

congestion = st.session_state.port_congestion

c1, c2, c3 = st.columns(3)
c1.metric("High congestion ports", int((congestion["congestion_level"] == "HIGH").sum()))
c2.metric("Medium congestion ports", int((congestion["congestion_level"] == "MEDIUM").sum()))
c3.metric("Total vessels inbound (fleet-wide)", int(congestion["vessels_converging"].sum()))

st.divider()
fig = px.bar(
    congestion, x="port", y="expected_wait_hours", color="congestion_level",
    color_discrete_map={"HIGH": "#E85C5C", "MEDIUM": "#F2C744", "LOW": "#5FD1A5"},
    title="Expected waiting time by port (hours)",
)
fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Port congestion detail")
st.dataframe(congestion, use_container_width=True, hide_index=True)

st.caption(
    "🧠 **Model note:** the congestion score is `max(0, waiting + converging − berths) × "
    "(service_hours / berths)`. Berth counts and service times are illustrative "
    "prototype assumptions, not real port capacity data."
)
