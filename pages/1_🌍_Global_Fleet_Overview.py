"""Global Fleet Overview page."""
from __future__ import annotations

import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from analytics.vessel_metrics import class_distribution, voyage_progress_summary
from utils.pipeline import initialize_state
from utils.ui_components import data_source_badge, render_kpi_cards
from utils.vessel_infographic import render_vessel_class_infographic

st.set_page_config(page_title="Fleet Overview", page_icon="🌍", layout="wide")
initialize_state()

st.title("🌍 Global Fleet Overview")
data_source_badge(st.session_state.provider.source_label)

snapshot = st.session_state.snapshot
render_kpi_cards(st.session_state.kpis)

st.divider()
st.subheader("📐 Vessel class reference")
components.html(render_vessel_class_infographic(), height=310, scrolling=False)

st.divider()
c1, c2 = st.columns(2)

with c1:
    st.subheader("Fleet composition by class")
    dist = class_distribution(snapshot)
    fig = px.pie(dist, names="vessel_class", values="count", hole=0.45,
                 color="vessel_class",
                 color_discrete_map={
                     "Panamax": "#4C9BE8", "Aframax": "#5FD1A5", "Suezmax": "#F2C744",
                     "VLCC": "#F28C44", "ULCC": "#E85C5C",
                 })
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Navigation status breakdown")
    status_counts = snapshot["nav_status"].value_counts().reset_index()
    status_counts.columns = ["nav_status", "count"]
    fig2 = px.bar(status_counts, x="nav_status", y="count", color="nav_status")
    fig2.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Voyage progress by vessel class")
st.dataframe(voyage_progress_summary(snapshot), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Speed distribution by class")
fig3 = px.box(snapshot, x="vessel_class", y="speed_knots", color="vessel_class",
              color_discrete_map={
                  "Panamax": "#4C9BE8", "Aframax": "#5FD1A5", "Suezmax": "#F2C744",
                  "VLCC": "#F28C44", "ULCC": "#E85C5C",
              })
fig3.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Cargo category mix")
cargo_counts = snapshot["cargo_category"].value_counts().reset_index()
cargo_counts.columns = ["cargo_category", "count"]
st.dataframe(cargo_counts, use_container_width=True, hide_index=True)
