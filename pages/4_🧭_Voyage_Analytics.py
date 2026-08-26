"""Voyage / movement analytics across the fleet."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from utils.pipeline import initialize_state
from utils.ui_components import data_source_badge

st.set_page_config(page_title="Voyage Analytics", page_icon="🧭", layout="wide")
initialize_state()

st.title("🧭 Voyage Analytics")
data_source_badge(st.session_state.provider.source_label)

snapshot = st.session_state.snapshot
route_df = st.session_state.route_df

st.subheader("Route deviation across the fleet")
merged = snapshot.merge(route_df[["mmsi", "deviation_km", "deviation_pct"]], on="mmsi", how="left", suffixes=("", "_r"))
fig = px.scatter(
    merged, x="voyage_progress_pct", y="deviation_km", color="vessel_class",
    size="speed_knots", hover_name="vessel_name",
    labels={"voyage_progress_pct": "Voyage progress (%)", "deviation_km": "Deviation from expected route (km)"},
    color_discrete_map={
        "Panamax": "#4C9BE8", "Aframax": "#5FD1A5", "Suezmax": "#F2C744",
        "VLCC": "#F28C44", "ULCC": "#E85C5C",
    },
)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Top routes by traffic volume")
route_counts = (
    snapshot.groupby(["origin", "destination"]).size().reset_index(name="vessel_count")
    .sort_values("vessel_count", ascending=False).head(15)
)
fig2 = px.bar(
    route_counts, x="vessel_count",
    y=route_counts.apply(lambda r: f"{r['origin']} → {r['destination']}", axis=1),
    orientation="h",
)
fig2.update_layout(yaxis_title="Route", xaxis_title="Vessel count", margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Speed vs. draught (loading condition proxy)")
fig3 = px.scatter(
    snapshot, x="draught_m", y="speed_knots", color="vessel_class", hover_name="vessel_name",
    color_discrete_map={
        "Panamax": "#4C9BE8", "Aframax": "#5FD1A5", "Suezmax": "#F2C744",
        "VLCC": "#F28C44", "ULCC": "#E85C5C",
    },
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.subheader("Fleet table — full deviation & progress detail")
st.dataframe(
    merged[[
        "mmsi", "vessel_name", "vessel_class", "origin", "destination",
        "voyage_progress_pct", "deviation_km", "deviation_pct", "speed_knots",
    ]].sort_values("deviation_km", ascending=False),
    use_container_width=True, hide_index=True,
)
