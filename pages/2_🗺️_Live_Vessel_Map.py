"""Live Vessel Map page with filters and route overlays."""
from __future__ import annotations

import streamlit as st

from utils.pipeline import apply_filters, initialize_state, run_pipeline
from utils.ui_components import data_source_badge, render_sidebar_filters, render_vessel_map

st.set_page_config(page_title="Live Vessel Map", page_icon="🗺️", layout="wide")
initialize_state()

st.title("🗺️ Live Vessel Tracking Map")
data_source_badge(st.session_state.provider.source_label)
st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S UTC')}")

if st.button("🔄 Refresh positions"):
    run_pipeline(advance_minutes=5)
    st.rerun()

snapshot = st.session_state.snapshot
filters = render_sidebar_filters(snapshot)
filtered = apply_filters(snapshot, filters)

st.caption(f"Showing {len(filtered)} of {len(snapshot)} tracked vessels")
fig = render_vessel_map(filtered)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Vessels in current view")
display_cols = [
    "mmsi", "vessel_name", "vessel_class", "flag", "speed_knots", "course_deg",
    "nav_status", "destination", "risk_score", "anomaly_score", "eta_deviation_hours",
]
st.dataframe(
    filtered[display_cols].sort_values("risk_score", ascending=False),
    use_container_width=True, hide_index=True,
)

csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Export filtered fleet (CSV)", csv, "fleet_snapshot.csv", "text/csv")
