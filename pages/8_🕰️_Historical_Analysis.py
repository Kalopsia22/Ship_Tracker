"""Historical Analysis: trajectory playback & fleet performance trends."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from utils.pipeline import initialize_state
from utils.ui_components import data_source_badge

st.set_page_config(page_title="Historical Analysis", page_icon="🕰️", layout="wide")
initialize_state()

st.title("🕰️ Historical Analysis")
data_source_badge(st.session_state.provider.source_label)

sim = st.session_state.simulator
history = sim.history_df()

if history.empty:
    st.info("No historical data generated yet.")
    st.stop()

st.subheader("Fleet-wide speed trend (last 72 hours)")
hourly = history.copy()
hourly["hour"] = hourly["timestamp"].dt.floor("h")
trend = hourly.groupby("hour")["speed_knots"].mean().reset_index()
fig = px.line(trend, x="hour", y="speed_knots", title="Average fleet speed over time (knots)")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Historical vessel density heatmap")
sample = history.sample(min(5000, len(history)), random_state=1)
fig2 = px.density_map(
    sample, lat="latitude", lon="longitude", radius=4, zoom=0.6,
    map_style="carto-darkmatter", title="Position density (last 72h, sampled)",
)
fig2.update_layout(margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Trajectory playback — select a vessel")
snapshot = st.session_state.snapshot
options = snapshot.apply(lambda r: f"{r['vessel_name']} — MMSI {r['mmsi']}", axis=1)
choice = st.selectbox("Vessel", options.tolist())
mmsi = choice.split("MMSI")[-1].strip()

vessel_history = history[history["mmsi"] == mmsi].sort_values("timestamp")
if not vessel_history.empty:
    fig3 = px.line_geo(
        vessel_history, lat="latitude", lon="longitude",
        title=f"Trajectory — {choice}",
    )
    fig3.update_geos(projection_type="natural earth", showland=True, landcolor="#1e293b",
                      showocean=True, oceancolor="#0b1220", bgcolor="#0b1220")
    fig3.update_traces(line=dict(color="#4C9BE8", width=3))
    fig3.update_layout(paper_bgcolor="#0b1220", geo_bgcolor="#0b1220", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.line(vessel_history, x="timestamp", y="speed_knots", title="Speed history")
    st.plotly_chart(fig4, use_container_width=True)

    csv = vessel_history.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export vessel history (CSV)", csv, f"{mmsi}_history.csv", "text/csv")
