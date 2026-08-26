"""Single-vessel deep dive with full AI explainability panels."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.vessel_metrics import historical_metrics
from data.database import log_audit_event
from models.anomaly_model import AnomalyDetector
from models.risk_model import explain_risk
from utils.pipeline import initialize_state
from utils.ui_components import data_source_badge, render_explanation_panel

st.set_page_config(page_title="Vessel Intelligence", page_icon="🔬", layout="wide")
initialize_state()

st.title("🔬 Vessel Intelligence")
data_source_badge(st.session_state.provider.source_label)

snapshot = st.session_state.snapshot
options = snapshot.apply(lambda r: f"{r['vessel_name']} — MMSI {r['mmsi']}", axis=1)
choice = st.selectbox("Select a vessel", options.tolist())
mmsi = choice.split("MMSI")[-1].strip()

row = snapshot[snapshot["mmsi"] == mmsi].iloc[0]
risk_row = st.session_state.risk_df[st.session_state.risk_df["mmsi"] == mmsi]

user = st.session_state.get("user")
if user:
    log_audit_event(user.username, "view_vessel_detail", details=f"mmsi={mmsi}")

c1, c2, c3 = st.columns(3)
c1.metric("Vessel", row["vessel_name"])
c1.caption(f"MMSI {row['mmsi']} · IMO {row.get('imo','-')} · Flag: {row['flag']}")
c2.metric("Current speed", f"{row['speed_knots']} kn")
c2.caption(f"Course {row['course_deg']}° · Heading {row.get('heading_deg','-')}°")
c3.metric("Risk score", f"{row.get('risk_score', 0):.0f}/100", row.get("risk_level", ""))

st.divider()
c4, c5, c6 = st.columns(3)
c4.metric("Destination", row["destination"])
c4.caption(f"Voyage progress: {row['voyage_progress_pct']}%")
c5.metric("Predicted ETA", pd.to_datetime(row["predicted_eta"]).strftime("%Y-%m-%d %H:%M UTC"))
c5.caption(f"Scheduled: {pd.to_datetime(row['eta_scheduled']).strftime('%Y-%m-%d %H:%M UTC')} "
           f"({row['eta_deviation_hours']:+.1f}h)")
c6.metric("Anomaly score", f"{row.get('anomaly_score', 0):.0f}/100")
c6.caption("Anomalous" if row.get("is_anomalous") else "Normal behavior")

st.divider()
st.subheader("🧠 Why this risk score? (AI explainability)")
if not risk_row.empty:
    r = risk_row.iloc[0]
    factors = explain_risk(r)
    render_explanation_panel(
        "Operational Risk Indicator", r["risk_level"], factors, disclaimer=(
            "This is an Operational Risk Indicator derived from observable AIS behavior. "
            "It is a decision-support signal, not a security or criminal classification, "
            "and does not represent vessel intent. Flag state, country, and ownership are "
            "explicitly excluded from this score."
        ),
    )

st.divider()
st.subheader("📈 Track history (last 72 hours)")
history = st.session_state.provider.get_vessel_history(mmsi, hours_back=72)
if not history.empty:
    metrics = historical_metrics(history)
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Avg speed", f"{metrics.get('avg_speed_knots','-')} kn")
    hc2.metric("Max speed", f"{metrics.get('max_speed_knots','-')} kn")
    hc3.metric("Distance travelled", f"{metrics.get('distance_travelled_km','-')} km")
    hc4.metric("Position reports", metrics.get("position_reports", 0))

    fig = px.line(history.sort_values("timestamp"), x="timestamp", y="speed_knots", title="Speed over time (knots)")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter_geo(
        history.sort_values("timestamp"), lat="latitude", lon="longitude",
        color="speed_knots", color_continuous_scale="Viridis", title="Track history",
    )
    fig2.update_geos(projection_type="natural earth", showland=True, landcolor="#1e293b",
                      showocean=True, oceancolor="#0b1220", bgcolor="#0b1220")
    fig2.update_layout(paper_bgcolor="#0b1220", geo_bgcolor="#0b1220")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No historical track data available for this vessel yet.")

st.divider()
st.subheader("📋 Vessel particulars")
particulars = {
    "Vessel class": row["vessel_class"], "Cargo category": row["cargo_category"],
    "Draught (m)": row["draught_m"], "Length (m)": row["length_m"], "Width (m)": row["width_m"],
    "Navigation status": row["nav_status"], "Origin port": row.get("origin", "-"),
    "Last AIS update": row["last_ais_update"],
}
particulars_df = pd.DataFrame(
    [(k, str(v)) for k, v in particulars.items()], columns=["Field", "Value"]
)
st.table(particulars_df)
