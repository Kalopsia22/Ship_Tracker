"""
Shared Streamlit UI components used across pages: sidebar filters, KPI
cards, the vessel map, and AI explanation panels.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CLASS_COLORS = {
    "Panamax": "#4C9BE8",
    "Aframax": "#5FD1A5",
    "Suezmax": "#F2C744",
    "VLCC": "#F28C44",
    "ULCC": "#E85C5C",
}

CLASS_SYMBOLS = {
    "Panamax": "circle",
    "Aframax": "square",
    "Suezmax": "diamond",
    "VLCC": "triangle-up",
    "ULCC": "star",
}


def render_sidebar_filters(snapshot: pd.DataFrame) -> dict:
    st.sidebar.header("🔎 Filters")

    vessel_classes = st.sidebar.multiselect(
        "Vessel class", sorted(snapshot["vessel_class"].unique()),
        default=list(sorted(snapshot["vessel_class"].unique())),
    )
    flags = st.sidebar.multiselect("Flag state", sorted(snapshot["flag"].unique()))

    speed_min, speed_max = float(snapshot["speed_knots"].min()), float(snapshot["speed_knots"].max())
    speed_range = st.sidebar.slider("Speed range (knots)", 0.0, max(20.0, speed_max), (0.0, max(20.0, speed_max)))

    draught_min, draught_max = float(snapshot["draught_m"].min()), float(snapshot["draught_m"].max())
    draught_range = st.sidebar.slider(
        "Draught range (m)", 0.0, max(25.0, draught_max), (0.0, max(25.0, draught_max))
    )

    destinations = ["(any)"] + sorted(snapshot["destination"].unique())
    destination = st.sidebar.selectbox("Destination port", destinations)

    search = st.sidebar.text_input("Search by IMO / MMSI / Name")

    st.sidebar.divider()
    st.sidebar.subheader("⚠️ AI alert thresholds")
    thresholds = st.session_state.get("thresholds", {})
    route_dev = st.sidebar.number_input(
        "Route deviation alert (km)", value=float(thresholds.get("route_deviation_km", 50)), step=5.0
    )
    eta_delay = st.sidebar.number_input(
        "ETA delay alert (hours)", value=float(thresholds.get("eta_delay_hours", 6)), step=1.0
    )
    ais_gap = st.sidebar.number_input(
        "AIS gap alert (minutes)", value=float(thresholds.get("ais_gap_minutes", 30)), step=5.0
    )
    risk_threshold = st.sidebar.slider(
        "Risk score alert threshold", 0, 100, int(thresholds.get("risk_score", 75))
    )
    st.session_state.thresholds = {
        "route_deviation_km": route_dev,
        "eta_delay_hours": eta_delay,
        "ais_gap_minutes": ais_gap,
        "risk_score": risk_threshold,
    }

    min_risk_filter = st.sidebar.slider("Minimum risk score to display", 0, 100, 0)

    return {
        "vessel_classes": vessel_classes,
        "flags": flags,
        "speed_range": speed_range,
        "draught_range": draught_range,
        "destination": None if destination == "(any)" else destination,
        "search": search,
        "risk_threshold": min_risk_filter,
    }


def render_kpi_cards(kpis: dict):
    cols = st.columns(4)
    cols[0].metric("Total vessels tracked", kpis.get("total_vessels", 0))
    cols[1].metric("Underway", kpis.get("underway", 0))
    cols[2].metric("At anchor", kpis.get("at_anchor", 0))
    cols[3].metric("In port", kpis.get("in_port", 0))

    cols2 = st.columns(4)
    cols2[0].metric("Abnormal behavior", kpis.get("abnormal_behavior", 0))
    cols2[1].metric("Predicted late arrivals", kpis.get("predicted_late", 0))
    cols2[2].metric("High-risk vessels", kpis.get("high_risk", 0))
    cols2[3].metric("AIS data freshness (avg min)", kpis.get("avg_data_freshness_min", 0))


def render_vessel_map(df: pd.DataFrame, height: int = 560) -> go.Figure:
    fig = go.Figure()

    for vessel_class, group in df.groupby("vessel_class"):
        fig.add_trace(go.Scattergeo(
            lon=group["longitude"], lat=group["latitude"],
            text=group.apply(
                lambda r: (
                    f"<b>{r['vessel_name']}</b> ({vessel_class})<br>"
                    f"MMSI: {r['mmsi']} | IMO: {r.get('imo','-')}<br>"
                    f"Speed: {r['speed_knots']} kn | Course: {r['course_deg']}°<br>"
                    f"Status: {r['nav_status']}<br>"
                    f"Destination: {r['destination']}<br>"
                    f"Risk: {r.get('risk_score', 'n/a')} | Anomaly: {r.get('anomaly_score','n/a')}"
                ), axis=1,
            ),
            mode="markers",
            marker=dict(
                size=9, color=CLASS_COLORS.get(vessel_class, "#999"),
                symbol=CLASS_SYMBOLS.get(vessel_class, "circle"), line=dict(width=0.5, color="white"),
            ),
            name=vessel_class,
            hoverinfo="text",
        ))

    fig.update_geos(
        projection_type="natural earth", showland=True, landcolor="#1e293b",
        showocean=True, oceancolor="#0b1220", showcountries=True,
        countrycolor="#334155", coastlinecolor="#334155", bgcolor="#0b1220",
    )
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        paper_bgcolor="#0b1220", geo_bgcolor="#0b1220",
    )
    return fig


def render_explanation_panel(title: str, level: str, factors: list, disclaimer: str | None = None):
    st.markdown(f"**{title}: {level}**")
    for f in factors:
        st.markdown(f"- {f.label}: `{f.value}` (weight {f.weight:.2f})")
    if disclaimer:
        st.caption(disclaimer)


def data_source_badge(label: str):
    color = "#f2c744" if label == "DEMO DATA" else "#5fd1a5"
    st.markdown(
        f"<span style='background-color:{color};color:#0b1220;padding:2px 10px;"
        f"border-radius:12px;font-size:0.75rem;font-weight:600;'>{label}</span>",
        unsafe_allow_html=True,
    )
