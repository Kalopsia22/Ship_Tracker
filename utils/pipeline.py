"""
Shared application state and processing pipeline.

Centralizes the flow:
  AIS Provider -> Validation -> ML Models -> Analytics -> Alerts
so every Streamlit page works from the same cached, consistent snapshot
instead of recomputing independently. Uses st.session_state + st.cache_data
so the fleet only regenerates on an explicit refresh.
"""
from __future__ import annotations

import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from analytics.port_analysis import port_congestion_estimate
from analytics.route_analysis import compute_route_deviation
from analytics.vessel_metrics import fleet_kpis
from data.ais_provider import get_provider
from data.schemas import Alert
from data.simulator import VesselSimulator
from data.validator import validate_positions
from models.anomaly_model import AnomalyDetector
from models.eta_model import ETAPredictor
from models.risk_model import compute_risk
from utils.configuration import config
from utils.logging import get_logger

logger = get_logger(__name__)


def _generate_alerts(risk_df: pd.DataFrame, thresholds: dict) -> list[dict]:
    alerts = []
    for _, row in risk_df.iterrows():
        if row["deviation_km"] > thresholds["route_deviation_km"]:
            alerts.append(_alert(row, "Route Deviation Alert", "WARNING",
                                  f"{row['vessel_name']} is {row['deviation_km']:.0f} km off its expected route."))
        if abs(row["eta_deviation_hours"]) > thresholds["eta_delay_hours"] and row["eta_deviation_hours"] > 0:
            alerts.append(_alert(row, "ETA Delay Alert", "WARNING",
                                  f"{row['vessel_name']} predicted {row['eta_deviation_hours']:.1f}h late vs schedule."))
        gap = row["_features"]["ais_gap_minutes"]
        if gap > thresholds["ais_gap_minutes"]:
            alerts.append(_alert(row, "AIS Gap Alert", "CRITICAL",
                                  f"{row['vessel_name']} has not reported AIS position in {gap:.0f} minutes."))
        if row["risk_score"] > thresholds["risk_score"]:
            alerts.append(_alert(row, "Operational Risk Alert", "CRITICAL",
                                  f"{row['vessel_name']} operational risk score is {row['risk_score']:.0f}/100."))
        if row["_features"]["is_loitering"]:
            alerts.append(_alert(row, "Unusual Movement Alert", "INFO",
                                  f"{row['vessel_name']} appears to be loitering (near-zero speed while under way)."))
    return alerts


def _alert(row, alert_type, severity, message) -> dict:
    return {
        "alert_id": str(uuid.uuid4())[:8],
        "mmsi": row["mmsi"],
        "vessel_name": row["vessel_name"],
        "alert_type": alert_type,
        "severity": severity,
        "message": message,
        "triggered_at": datetime.utcnow(),
        "acknowledged": False,
    }


def initialize_state():
    """Idempotent — safe to call at the top of every page."""
    if "simulator" not in st.session_state:
        logger.info("Initializing simulator with fleet size %d", config.simulated_fleet_size)
        sim = VesselSimulator(fleet_size=config.simulated_fleet_size, seed=config.random_seed)
        sim.generate_history(hours_back=72)
        st.session_state.simulator = sim
        st.session_state.provider = get_provider(simulator=sim)
        st.session_state.eta_model = ETAPredictor()
        st.session_state.anomaly_model = AnomalyDetector()
        st.session_state.alerts_log = []
        st.session_state.last_refresh = datetime.utcnow()
        st.session_state.thresholds = {
            "route_deviation_km": config.default_route_deviation_km,
            "eta_delay_hours": config.default_eta_delay_hours,
            "ais_gap_minutes": config.default_ais_gap_minutes,
            "risk_score": config.default_risk_score_threshold,
        }
        run_pipeline()


def run_pipeline(advance_minutes: float = 0.0):
    """Execute the full ingestion -> validation -> ML -> alerts pipeline."""
    sim = st.session_state.simulator
    if advance_minutes:
        sim.step(minutes=advance_minutes)

    provider = st.session_state.provider
    raw_snapshot = provider.get_fleet_snapshot()

    clean_snapshot, validation_report = validate_positions(raw_snapshot)

    eta_df = st.session_state.eta_model.predict(clean_snapshot)
    anomaly_df = st.session_state.anomaly_model.score(clean_snapshot)
    route_df = compute_route_deviation(sim, clean_snapshot)
    risk_df = compute_risk(anomaly_df, eta_df, route_df)

    # merge everything into one convenient working table
    merged = clean_snapshot.merge(eta_df[["mmsi", "predicted_eta", "eta_deviation_hours", "confidence"]], on="mmsi", how="left")
    merged = merged.merge(anomaly_df[["mmsi", "anomaly_score", "is_anomalous"]], on="mmsi", how="left")
    merged = merged.merge(route_df[["mmsi", "deviation_km", "deviation_pct"]], on="mmsi", how="left")
    merged = merged.merge(risk_df[["mmsi", "risk_score", "risk_level"]], on="mmsi", how="left")

    new_alerts = _generate_alerts(risk_df, st.session_state.thresholds)
    st.session_state.alerts_log = (new_alerts + st.session_state.alerts_log)[:200]

    st.session_state.snapshot = merged
    st.session_state.eta_df = eta_df
    st.session_state.anomaly_df = anomaly_df
    st.session_state.route_df = route_df
    st.session_state.risk_df = risk_df
    st.session_state.validation_report = validation_report
    st.session_state.kpis = fleet_kpis(merged, risk_df, None)
    st.session_state.port_congestion = port_congestion_estimate(merged)
    st.session_state.last_refresh = datetime.utcnow()
    logger.info("Pipeline run complete: %d vessels, %d alerts", len(merged), len(new_alerts))


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    if filters.get("vessel_classes"):
        out = out[out["vessel_class"].isin(filters["vessel_classes"])]
    if filters.get("flags"):
        out = out[out["flag"].isin(filters["flags"])]
    if filters.get("speed_range"):
        lo, hi = filters["speed_range"]
        out = out[out["speed_knots"].between(lo, hi)]
    if filters.get("draught_range"):
        lo, hi = filters["draught_range"]
        out = out[out["draught_m"].between(lo, hi)]
    if filters.get("destination"):
        out = out[out["destination"] == filters["destination"]]
    if filters.get("risk_threshold") is not None:
        out = out[out["risk_score"] >= filters["risk_threshold"]]
    if filters.get("search"):
        q = filters["search"].strip().lower()
        out = out[
            out["mmsi"].str.lower().str.contains(q)
            | out["vessel_name"].str.lower().str.contains(q)
            | out["imo"].str.lower().str.contains(q)
        ]
    return out
