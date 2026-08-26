"""
Fleet-level and vessel-level metric calculations.
"""
from __future__ import annotations

import pandas as pd


def fleet_kpis(snapshot: pd.DataFrame, risk_df: pd.DataFrame | None, alert_df: pd.DataFrame | None) -> dict:
    total = len(snapshot)
    underway = int((snapshot["nav_status"] == "Under way using engine").sum())
    at_anchor = int((snapshot["nav_status"] == "At anchor").sum())
    in_port = int(snapshot["nav_status"].isin(["Moored", "In port"]).sum())

    abnormal = 0
    high_risk = 0
    if risk_df is not None and not risk_df.empty:
        abnormal = int((risk_df["anomaly_score"] > 50).sum())
        high_risk = int((risk_df["risk_level"] == "HIGH").sum())

    late_predicted = 0
    if "eta_deviation_hours" in snapshot.columns:
        late_predicted = int((snapshot["eta_deviation_hours"] > 6).sum())

    now = pd.Timestamp.utcnow().tz_localize(None)
    freshness_minutes = (now - pd.to_datetime(snapshot["last_ais_update"])).dt.total_seconds().mean() / 60 if total else 0

    return {
        "total_vessels": total,
        "underway": underway,
        "at_anchor": at_anchor,
        "in_port": in_port,
        "abnormal_behavior": abnormal,
        "predicted_late": late_predicted,
        "high_risk": high_risk,
        "avg_data_freshness_min": round(freshness_minutes, 1),
    }


def class_distribution(snapshot: pd.DataFrame) -> pd.DataFrame:
    return snapshot["vessel_class"].value_counts().rename_axis("vessel_class").reset_index(name="count")


def speed_distribution(snapshot: pd.DataFrame) -> pd.DataFrame:
    return snapshot[["vessel_class", "speed_knots"]]


def voyage_progress_summary(snapshot: pd.DataFrame) -> pd.DataFrame:
    return (
        snapshot.groupby("vessel_class")
        .agg(
            avg_progress_pct=("voyage_progress_pct", "mean"),
            avg_remaining_km=("remaining_km", "mean"),
            count=("mmsi", "count"),
        )
        .reset_index()
        .round(1)
    )


def historical_metrics(history: pd.DataFrame) -> dict:
    if history.empty:
        return {}
    distance_km = 0.0
    if len(history) > 1:
        h = history.sort_values("timestamp")
        # approximate distance from speed * time deltas
        dt_hours = h["timestamp"].diff().dt.total_seconds().fillna(0) / 3600
        distance_km = float((h["speed_knots"] * 1.852 * dt_hours).sum())
    return {
        "avg_speed_knots": round(history["speed_knots"].mean(), 1),
        "max_speed_knots": round(history["speed_knots"].max(), 1),
        "min_speed_knots": round(history["speed_knots"].min(), 1),
        "distance_travelled_km": round(distance_km, 1),
        "position_reports": len(history),
    }
