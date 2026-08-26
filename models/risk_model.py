"""
Vessel Operational Risk Indicator.

Deliberately explainable (not a black box): a weighted composite of
observable operational indicators. Explicitly EXCLUDES flag state, country,
owner nationality, or other proxy variables from the scoring formula, per
the project's fairness requirements — these are behavioral/operational
signals only.
"""
from __future__ import annotations

import pandas as pd

from data.schemas import RiskFactorContribution
from utils.logging import get_logger

logger = get_logger(__name__)

MODEL_VERSION = "risk-composite-v0.2-prototype"

# Weights are transparent and documented — this is a rules-based explainable
# model rather than an opaque learned classifier, by design (see spec §7-8).
WEIGHTS = {
    "route_deviation": 0.25,
    "anomaly_score": 0.30,
    "ais_gap": 0.20,
    "eta_deviation": 0.15,
    "loitering": 0.10,
}


def compute_risk(
    anomaly_df: pd.DataFrame,
    eta_df: pd.DataFrame,
    route_deviation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine anomaly, ETA, and route-deviation outputs into a single risk score."""
    merged = anomaly_df.merge(eta_df[["mmsi", "eta_deviation_hours"]], on="mmsi", how="left")
    merged = merged.merge(route_deviation_df[["mmsi", "deviation_km", "deviation_pct"]], on="mmsi", how="left")

    merged["eta_deviation_hours"] = merged["eta_deviation_hours"].fillna(0)
    merged["deviation_km"] = merged["deviation_km"].fillna(0)

    norm_anomaly = merged["anomaly_score"] / 100
    norm_route = (merged["deviation_km"] / 100).clip(0, 1)
    norm_ais_gap = merged["_features"].apply(lambda f: min(1.0, f["ais_gap_minutes"] / 120))
    norm_eta = (merged["eta_deviation_hours"].abs() / 24).clip(0, 1)
    norm_loiter = merged["_features"].apply(lambda f: 1.0 if f["is_loitering"] else 0.0)

    risk = (
        WEIGHTS["route_deviation"] * norm_route
        + WEIGHTS["anomaly_score"] * norm_anomaly
        + WEIGHTS["ais_gap"] * norm_ais_gap
        + WEIGHTS["eta_deviation"] * norm_eta
        + WEIGHTS["loitering"] * norm_loiter
    ) * 100

    merged["risk_score"] = risk.round(1)
    merged["risk_level"] = pd.cut(
        merged["risk_score"], bins=[-1, 35, 65, 101], labels=["LOW", "MEDIUM", "HIGH"]
    )
    merged["model_version"] = MODEL_VERSION
    return merged


def explain_risk(row: pd.Series) -> list[RiskFactorContribution]:
    factors = []
    if row["deviation_km"] and row["deviation_km"] > 5:
        factors.append(RiskFactorContribution(
            label="Route deviation",
            value=f"{row['deviation_km']:.0f} km off expected route",
            weight=WEIGHTS["route_deviation"] * min(1.0, row["deviation_km"] / 100),
        ))
    if row["anomaly_score"] > 40:
        factors.append(RiskFactorContribution(
            label="Elevated behavioral anomaly score",
            value=f"{row['anomaly_score']:.0f}/100",
            weight=WEIGHTS["anomaly_score"] * (row["anomaly_score"] / 100),
        ))
    gap = row["_features"]["ais_gap_minutes"]
    if gap > 30:
        factors.append(RiskFactorContribution(
            label="AIS transmission gap",
            value=f"{gap:.0f} minutes",
            weight=WEIGHTS["ais_gap"] * min(1.0, gap / 120),
        ))
    if abs(row["eta_deviation_hours"]) > 3:
        factors.append(RiskFactorContribution(
            label="ETA deviation",
            value=f"{row['eta_deviation_hours']:+.1f} hours vs schedule",
            weight=WEIGHTS["eta_deviation"] * min(1.0, abs(row["eta_deviation_hours"]) / 24),
        ))
    if row["_features"]["is_loitering"]:
        factors.append(RiskFactorContribution(
            label="Prolonged loitering",
            value="Near-zero speed while marked under way",
            weight=WEIGHTS["loitering"],
        ))
    if not factors:
        factors.append(RiskFactorContribution(
            label="No elevated risk indicators detected",
            value="Operating within normal parameters",
            weight=0.05,
        ))
    return sorted(factors, key=lambda f: -f.weight)
