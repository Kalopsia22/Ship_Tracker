"""
Anomaly Detection model.

Uses an Isolation Forest (unsupervised) over operational features — speed,
course variability, AIS reporting gaps, loitering indicators — to produce a
0-100 Anomaly Score per vessel, plus a human-readable explanation of which
features drove the score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from data.schemas import RiskFactorContribution
from utils.logging import get_logger

logger = get_logger(__name__)

MODEL_VERSION = "anomaly-isoforest-v0.2-prototype"

FEATURE_COLUMNS = [
    "speed_knots", "speed_deviation_from_class_norm", "ais_gap_minutes",
    "course_volatility", "is_loitering",
]

CLASS_NORMAL_SPEED = {
    "Panamax": 14.0, "Aframax": 13.0, "Suezmax": 13.0, "VLCC": 12.5, "ULCC": 12.0,
}


def _engineer_features(snapshot: pd.DataFrame) -> pd.DataFrame:
    df = snapshot.copy()
    class_norm = df["vessel_class"].map(CLASS_NORMAL_SPEED).fillna(13.0)
    df["speed_deviation_from_class_norm"] = (df["speed_knots"] - class_norm).abs()

    now = pd.Timestamp.utcnow().tz_localize(None)
    last_update = pd.to_datetime(df["last_ais_update"])
    df["ais_gap_minutes"] = (now - last_update).dt.total_seconds() / 60

    # Course volatility proxy: heading vs course divergence (real system
    # would use rolling std of heading over recent AIS reports)
    df["course_volatility"] = (df["heading_deg"] - df["course_deg"]).abs().clip(upper=180)

    df["is_loitering"] = ((df["speed_knots"] < 1.0) & (df["nav_status"] == "Under way using engine")).astype(int)
    return df


class AnomalyDetector:
    def __init__(self, contamination: float = 0.12):
        self.model = IsolationForest(
            n_estimators=150, contamination=contamination, random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, snapshot: pd.DataFrame):
        df = _engineer_features(snapshot)
        X = self.scaler.fit_transform(df[FEATURE_COLUMNS])
        self.model.fit(X)
        self.is_fitted = True
        logger.info("Anomaly model fitted on %d rows", len(df))

    def score(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        df = _engineer_features(snapshot)
        if not self.is_fitted:
            self.fit(snapshot)

        X = self.scaler.transform(df[FEATURE_COLUMNS])
        raw_scores = self.model.score_samples(X)  # higher = more normal
        # Rescale to 0-100 where 100 = most anomalous
        normalized = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
        anomaly_score = ((1 - normalized) * 100).round(1)

        is_anomalous = self.model.predict(X) == -1

        result = snapshot[["mmsi", "vessel_name", "vessel_class"]].copy()
        result["anomaly_score"] = anomaly_score
        result["is_anomalous"] = is_anomalous
        result["model_version"] = MODEL_VERSION
        result["_features"] = df[FEATURE_COLUMNS].to_dict("records")
        return result

    @staticmethod
    def explain(feature_row: dict) -> list[RiskFactorContribution]:
        factors = []
        if feature_row["speed_deviation_from_class_norm"] > 4:
            factors.append(RiskFactorContribution(
                label="Speed deviates from class-normal cruise speed",
                value=f"{feature_row['speed_deviation_from_class_norm']:.1f} kn deviation",
                weight=min(1.0, feature_row["speed_deviation_from_class_norm"] / 12),
            ))
        if feature_row["ais_gap_minutes"] > 30:
            factors.append(RiskFactorContribution(
                label="AIS transmission gap",
                value=f"{feature_row['ais_gap_minutes']:.0f} minutes since last report",
                weight=min(1.0, feature_row["ais_gap_minutes"] / 120),
            ))
        if feature_row["course_volatility"] > 20:
            factors.append(RiskFactorContribution(
                label="Heading/course divergence",
                value=f"{feature_row['course_volatility']:.0f}° divergence",
                weight=min(1.0, feature_row["course_volatility"] / 90),
            ))
        if feature_row["is_loitering"]:
            factors.append(RiskFactorContribution(
                label="Loitering behavior",
                value="Near-zero speed while marked under way",
                weight=0.8,
            ))
        if not factors:
            factors.append(RiskFactorContribution(
                label="No significant deviation from normal operating profile",
                value="Within expected range",
                weight=0.1,
            ))
        return factors
