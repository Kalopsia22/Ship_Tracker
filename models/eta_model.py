"""
ETA Prediction model.

A lightweight, explainable regression model that predicts arrival time
based on current position, speed, distance remaining, vessel class, and a
simulated congestion factor. Ships with a heuristic fallback (great-circle
distance / current speed) so the app works even before a model has been
"trained", and a simple RandomForest option to demonstrate the ML pathway.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data.schemas import RiskFactorContribution
from utils.logging import get_logger

logger = get_logger(__name__)

MODEL_VERSION = "eta-rf-v0.3-prototype"

VESSEL_CLASS_CONGESTION_FACTOR = {
    "Panamax": 1.02, "Aframax": 1.03, "Suezmax": 1.05, "VLCC": 1.08, "ULCC": 1.10,
}


class ETAPredictor:
    """
    Predicts arrival time. In this prototype the model is trained on-the-fly
    from the simulated fleet's own historical speed profile (a stand-in for
    a real model trained on months of historical voyages).
    """

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42)
        self.is_fitted = False

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        class_dummies = pd.get_dummies(df["vessel_class"], prefix="class")
        feats = pd.DataFrame({
            "remaining_km": df["remaining_km"],
            "speed_knots": df["speed_knots"].replace(0, 0.1),
            "draught_m": df["draught_m"],
            "voyage_progress_pct": df["voyage_progress_pct"],
        })
        return pd.concat([feats, class_dummies], axis=1)

    def fit_on_synthetic_history(self, snapshot: pd.DataFrame):
        """
        Train against a synthetically-labeled target (actual hours-to-arrival
        implied by remaining distance / cruise speed with noise), simulating
        what a real historical-voyage-duration dataset would provide.
        """
        df = snapshot.copy()
        rng = np.random.default_rng(7)
        congestion = df["vessel_class"].map(VESSEL_CLASS_CONGESTION_FACTOR).fillna(1.0)
        true_hours = (df["remaining_km"] / df["speed_knots"].replace(0, 0.1) / 1.852) * congestion
        true_hours += rng.normal(0, 2.5, size=len(df))
        true_hours = true_hours.clip(lower=0.5)

        X = self._features(df)
        self.model.fit(X, true_hours)
        self.is_fitted = True
        logger.info("ETA model fitted on %d synthetic training rows", len(df))

    def predict(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            self.fit_on_synthetic_history(snapshot)

        X = self._features(snapshot)
        pred_hours = self.model.predict(X)

        # Confidence: derived from tree-prediction variance (lower spread = higher confidence)
        X_values = X.values
        tree_preds = np.stack([t.predict(X_values) for t in self.model.estimators_])
        std = tree_preds.std(axis=0)
        confidence = np.clip(1 - (std / (pred_hours + 1e-6)), 0.35, 0.97)

        now = pd.Timestamp.utcnow().tz_localize(None)
        predicted_eta = now + pd.to_timedelta(pred_hours, unit="h")

        result = snapshot[["mmsi", "vessel_name", "eta_scheduled"]].copy()
        result["predicted_eta"] = predicted_eta
        result["predicted_hours_remaining"] = pred_hours.round(1)
        result["confidence"] = confidence.round(2)

        scheduled = pd.to_datetime(result["eta_scheduled"])
        deviation = (result["predicted_eta"] - scheduled).dt.total_seconds() / 3600
        result["eta_deviation_hours"] = deviation.round(1)
        result["model_version"] = MODEL_VERSION
        return result

    @staticmethod
    def explain(row: pd.Series) -> list[RiskFactorContribution]:
        factors = []
        if pd.notna(row.get("eta_deviation_hours")):
            factors.append(RiskFactorContribution(
                label="ETA deviation from schedule",
                value=f"{row['eta_deviation_hours']:+.1f} hours",
                weight=min(1.0, abs(row["eta_deviation_hours"]) / 24),
            ))
        return factors
