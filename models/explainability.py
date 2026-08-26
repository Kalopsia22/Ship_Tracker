"""
Explainability helpers shared across models.

Provides a consistent way to render "why did the AI say this" panels,
including feature-importance extraction from tree-based models and
formatted natural-language explanations.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from data.schemas import RiskFactorContribution


def feature_importance_table(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract global feature importance from a fitted tree-based model."""
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def format_explanation(
    headline: str,
    level: str,
    factors: list[RiskFactorContribution],
) -> str:
    """Render a human-readable explanation block, e.g. for alert messages."""
    lines = [f"{headline}: {level}", "", "Primary contributing factors:"]
    for f in sorted(factors, key=lambda x: -x.weight)[:5]:
        lines.append(f"• {f.label}: {f.value}")
    return "\n".join(lines)


def confidence_band(confidence: float) -> str:
    if confidence >= 0.8:
        return "High"
    if confidence >= 0.55:
        return "Moderate"
    return "Low"


MODEL_CARD = {
    "eta-rf-v0.3-prototype": {
        "model_type": "Random Forest Regressor",
        "training_data": "Synthetic voyage-duration data derived from simulated fleet speed profiles (prototype). In production: historical AIS voyage completions.",
        "known_limitations": [
            "Does not yet incorporate live weather or sea-state data",
            "Port congestion is a static per-class factor, not a live queue model",
            "Confidence intervals are approximated from tree-prediction variance",
        ],
    },
    "anomaly-isoforest-v0.2-prototype": {
        "model_type": "Isolation Forest (unsupervised)",
        "training_data": "Fitted on the current fleet snapshot's operational features (prototype). In production: rolling window of historical fleet behavior.",
        "known_limitations": [
            "Contamination rate (expected anomaly fraction) is a fixed assumption (12%)",
            "No distinction yet between benign anomalies (e.g. weather routing) and operationally significant ones",
        ],
    },
    "risk-composite-v0.2-prototype": {
        "model_type": "Explainable weighted composite (rules-based)",
        "training_data": "Not a learned model — weights are analyst-defined and documented for transparency.",
        "known_limitations": [
            "Weights are illustrative and should be calibrated against real incident data before production use",
            "Deliberately excludes flag/country/ownership to avoid proxy discrimination — see Responsible AI page",
        ],
    },
}
