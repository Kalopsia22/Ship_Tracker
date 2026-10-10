"""Operational Risk Indicator: a transparent weighted composite.

Every component is a documented 0-1 normalisation of a behavioural signal;
the score is the weighted sum x 100. No learned weights, no hidden inputs.
Flag state, ownership, nationality and vessel name are NOT inputs, and
`FORBIDDEN_INPUTS` is enforced in code and tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORBIDDEN_INPUTS = {"flag", "flag_state", "country", "nationality", "owner", "operator", "vessel_name", "mmsi"}

DEFAULT_WEIGHTS = {"behaviour": 0.35, "route": 0.25, "comms": 0.20, "speed": 0.10, "stall": 0.10}
# component -> (input column, value at which the component saturates at 1.0)
COMPONENTS = {
    "behaviour": ("anomaly_score", 1.0),
    "route": ("xtrack_last_nm", 100.0),
    "comms": ("max_gap_h", 12.0),
    "speed": ("sog_std", 4.0),
    "stall": ("stationary_frac", 0.5),
}
LABELS = {
    "behaviour": "Unusual overall behaviour (ML)", "route": "Distance off planned route",
    "comms": "AIS reporting gap", "speed": "Speed instability", "stall": "Stopped at sea",
}
# Thresholds were set from the synthetic score distribution (normal fleet max ~25); recalibrate on real data.
LEVELS = [(50, "Critical"), (30, "High"), (15, "Medium"), (0, "Low")]

assert not ({c for c, _ in COMPONENTS.values()} & FORBIDDEN_INPUTS), "forbidden risk input"


class RiskModel:
    def __init__(self, weights: dict | None = None):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        if set(self.weights) != set(COMPONENTS):
            raise ValueError(f"weights must define exactly {sorted(COMPONENTS)}")
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("risk weights must sum to 1.0")

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-component weighted points, risk_score (0-100), level, top factor."""
        out = pd.DataFrame(index=df.index)
        for name, (col, sat) in COMPONENTS.items():
            norm = np.clip(df[col].fillna(0.0).to_numpy(float) / sat, 0, 1)
            out[f"pts_{name}"] = 100.0 * self.weights[name] * norm
        pts = out[[f"pts_{n}" for n in COMPONENTS]]
        out["risk_score"] = pts.sum(axis=1)
        out["risk_level"] = out["risk_score"].map(level_for)
        top = pts.idxmax(axis=1).str.removeprefix("pts_")
        out["top_factor"] = np.where(pts.max(axis=1) > 1.0, top.map(LABELS), "None significant")
        return out

    def explain(self, row: pd.Series) -> pd.DataFrame:
        """Contribution table for one vessel (row from `score` + inputs)."""
        recs = [{"factor": LABELS[n], "input": COMPONENTS[n][0], "value": float(row.get(COMPONENTS[n][0], 0) or 0),
                 "saturates_at": COMPONENTS[n][1], "weight": self.weights[n], "points": float(row[f"pts_{n}"])}
                for n in COMPONENTS]
        return pd.DataFrame(recs).sort_values("points", ascending=False).reset_index(drop=True)


def level_for(score: float) -> str:
    return next(label for floor, label in LEVELS if score >= floor)
