"""Behavioural anomaly detection (Isolation Forest) with a 0-1 normalised score."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from maritime.models.features import ANOMALY_FEATURES


@dataclass
class AnomalyModel:
    forest: IsolationForest
    lo: float  # raw-score calibration (from training distribution)
    hi: float

    def score(self, feats: pd.DataFrame) -> pd.DataFrame:
        raw = -self.forest.score_samples(feats[ANOMALY_FEATURES].fillna(0.0))
        norm = np.clip((raw - self.lo) / max(self.hi - self.lo, 1e-9), 0.0, 1.0)
        flag = self.forest.predict(feats[ANOMALY_FEATURES].fillna(0.0)) == -1
        return pd.DataFrame({"anomaly_score": norm, "is_anomaly": flag}, index=feats.index)


def train_anomaly(train_feats: pd.DataFrame, contamination: float = 0.1, seed: int = 0) -> AnomalyModel:
    X = train_feats[ANOMALY_FEATURES].fillna(0.0)
    forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed).fit(X)
    raw = -forest.score_samples(X)
    return AnomalyModel(forest, float(np.percentile(raw, 5)), float(np.percentile(raw, 99)))


def evaluate_anomaly(model: AnomalyModel, feats: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Evaluate against injected labels on a fleet the model never saw."""
    t = truth.set_index("mmsi").loc[feats.index]
    scored = model.score(feats)
    y = t["is_anomalous"].to_numpy(bool)
    per_type = {
        a: float(scored["is_anomaly"][t["anomaly_type"] == a].mean())
        for a in t["anomaly_type"].unique() if a != "none"
    }
    return {
        "roc_auc": float(roc_auc_score(y, scored["anomaly_score"])),
        "precision": float(precision_score(y, scored["is_anomaly"], zero_division=0)),
        "recall": float(recall_score(y, scored["is_anomaly"], zero_division=0)),
        "n_eval_vessels": int(len(y)), "n_eval_anomalous": int(y.sum()),
        "recall_by_type": per_type,
        "caveat": "Labels come from the simulator's injected anomalies; real-world performance is unknown.",
    }
