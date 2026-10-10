"""ETA model: Random Forest + split-conformal intervals + naive baseline.

Honest evaluation: chronological split *by voyage* (no snapshot leakage),
a naive distance/speed baseline for comparison, and empirical coverage of the
stated interval on held-out voyages.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from maritime.data.schema import VESSEL_CLASSES

NUMERIC = ["dist_remaining_nm", "dist_total_nm", "sog_now", "sog_mean_6h", "draught_ratio", "dest_congestion_idx", "hour_utc"]
CATEGORICAL = ["vessel_class"]
FEATURES = NUMERIC + CATEGORICAL
INTERVAL_LEVEL = 0.80


def naive_eta_hours(df: pd.DataFrame) -> np.ndarray:
    """Baseline: remaining great-circle distance at current speed."""
    return df["dist_remaining_nm"].to_numpy(float) / np.maximum(df["sog_now"].to_numpy(float), 1.0)


@dataclass
class EtaModel:
    pipeline: Pipeline
    rel_q: float  # conformal quantile of |y - yhat| / max(yhat, 1)
    level: float = INTERVAL_LEVEL

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        yhat = self.pipeline.predict(X[FEATURES])
        half = self.rel_q * np.maximum(yhat, 1.0)
        return pd.DataFrame({"eta_hours": yhat, "eta_low_h": np.maximum(yhat - half, 0), "eta_high_h": yhat + half}, index=X.index)

    def feature_importance(self) -> pd.Series:
        pre, rf = self.pipeline.named_steps["pre"], self.pipeline.named_steps["rf"]
        return pd.Series(rf.feature_importances_, index=pre.get_feature_names_out()).sort_values(ascending=False)


def _build_pipeline(seed: int) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(categories=[list(VESSEL_CLASSES) + ["Tanker", "Cargo", "Other"]], handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    rf = RandomForestRegressor(n_estimators=120, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    return Pipeline([("pre", pre), ("rf", rf)])


def train_eta(snapshots: pd.DataFrame, seed: int = 0) -> tuple[EtaModel, dict]:
    voyages = np.sort(snapshots["voyage_id"].unique())
    n = len(voyages)
    train_v, cal_v, test_v = voyages[: int(.6 * n)], voyages[int(.6 * n): int(.8 * n)], voyages[int(.8 * n):]
    tr, ca, te = (snapshots[snapshots["voyage_id"].isin(v)] for v in (train_v, cal_v, test_v))

    pipe = _build_pipeline(seed).fit(tr[FEATURES], tr["target_hours"])

    # split-conformal calibration on a held-out calibration set
    cal_pred = pipe.predict(ca[FEATURES])
    scores = np.abs(ca["target_hours"].to_numpy() - cal_pred) / np.maximum(cal_pred, 1.0)
    k = int(np.ceil((len(scores) + 1) * INTERVAL_LEVEL))
    rel_q = float(np.sort(scores)[min(k, len(scores)) - 1])
    model = EtaModel(pipe, rel_q)

    pred = model.predict(te)
    y = te["target_hours"].to_numpy()
    base = naive_eta_hours(te)
    mae, base_mae = float(np.mean(np.abs(y - pred["eta_hours"]))), float(np.mean(np.abs(y - base)))
    per_class = {
        c: float(np.mean(np.abs(y[m] - pred["eta_hours"].to_numpy()[m])))
        for c in te["vessel_class"].unique() if (m := (te["vessel_class"] == c).to_numpy()).sum() >= 20
    }
    metrics = {
        "n_train_snapshots": int(len(tr)), "n_calibration_snapshots": int(len(ca)), "n_test_snapshots": int(len(te)),
        "test_mae_hours": mae, "baseline_mae_hours": base_mae,
        "improvement_vs_baseline_pct": 100 * (1 - mae / base_mae),
        "interval_level": INTERVAL_LEVEL,
        "interval_empirical_coverage": float(np.mean((y >= pred["eta_low_h"]) & (y <= pred["eta_high_h"]))),
        "mae_by_class_hours": per_class,
        "split": "chronological by voyage: 60% train / 20% calibration / 20% test",
    }
    return model, metrics
