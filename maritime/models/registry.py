"""Versioned model artifacts with integrity checks and provenance metadata.

Models are trained offline (`scripts/train.py`) and loaded here. If no
artifacts exist (first run) a demo model set is trained once and persisted so
the app starts out of the box without refitting on every Streamlit rerun.

Security note: artifacts are joblib pickles. Only load files you produced; each
file's SHA-256 is recorded in metadata.json and verified before loading.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from maritime.data.simulator import simulate_fleet, simulate_voyage_snapshots
from maritime.models.anomaly import AnomalyModel, evaluate_anomaly, train_anomaly
from maritime.models.eta import EtaModel, train_eta
from maritime.models.features import ANOMALY_FEATURES, compute_vessel_features
from maritime.models.risk import DEFAULT_WEIGHTS, RiskModel

ETA_FILE, ANOMALY_FILE, META_FILE = "eta_model.joblib", "anomaly_model.joblib", "metadata.json"


class ArtifactIntegrityError(RuntimeError):
    pass


@dataclass
class ModelBundle:
    eta: EtaModel
    anomaly: AnomalyModel
    risk: RiskModel
    metadata: dict


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _anomaly_training_features(n_fleets: int, fleet_size: int, seed: int, now: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for k in range(n_fleets):
        sim = simulate_fleet(fleet_size, seed=seed + 1000 + k, now=now)
        f = compute_vessel_features(sim.positions, now=now)
        f.index = f"{k}-" + f.index
        parts.append(f)
    return pd.concat(parts)


def train_and_save(
    artifacts_dir: Path,
    anomaly_features: pd.DataFrame | None = None,
    anomaly_source: str = "synthetic simulator",
    n_voyages: int = 3000,
    seed: int = 0,
    risk_weights: dict | None = None,
) -> ModelBundle:
    """Train ETA (synthetic voyages) + anomaly model, evaluate, persist with metadata."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp("2025-01-01 12:00", tz="UTC")  # fixed: training must be reproducible

    eta, eta_metrics = train_eta(simulate_voyage_snapshots(n_voyages, seed=seed + 7), seed=seed)

    train_feats = anomaly_features if anomaly_features is not None else _anomaly_training_features(8, 120, seed, now)
    anomaly = train_anomaly(train_feats, seed=seed)
    holdout = simulate_fleet(300, seed=seed + 99_999, now=now)  # unseen synthetic fleet
    anomaly_metrics = evaluate_anomaly(anomaly, compute_vessel_features(holdout.positions, now=now), holdout.truth)

    joblib.dump(eta, artifacts_dir / ETA_FILE)
    joblib.dump(anomaly, artifacts_dir / ANOMALY_FILE)
    meta = {
        "version": datetime.now(timezone.utc).strftime("%Y.%m.%d-%H%M%S"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "seed": seed,
        "eta": {"algorithm": "RandomForestRegressor + split-conformal intervals", "data_source": "synthetic voyage simulator",
                "features": eta_feature_list(), "metrics": eta_metrics},
        "anomaly": {"algorithm": "IsolationForest", "data_source": anomaly_source, "features": ANOMALY_FEATURES,
                    "n_training_vessels": int(len(train_feats)), "metrics_on_synthetic_holdout": anomaly_metrics},
        "risk": {"type": "transparent weighted composite", "weights": risk_weights or DEFAULT_WEIGHTS},
        "sha256": {ETA_FILE: _sha256(artifacts_dir / ETA_FILE), ANOMALY_FILE: _sha256(artifacts_dir / ANOMALY_FILE)},
        "synthetic_data_notice": "ETA is always trained on synthetic voyages; reported metrics are NOT real-world accuracy.",
    }
    (artifacts_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    return ModelBundle(eta, anomaly, RiskModel(risk_weights), meta)


def eta_feature_list() -> list[str]:
    from maritime.models.eta import FEATURES
    return list(FEATURES)


def load_models(artifacts_dir: Path, risk_weights: dict | None = None) -> ModelBundle:
    meta_path = artifacts_dir / META_FILE
    meta = json.loads(meta_path.read_text())
    for fname, digest in meta["sha256"].items():
        if _sha256(artifacts_dir / fname) != digest:
            raise ArtifactIntegrityError(f"{fname} does not match recorded checksum; refusing to load")
    if meta.get("sklearn_version") != sklearn.__version__:
        raise ArtifactIntegrityError(
            f"artifacts built with scikit-learn {meta.get('sklearn_version')}, running {sklearn.__version__}; retrain"
        )
    return ModelBundle(joblib.load(artifacts_dir / ETA_FILE), joblib.load(artifacts_dir / ANOMALY_FILE), RiskModel(risk_weights), meta)


def load_or_train(artifacts_dir: Path, risk_weights: dict | None = None) -> ModelBundle:
    try:
        return load_models(artifacts_dir, risk_weights)
    except (FileNotFoundError, KeyError, ArtifactIntegrityError, json.JSONDecodeError):
        return train_and_save(artifacts_dir, risk_weights=risk_weights)
