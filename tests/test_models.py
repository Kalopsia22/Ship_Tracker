import json

import pandas as pd
import pytest

from maritime.data.simulator import simulate_fleet
from maritime.models import risk as risk_mod
from maritime.models.anomaly import evaluate_anomaly
from maritime.models.eta import FEATURES
from maritime.models.features import ANOMALY_FEATURES, compute_vessel_features
from maritime.models.registry import ArtifactIntegrityError, load_models
from maritime.models.risk import COMPONENTS, FORBIDDEN_INPUTS, RiskModel


def test_eta_beats_naive_baseline_and_interval_is_calibrated(bundle):
    m = bundle.metadata["eta"]["metrics"]
    assert m["test_mae_hours"] < 0.75 * m["baseline_mae_hours"]
    assert 0.68 <= m["interval_empirical_coverage"] <= 0.92  # nominal 0.80


def test_eta_prediction_is_ordered_and_positive(bundle):
    X = pd.DataFrame([{"vessel_class": "VLCC", "dist_remaining_nm": 3000, "dist_total_nm": 6000, "sog_now": 14, "sog_mean_6h": 14,
                       "draught_ratio": 0.8, "dest_congestion_idx": 0.5, "hour_utc": 6}])
    p = bundle.eta.predict(X).iloc[0]
    assert 0 <= p.eta_low_h <= p.eta_hours <= p.eta_high_h
    assert set(FEATURES) == set(X.columns)


def test_anomaly_model_generalises_to_unseen_fleet(bundle, now):
    sim = simulate_fleet(250, seed=424242, now=now)
    res = evaluate_anomaly(bundle.anomaly, compute_vessel_features(sim.positions, now=now), sim.truth)
    assert res["roc_auc"] > 0.9 and res["precision"] > 0.6


def test_anomaly_scores_are_bounded(bundle, now):
    f = compute_vessel_features(simulate_fleet(40, seed=2, now=now).positions, now=now)
    s = bundle.anomaly.score(f)
    assert s["anomaly_score"].between(0, 1).all()


def test_no_forbidden_inputs_anywhere():
    used = {c for c, _ in COMPONENTS.values()} | set(ANOMALY_FEATURES) | set(FEATURES)
    assert not (used & FORBIDDEN_INPUTS)
    assert "flag" not in " ".join(sorted(used)).lower()


def test_risk_is_monotonic_bounded_and_explainable():
    rm = RiskModel()
    calm = {"anomaly_score": 0, "xtrack_last_nm": 0, "max_gap_h": 0, "sog_std": 0, "stationary_frac": 0}
    bad = {"anomaly_score": 1, "xtrack_last_nm": 500, "max_gap_h": 99, "sog_std": 50, "stationary_frac": 1}
    out = rm.score(pd.DataFrame([calm, bad]))
    assert out.loc[0, "risk_score"] == 0 and out.loc[1, "risk_score"] == pytest.approx(100)
    assert out.loc[0, "risk_level"] == "Low" and out.loc[1, "risk_level"] == "Critical"
    row = pd.concat([pd.Series(bad), out.loc[1]])
    assert rm.explain(row)["points"].sum() == pytest.approx(out.loc[1, "risk_score"])


def test_risk_weights_validated():
    with pytest.raises(ValueError, match="sum to 1"):
        RiskModel({**risk_mod.DEFAULT_WEIGHTS, "route": 0.9})
    with pytest.raises(ValueError, match="exactly"):
        RiskModel({"route": 1.0})


def test_registry_round_trip_and_metadata(bundle, tmp_path):
    from maritime.models.registry import train_and_save
    b = train_and_save(tmp_path, n_voyages=600, seed=1)
    loaded = load_models(tmp_path)
    assert loaded.metadata["version"] == b.metadata["version"]
    assert "synthetic_data_notice" in loaded.metadata


def test_registry_detects_tampering(tmp_path):
    from maritime.models.registry import ANOMALY_FILE, train_and_save
    train_and_save(tmp_path, n_voyages=600, seed=1)
    (tmp_path / ANOMALY_FILE).write_bytes((tmp_path / ANOMALY_FILE).read_bytes() + b"x")
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        load_models(tmp_path)


def test_registry_rejects_sklearn_version_mismatch(tmp_path):
    from maritime.models.registry import META_FILE, train_and_save
    train_and_save(tmp_path, n_voyages=600, seed=1)
    meta = json.loads((tmp_path / META_FILE).read_text()); meta["sklearn_version"] = "0.0.1"
    (tmp_path / META_FILE).write_text(json.dumps(meta))
    with pytest.raises(ArtifactIntegrityError, match="retrain"):
        load_models(tmp_path)
