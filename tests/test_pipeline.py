"""Basic sanity tests for the core data/model pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.simulator import VesselSimulator
from data.validator import validate_positions
from models.anomaly_model import AnomalyDetector
from models.eta_model import ETAPredictor
from models.risk_model import compute_risk
from analytics.route_analysis import compute_route_deviation
from analytics.port_analysis import port_congestion_estimate


def test_simulator_snapshot_shape():
    sim = VesselSimulator(fleet_size=20, seed=1)
    snap = sim.snapshot()
    assert len(snap) == 20
    assert {"mmsi", "latitude", "longitude", "speed_knots"}.issubset(snap.columns)
    assert snap["latitude"].between(-90, 90).all()
    assert snap["longitude"].between(-180, 180).all()


def test_validator_flags_bad_rows():
    sim = VesselSimulator(fleet_size=10, seed=2)
    snap = sim.snapshot()
    snap.loc[0, "latitude"] = 999  # inject a bad row
    clean, report = validate_positions(snap)
    assert report.invalid_coordinates >= 1


def test_eta_predictor_runs():
    sim = VesselSimulator(fleet_size=15, seed=3)
    snap = sim.snapshot()
    model = ETAPredictor()
    result = model.predict(snap)
    assert len(result) == 15
    assert "predicted_eta" in result.columns
    assert result["confidence"].between(0, 1).all()


def test_anomaly_detector_runs():
    sim = VesselSimulator(fleet_size=30, seed=4)
    snap = sim.snapshot()
    model = AnomalyDetector()
    result = model.score(snap)
    assert len(result) == 30
    assert result["anomaly_score"].between(0, 100).all()


def test_risk_model_end_to_end():
    sim = VesselSimulator(fleet_size=25, seed=5)
    snap = sim.snapshot()
    eta_model = ETAPredictor()
    anomaly_model = AnomalyDetector()
    eta_df = eta_model.predict(snap)
    anomaly_df = anomaly_model.score(snap)
    route_df = compute_route_deviation(sim, snap)
    risk_df = compute_risk(anomaly_df, eta_df, route_df)
    assert risk_df["risk_score"].between(0, 100).all()
    assert set(risk_df["risk_level"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})


def test_port_congestion_runs():
    sim = VesselSimulator(fleet_size=40, seed=6)
    snap = sim.snapshot()
    congestion = port_congestion_estimate(snap)
    assert not congestion.empty
    assert set(congestion["congestion_level"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})


if __name__ == "__main__":
    test_simulator_snapshot_shape()
    test_validator_flags_bad_rows()
    test_eta_predictor_runs()
    test_anomaly_detector_runs()
    test_risk_model_end_to_end()
    test_port_congestion_runs()
    print("All tests passed.")
