import json

import pandas as pd
import pytest

from maritime.config import Settings
from maritime.data.providers import CsvProvider, DemoProvider, HttpJsonProvider, ProviderError, build_provider
from maritime.data.schema import POSITION_COLUMNS
from maritime.data.simulator import simulate_fleet, simulate_voyage_snapshots
from maritime.data.validator import validate_positions


def test_simulator_is_deterministic_for_seed_and_clock(now):
    a = simulate_fleet(30, seed=5, now=now).positions
    b = simulate_fleet(30, seed=5, now=now).positions
    pd.testing.assert_frame_equal(a, b)


def test_simulator_schema_truth_and_clean(now):
    sim = simulate_fleet(100, seed=1, now=now, anomaly_rate=0.2)
    assert list(sim.positions.columns) == POSITION_COLUMNS
    assert 5 <= sim.truth["is_anomalous"].sum() <= 40
    _, rep = validate_positions(sim.positions)
    assert rep.score == 100.0


def test_vessels_move_with_the_clock(now):
    a = simulate_fleet(20, seed=3, now=now).positions.groupby("mmsi").tail(1).set_index("mmsi")
    b = simulate_fleet(20, seed=3, now=now + pd.Timedelta(hours=6)).positions.groupby("mmsi").tail(1).set_index("mmsi")
    common = a.index.intersection(b.index)
    assert ((a.loc[common, "lat"] - b.loc[common, "lat"]).abs() + (a.loc[common, "lon"] - b.loc[common, "lon"]).abs()).median() > 0.1


def test_voyage_snapshots_have_expected_shape():
    v = simulate_voyage_snapshots(50, 3, seed=1)
    assert len(v) == 150 and v["target_hours"].min() > 0 and v["voyage_id"].nunique() == 50


def test_csv_provider_maps_noaa_columns(tmp_path):
    p = tmp_path / "ais.csv"
    pd.DataFrame({"MMSI": [1, 1], "BaseDateTime": ["2026-01-01T00:00:00", "2026-01-01T01:00:00"], "LAT": [10, 10.1], "LON": [20, 20.1],
                  "SOG": [10, 11], "COG": [90, 91], "Heading": [90, 91], "VesselName": ["A", "A"], "Status": [0, 0], "Draft": [9, 9], "VesselType": [80, 80]}).to_csv(p, index=False)
    df = CsvProvider(str(p)).positions()
    assert set(["mmsi", "timestamp", "lat", "lon", "vessel_class"]).issubset(df.columns) and (df["vessel_class"] == "Tanker").all()


def test_csv_provider_missing_file_is_explicit_error():
    with pytest.raises(ProviderError, match="not found"):
        CsvProvider("/nope.csv").positions()


class _Resp:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): pass
    def json(self): return self.payload


def test_http_provider_maps_fields_and_result_path(monkeypatch):
    import requests
    payload = {"data": {"ships": [{"id": "9", "t": "2026-01-01T00:00:00Z", "la": 1.0, "lo": 2.0, "kn": 5.0}]}}
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(payload))
    p = HttpJsonProvider("http://x", "k", {"id": "mmsi", "t": "timestamp", "la": "lat", "lo": "lon", "kn": "sog"}, "data.ships")
    df = p.positions()
    assert df.loc[0, "mmsi"] == "9" and df.loc[0, "sog"] == 5.0


def test_http_provider_errors_are_not_swallowed(monkeypatch):
    import requests

    def boom(*a, **k): raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(ProviderError, match="failed"):
        HttpJsonProvider("http://x", None, {"a": "mmsi"}).positions()
    with pytest.raises(ProviderError):
        HttpJsonProvider("", None, {})


def test_build_provider_modes():
    assert isinstance(build_provider(Settings(data_mode="demo")), DemoProvider)
    assert isinstance(build_provider(Settings(data_mode="csv", csv_path="x.csv")), CsvProvider)
    s = Settings(data_mode="live", api_url="http://x", api_field_map_json=json.dumps({"a": "mmsi"}))
    assert isinstance(build_provider(s), HttpJsonProvider)
