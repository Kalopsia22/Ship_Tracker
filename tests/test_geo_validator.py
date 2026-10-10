import numpy as np
import pandas as pd
import pytest

from maritime.data.validator import validate_positions
from maritime.geo import bearing_deg, cross_track_nm, destination_point, haversine_nm, interpolate_gc


def test_haversine_known_distance():
    # London -> New York is about 2,999 nm
    assert haversine_nm(51.5, -0.12, 40.71, -74.0) == pytest.approx(2999, rel=0.01)


def test_interpolation_endpoints_and_midpoint_distance():
    lat, lon = interpolate_gc(0, 0, 0, 60, np.array([0, 0.5, 1]))
    assert lon[0] == pytest.approx(0, abs=1e-6) and lon[2] == pytest.approx(60, abs=1e-6)
    assert lon[1] == pytest.approx(30, abs=1e-6)


def test_cross_track_zero_on_route_and_matches_offset():
    la, lo = interpolate_gc(10, 0, 10, 50, np.array([0.4]))
    assert cross_track_nm(la, lo, 10, 0, 10, 50)[0] < 1.0
    brg = bearing_deg(la, lo, 10, 50)
    la2, lo2 = destination_point(la, lo, (brg + 90) % 360, 100.0)
    assert cross_track_nm(la2, lo2, 10, 0, 10, 50)[0] == pytest.approx(100, rel=0.03)


def _row(**kw):
    base = dict(mmsi="1", timestamp="2026-01-01T00:00:00Z", lat=10.0, lon=20.0, sog=12.0, cog=90.0)
    return {**base, **kw}


def test_validator_drops_each_invalid_class_and_scores():
    df = pd.DataFrame([
        _row(), _row(timestamp="2026-01-01T01:00:00Z"),
        _row(mmsi="2", lat=95.0), _row(mmsi="3", lat=0.0, lon=0.0), _row(mmsi="4", sog=99.0),
        _row(mmsi="5", lat=None), _row(),  # duplicate of first
    ])
    clean, rep = validate_positions(df)
    assert len(clean) == 2
    assert rep.dropped == {"missing_required_value": 1, "coordinates_out_of_range": 1, "null_island": 1,
                           "implausible_speed": 1, "duplicate_report": 1}
    assert rep.score == pytest.approx(100 * 2 / 7)


def test_validator_requires_core_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_positions(pd.DataFrame({"mmsi": ["1"]}))


def test_validator_does_not_mutate_input():
    df = pd.DataFrame([_row(lat=95.0)])
    before = df.copy()
    validate_positions(df)
    pd.testing.assert_frame_equal(df, before)
