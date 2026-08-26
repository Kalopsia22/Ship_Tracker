"""
Route deviation analysis.

Compares each vessel's actual position against the great-circle route
between its origin and declared destination, producing a deviation
distance and percentage.
"""
from __future__ import annotations

import math

import pandas as pd

from data.simulator import haversine_km, initial_bearing_deg, destination_point

DEFAULT_DEVIATION_THRESHOLD_KM = 50.0


def _cross_track_distance_km(origin, dest, point) -> float:
    """
    Cross-track distance of `point` from the great-circle path origin->dest,
    using the standard spherical cross-track-distance formula.
    """
    r = 6371.0
    d13 = haversine_km(origin["lat"], origin["lon"], point["lat"], point["lon"]) / r
    brng13 = math.radians(initial_bearing_deg(origin["lat"], origin["lon"], point["lat"], point["lon"]))
    brng12 = math.radians(initial_bearing_deg(origin["lat"], origin["lon"], dest["lat"], dest["lon"]))
    xt = math.asin(math.sin(d13) * math.sin(brng13 - brng12)) * r
    return abs(xt)


def compute_route_deviation(simulator, snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    For each vessel in the snapshot, compute how far its current position
    lies from the expected great-circle route between origin and
    destination.
    """
    rows = []
    for _, row in snapshot.iterrows():
        vessel = simulator.fleet.get(row["mmsi"])
        if vessel is None:
            continue
        origin = vessel["origin"]
        dest = vessel["destination_port"]
        point = {"lat": row["latitude"], "lon": row["longitude"]}
        deviation_km = _cross_track_distance_km(origin, dest, point)
        total_km = vessel["total_km"]
        deviation_pct = round(100 * deviation_km / total_km, 2) if total_km else 0.0
        rows.append({
            "mmsi": row["mmsi"],
            "vessel_name": row["vessel_name"],
            "origin": origin["name"],
            "destination": dest["name"],
            "deviation_km": round(deviation_km, 1),
            "deviation_pct": deviation_pct,
            "expected_route_km": round(total_km, 1),
        })
    return pd.DataFrame(rows)
