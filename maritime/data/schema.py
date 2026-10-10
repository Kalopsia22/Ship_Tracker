"""The canonical position-record contract every provider must satisfy."""
from __future__ import annotations

from dataclasses import dataclass

POSITION_COLUMNS = [
    "mmsi", "vessel_name", "vessel_class", "timestamp", "lat", "lon",
    "sog", "cog", "heading", "draught", "status", "origin", "destination",
]
REQUIRED_COLUMNS = ["mmsi", "timestamp", "lat", "lon"]


@dataclass(frozen=True)
class VesselClass:
    name: str
    service_speed_kn: float
    max_draught_m: float


VESSEL_CLASSES: dict[str, VesselClass] = {
    c.name: c
    for c in [
        VesselClass("VLCC", 14.5, 22.0),
        VesselClass("Suezmax", 14.0, 17.0),
        VesselClass("Aframax", 13.5, 15.0),
        VesselClass("Capesize", 13.0, 18.5),
        VesselClass("Panamax", 12.5, 14.0),
    ]
}
DEFAULT_SERVICE_SPEED = 13.0


def service_speed(vessel_class: str) -> float:
    vc = VESSEL_CLASSES.get(vessel_class)
    return vc.service_speed_kn if vc else DEFAULT_SERVICE_SPEED


def max_draught(vessel_class: str) -> float:
    vc = VESSEL_CLASSES.get(vessel_class)
    return vc.max_draught_m if vc else 15.0
