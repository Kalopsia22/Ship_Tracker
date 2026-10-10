"""Reference port list and the shared congestion-delay assumption."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Port:
    name: str
    lat: float
    lon: float
    capacity: int  # simplified: arrivals a port can absorb per 48 h window


PORTS: dict[str, Port] = {
    p.name: p
    for p in [
        Port("Rotterdam", 51.95, 4.14, 5),
        Port("Antwerp", 51.27, 4.35, 4),
        Port("Singapore", 1.26, 103.84, 6),
        Port("Fujairah", 25.12, 56.36, 3),
        Port("Ras Tanura", 26.64, 50.16, 3),
        Port("Houston", 29.73, -95.02, 4),
        Port("Corpus Christi", 27.81, -97.40, 3),
        Port("Shanghai", 31.23, 121.49, 6),
        Port("Ningbo-Zhoushan", 29.87, 121.95, 5),
        Port("Qingdao", 36.07, 120.32, 4),
        Port("Yokohama", 35.45, 139.65, 3),
        Port("Santos", -23.98, -46.30, 3),
        Port("Port Hedland", -20.31, 118.58, 4),
        Port("Durban", -29.87, 31.03, 3),
        Port("Mumbai (JNPT)", 18.95, 72.95, 3),
    ]
}
PORT_NAMES = list(PORTS)


def congestion_index(inbound_48h: float, capacity: float) -> float:
    return float(inbound_48h) / max(float(capacity), 1.0)


def congestion_delay_hours(idx):
    """Expected berth waiting time (h) for a congestion index.

    This is a *stated assumption*, used identically by the demo simulator (to
    generate ground truth) and by the Port Intelligence page (to report an
    estimate). On real data it is a heuristic that should be recalibrated
    against observed waiting times.
    """
    return 14.0 * np.maximum(np.asarray(idx, dtype=float) - 0.7, 0.0) ** 1.5
