"""
Port congestion / waiting-time intelligence.

Estimates expected waiting time and berth delay per port based on how many
vessels are currently converging on it, using a simple queueing heuristic.
A production version would use historical berth-occupancy and port-call
duration data.
"""
from __future__ import annotations

import pandas as pd

from data.simulator import PORTS

# Assumed average berths available per port (illustrative, prototype-only)
DEFAULT_BERTHS = 6
BASE_SERVICE_HOURS = 30  # avg time to unload/load a tanker-class vessel


def port_congestion_estimate(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate congestion for each destination port based on the number of
    vessels currently inbound (within the last leg of their voyage) plus
    vessels already at anchor/in port there.
    """
    rows = []
    for port in PORTS:
        name = port["name"]
        inbound = snapshot[
            (snapshot["destination"] == name) & (snapshot["voyage_progress_pct"] > 70)
        ]
        at_port_now = snapshot[
            (snapshot["destination"] == name)
            & (snapshot["nav_status"].isin(["At anchor", "Moored", "In port"]))
        ]
        vessels_converging = len(inbound)
        vessels_waiting = len(at_port_now)

        # simple queueing heuristic: waiting time grows once demand exceeds berths
        excess = max(0, vessels_waiting + vessels_converging - DEFAULT_BERTHS)
        expected_wait_hours = excess * (BASE_SERVICE_HOURS / max(1, DEFAULT_BERTHS))

        rows.append({
            "port": name,
            "country": port["country"],
            "vessels_converging": vessels_converging,
            "vessels_currently_waiting": vessels_waiting,
            "estimated_berths": DEFAULT_BERTHS,
            "expected_wait_hours": round(expected_wait_hours, 1),
            "congestion_level": (
                "HIGH" if expected_wait_hours > 24 else "MEDIUM" if expected_wait_hours > 6 else "LOW"
            ),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("expected_wait_hours", ascending=False).reset_index(drop=True)
