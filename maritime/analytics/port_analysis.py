"""Port congestion estimate from inbound traffic (transparent heuristic)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from maritime.ports import PORTS, congestion_delay_hours, congestion_index


def naive_arrival_hours(dist_nm: pd.Series, sog: pd.Series) -> pd.Series:
    return dist_nm / np.maximum(sog.astype(float), 3.0)


def port_congestion(vessels: pd.DataFrame) -> pd.DataFrame:
    """`vessels` needs: destination, dist_to_dest_nm, sog (naive ETA avoids circularity with the ETA model)."""
    eta = naive_arrival_hours(vessels["dist_to_dest_nm"], vessels["sog"])
    rows = []
    for name, port in PORTS.items():
        m = (vessels["destination"] == name).to_numpy()
        e = eta[m]
        n48 = int((e <= 48).sum())
        idx = congestion_index(n48, port.capacity)
        rows.append({
            "port": name, "lat": port.lat, "lon": port.lon, "capacity": port.capacity,
            "inbound_total": int(m.sum()), "inbound_24h": int((e <= 24).sum()), "inbound_48h": n48,
            "inbound_72h": int((e <= 72).sum()), "congestion_idx": idx,
            "expected_wait_h": float(congestion_delay_hours(idx)),
        })
    return pd.DataFrame(rows).sort_values("congestion_idx", ascending=False).reset_index(drop=True)
