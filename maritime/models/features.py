"""Behavioural feature engineering. Inputs are kinematics + planned route only.

No flag state, owner, nationality or vessel name is ever used (see risk.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from maritime.data.schema import service_speed
from maritime.geo import cross_track_nm, haversine_nm
from maritime.ports import PORTS

ANOMALY_FEATURES = [
    "sog_mean", "sog_std", "sog_ratio", "recent_sog_ratio", "cog_turn_mean",
    "xtrack_max_nm", "xtrack_last_nm", "max_gap_h", "stationary_frac",
]
PORT_RADIUS_NM = 25.0  # slow/stopped near a known port is normal, not a signal


def _port_coords(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    lat = series.map(lambda n: PORTS[n].lat if n in PORTS else np.nan).to_numpy(float)
    lon = series.map(lambda n: PORTS[n].lon if n in PORTS else np.nan).to_numpy(float)
    return lat, lon


def latest_positions(positions: pd.DataFrame) -> pd.DataFrame:
    return positions.sort_values("timestamp").groupby("mmsi", as_index=False).tail(1).reset_index(drop=True)


def add_route_geometry(pos: pd.DataFrame) -> pd.DataFrame:
    """Adds xtrack_nm, dist_to_dest_nm, dist_to_origin_nm (NaN when route unknown)."""
    out = pos.copy()
    o_lat, o_lon = _port_coords(out["origin"])
    d_lat, d_lon = _port_coords(out["destination"])
    out["xtrack_nm"] = cross_track_nm(out["lat"].to_numpy(float), out["lon"].to_numpy(float), o_lat, o_lon, d_lat, d_lon)
    out["dist_to_dest_nm"] = haversine_nm(out["lat"].to_numpy(float), out["lon"].to_numpy(float), d_lat, d_lon)
    out["dist_to_origin_nm"] = haversine_nm(out["lat"].to_numpy(float), out["lon"].to_numpy(float), o_lat, o_lon)
    return out


def _wrapped_turns(cog: np.ndarray) -> np.ndarray:
    cog = cog[~np.isnan(cog)]
    if len(cog) < 2:
        return np.array([0.0])
    return np.abs((np.diff(cog) + 180) % 360 - 180)


def compute_vessel_features(positions: pd.DataFrame, now: pd.Timestamp | None = None, window_hours: float = 24.0) -> pd.DataFrame:
    """One row per vessel summarising the last `window_hours` of behaviour."""
    now = pd.Timestamp(now) if now is not None else positions["timestamp"].max()
    pos = add_route_geometry(positions[positions["timestamp"] > now - pd.Timedelta(hours=window_hours)])
    rows = []
    for mmsi, g in pos.sort_values("timestamp").groupby("mmsi"):
        spd = service_speed(str(g["vessel_class"].iloc[-1]))
        sog = g["sog"].astype(float).to_numpy()
        t = g["timestamp"]
        gaps = t.diff().dt.total_seconds().div(3600).dropna()
        since_last = (now - t.iloc[-1]).total_seconds() / 3600
        near_port = (g["dist_to_dest_nm"].fillna(1e9) < PORT_RADIUS_NM) | (g["dist_to_origin_nm"].fillna(1e9) < PORT_RADIUS_NM)
        at_sea = ~near_port.to_numpy()
        stopped = (sog < 1.0) & at_sea & (g["status"].astype(str) != "Moored").to_numpy()
        recent = sog[(now - t).dt.total_seconds().to_numpy() <= 6 * 3600]
        xt = g["xtrack_nm"].to_numpy(float)
        rows.append({
            "mmsi": mmsi,
            "n_reports": len(g),
            "sog_mean": float(np.nanmean(sog)) if len(sog) else 0.0,
            "sog_std": float(np.nanstd(sog)) if len(sog) else 0.0,
            "sog_ratio": float(np.nanmean(sog)) / spd if len(sog) else 0.0,
            "recent_sog_ratio": (float(np.nanmean(recent)) / spd) if len(recent) else 0.0,
            "cog_turn_mean": float(_wrapped_turns(g["cog"].to_numpy(float)).mean()),
            "xtrack_max_nm": float(np.nanmax(xt)) if np.isfinite(xt).any() else 0.0,
            "xtrack_last_nm": float(xt[-1]) if np.isfinite(xt[-1]) else 0.0,
            "route_known": bool(np.isfinite(xt).any()),
            "max_gap_h": float(max(gaps.max() if len(gaps) else 0.0, since_last)),
            "hours_since_last": float(since_last),
            "stationary_frac": float(stopped.mean()) if len(stopped) else 0.0,
        })
    return pd.DataFrame(rows).set_index("mmsi")
