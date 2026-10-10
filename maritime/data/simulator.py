"""Synthetic AIS simulator.

Used for demo mode and for generating *labelled* training/evaluation data.
Everything it produces is synthetic: metrics computed on its output measure
how well models recover the simulator's own rules, not real-world accuracy.
The README and the Governance page say so explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from maritime.data.schema import POSITION_COLUMNS, VESSEL_CLASSES, max_draught, service_speed
from maritime.geo import bearing_deg, destination_point, haversine_nm, interpolate_gc
from maritime.ports import PORT_NAMES, PORTS, congestion_delay_hours

EPOCH = pd.Timestamp("2024-01-01", tz="UTC")
ANOMALY_TYPES = ["loitering", "route_deviation", "ais_gap", "erratic_speed"]
_NAME_A = ["Atlantic", "Pacific", "Nordic", "Coral", "Orient", "Polar", "Golden", "Silver", "Eastern", "Ocean"]
_NAME_B = ["Voyager", "Pioneer", "Spirit", "Horizon", "Star", "Mariner", "Titan", "Venture", "Dawn", "Crest"]


@dataclass
class FleetSimulation:
    positions: pd.DataFrame
    truth: pd.DataFrame  # mmsi, anomaly_type ("none" or one of ANOMALY_TYPES), is_anomalous


def _pick_route(rng: np.random.Generator) -> tuple[str, str, float]:
    while True:
        o, d = rng.choice(PORT_NAMES, size=2, replace=False)
        dist = float(haversine_nm(PORTS[o].lat, PORTS[o].lon, PORTS[d].lat, PORTS[d].lon))
        if 1500 <= dist <= 9500:
            return str(o), str(d), dist


def simulate_fleet(
    n_vessels: int = 120,
    hours: int = 72,
    step_hours: float = 1.0,
    seed: int = 42,
    now: pd.Timestamp | None = None,
    anomaly_rate: float = 0.12,
) -> FleetSimulation:
    """Simulate `hours` of hourly AIS reports ending at `now`.

    Vessel parameters depend only on `seed`; vessel *progress* depends on the
    clock, so the demo fleet visibly moves between refreshes.
    """
    rng = np.random.default_rng(seed)
    if now is None:
        now = pd.Timestamp.now(tz="UTC").floor("5min")
    now = pd.Timestamp(now)
    elapsed_h = (now - EPOCH).total_seconds() / 3600.0
    class_names = list(VESSEL_CLASSES)

    rows, truth = [], []
    for i in range(n_vessels):
        mmsi = str(200_000_000 + int(rng.integers(0, 99_999_999)) + i)
        cls = str(rng.choice(class_names))
        speed_nom = service_speed(cls)
        origin, dest, dist_total = _pick_route(rng)
        o, d = PORTS[origin], PORTS[dest]
        p0 = float(rng.uniform(0, 1))
        p_now = (p0 + speed_nom * elapsed_h / dist_total) % 1.0
        draught = max_draught(cls) * float(rng.uniform(0.55, 1.0))
        name = f"{rng.choice(_NAME_A)} {rng.choice(_NAME_B)} {i % 97:02d}"
        anomaly = "none"
        if rng.random() < anomaly_rate:
            anomaly = str(rng.choice(ANOMALY_TYPES))
        dev_sign = float(rng.choice([-1, 1]))

        ts = [now - pd.Timedelta(hours=h) for h in np.arange(hours, -1e-9, -step_hours)]
        age_h = np.array([(now - t).total_seconds() / 3600 for t in ts])
        weather = np.clip(rng.normal(1.0, 0.05, len(ts)), 0.8, 1.08)
        sog = speed_nom * weather * (1 - 0.05 * draught / max_draught(cls)) + rng.normal(0, 0.3, len(ts))
        progress = p_now - (speed_nom * age_h) / dist_total  # fraction along route

        # --- inject anomalies on the most recent window -------------------
        offset_nm = np.zeros(len(ts))
        keep = np.ones(len(ts), dtype=bool)
        if anomaly == "loitering":
            m = age_h <= 12
            stall_p = p_now - (speed_nom * 12) / dist_total
            progress = np.where(m, stall_p, progress)
            sog = np.where(m, np.abs(rng.normal(0.6, 0.4, len(ts))), sog)
        elif anomaly == "route_deviation":
            m = age_h <= 24
            offset_nm = np.where(m, dev_sign * 130 * (1 - age_h / 24), 0.0)
        elif anomaly == "ais_gap":
            keep = ~((age_h < 14) & (age_h > 1))
        elif anomaly == "erratic_speed":
            m = age_h <= 18
            sog = np.where(m, rng.uniform(2, 21, len(ts)), sog)

        valid = (progress >= 0) & (progress <= 1) & keep
        f = np.clip(progress, 0, 1)
        lat, lon = interpolate_gc(o.lat, o.lon, d.lat, d.lon, f)
        brg = bearing_deg(lat, lon, d.lat, d.lon)
        if offset_nm.any():
            lat, lon = destination_point(lat, lon, (brg + 90) % 360, offset_nm)
        lat = lat + rng.normal(0, 0.002, len(ts))
        lon = lon + rng.normal(0, 0.002, len(ts))
        cog = (brg + rng.normal(0, 1.5, len(ts))) % 360
        sog = np.clip(sog, 0, 30)
        status = np.where(sog < 1.0, "At anchor", "Under way")

        v = pd.DataFrame({
            "mmsi": mmsi, "vessel_name": name, "vessel_class": cls, "timestamp": ts,
            "lat": lat, "lon": lon, "sog": sog, "cog": cog, "heading": cog,
            "draught": draught, "status": status, "origin": origin, "destination": dest,
        })[valid]
        rows.append(v)
        truth.append({"mmsi": mmsi, "anomaly_type": anomaly, "is_anomalous": anomaly != "none"})

    positions = pd.concat(rows, ignore_index=True)[POSITION_COLUMNS]
    return FleetSimulation(positions, pd.DataFrame(truth))


def simulate_voyage_snapshots(n_voyages: int = 3000, snapshots_per_voyage: int = 4, seed: int = 7) -> pd.DataFrame:
    """Labelled ETA training data: snapshot features + true hours remaining.

    The target includes (a) a per-route sea-vs-great-circle detour the model
    can only infer via distance/route features, (b) autocorrelated weather,
    and (c) berth waiting from `congestion_delay_hours` plus random delay.
    """
    rng = np.random.default_rng(seed)
    class_names = list(VESSEL_CLASSES)
    rows = []
    for vid in range(n_voyages):
        cls = str(rng.choice(class_names))
        spd = service_speed(cls)
        o_name, d_name, dist_total = _pick_route(rng)
        oi, di = PORT_NAMES.index(o_name), PORT_NAMES.index(d_name)
        detour = 1.0 + 0.2 * ((oi * 7 + di * 13) % 10) / 10 + rng.normal(0, 0.02)
        cong = float(np.clip(rng.beta(2, 3) * 1.8, 0, 2.0))
        delay_mean = float(congestion_delay_hours(cong))
        draught_ratio = float(rng.uniform(0.55, 1.0))
        for _ in range(snapshots_per_voyage):
            frac = float(rng.uniform(0.05, 0.95))
            rem = dist_total * (1 - frac)
            w_now = float(np.clip(rng.normal(1.0, 0.06), 0.8, 1.08))
            w_future = float(np.clip(0.6 * w_now + 0.4 * rng.normal(1.0, 0.06), 0.8, 1.08))
            draft_eff = 1 - 0.05 * draught_ratio
            sog_now = spd * w_now * draft_eff + rng.normal(0, 0.3)
            sog_6h = spd * (0.7 * w_now + 0.3) * draft_eff + rng.normal(0, 0.2)
            target = rem * detour / (spd * w_future * draft_eff) + delay_mean + float(rng.exponential(1.5))
            rows.append({
                "voyage_id": vid, "vessel_class": cls, "dist_total_nm": dist_total,
                "dist_remaining_nm": rem, "sog_now": max(sog_now, 0.5), "sog_mean_6h": max(sog_6h, 0.5),
                "draught_ratio": draught_ratio, "dest_congestion_idx": cong,
                "hour_utc": int(rng.integers(0, 24)), "target_hours": target,
            })
    return pd.DataFrame(rows)
