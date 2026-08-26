"""
Synthetic AIS data generator (DEMO MODE).

Generates a realistic fleet of tanker/bulker-class vessels with plausible
routes between major oceanic ports, historical trajectories, and a small
number of intentionally injected anomalies (course changes, AIS gaps,
loitering, speed drops) so the AI/ML layer has something to detect.

All data produced here is clearly synthetic and is labeled DataSource.DEMO
everywhere it is used downstream.
"""
from __future__ import annotations

import math
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Reference data: major ports and typical vessel-class speed/size envelopes
# ---------------------------------------------------------------------------

PORTS = [
    {"name": "Rotterdam", "country": "Netherlands", "lat": 51.95, "lon": 4.14},
    {"name": "Singapore", "country": "Singapore", "lat": 1.29, "lon": 103.85},
    {"name": "Houston", "country": "USA", "lat": 29.75, "lon": -95.10},
    {"name": "Fujairah", "country": "UAE", "lat": 25.11, "lon": 56.34},
    {"name": "Ningbo-Zhoushan", "country": "China", "lat": 29.87, "lon": 121.90},
    {"name": "Ras Tanura", "country": "Saudi Arabia", "lat": 26.64, "lon": 50.16},
    {"name": "Santos", "country": "Brazil", "lat": -23.96, "lon": -46.33},
    {"name": "Yokohama", "country": "Japan", "lat": 35.44, "lon": 139.64},
    {"name": "Sullom Voe", "country": "UK", "lat": 60.48, "lon": -1.31},
    {"name": "Panama Canal (Balboa)", "country": "Panama", "lat": 8.95, "lon": -79.57},
    {"name": "Suez", "country": "Egypt", "lat": 29.97, "lon": 32.55},
    {"name": "Durban", "country": "South Africa", "lat": -29.87, "lon": 31.04},
    {"name": "Galveston", "country": "USA", "lat": 29.30, "lon": -94.80},
    {"name": "Ulsan", "country": "South Korea", "lat": 35.50, "lon": 129.38},
    {"name": "Sikka", "country": "India", "lat": 22.43, "lon": 69.83},
]

VESSEL_CLASS_PROFILES = {
    "Panamax": {"loa": (225, 275), "beam": (32, 32.3), "draught": (12, 14.5), "cruise_speed": (13, 15)},
    "Aframax": {"loa": (230, 250), "beam": (42, 44), "draught": (13, 15), "cruise_speed": (12, 14.5)},
    "Suezmax": {"loa": (270, 285), "beam": (46, 50), "draught": (16, 17.5), "cruise_speed": (12, 14)},
    "VLCC": {"loa": (300, 333), "beam": (58, 60), "draught": (20, 22.5), "cruise_speed": (11, 13.5)},
    "ULCC": {"loa": (350, 415), "beam": (63, 68), "draught": (22, 24.5), "cruise_speed": (10, 13)},
}

CARGO_BY_CLASS = {
    "Panamax": ["Grain", "Coal", "Refined Products"],
    "Aframax": ["Crude Oil", "Refined Products"],
    "Suezmax": ["Crude Oil"],
    "VLCC": ["Crude Oil"],
    "ULCC": ["Crude Oil"],
}

FLAGS = [
    "Panama", "Marshall Islands", "Liberia", "Singapore", "Malta",
    "Hong Kong", "Greece", "Japan", "China", "South Korea", "Bahamas",
]

NAV_STATUSES = [
    "Under way using engine", "At anchor", "Moored", "In port",
    "Restricted manoeuvrability",
]


def _random_vessel_name(rng: random.Random) -> str:
    prefixes = ["Ocean", "Atlantic", "Pacific", "Star", "Global", "Horizon",
                "Nordic", "Sovereign", "Endeavour", "Meridian", "Crest", "Voyager"]
    suffixes = ["Pioneer", "Spirit", "Trader", "Navigator", "Falcon",
                "Explorer", "Legacy", "Odyssey", "Venture", "Prestige"]
    return f"{rng.choice(prefixes)} {rng.choice(suffixes)}"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def initial_bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def destination_point(lat, lon, bearing_deg, distance_km):
    r = 6371.0
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_km / r)
        + math.cos(lat1) * math.sin(distance_km / r) * math.cos(br)
    )
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(distance_km / r) * math.cos(lat1),
        math.cos(distance_km / r) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


class VesselSimulator:
    """
    Maintains a fleet of simulated vessels, each on a great-circle route
    between two ports, and can advance the fleet state through time while
    injecting occasional anomalies.
    """

    def __init__(self, fleet_size: int = 120, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.fleet_size = fleet_size
        self.fleet: dict[str, dict] = {}
        self._init_fleet()

    # -- fleet initialization ------------------------------------------------

    def _init_fleet(self):
        classes = list(VESSEL_CLASS_PROFILES.keys())
        for i in range(self.fleet_size):
            vessel_class = classes[i % len(classes)]  # even distribution across the 5 classes
            profile = VESSEL_CLASS_PROFILES[vessel_class]
            origin, dest = self.rng.sample(PORTS, 2)

            total_km = haversine_km(origin["lat"], origin["lon"], dest["lat"], dest["lon"])
            cruise_speed = self.rng.uniform(*profile["cruise_speed"])
            progress_km = self.rng.uniform(0, total_km)

            mmsi = f"{self.rng.randint(200, 775)}{self.rng.randint(100000, 999999)}"
            imo = f"IMO{self.rng.randint(9100000, 9999999)}"

            is_anomalous = self.rng.random() < 0.12  # ~12% of fleet has an active anomaly
            anomaly_type = self.rng.choice(
                ["speed_drop", "course_deviation", "ais_gap", "loitering"]
            ) if is_anomalous else None

            vessel = {
                "mmsi": mmsi,
                "imo": imo,
                "vessel_name": _random_vessel_name(self.rng),
                "vessel_class": vessel_class,
                "flag": self.rng.choice(FLAGS),
                "origin": origin,
                "destination_port": dest,
                "total_km": total_km,
                "progress_km": progress_km,
                "cruise_speed": cruise_speed,
                "current_speed": cruise_speed,
                "nav_status": "Under way using engine",
                "draught_m": round(self.rng.uniform(*profile["draught"]), 1),
                "length_m": round(self.rng.uniform(*profile["loa"]), 1),
                "width_m": round(self.rng.uniform(*profile["beam"]), 1),
                "cargo_category": self.rng.choice(CARGO_BY_CLASS[vessel_class]),
                "eta_scheduled": datetime.utcnow()
                + timedelta(hours=(total_km - progress_km) / cruise_speed + self.rng.uniform(-6, 6)),
                "last_ais_update": datetime.utcnow() - timedelta(minutes=self.rng.uniform(0, 5)),
                "is_anomalous": is_anomalous,
                "anomaly_type": anomaly_type,
                "ais_gap_minutes": self.rng.uniform(35, 90) if anomaly_type == "ais_gap" else 0,
                "course_noise_deg": self.rng.uniform(15, 45) if anomaly_type == "course_deviation" else self.rng.uniform(0, 3),
                "history": [],  # populated by generate_history()
            }
            self.fleet[mmsi] = vessel

    # -- position calculation -------------------------------------------------

    def _current_position(self, vessel: dict, progress_km: float | None = None):
        progress_km = vessel["progress_km"] if progress_km is None else progress_km
        progress_km = max(0.0, min(progress_km, vessel["total_km"]))
        frac = progress_km / vessel["total_km"] if vessel["total_km"] > 0 else 0
        bearing = initial_bearing_deg(
            vessel["origin"]["lat"], vessel["origin"]["lon"],
            vessel["destination_port"]["lat"], vessel["destination_port"]["lon"],
        )
        lat, lon = destination_point(vessel["origin"]["lat"], vessel["origin"]["lon"], bearing, progress_km)
        return lat, lon, bearing, frac

    def generate_history(self, hours_back: int = 72, step_minutes: int = 30):
        """Backfill a plausible trajectory for every vessel for historical analytics."""
        for mmsi, vessel in self.fleet.items():
            history = []
            steps = int(hours_back * 60 / step_minutes)
            start_progress = max(0.0, vessel["progress_km"] - vessel["cruise_speed"] * 1.852 * hours_back)
            for s in range(steps + 1):
                t = datetime.utcnow() - timedelta(hours=hours_back) + timedelta(minutes=s * step_minutes)
                p = start_progress + (vessel["progress_km"] - start_progress) * (s / steps if steps else 1)
                lat, lon, bearing, _ = self._current_position(vessel, progress_km=p)
                speed = max(0, vessel["cruise_speed"] + self.np_rng.normal(0, 0.6))
                history.append({
                    "mmsi": mmsi, "timestamp": t, "latitude": round(lat, 4),
                    "longitude": round(lon, 4), "speed_knots": round(speed, 1),
                    "course_deg": round(bearing % 360, 1),
                })
            vessel["history"] = history
        logger.info("Generated %d-hour history for %d vessels", hours_back, len(self.fleet))

    # -- time stepping ----------------------------------------------------

    def step(self, minutes: float = 5.0):
        """Advance the whole fleet forward in simulated time."""
        for vessel in self.fleet.values():
            if vessel["anomaly_type"] == "loitering":
                speed = self.np_rng.normal(0.4, 0.3)
            elif vessel["anomaly_type"] == "speed_drop":
                speed = vessel["cruise_speed"] * 0.35 + self.np_rng.normal(0, 0.3)
            else:
                speed = vessel["cruise_speed"] + self.np_rng.normal(0, 0.5)
            speed = max(0.0, speed)
            vessel["current_speed"] = round(speed, 1)

            distance_km = speed * 1.852 * (minutes / 60)
            vessel["progress_km"] = min(vessel["total_km"], vessel["progress_km"] + distance_km)

            if vessel["progress_km"] >= vessel["total_km"] * 0.995:
                vessel["nav_status"] = self.rng.choice(["At anchor", "Moored", "In port"])
                vessel["current_speed"] = round(self.np_rng.uniform(0, 0.5), 1)
            else:
                vessel["nav_status"] = "Under way using engine"

            if vessel["anomaly_type"] != "ais_gap" or self.rng.random() > 0.3:
                vessel["last_ais_update"] = datetime.utcnow()

    # -- current snapshot as a DataFrame -----------------------------------

    def snapshot(self) -> pd.DataFrame:
        rows = []
        for mmsi, vessel in self.fleet.items():
            lat, lon, bearing, frac = self._current_position(vessel)
            heading = (bearing + self.np_rng.normal(0, vessel["course_noise_deg"])) % 360
            rows.append({
                "mmsi": mmsi,
                "imo": vessel["imo"],
                "vessel_name": vessel["vessel_name"],
                "vessel_class": vessel["vessel_class"],
                "flag": vessel["flag"],
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "speed_knots": vessel["current_speed"],
                "course_deg": round(bearing % 360, 1),
                "heading_deg": round(heading, 1),
                "nav_status": vessel["nav_status"],
                "destination": vessel["destination_port"]["name"],
                "origin": vessel["origin"]["name"],
                "eta_scheduled": vessel["eta_scheduled"],
                "draught_m": vessel["draught_m"],
                "length_m": vessel["length_m"],
                "width_m": vessel["width_m"],
                "cargo_category": vessel["cargo_category"],
                "last_ais_update": vessel["last_ais_update"],
                "voyage_progress_pct": round(frac * 100, 1),
                "total_route_km": round(vessel["total_km"], 1),
                "remaining_km": round(vessel["total_km"] - vessel["progress_km"], 1),
                "is_anomalous_ground_truth": vessel["is_anomalous"],
                "anomaly_type_ground_truth": vessel["anomaly_type"],
                "data_source": "DEMO DATA",
                "source_timestamp": datetime.utcnow(),
            })
        return pd.DataFrame(rows)

    def history_df(self) -> pd.DataFrame:
        rows = []
        for vessel in self.fleet.values():
            rows.extend(vessel["history"])
        return pd.DataFrame(rows)
