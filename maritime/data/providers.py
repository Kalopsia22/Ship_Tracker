"""Data providers: one interface, three implementations.

* DemoProvider      - synthetic fleet (default; no credentials needed)
* CsvProvider       - real AIS exports (default column map = NOAA MarineCadastre)
* HttpJsonProvider  - any JSON REST vendor, via an env-supplied field map
                      (generic; verify against your vendor before relying on it)
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

import pandas as pd

from maritime.config import Settings
from maritime.data.schema import POSITION_COLUMNS
from maritime.data.simulator import simulate_fleet


class ProviderError(RuntimeError):
    """Raised when a provider cannot return data. Never silently swallowed."""


class AISProvider(ABC):
    name: str = "base"
    is_synthetic: bool = False

    @abstractmethod
    def positions(self, hours: int = 72) -> pd.DataFrame:
        """Raw position reports (any superset of the schema; validator cleans it)."""

    def describe(self) -> str:
        return self.name


class DemoProvider(AISProvider):
    name = "demo (synthetic simulator)"
    is_synthetic = True

    def __init__(self, fleet_size: int = 120, seed: int = 42, now: pd.Timestamp | None = None):
        self.fleet_size, self.seed, self.now = fleet_size, seed, now

    def positions(self, hours: int = 72) -> pd.DataFrame:
        return simulate_fleet(self.fleet_size, hours=hours, seed=self.seed, now=self.now).positions


NOAA_MAP = {
    "MMSI": "mmsi", "BaseDateTime": "timestamp", "LAT": "lat", "LON": "lon", "SOG": "sog",
    "COG": "cog", "Heading": "heading", "VesselName": "vessel_name", "Status": "status",
    "Draft": "draught", "VesselType": "vessel_type_code",
}


def _class_from_type_code(code) -> str:
    try:
        c = int(code)
    except (TypeError, ValueError):
        return "Other"
    if 80 <= c <= 89:
        return "Tanker"
    if 70 <= c <= 79:
        return "Cargo"
    return "Other"


class CsvProvider(AISProvider):
    name = "csv (real AIS export)"

    def __init__(self, path: str, column_map: dict[str, str] | None = None):
        if not path:
            raise ProviderError("AIS_CSV_PATH is not set")
        self.path, self.column_map = path, column_map or NOAA_MAP

    def positions(self, hours: int = 72) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.path)
        except FileNotFoundError as e:
            raise ProviderError(f"CSV not found: {self.path}") from e
        df = df.rename(columns=self.column_map)
        if "vessel_type_code" in df.columns:
            df["vessel_class"] = df["vessel_type_code"].map(_class_from_type_code)
        # Real exports carry no planned origin/destination port: route-based
        # features (cross-track, ETA) are unavailable and are handled as NaN.
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        cutoff = df["timestamp"].max() - pd.Timedelta(hours=hours)
        return df[df["timestamp"] >= cutoff]


class HttpJsonProvider(AISProvider):
    name = "live (generic HTTP/JSON)"

    def __init__(self, url: str, api_key: str | None, field_map: dict[str, str], result_path: str = "", timeout: int = 20):
        if not url or not field_map:
            raise ProviderError("AIS_API_URL and AIS_FIELD_MAP must both be set for live mode")
        self.url, self.api_key, self.field_map = url, api_key, field_map
        self.result_path, self.timeout = result_path, timeout

    def positions(self, hours: int = 72) -> pd.DataFrame:
        import requests

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = requests.get(self.url, headers=headers, params={"hours": hours}, timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as e:
            raise ProviderError(f"Live AIS request failed: {e}") from e
        for key in filter(None, self.result_path.split(".")):
            try:
                payload = payload[key]
            except (KeyError, TypeError) as e:
                raise ProviderError(f"AIS_RESULT_PATH segment {key!r} not found in response") from e
        if not isinstance(payload, list):
            raise ProviderError("Live AIS response did not resolve to a list of records")
        df = pd.DataFrame(payload).rename(columns=self.field_map)
        for col in POSITION_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df


def build_provider(settings: Settings) -> AISProvider:
    if settings.data_mode == "demo":
        return DemoProvider(settings.fleet_size, settings.sim_seed)
    if settings.data_mode == "csv":
        return CsvProvider(settings.csv_path or "")
    fmap = json.loads(settings.api_field_map_json or "{}")
    return HttpJsonProvider(settings.api_url or "", settings.api_key, fmap, settings.api_result_path)
