"""
AIS Data Provider adapter architecture.

Defines a common `AISProvider` interface so the dashboard never talks to a
specific vendor API directly. In demo mode, `SimulatedAISProvider` wraps the
synthetic data generator. In live mode, `LiveAISProvider` is a template for
connecting to a real AIS data vendor (e.g. Spire, MarineTraffic, Datalastic,
exactEarth) — fill in `_fetch_from_api()` with the vendor's actual endpoint.

No credentials are ever hard-coded; they are read from environment
variables / Streamlit secrets via utils.configuration.
"""
from __future__ import annotations

import abc
from typing import Optional

import pandas as pd

from utils.configuration import config
from utils.logging import get_logger

logger = get_logger(__name__)


class AISProvider(abc.ABC):
    """Common interface every AIS data source must implement."""

    @abc.abstractmethod
    def get_fleet_snapshot(self) -> pd.DataFrame:
        """Return the current position of every tracked vessel."""

    @abc.abstractmethod
    def get_vessel_history(self, mmsi: str, hours_back: int = 72) -> pd.DataFrame:
        """Return historical positions for a single vessel."""

    @property
    @abc.abstractmethod
    def source_label(self) -> str:
        """Either 'DEMO DATA' or 'LIVE AIS' — always shown in the UI."""


class SimulatedAISProvider(AISProvider):
    """Wraps the synthetic vessel simulator for demo mode."""

    def __init__(self, simulator):
        self.simulator = simulator

    def get_fleet_snapshot(self) -> pd.DataFrame:
        return self.simulator.snapshot()

    def get_vessel_history(self, mmsi: str, hours_back: int = 72) -> pd.DataFrame:
        history = self.simulator.history_df()
        if history.empty:
            return history
        vessel_history = history[history["mmsi"] == mmsi].copy()
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=hours_back)
        return vessel_history[vessel_history["timestamp"] >= cutoff]

    @property
    def source_label(self) -> str:
        return "DEMO DATA"


class LiveAISProvider(AISProvider):
    """
    Template adapter for a real AIS data vendor.

    To connect a real provider:
      1. Set AIS_DATA_MODE=live and AIS_API_KEY / AIS_API_BASE_URL in your
         environment or Streamlit secrets (see .env.example).
      2. Implement `_fetch_from_api()` below using the vendor's REST/WS API.
      3. Map the vendor's response schema onto the columns expected by
         `data/schemas.py::AISPosition` (mmsi, imo, vessel_name, latitude,
         longitude, speed_knots, course_deg, nav_status, etc).
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or config.ais_api_key
        self.base_url = base_url or config.ais_api_base_url
        if not self.api_key or not self.base_url:
            logger.warning(
                "Live AIS provider selected but AIS_API_KEY/AIS_API_BASE_URL "
                "are not configured. Falls back to raising on use."
            )

    def _fetch_from_api(self, endpoint: str, params: dict | None = None) -> dict:
        import httpx

        if not self.api_key or not self.base_url:
            raise RuntimeError(
                "Live AIS mode requires AIS_API_KEY and AIS_API_BASE_URL to be set."
            )
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(base_url=self.base_url, headers=headers, timeout=10.0) as client:
            response = client.get(endpoint, params=params or {})
            response.raise_for_status()
            return response.json()

    def get_fleet_snapshot(self) -> pd.DataFrame:
        # Example vendor call — replace `/v1/vessels/positions` and the
        # field mapping below with the real provider's contract.
        payload = self._fetch_from_api("/v1/vessels/positions")
        df = pd.DataFrame(payload.get("vessels", []))
        return df

    def get_vessel_history(self, mmsi: str, hours_back: int = 72) -> pd.DataFrame:
        payload = self._fetch_from_api(f"/v1/vessels/{mmsi}/history", params={"hours": hours_back})
        return pd.DataFrame(payload.get("positions", []))

    @property
    def source_label(self) -> str:
        return "LIVE AIS"


def get_provider(simulator=None) -> AISProvider:
    """Factory that returns the correct provider based on current config."""
    if config.is_demo_mode:
        if simulator is None:
            raise ValueError("A simulator instance is required for demo mode")
        return SimulatedAISProvider(simulator)
    return LiveAISProvider()
