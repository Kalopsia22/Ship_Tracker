"""
Application configuration.

Loads settings from environment variables / a local .env file / Streamlit
secrets. No credentials are ever hard-coded here — this module only reads
them from the environment at runtime.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load a local .env file if python-dotenv is available. This is optional —
# in production, real environment variables or Streamlit secrets are used
# instead.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "ship_tracking.db"


def _get_secret(key: str, default: str | None = None) -> str | None:
    """
    Read a config value, preferring Streamlit secrets (when running inside
    Streamlit and a secrets.toml is present) and falling back to plain
    environment variables. Never raises if secrets aren't configured.
    """
    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


@dataclass
class AppConfig:
    """Central application configuration object."""

    # Data mode: "demo" uses the synthetic simulator, "live" uses a real
    # AIS provider adapter (see data/ais_provider.py).
    data_mode: str = field(default_factory=lambda: _get_secret("AIS_DATA_MODE", "demo"))

    # Credentials for a real AIS provider — only read, never stored.
    ais_api_key: str | None = field(default_factory=lambda: _get_secret("AIS_API_KEY"))
    ais_api_base_url: str | None = field(
        default_factory=lambda: _get_secret("AIS_API_BASE_URL")
    )

    # Database connection string. Defaults to a local SQLite file, but is
    # designed to be swapped for a PostgreSQL/PostGIS DSN in production,
    # e.g. postgresql+psycopg2://user:pass@host:5432/shiptracking
    database_url: str = field(
        default_factory=lambda: _get_secret("DATABASE_URL", f"sqlite:///{DB_PATH}")
    )

    # Simulation parameters (demo mode only)
    simulated_fleet_size: int = field(
        default_factory=lambda: int(_get_secret("SIMULATED_FLEET_SIZE", "120"))
    )
    random_seed: int = field(default_factory=lambda: int(_get_secret("RANDOM_SEED", "42")))

    # Auto-refresh interval for the live map, in seconds
    refresh_interval_seconds: int = field(
        default_factory=lambda: int(_get_secret("REFRESH_INTERVAL_SECONDS", "15"))
    )

    # Default alert thresholds — user-configurable in the sidebar at runtime
    default_route_deviation_km: float = 50.0
    default_eta_delay_hours: float = 6.0
    default_ais_gap_minutes: float = 30.0
    default_risk_score_threshold: float = 75.0

    app_name: str = "Maritime Intelligence Dashboard"
    app_version: str = "0.1.0-prototype"

    @property
    def is_demo_mode(self) -> bool:
        return self.data_mode.lower() != "live"


config = AppConfig()
