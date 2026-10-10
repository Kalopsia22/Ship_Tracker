"""Central configuration, read from environment variables or Streamlit secrets.

Nothing sensitive is hard-coded. Defaults are safe: the app only enables
demo credentials when it is *also* running on synthetic data.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is not None:
        return value
    try:  # Streamlit secrets are optional (raises if no secrets file exists)
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    data_mode: str = "demo"  # demo | csv | live
    fleet_size: int = 120
    sim_seed: int = 42
    csv_path: str | None = None
    database_url: str = f"sqlite:///{ROOT / 'data_cache' / 'maritime.db'}"
    artifacts_dir: Path = ROOT / "artifacts"
    refresh_seconds: int = 300
    # auth
    auth_mode: str = "demo"  # demo | strict
    auth_users_json: str | None = None
    max_login_attempts: int = 5
    lockout_seconds: int = 300
    # live HTTP provider
    api_url: str | None = None
    api_key: str | None = None
    api_field_map_json: str | None = None
    api_result_path: str = ""
    # risk weights + alert thresholds live in their modules; overridable here
    risk_weights_json: str | None = None
    alert_rules_json: str | None = None
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        data_mode = (_get("AIS_DATA_MODE", "demo") or "demo").lower()
        if data_mode not in {"demo", "csv", "live"}:
            raise ValueError(f"AIS_DATA_MODE must be demo|csv|live, got {data_mode!r}")
        # Fail closed: demo credentials are only ever the default for demo data.
        default_auth = "demo" if data_mode == "demo" else "strict"
        auth_mode = (_get("AUTH_MODE", default_auth) or default_auth).lower()
        if auth_mode not in {"demo", "strict"}:
            raise ValueError(f"AUTH_MODE must be demo|strict, got {auth_mode!r}")
        if auth_mode == "demo" and data_mode != "demo":
            raise ValueError(
                "AUTH_MODE=demo (built-in demo accounts) is only allowed with "
                "AIS_DATA_MODE=demo. Configure AUTH_USERS_JSON and use AUTH_MODE=strict."
            )
        return cls(
            data_mode=data_mode,
            fleet_size=int(_get("SIMULATED_FLEET_SIZE", "120")),
            sim_seed=int(_get("SIMULATION_SEED", "42")),
            csv_path=_get("AIS_CSV_PATH"),
            database_url=_get("DATABASE_URL", cls.database_url) or cls.database_url,
            artifacts_dir=Path(_get("ARTIFACTS_DIR", str(cls.artifacts_dir))),
            refresh_seconds=int(_get("REFRESH_INTERVAL_SECONDS", "300")),
            auth_mode=auth_mode,
            auth_users_json=_get("AUTH_USERS_JSON"),
            max_login_attempts=int(_get("MAX_LOGIN_ATTEMPTS", "5")),
            lockout_seconds=int(_get("LOCKOUT_SECONDS", "300")),
            api_url=_get("AIS_API_URL"),
            api_key=_get("AIS_API_KEY"),
            api_field_map_json=_get("AIS_FIELD_MAP"),
            api_result_path=_get("AIS_RESULT_PATH", "") or "",
            risk_weights_json=_get("RISK_WEIGHTS"),
            alert_rules_json=_get("ALERT_RULES"),
        )

    def alert_rules(self) -> dict:
        return json.loads(self.alert_rules_json) if self.alert_rules_json else {}

    def risk_weights(self) -> dict | None:
        return json.loads(self.risk_weights_json) if self.risk_weights_json else None
