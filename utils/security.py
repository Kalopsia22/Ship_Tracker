"""
Security helpers.

This module documents and implements the *architecture* for authentication,
role-based access, and credential handling. In this prototype, roles are
simulated locally (there is no real user database or login server) — but
the interfaces here are designed so that a real identity provider
(OKTA, Auth0, Azure AD, etc.) could be dropped in without changing callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    VIEWER = "viewer"          # Read-only access to dashboards
    ANALYST = "analyst"        # Can configure alerts/thresholds, export data
    ADMIN = "admin"            # Can manage model versions, users, audit logs


ROLE_PERMISSIONS = {
    Role.VIEWER: {"view_dashboard", "view_vessel_detail"},
    Role.ANALYST: {
        "view_dashboard",
        "view_vessel_detail",
        "configure_alerts",
        "export_data",
        "acknowledge_alerts",
    },
    Role.ADMIN: {
        "view_dashboard",
        "view_vessel_detail",
        "configure_alerts",
        "export_data",
        "acknowledge_alerts",
        "manage_models",
        "manage_users",
        "view_audit_log",
    },
}


@dataclass
class SimulatedUser:
    """
    A stand-in for an authenticated user session. In production this would
    be populated by an SSO/identity-provider integration (e.g. via
    streamlit-authenticator, OAuth, or a reverse-proxy header) rather than
    a sidebar dropdown.
    """

    username: str
    role: Role

    def has_permission(self, permission: str) -> bool:
        return permission in ROLE_PERMISSIONS.get(self.role, set())


def mask_secret(value: str | None, visible_chars: int = 4) -> str:
    """Mask a credential for safe display/logging, e.g. in diagnostics UI."""
    if not value:
        return "(not set)"
    if len(value) <= visible_chars:
        return "*" * len(value)
    return f"{'*' * (len(value) - visible_chars)}{value[-visible_chars:]}"
