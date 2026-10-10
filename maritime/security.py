"""Authentication and role-based access control (pure logic, no UI).

* Passwords: PBKDF2-HMAC-SHA256, per-user random salt, constant-time compare.
* Users come from AUTH_USERS_JSON / Streamlit secrets:
    {"alice": {"password_hash": "pbkdf2_sha256$...", "role": "analyst"}}
  Generate hashes with `python scripts/hash_password.py`.
* Demo accounts exist ONLY when AUTH_MODE=demo (which requires demo data).
* Failed logins are throttled per username, process-wide (survives page reloads).

Still not a substitute for an identity provider (SSO/OIDC) in production: there
is no MFA, password reset or session expiry. See README.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass
from enum import IntEnum

from maritime.config import Settings

ITERATIONS = 310_000
DEMO_ACCOUNTS = {"viewer": ("viewer-demo", "viewer"), "analyst": ("analyst-demo", "analyst"), "admin": ("admin-demo", "admin")}


class Role(IntEnum):
    VIEWER = 1
    ANALYST = 2
    ADMIN = 3

    @classmethod
    def parse(cls, value: str) -> "Role":
        try:
            return cls[value.strip().upper()]
        except KeyError as e:
            raise ValueError(f"unknown role {value!r}; use viewer|analyst|admin") from e


@dataclass(frozen=True)
class User:
    username: str
    role: Role
    password_hash: str


def hash_password(password: str, iterations: int = ITERATIONS) -> str:
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt_b64), int(iters))
        return hmac.compare_digest(dk, base64.b64decode(hash_b64))
    except (ValueError, TypeError):
        return False


class LoginThrottle:
    def __init__(self, max_attempts: int = 5, lockout_seconds: int = 300, clock=time.monotonic):
        self.max_attempts, self.lockout, self.clock = max_attempts, lockout_seconds, clock
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, username: str) -> list[float]:
        cutoff = self.clock() - self.lockout
        self._fails[username] = [t for t in self._fails.get(username, []) if t > cutoff]
        return self._fails[username]

    def is_locked(self, username: str) -> bool:
        with self._lock:
            return len(self._recent(username.lower())) >= self.max_attempts

    def record_failure(self, username: str) -> None:
        with self._lock:
            self._recent(username.lower()).append(self.clock())

    def reset(self, username: str) -> None:
        with self._lock:
            self._fails.pop(username.lower(), None)


class Authenticator:
    def __init__(self, users: dict[str, User], throttle: LoginThrottle | None = None):
        self.users = {k.lower(): v for k, v in users.items()}
        self.throttle = throttle or LoginThrottle()
        self._dummy = hash_password("not-a-real-password", iterations=ITERATIONS)  # equalise timing for unknown users

    def authenticate(self, username: str, password: str) -> tuple[User | None, str]:
        """Return (user, message). Messages never reveal whether a username exists."""
        key = (username or "").strip().lower()
        if self.throttle.is_locked(key):
            return None, "Too many failed attempts. Try again later."
        user = self.users.get(key)
        ok = verify_password(password or "", user.password_hash if user else self._dummy)
        if user and ok:
            self.throttle.reset(key)
            return user, "ok"
        self.throttle.record_failure(key)
        return None, "Invalid username or password."


def load_users(settings: Settings) -> dict[str, User]:
    if settings.auth_mode == "demo":
        return {n: User(n, Role.parse(r), hash_password(pw)) for n, (pw, r) in DEMO_ACCOUNTS.items()}
    if not settings.auth_users_json:
        raise RuntimeError("AUTH_MODE=strict requires AUTH_USERS_JSON (see README / scripts/hash_password.py). Refusing to start without users.")
    raw = json.loads(settings.auth_users_json)
    users = {}
    for name, spec in raw.items():
        if not str(spec.get("password_hash", "")).startswith("pbkdf2_sha256$"):
            raise ValueError(f"user {name!r}: password_hash must be a pbkdf2_sha256 hash, never a plaintext password")
        users[name] = User(name, Role.parse(spec["role"]), spec["password_hash"])
    if not users:
        raise RuntimeError("AUTH_USERS_JSON defines no users")
    return users


def has_role(actual: Role, required: Role) -> bool:
    return actual >= required
