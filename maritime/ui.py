"""Streamlit glue: cached resources, login gate, shared page chrome.

Heavy objects (models, DB engine, authenticator, pipeline result) live in
`st.cache_resource`, so a Streamlit rerun never refits models or recomputes
the pipeline; the pipeline result refreshes on a TTL.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from maritime.alerts import AlertRules
from maritime.config import Settings
from maritime.data.providers import ProviderError, build_provider
from maritime.database import Repository, make_engine
from maritime.models.features import compute_vessel_features
from maritime.models.registry import ModelBundle, load_or_train, train_and_save
from maritime.pipeline import PipelineResult, run_pipeline
from maritime.security import Authenticator, LoginThrottle, Role, User, has_role, load_users

SETTINGS = Settings.from_env()
RISK_COLORS = {"Low": [46, 160, 67], "Medium": [214, 158, 46], "High": [219, 97, 36], "Critical": [200, 30, 30]}


@st.cache_resource
def get_authenticator() -> Authenticator:
    return Authenticator(load_users(SETTINGS), LoginThrottle(SETTINGS.max_login_attempts, SETTINGS.lockout_seconds))


@st.cache_resource
def get_models() -> ModelBundle:
    with st.spinner("Loading models (first run trains and saves them; later runs load from disk)..."):
        return load_or_train(SETTINGS.artifacts_dir, SETTINGS.risk_weights())


@st.cache_resource
def get_repo() -> Repository:
    return Repository(make_engine(SETTINGS.database_url))


def alert_rules() -> AlertRules:
    return AlertRules(**SETTINGS.alert_rules())


@st.cache_resource(ttl=SETTINGS.refresh_seconds, show_spinner="Refreshing fleet data...")
def _pipeline() -> PipelineResult:
    result = run_pipeline(build_provider(SETTINGS), get_models(), alert_rules())
    get_repo().sync_alerts(result.alerts)
    return result


def get_result() -> PipelineResult:
    """Pipeline result (do not mutate; copy first). Provider failures are shown, never hidden."""
    try:
        return _pipeline()
    except ProviderError as e:
        st.error(f"Data provider error: {e}")
        st.stop()


def retrain_models(user: User) -> dict:
    """Admin action: retrain, persist, audit, and invalidate caches."""
    result = get_result()
    if result.is_synthetic or len(result.vessels) < 50:
        bundle = train_and_save(SETTINGS.artifacts_dir, risk_weights=SETTINGS.risk_weights())
    else:
        feats = compute_vessel_features(result.positions)
        bundle = train_and_save(SETTINGS.artifacts_dir, anomaly_features=feats,
                                anomaly_source=f"provider data: {result.provider}", risk_weights=SETTINGS.risk_weights())
    get_repo().audit(user.username, "model.retrain", bundle.metadata["version"])
    st.cache_resource.clear()
    return bundle.metadata


# --------------------------------------------------------------------------- auth
def _current_user() -> User | None:
    name = st.session_state.get("auth_user")
    return get_authenticator().users.get(name) if name else None


def require_login(min_role: Role = Role.VIEWER) -> User:
    """Gate a page. Renders the login form and halts until authenticated."""
    auth = get_authenticator()
    user = _current_user()
    if user is None:
        st.title("🚢 Maritime Intelligence")
        st.subheader("Sign in")
        if SETTINGS.auth_mode == "demo":
            st.info("**Demo mode**: synthetic data and built-in demo accounts.\n\n"
                    "`viewer` / `viewer-demo` · `analyst` / `analyst-demo` · `admin` / `admin-demo`")
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            found, msg = auth.authenticate(username, password)
            if found:
                st.session_state["auth_user"] = found.username.lower()
                get_repo().audit(found.username, "auth.login")
                st.rerun()
            get_repo().audit(username or "?", "auth.login_failed")
            st.error(msg)
        st.stop()
    if not has_role(user.role, min_role):
        st.error(f"This page requires the **{min_role.name.lower()}** role. You are signed in as **{user.role.name.lower()}**.")
        st.stop()
    with st.sidebar:
        st.caption(f"Signed in as **{user.username}** ({user.role.name.lower()})")
        if st.button("Sign out"):
            get_repo().audit(user.username, "auth.logout")
            st.session_state.pop("auth_user", None)
            st.rerun()
    return user


def page_header(title: str, result: PipelineResult | None = None) -> None:
    st.title(title)
    if SETTINGS.data_mode == "demo":
        st.warning("**DEMO MODE**: all vessels, positions and incidents are synthetic. Do not use for operational decisions.", icon="⚠️")
    if result is not None:
        st.caption(f"Source: {result.provider} · data as of {result.generated_at:%Y-%m-%d %H:%M} UTC · "
                   f"quality {result.quality.score:.1f}% · refreshes every {SETTINGS.refresh_seconds // 60} min")


def refresh_button() -> None:
    with st.sidebar:
        if st.button("↻ Refresh data now"):
            _pipeline.clear()
            st.rerun()


def vessel_label(mmsi: str, name: str) -> str:
    return f"{name} ({mmsi})"


def colorize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["color"] = out["risk_level"].map(RISK_COLORS)
    return out
