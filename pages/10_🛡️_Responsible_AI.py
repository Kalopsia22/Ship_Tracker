"""Responsible AI / Model Governance page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data.database import get_session, AuditLog
from models.explainability import MODEL_CARD
from utils.pipeline import initialize_state
from utils.security import Role
from utils.ui_components import data_source_badge

st.set_page_config(page_title="Responsible AI", page_icon="🛡️", layout="wide")
initialize_state()

st.title("🛡️ Responsible AI & Model Governance")
data_source_badge(st.session_state.provider.source_label)

st.markdown(
    """
    This page documents how the AI/ML components of this dashboard are built and governed,
    consistent with the project's Responsible AI requirements.
    """
)

st.divider()
st.subheader("1. Purpose & scope of AI outputs")
st.markdown(
    """
    Every AI-derived score on this dashboard — **ETA predictions**, **anomaly scores**,
    and the **Operational Risk Indicator** — is a **decision-support signal for maritime
    operations, logistics planning, and safety monitoring**. None of these scores are:

    - A determination of illegal activity, sanctions evasion, or security threat
    - A substitute for human review before any operational or legal action
    - Based on the vessel operator's, owner's, or flag state's nationality
    """
)

st.divider()
st.subheader("2. Fairness — what the Operational Risk Indicator does *not* use")
st.markdown(
    """
    The risk model (`models/risk_model.py`) is a transparent, weighted composite of
    **behavioral and operational signals only**:

    - Route deviation from expected great-circle path
    - Behavioral anomaly score (speed/course/AIS-gap based)
    - AIS reporting gaps
    - ETA deviation from schedule
    - Loitering indicators

    It **deliberately excludes** flag state, country of registration, owner nationality,
    or vessel name/language as inputs, to avoid encoding geographic or national-origin
    bias into an operational score. Analysts reviewing high-risk vessels are encouraged
    to independently verify context (weather routing, mechanical issues, scheduled
    anchorage) before escalating.
    """
)

st.divider()
st.subheader("3. Explainability")
st.markdown(
    "Every AI score shown to a user is accompanied by an explanation panel listing the "
    "specific contributing factors and their relative weight — see the **Vessel "
    "Intelligence** page for a live example. No score is presented without its "
    "supporting factors."
)

st.divider()
st.subheader("4. Model cards")
for model_id, card in MODEL_CARD.items():
    with st.expander(f"📇 {model_id}"):
        st.markdown(f"**Type:** {card['model_type']}")
        st.markdown(f"**Training data:** {card['training_data']}")
        st.markdown("**Known limitations:**")
        for lim in card["known_limitations"]:
            st.markdown(f"- {lim}")

st.divider()
st.subheader("5. Human-in-the-loop & access control")
st.markdown(
    """
    - Alerts require explicit human **acknowledgment** before being cleared (see AI Alerts page)
    - Only **Analyst** and **Admin** roles can acknowledge alerts or configure thresholds
    - Only **Admin** role can view the audit log below or manage model versions
    - All vessel-detail views and alert acknowledgments are logged to the audit trail
    """
)
st.caption(f"Current session role: **{st.session_state.user.role.value if st.session_state.get('user') else 'unknown'}**")

st.divider()
st.subheader("6. Audit log")
user = st.session_state.get("user")
if user and user.has_permission("view_audit_log"):
    session = get_session()
    try:
        logs = session.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(50).all()
        if logs:
            df = pd.DataFrame([{
                "timestamp": l.timestamp, "username": l.username,
                "action": l.action, "details": l.details,
            } for l in logs])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No audit events recorded yet this session.")
    finally:
        session.close()
else:
    st.warning("🔒 Admin role required to view the audit log. Switch role in the sidebar to view.")

st.divider()
st.subheader("7. Data provenance")
st.markdown(
    f"""
    - **Current data source:** `{st.session_state.provider.source_label}`
    - Demo mode uses a clearly-labeled synthetic AIS simulator (`data/simulator.py`);
      no real vessel positions are used or implied.
    - Live mode requires explicit configuration of a licensed AIS data provider
      (see `data/ais_provider.py` and `.env.example`) — credentials are never
      hard-coded and are read only from environment variables or Streamlit secrets.
    """
)

st.divider()
st.subheader("8. Known limitations of this prototype")
st.markdown(
    """
    - Runs on synthetic data by default; ML models are fitted on-the-fly against the
      current session's simulated fleet rather than a large historical corpus.
    - Anomaly/risk model weights are illustrative and should be calibrated against
      real historical incident data before any production deployment.
    - No live weather, currents, or port-authority berth data are integrated.
    - Single-process Streamlit prototype — not yet architected for the throughput of
      thousands of concurrent live vessels (see README for the production scaling notes).
    """
)
