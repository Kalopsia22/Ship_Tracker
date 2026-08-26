"""AI Alerts feed with acknowledgment workflow."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data.database import log_audit_event
from utils.pipeline import initialize_state, run_pipeline
from utils.security import Role
from utils.ui_components import data_source_badge

st.set_page_config(page_title="AI Alerts", page_icon="🚨", layout="wide")
initialize_state()

st.title("🚨 AI Alerts")
data_source_badge(st.session_state.provider.source_label)

if st.button("🔄 Re-run detection now"):
    run_pipeline(advance_minutes=5)
    st.rerun()

alerts = st.session_state.alerts_log
df = pd.DataFrame(alerts)

c1, c2, c3 = st.columns(3)
if not df.empty:
    c1.metric("Critical", int((df["severity"] == "CRITICAL").sum()))
    c2.metric("Warning", int((df["severity"] == "WARNING").sum()))
    c3.metric("Info", int((df["severity"] == "INFO").sum()))
else:
    c1.metric("Critical", 0)
    c2.metric("Warning", 0)
    c3.metric("Info", 0)

st.divider()

severity_filter = st.multiselect(
    "Filter by severity", ["CRITICAL", "WARNING", "INFO"], default=["CRITICAL", "WARNING", "INFO"]
)
show_ack = st.checkbox("Show acknowledged alerts", value=False)

if df.empty:
    st.success("No alerts triggered — fleet operating within configured thresholds.")
else:
    filtered = df[df["severity"].isin(severity_filter)]
    if not show_ack:
        filtered = filtered[~filtered["acknowledged"]]

    user = st.session_state.get("user")
    can_ack = user and user.has_permission("acknowledge_alerts")

    for _, alert in filtered.sort_values("triggered_at", ascending=False).iterrows():
        color = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}[alert["severity"]]
        with st.container(border=True):
            cols = st.columns([6, 2, 2])
            with cols[0]:
                st.markdown(f"{color} **{alert['alert_type']}** — {alert['vessel_name']} (MMSI {alert['mmsi']})")
                st.caption(alert["message"])
                st.caption(f"Triggered: {pd.to_datetime(alert['triggered_at']).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            with cols[1]:
                st.markdown(f"Severity: **{alert['severity']}**")
            with cols[2]:
                if not alert["acknowledged"]:
                    if can_ack and st.button("Acknowledge", key=f"ack_{alert['alert_id']}"):
                        for a in st.session_state.alerts_log:
                            if a["alert_id"] == alert["alert_id"]:
                                a["acknowledged"] = True
                        log_audit_event(user.username, "acknowledge_alert", details=alert["alert_id"])
                        st.rerun()
                    elif not can_ack:
                        st.caption("🔒 Analyst/Admin role required")
                else:
                    st.caption("✅ Acknowledged")

if not df.empty:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export alert log (CSV)", csv, "alert_log.csv", "text/csv")

st.divider()
with st.expander("ℹ️ How alert thresholds work"):
    st.markdown(
        """
        Alert thresholds are configured from the **Live Vessel Map** page sidebar and apply
        fleet-wide:
        - **Route deviation** — vessel is more than X km from its expected great-circle route
        - **ETA delay** — predicted arrival is more than X hours later than scheduled
        - **AIS gap** — no position report received for more than X minutes
        - **Risk score** — composite Operational Risk Indicator exceeds X/100
        - **Unusual movement** — always flagged as INFO regardless of threshold (loitering detection)
        """
    )
