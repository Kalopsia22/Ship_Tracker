import pandas as pd
import streamlit as st

st.set_page_config(page_title="Alerts · Maritime Intelligence", page_icon="🚨", layout="wide")

from maritime import ui  # noqa: E402
from maritime.alerts import AlertRules, generate_alerts  # noqa: E402
from maritime.security import Role, has_role  # noqa: E402

user = ui.require_login(Role.VIEWER)
result = ui.get_result()
repo = ui.get_repo()
ui.refresh_button()
ui.page_header("AI Alerts", result)

feed, preview = st.tabs(["Active alerts", "Threshold preview"])

with feed:
    status = st.radio("Show", ["open", "acknowledged"], horizontal=True)
    sev = st.multiselect("Severity", ["Critical", "Warning", "Info"], default=["Critical", "Warning", "Info"])
    alerts = [a for a in repo.list_alerts((status,)) if a["severity"] in sev]
    st.caption(f"{len(alerts)} {status} alert(s). Alerts auto-resolve when the underlying condition clears.")
    can_ack = has_role(user.role, Role.ANALYST)
    if not can_ack:
        st.caption("Acknowledging alerts requires the analyst role.")
    for a in alerts:
        icon = {"Critical": "🔴", "Warning": "🟠", "Info": "🔵"}[a["severity"]]
        cols = st.columns([8, 2])
        cols[0].markdown(f"{icon} **{a['type']}** · {a['message']}  \n<small>raised {a['created_at']:%d %b %H:%M} UTC"
                         + (f" · acknowledged by {a['ack_by']}" if a["ack_by"] else "") + "</small>", unsafe_allow_html=True)
        if status == "open" and can_ack and cols[1].button("Acknowledge", key=f"ack{a['id']}"):
            if repo.acknowledge(a["id"], user.username):
                st.rerun()

with preview:
    st.write("Try different thresholds. This **previews** how many alerts would fire; it does not change the deployed rules.")
    base = ui.alert_rules()
    r = AlertRules(
        risk_score_min=st.slider("Risk score ≥", 0, 100, int(base.risk_score_min)),
        ais_gap_hours=st.slider("AIS gap ≥ (h)", 1, 24, int(base.ais_gap_hours)),
        route_deviation_nm=st.slider("Route deviation ≥ (nm)", 10, 300, int(base.route_deviation_nm)),
        port_congestion_idx=st.slider("Port congestion index ≥", 0.5, 2.0, float(base.port_congestion_idx), 0.1),
    )
    cand = generate_alerts(result.vessels, result.ports, r)
    st.metric("Alerts that would fire", len(cand))
    if cand:
        st.dataframe(pd.DataFrame([{"type": c.alert_type, "severity": c.severity, "message": c.message} for c in cand]), hide_index=True, width="stretch")
