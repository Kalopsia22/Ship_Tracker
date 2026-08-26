"""Business Intelligence: translating AI outputs into operational decisions."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.pipeline import initialize_state
from utils.ui_components import data_source_badge

st.set_page_config(page_title="Business Intelligence", page_icon="💼", layout="wide")
initialize_state()

st.title("💼 Business Intelligence")
data_source_badge(st.session_state.provider.source_label)
st.caption("How AI outputs on this dashboard map to concrete operational and commercial decisions.")

snapshot = st.session_state.snapshot
risk_df = st.session_state.risk_df
eta_df = st.session_state.eta_df

st.subheader("Decision-support summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Charter parties at risk of demurrage", int((eta_df["eta_deviation_hours"] > 12).sum()),
          help="Vessels predicted >12h late — potential laytime/demurrage exposure")
c2.metric("Vessels needing route review", int((risk_df["deviation_km"] > 50).sum()))
c3.metric("Estimated fleet-wide risk exposure", f"{risk_df['risk_score'].mean():.0f}/100")
c4.metric("Ports flagged high-congestion", int((st.session_state.port_congestion["congestion_level"] == "HIGH").sum()))

st.divider()
st.subheader("Cost-impact view: predicted delay by cargo category")
delay_by_cargo = snapshot.merge(eta_df[["mmsi", "eta_deviation_hours"]], on="mmsi", suffixes=("", "_eta"))
agg = delay_by_cargo.groupby("cargo_category")["eta_deviation_hours"].mean().reset_index()
fig = px.bar(agg, x="cargo_category", y="eta_deviation_hours",
             title="Average predicted ETA deviation by cargo type (hours)")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Risk-adjusted vessel watchlist")
watchlist = risk_df.merge(
    snapshot[["mmsi", "destination", "cargo_category"]], on="mmsi"
).sort_values("risk_score", ascending=False).head(20)
st.dataframe(
    watchlist[["mmsi", "vessel_name", "destination", "cargo_category", "risk_score", "risk_level"]],
    use_container_width=True, hide_index=True,
)

st.divider()
st.subheader("Suggested actions")
st.markdown(
    """
    - **High-risk vessels** → flag to operations for manual AIS cross-check and, where
      appropriate, contact the vessel operator directly.
    - **Predicted demurrage exposure** → notify chartering desk to review laytime clauses
      and proactively communicate with counterparties.
    - **High port congestion** → consider adjusting speed (virtual arrival / just-in-time
      steaming) to reduce anchorage time and fuel burn.
    - **Route deviations** → cross-reference with weather routing services before
      escalating — deviation alone is not evidence of an incident.
    """
)
st.caption(
    "⚠️ These are decision-support suggestions generated from model outputs, not "
    "automated actions. See the Responsible AI page for governance and human-in-the-loop policy."
)
