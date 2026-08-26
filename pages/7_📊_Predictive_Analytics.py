"""Predictive Analytics: ETA model performance & fuel-efficiency intelligence."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from models.explainability import MODEL_CARD, confidence_band
from utils.pipeline import initialize_state
from utils.ui_components import data_source_badge

st.set_page_config(page_title="Predictive Analytics", page_icon="📊", layout="wide")
initialize_state()

st.title("📊 Predictive Analytics")
data_source_badge(st.session_state.provider.source_label)

eta_df = st.session_state.eta_df
snapshot = st.session_state.snapshot

st.subheader("ETA prediction summary")
c1, c2, c3 = st.columns(3)
c1.metric("Avg predicted deviation", f"{eta_df['eta_deviation_hours'].mean():+.1f} h")
c2.metric("Vessels predicted late (>6h)", int((eta_df["eta_deviation_hours"] > 6).sum()))
c3.metric("Avg model confidence", f"{eta_df['confidence'].mean():.0%}")

st.divider()
fig = px.histogram(eta_df, x="eta_deviation_hours", nbins=40, title="Distribution of predicted ETA deviation (hours)")
fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig, use_container_width=True)

eta_df_display = eta_df.copy()
eta_df_display["confidence_band"] = eta_df_display["confidence"].apply(confidence_band)
st.divider()
st.subheader("Prediction confidence distribution")
conf_counts = eta_df_display["confidence_band"].value_counts().reset_index()
conf_counts.columns = ["confidence_band", "count"]
fig2 = px.bar(conf_counts, x="confidence_band", y="count",
              category_orders={"confidence_band": ["Low", "Moderate", "High"]})
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Vessels most at risk of significant delay")
top_delays = eta_df_display.merge(
    snapshot[["mmsi", "vessel_class", "destination"]], on="mmsi"
).sort_values("eta_deviation_hours", ascending=False).head(15)
st.dataframe(
    top_delays[["vessel_name", "vessel_class", "destination", "eta_deviation_hours", "confidence_band"]],
    use_container_width=True, hide_index=True,
)

st.divider()
st.subheader("🧠 Model card: ETA prediction")
card = MODEL_CARD.get("eta-rf-v0.3-prototype", {})
st.markdown(f"**Model type:** {card.get('model_type')}")
st.markdown(f"**Training data:** {card.get('training_data')}")
st.markdown("**Known limitations:**")
for lim in card.get("known_limitations", []):
    st.markdown(f"- {lim}")

st.divider()
st.subheader("⛽ Fuel-efficiency intelligence (illustrative)")
st.caption(
    "Fuel consumption is estimated from a cubic speed-power relationship — a standard "
    "naval-architecture approximation, not vessel-specific telemetry."
)
snap = snapshot.copy()
# Illustrative cubic fuel-consumption relationship (relative units)
snap["relative_fuel_burn"] = (snap["speed_knots"] / snap["speed_knots"].median()).clip(lower=0.1) ** 3
fig3 = px.scatter(
    snap, x="speed_knots", y="relative_fuel_burn", color="vessel_class", hover_name="vessel_name",
    labels={"relative_fuel_burn": "Relative fuel burn (vs fleet median speed)"},
)
st.plotly_chart(fig3, use_container_width=True)
