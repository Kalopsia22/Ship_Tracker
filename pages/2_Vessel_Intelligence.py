import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Vessel Intelligence · Maritime Intelligence", page_icon="🔍", layout="wide")

from maritime import ui  # noqa: E402
from maritime.geo import interpolate_gc  # noqa: E402
from maritime.ports import PORTS  # noqa: E402
from maritime.security import Role  # noqa: E402

user = ui.require_login(Role.VIEWER)
result = ui.get_result()
models = ui.get_models()
ui.refresh_button()
ui.page_header("Vessel Intelligence", result)

v = result.vessels.sort_values("risk_score", ascending=False)
labels = {m: ui.vessel_label(m, r["vessel_name"]) for m, r in v.iterrows()}
mmsi = st.selectbox("Vessel (sorted by risk)", list(labels), format_func=labels.get)
row = v.loc[mmsi]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk score", f"{row['risk_score']:.0f} / 100", row["risk_level"], delta_color="off")
c2.metric("Behaviour", "Anomalous" if row["is_anomaly"] else "Normal", f"score {row['anomaly_score']:.2f}", delta_color="off")
if pd.notna(row["eta_hours"]):
    c3.metric("ETA (model)", f"{row['eta_utc']:%d %b %H:%M} UTC", f"{row['eta_hours']:.0f} h", delta_color="off")
    c4.metric("80% interval", f"{row['eta_low_h']:.0f}–{row['eta_high_h']:.0f} h")
    if not row["eta_reliable"]:
        st.warning("ETA reliability is **low**: this vessel is flagged anomalous or is not making way. "
                   "The model was trained on normal voyages and may be unreliable here.")
else:
    c3.metric("ETA (model)", "n/a")
    c4.metric("Route known", "No")

st.subheader("Why this risk score?")
expl = models.risk.explain(row)
left, right = st.columns([3, 2])
with left:
    st.dataframe(expl, hide_index=True, width="stretch", column_config={
        "points": st.column_config.ProgressColumn("Points", min_value=0, max_value=35, format="%.1f"),
        "value": st.column_config.NumberColumn(format="%.2f"), "weight": st.column_config.NumberColumn(format="%.2f")})
with right:
    st.bar_chart(expl.set_index("factor")["points"])
st.caption("Score = Σ weight × min(value / saturation, 1) × 100. Inputs are behavioural only: no flag, owner, nationality or name.")

st.subheader("Track (last 72 h) and planned route")
track = result.positions[result.positions["mmsi"] == mmsi].sort_values("timestamp")
layers = [pdk.Layer("PathLayer", data=pd.DataFrame({"path": [track[["lon", "lat"]].values.tolist()]}), get_path="path",
                    get_color=[30, 100, 200], width_min_pixels=3),
          pdk.Layer("ScatterplotLayer", data=pd.DataFrame({"lon": [row["lon"]], "lat": [row["lat"]]}), get_position=["lon", "lat"],
                    get_fill_color=ui.RISK_COLORS[row["risk_level"]], get_radius=60000, radius_min_pixels=7)]
if row["route_known"]:
    o, d = PORTS[row["origin"]], PORTS[row["destination"]]
    la, lo = interpolate_gc(o.lat, o.lon, d.lat, d.lon, np.linspace(0, 1, 60))
    layers.insert(0, pdk.Layer("PathLayer", data=pd.DataFrame({"path": [np.c_[lo, la].tolist()]}), get_path="path",
                               get_color=[150, 150, 150], width_min_pixels=2))
    st.caption(f"Planned: {row['origin']} → {row['destination']} (straight great-circle line: ignores land and shipping lanes). "
               f"Off-route now: {row['xtrack_last_nm']:.0f} nm.")
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=float(row["lat"]), longitude=float(row["lon"]), zoom=2.5)))

st.subheader("Speed over ground")
st.line_chart(track.set_index("timestamp")["sog"], y_label="knots")

with st.expander("Raw behavioural features"):
    st.dataframe(row[["sog_mean", "sog_std", "sog_ratio", "recent_sog_ratio", "cog_turn_mean", "xtrack_max_nm",
                      "xtrack_last_nm", "max_gap_h", "stationary_frac", "n_reports"]].astype(float).to_frame("value"), width="stretch")
