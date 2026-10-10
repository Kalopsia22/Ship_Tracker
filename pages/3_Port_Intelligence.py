import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Port Intelligence · Maritime Intelligence", page_icon="⚓", layout="wide")

from maritime import ui  # noqa: E402
from maritime.security import Role  # noqa: E402

user = ui.require_login(Role.VIEWER)
result = ui.get_result()
ui.refresh_button()
ui.page_header("Port Intelligence", result)
st.info("Congestion = vessels expected within 48 h (distance ÷ current speed) ÷ a simplified port capacity. "
        "Waiting time follows a stated heuristic (`maritime/ports.py`) that must be recalibrated against real berth data before operational use.")

p = result.ports
c1, c2 = st.columns([3, 2])
with c1:
    st.dataframe(p[["port", "inbound_24h", "inbound_48h", "inbound_72h", "capacity", "congestion_idx", "expected_wait_h"]],
                 hide_index=True, width="stretch", column_config={
                     "congestion_idx": st.column_config.ProgressColumn("Congestion", min_value=0, max_value=2, format="%.2f"),
                     "expected_wait_h": st.column_config.NumberColumn("Est. wait (h)", format="%.1f")})
with c2:
    st.bar_chart(p.set_index("port")["congestion_idx"], horizontal=True)

st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ScatterplotLayer", data=p.round(2), get_position=["lon", "lat"], get_fill_color=[30, 100, 200, 160],
                                           get_radius="congestion_idx * 250000 + 60000", radius_min_pixels=5, pickable=True)],
                         initial_view_state=pdk.ViewState(latitude=20, longitude=40, zoom=1.2),
                         tooltip={"text": "{port}\n48h inbound: {inbound_48h} / capacity {capacity}"}))

port = st.selectbox("Inbound vessels for", p["port"])
v = result.vessels[result.vessels["destination"] == port].sort_values("dist_to_dest_nm")
st.dataframe(v.reset_index()[["vessel_name", "mmsi", "vessel_class", "dist_to_dest_nm", "sog", "eta_utc", "eta_reliable", "risk_level"]],
             hide_index=True, width="stretch", column_config={"dist_to_dest_nm": st.column_config.NumberColumn("Dist (nm)", format="%.0f")})
