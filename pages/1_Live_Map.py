import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Live Map · Maritime Intelligence", page_icon="🗺️", layout="wide")

from maritime import ui  # noqa: E402
from maritime.security import Role, has_role  # noqa: E402

user = ui.require_login(Role.VIEWER)
result = ui.get_result()
ui.refresh_button()
ui.page_header("Live Vessel Map", result)

v = result.vessels.reset_index()
with st.sidebar:
    classes = st.multiselect("Vessel class", sorted(v["vessel_class"].dropna().unique()))
    levels = st.multiselect("Risk level", ["Low", "Medium", "High", "Critical"])
    query = st.text_input("Search name or MMSI")
if classes:
    v = v[v["vessel_class"].isin(classes)]
if levels:
    v = v[v["risk_level"].isin(levels)]
if query:
    q = query.lower()
    v = v[v["vessel_name"].str.lower().str.contains(q, na=False) | v["mmsi"].str.contains(q)]

st.caption(f"{len(v)} vessels shown")
layer = pdk.Layer("ScatterplotLayer", data=ui.colorize(v)[["mmsi", "vessel_name", "vessel_class", "lat", "lon", "sog", "risk_score", "risk_level", "color"]].round(2),
                  get_position=["lon", "lat"], get_fill_color="color", get_radius=40000, radius_min_pixels=4, radius_max_pixels=12, pickable=True)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=20, longitude=40, zoom=1.2),
                         tooltip={"text": "{vessel_name}\n{vessel_class} · {sog} kn\nRisk {risk_score} ({risk_level})"}))
st.markdown(" ".join(f"<span style='color:rgb{tuple(c)}'>●</span> {k}" for k, c in ui.RISK_COLORS.items()), unsafe_allow_html=True)

if has_role(user.role, Role.ANALYST):
    st.download_button("Download filtered vessels (CSV)", v.drop(columns=["eta_utc"], errors="ignore").to_csv(index=False), "vessels.csv", "text/csv",
                       on_click=lambda: ui.get_repo().audit(user.username, "export.vessels_csv", detail=f"{len(v)} rows"))
else:
    st.caption("CSV export requires the analyst role.")
