"""Entry point: Fleet Overview. Other pages live in pages/."""
import streamlit as st

st.set_page_config(page_title="Maritime Intelligence", page_icon="🚢", layout="wide")

from maritime import ui  # noqa: E402  (after set_page_config)
from maritime.security import Role  # noqa: E402

user = ui.require_login(Role.VIEWER)
result = ui.get_result()
ui.refresh_button()
ui.page_header("Global Fleet Overview", result)

v = result.vessels
open_alerts = len(ui.get_repo().list_alerts(("open",)))
elevated = int(v["risk_level"].isin(["High", "Critical"]).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Vessels tracked", len(v))
c2.metric("Avg speed (kn)", f"{v['sog'].mean():.1f}")
c3.metric("High / Critical risk", elevated)
c4.metric("Open alerts", open_alerts)
c5.metric("Data quality", f"{result.quality.score:.1f}%")

left, right = st.columns(2)
with left:
    st.subheader("Fleet by class")
    st.bar_chart(v["vessel_class"].value_counts())
with right:
    st.subheader("Risk level distribution")
    order = ["Low", "Medium", "High", "Critical"]
    st.bar_chart(v["risk_level"].value_counts().reindex(order, fill_value=0))

st.subheader("Highest-risk vessels")
top = v.sort_values("risk_score", ascending=False).head(10).reset_index()
st.dataframe(
    top[["vessel_name", "mmsi", "vessel_class", "risk_score", "risk_level", "top_factor", "origin", "destination"]],
    hide_index=True, width="stretch",
    column_config={"risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f")},
)

for note in result.notes:
    st.info(note)
if result.quality.dropped:
    with st.expander("Data-quality details"):
        st.json(result.quality.as_dict())
