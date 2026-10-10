from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from maritime.data.providers import AISProvider, DemoProvider
from maritime.data.simulator import simulate_fleet
from maritime.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent


def test_pipeline_end_to_end(bundle, now):
    r = run_pipeline(DemoProvider(80, 42, now), bundle)
    v = r.vessels
    assert len(v) >= 70 and r.quality.score == 100
    assert v["risk_score"].between(0, 100).all() and v["eta_hours"].notna().all()
    assert (v["eta_low_h"] <= v["eta_hours"]).all() and (v["eta_hours"] <= v["eta_high_h"]).all()
    assert set(r.ports.columns) >= {"port", "congestion_idx", "expected_wait_h"}


def test_pipeline_flags_injected_anomalies(bundle, now):
    sim = simulate_fleet(200, seed=11, now=now)

    class P(AISProvider):
        name, is_synthetic = "t", True
        def positions(self, hours=72): return sim.positions

    r = run_pipeline(P(), bundle)
    t = sim.truth.set_index("mmsi").loc[r.vessels.index]
    assert r.vessels.loc[t["is_anomalous"], "risk_score"].mean() > 3 * r.vessels.loc[~t["is_anomalous"], "risk_score"].mean()


def test_pipeline_degrades_gracefully_without_routes(bundle, now):
    pos = simulate_fleet(40, seed=3, now=now).positions.assign(origin=None, destination=None)

    class P(AISProvider):
        name = "no-routes"
        def positions(self, hours=72): return pos

    r = run_pipeline(P(), bundle)
    assert r.vessels["eta_hours"].isna().all() and r.notes and len(r.vessels) > 0
    assert not any(a.alert_type == "ROUTE_DEVIATION" for a in r.alerts)


def test_pipeline_rejects_all_invalid_data(bundle):
    class P(AISProvider):
        name = "bad"
        def positions(self, hours=72):
            return pd.DataFrame({"mmsi": ["1"], "timestamp": ["2026-01-01"], "lat": [999.0], "lon": [0.0]})

    with pytest.raises(ValueError, match="No valid"):
        run_pipeline(P(), bundle)


# ---- Streamlit page smoke + access-control tests -----------------------------
def _page(path: str, user: str | None = None) -> AppTest:
    at = AppTest.from_file(str(ROOT / path), default_timeout=180)
    if user:
        at.session_state["auth_user"] = user
    return at.run()


@pytest.mark.parametrize("page", ["app.py", "pages/1_Live_Map.py", "pages/4_Alerts.py", "pages/5_Model_Governance.py"])
def test_pages_require_login(page):
    at = _page(page)
    assert not at.exception
    assert any("Sign in" in s.value for s in at.subheader)  # login form, no data


@pytest.mark.parametrize("page", ["app.py", "pages/1_Live_Map.py", "pages/2_Vessel_Intelligence.py", "pages/3_Port_Intelligence.py",
                                  "pages/4_Alerts.py", "pages/5_Model_Governance.py"])
def test_pages_render_for_each_role(page):
    for user in ("viewer", "admin"):
        at = _page(page, user)
        assert not at.exception, f"{page} as {user}: {at.exception}"


def test_login_form_rejects_bad_password_and_accepts_good():
    at = _page("app.py")
    at.text_input[0].set_value("viewer"); at.text_input[1].set_value("wrong-password"); at.button[0].click().run()
    assert at.error and "Invalid" in at.error[0].value
    at.text_input[0].set_value("viewer"); at.text_input[1].set_value("viewer-demo"); at.button[0].click().run()
    assert at.session_state["auth_user"] == "viewer" and not at.exception


def test_viewer_cannot_ack_or_see_admin_features():
    at = _page("pages/4_Alerts.py", "viewer")
    assert not any(b.label == "Acknowledge" for b in at.button)
    gov = _page("pages/5_Model_Governance.py", "viewer")
    assert any("requires the admin role" in i.value for i in gov.info)
    assert not any(b.label == "Retrain models now" for b in gov.button)


def test_analyst_can_ack_and_admin_sees_audit():
    at = _page("pages/4_Alerts.py", "analyst")
    acks = [b for b in at.button if b.label == "Acknowledge"]
    assert acks, "demo fleet should produce at least one open alert"
    acks[0].click().run()
    assert not at.exception
    admin = _page("pages/5_Model_Governance.py", "admin")
    assert any(b.label == "Retrain models now" for b in admin.button)
