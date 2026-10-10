import pandas as pd
import streamlit as st

st.set_page_config(page_title="Governance · Maritime Intelligence", page_icon="🧭", layout="wide")

from maritime import ui  # noqa: E402
from maritime.models.risk import COMPONENTS, FORBIDDEN_INPUTS, LABELS  # noqa: E402
from maritime.security import Role, has_role  # noqa: E402

user = ui.require_login(Role.VIEWER)
result = ui.get_result()
models = ui.get_models()
meta = models.metadata
ui.page_header("Responsible AI & Model Governance", result)

st.error("**Read this first.** ETA models are trained on **synthetic** voyages, and the anomaly model's evaluation uses **injected synthetic anomalies**. "
         "The metrics below show the pipeline works as designed. They are **not** evidence of real-world accuracy.", icon="🧪")

tabs = st.tabs(["Model cards", "Fairness & limits", "Data quality", "Audit log", "Admin"])

with tabs[0]:
    st.caption(f"Model set `{meta['version']}` · trained {meta['trained_at'][:19]} UTC · scikit-learn {meta['sklearn_version']} · artifact checksums verified at load")
    e, a = meta["eta"], meta["anomaly"]
    em = e["metrics"]
    st.subheader("ETA prediction")
    st.write(f"{e['algorithm']} · data: {e['data_source']} · split: {em['split']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test MAE", f"{em['test_mae_hours']:.1f} h")
    c2.metric("Naive baseline MAE", f"{em['baseline_mae_hours']:.1f} h")
    c3.metric("Improvement vs baseline", f"{em['improvement_vs_baseline_pct']:.0f}%")
    c4.metric(f"{int(em['interval_level'] * 100)}% interval coverage", f"{em['interval_empirical_coverage'] * 100:.0f}%", "target 80%", delta_color="off")
    col_a, col_b = st.columns(2)
    col_a.write("MAE by vessel class (h)")
    col_a.bar_chart(pd.Series(em["mae_by_class_hours"]))
    col_b.write("Feature importance")
    col_b.bar_chart(models.eta.feature_importance().head(10), horizontal=True)

    st.subheader("Behavioural anomaly detection")
    am = a["metrics_on_synthetic_holdout"]
    st.write(f"{a['algorithm']} · trained on: {a['data_source']} ({a['n_training_vessels']} vessels) · evaluated on an unseen synthetic fleet")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{am['roc_auc']:.3f}")
    c2.metric("Precision", f"{am['precision']:.2f}")
    c3.metric("Recall", f"{am['recall']:.2f}")
    st.write("Recall by injected anomaly type (weak spots are visible, not hidden)")
    st.bar_chart(pd.Series(am["recall_by_type"]))

    st.subheader("Operational Risk Indicator")
    st.write("Transparent weighted composite. No learned weights, so every point is attributable to a named signal.")
    st.dataframe(pd.DataFrame([{"factor": LABELS[n], "input": COMPONENTS[n][0], "saturates_at": COMPONENTS[n][1], "weight": models.risk.weights[n]}
                               for n in COMPONENTS]), hide_index=True, width="stretch")

with tabs[1]:
    st.markdown(f"""
**Fairness.** The risk indicator and anomaly model use kinematic and route signals only. These attributes are never inputs
(enforced in code and unit tests): `{", ".join(sorted(FORBIDDEN_INPUTS))}`. Scores describe *behaviour*, not the
vessel's owner or nationality.

**Human in the loop.** Alerts are advisory. Analysts acknowledge them, each acknowledgment is audit-logged, and ETA output
is marked *low reliability* for anomalous or stationary vessels.

**Known limitations**
- Planned routes are straight great-circle lines: they ignore land, canals, and traffic-separation schemes, so legitimate
  detours can look like deviations and ETAs on canal routes are optimistic.
- Congestion/waiting time is a stated heuristic, not a learned model.
- Risk-level thresholds (15/30/50) were set from the synthetic score distribution and need recalibration on real data.
- ETA and route features need origin/destination ports. Plain AIS exports without them fall back to behaviour-only scoring.
- The 80% ETA interval is conformal-calibrated but empirically covers slightly under target on the held-out set.
""")

with tabs[2]:
    st.json(result.quality.as_dict())

with tabs[3]:
    if has_role(user.role, Role.ADMIN):
        log = ui.get_repo().audit_log()
        st.dataframe(pd.DataFrame(log), hide_index=True, width="stretch") if log else st.write("No entries yet.")
    else:
        st.info("The audit log requires the admin role.")

with tabs[4]:
    if has_role(user.role, Role.ADMIN):
        st.write("Retrain from the current data source. Synthetic mode retrains on the simulator; real-data modes retrain the anomaly model on the live fleet (ETA stays synthetic-trained).")
        if st.button("Retrain models now"):
            with st.spinner("Training..."):
                new = ui.retrain_models(user)
            st.success(f"Saved model set {new['version']}")
            st.rerun()
    else:
        st.info("Retraining requires the admin role.")
