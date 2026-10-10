# Maritime Intelligence

An explainable AI/ML dashboard over AIS-style vessel data (Streamlit): ETA prediction with calibrated
intervals, behavioural anomaly detection, a transparent operational risk score, port congestion
estimates, a persistent alert workflow, and a Responsible-AI governance page.

> **Runs in demo mode by default** on a synthetic fleet: no credentials, no data licence.
> Everything in demo mode is simulated. The app says so on every page.

## What changed from v1 (and why)

This is a ground-up refinement of the original prototype, targeting its main weaknesses:

| v1 weakness | What this version does |
|---|---|
| Models trained & scored on synthetic data only | Real-data path: `CsvProvider` (NOAA MarineCadastre columns) and a generic `HttpJsonProvider`; `scripts/train.py --source csv` trains the anomaly model on **real tracks**. Synthetic metrics are labelled as such everywhere (UI, metadata, this README). |
| Models fit on the fly per session | Models are **trained offline, versioned, and persisted** with provenance metadata + SHA-256 checksums verified on load. Pipeline results cached with a TTL; no refits on rerun. |
| Weak evaluation | ETA: chronological split **by voyage** (no leakage), **naive baseline** comparison, **split-conformal 80% intervals** with measured coverage. Anomaly: evaluated on an **unseen** fleet with per-type recall. |
| One test script | **50 pytest tests** + CI (ruff + pytest on 3.11/3.12): geo math, validator, providers, models, registry tamper-detection, auth, DB, pipeline, and Streamlit page/role tests. |
| Simulated role selector | **Real authentication**: PBKDF2-SHA256 hashes, constant-time compare, per-user lockout, and **RBAC enforced server-side** (ack alerts = analyst, audit log/retrain = admin, CSV export = analyst). Fails closed: built-in demo accounts exist *only* with demo data. |
| Missing hygiene | `.env.example`, LICENSE, Dockerfile (trains at build, non-root), Makefile, CI, `.gitignore`, pinned tested versions. |
| Alerts not persistent | SQLite/PostgreSQL-backed alerts with dedupe, auto-resolve, acknowledgment, and an audit trail. |

**Honest caveat:** v1's source code was not available to me beyond its README; this is a clean rebuild guided by its
feature list and architecture, not a patch of its code. Pages are consolidated from 10 to 6: Voyage Analytics, Predictive Analytics and Historical Analysis were folded into
the Vessel Intelligence and Governance pages, and Business Intelligence (demurrage/routing what-ifs) was dropped
because it could not be grounded in real cost data. Every remaining page is tested.

## Quick start

```bash
pip install -r requirements.txt      # add -c constraints-tested.txt for the exact tested versions
streamlit run app.py
```

First launch trains and saves models (~10 s); afterwards they load from `artifacts/`.
Demo logins: `viewer / viewer-demo`, `analyst / analyst-demo`, `admin / admin-demo`.

```bash
make test      # pytest (≈ 1 min)      make lint     # ruff
make train     # offline training      make docker   # build & run container
```

## Pages

1. **Fleet Overview**: KPIs, class and risk distribution, top-risk vessels, data-quality details
2. **Live Map**: filters/search, risk-coloured vessels, audited CSV export (analyst+)
3. **Vessel Intelligence**: ETA + interval, risk-factor breakdown, 72 h track vs planned route
4. **Port Intelligence**: congestion index, estimated waiting time, inbound vessels
5. **Alerts**: persistent feed, acknowledgment workflow, threshold *preview*
6. **Responsible AI & Model Governance**: model cards, honest metrics, fairness & limits, audit log (admin), retrain (admin)

## Using real data

**CSV (e.g. NOAA MarineCadastre):**
```bash
export AIS_DATA_MODE=csv AIS_CSV_PATH=data_cache/ais.csv
python scripts/hash_password.py          # create users; put output in AUTH_USERS_JSON
export AUTH_MODE=strict AUTH_USERS_JSON='{"alice":{"password_hash":"pbkdf2_sha256$...","role":"admin"}}'
python scripts/train.py --source csv --csv data_cache/ais.csv
streamlit run app.py
```
Plain AIS exports carry no planned origin/destination, so **route-based features (cross-track, ETA, port
inbound counts) are unavailable** and the pipeline degrades to behaviour-only scoring with an on-screen note.

**Live HTTP/JSON vendor:** set `AIS_API_URL`, `AIS_API_KEY`, `AIS_FIELD_MAP` (vendor field → schema field) and
`AIS_RESULT_PATH`. This adapter is generic and unit-tested against a mock; **verify it against your vendor** before
relying on it. Provider failures are shown as errors, never silently replaced by demo data.

## Configuration

See [`.env.example`](.env.example). Key variables: `AIS_DATA_MODE`, `AUTH_MODE`, `AUTH_USERS_JSON`,
`DATABASE_URL` (SQLite default; PostgreSQL supported via SQLAlchemy), `REFRESH_INTERVAL_SECONDS`,
`ALERT_RULES`, `RISK_WEIGHTS`. Settings are read from the environment or Streamlit secrets.

## Architecture

```
app.py, pages/            Streamlit UI (thin; login gate + role checks on every page)
maritime/
  ui.py                   cached resources, login gate, page chrome
  pipeline.py             ingest → validate → features → ML → risk → ports → alerts
  data/                   schema, validator, simulator, providers (demo/csv/http)
  models/                 features, eta (RF + conformal), anomaly (IsolationForest),
                          risk (transparent composite), registry (versioned artifacts)
  analytics/port_analysis congestion heuristic
  security.py             hashing, throttling, RBAC (pure logic, fully unit-tested)
  database.py, alerts.py  persistent alerts + audit log
scripts/                  train.py, hash_password.py
tests/                    50 tests
```

## Models & evaluation (demo/synthetic; reproduce with `python scripts/train.py`)

| Model | Method | Result on held-out *synthetic* data |
|---|---|---|
| ETA | Random Forest + split-conformal interval | MAE ≈ 11 h vs 22 h naive baseline (~48% better); 80% interval covers ≈ 77% |
| Anomaly | Isolation Forest on 9 behavioural features | ROC-AUC ≈ 0.99; recall is weakest on `ais_gap` (~0.4) and `erratic_speed` (~0.6) |
| Risk | Weighted sum of 5 normalised signals (no learned weights) | Every point attributable to a named input |

**These numbers show the pipeline works as designed, not real-world accuracy.** The simulator defines both the
data and the anomalies, so recovering them is the easy case.

## Responsible AI

Risk and anomaly inputs are kinematic/route signals only. `flag, country, nationality, owner, operator,
vessel_name, mmsi` are forbidden inputs, enforced in code and tests. Alerts are advisory, acknowledgments are
audit-logged, and ETAs are marked *low reliability* for anomalous or stationary vessels.

## Known limitations

- Planned routes are great-circle lines: they ignore land, canals and traffic separation schemes, so legitimate
  detours can read as deviations.
- Port waiting time is a stated heuristic (`maritime/ports.py`), not learned; recalibrate on real berth data.
- Risk-level thresholds (15/30/50) were set from the synthetic score distribution.
- ETA is always trained on synthetic voyages (real AIS lacks completed-voyage labels without a segmentation step).
- Auth is real but basic: no SSO/OIDC, MFA, password reset or session expiry. Put the app behind your identity
  provider / reverse proxy for production. Login throttling is in-process (not shared across replicas).
- Single-process Streamlit app: for thousands of vessels, move ingestion to a worker + stream and serve results from
  the database.
- Model artifacts are joblib pickles. They are checksum-verified, but only ever load artifacts you built yourself.
- Tested with the versions in `constraints-tested.txt` (notably pandas 3.x, scikit-learn 1.8); other versions are
  allowed by `requirements.txt` but not verified.

## License

MIT
