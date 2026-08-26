# AI-Powered Maritime Intelligence Dashboard (Prototype)

A Streamlit-based live ship-tracking dashboard demonstrating an AI/ML layer over
AIS-style vessel data: ETA prediction, behavioral anomaly detection, an
explainable operational risk score, route-deviation analysis, port congestion
forecasting, and a configurable real-time alert feed — plus a dedicated
Responsible AI / Model Governance page.

**This prototype ships in DEMO MODE by default**, running entirely on a
synthetic AIS simulator so it works out of the box with no credentials or
external data license. It is architected so a real AIS data provider can be
plugged in later (see "Connecting live AIS data" below) without touching the
UI or model code.

## What's included

| Area | Where |
|---|---|
| Synthetic AIS simulator (120-vessel fleet, 5 tanker/bulker classes, 15 global ports, injected anomalies) | `data/simulator.py` |
| Data validation & quality scoring | `data/validator.py` |
| Live/demo provider abstraction | `data/ais_provider.py` |
| SQLAlchemy schema (vessels, positions, voyages, alerts, audit log, model versions) | `data/database.py` |
| ETA prediction (Random Forest, explainable) | `models/eta_model.py` |
| Anomaly detection (Isolation Forest) | `models/anomaly_model.py` |
| Explainable Operational Risk Indicator (transparent weighted composite) | `models/risk_model.py` |
| Route deviation analysis | `analytics/route_analysis.py` |
| Port congestion estimate | `analytics/port_analysis.py` |
| Fleet KPIs / metrics | `analytics/vessel_metrics.py` |
| Shared ingestion → validation → ML → alerts pipeline | `utils/pipeline.py` |
| Role-based access architecture (Viewer/Analyst/Admin) | `utils/security.py` |
| Main dashboard + 10 pages | `app.py`, `pages/` |

## Pages

1. **Global Fleet Overview** — fleet composition, status breakdown, progress by class
2. **Live Vessel Map** — interactive global map with filters, search, CSV export
3. **Vessel Intelligence** — single-vessel deep dive with full AI explainability panel
4. **Voyage Analytics** — route deviation, traffic volume, speed/draught relationships
5. **Port Intelligence** — predictive congestion & waiting-time estimates
6. **AI Alerts** — configurable thresholds, severity feed, acknowledgment workflow
7. **Predictive Analytics** — ETA model performance, confidence bands, fuel-burn intelligence
8. **Historical Analysis** — 72h trajectory playback, density heatmap, speed trends
9. **Business Intelligence** — AI outputs translated into demurrage/routing/congestion decisions
10. **Responsible AI / Model Governance** — fairness, explainability, model cards, audit log

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

No configuration is required for demo mode — the app generates its own
synthetic fleet on first load.

## Configuration

Copy `.env.example` to `.env` to customize:

```bash
cp .env.example .env
```

Key settings:

- `AIS_DATA_MODE` — `demo` (default) or `live`
- `SIMULATED_FLEET_SIZE` — number of simulated vessels (default 120)
- `DATABASE_URL` — defaults to local SQLite; point at PostgreSQL/PostGIS for production
- `REFRESH_INTERVAL_SECONDS` — auto-refresh cadence on the main dashboard

## Connecting live AIS data

This prototype does **not** ship with a live AIS vendor integration (that
requires a commercial data license). To connect one:

1. Choose a provider (e.g. Spire, MarineTraffic, Datalastic, exactEarth).
2. Set `AIS_DATA_MODE=live`, `AIS_API_KEY`, and `AIS_API_BASE_URL` in `.env`
   or your deployment's secrets manager.
3. Implement `LiveAISProvider._fetch_from_api()` and the field mapping in
   `data/ais_provider.py` for your vendor's response schema.
4. Everything downstream (validation, ML models, map, alerts) works unchanged
   — they only depend on the common `AISProvider` interface.

Credentials are **never hard-coded**; they're read only from environment
variables or Streamlit secrets (`utils/configuration.py`).

## Architecture notes / production scaling

This is a single-process Streamlit prototype intended to demonstrate the full
feature set end-to-end. For a production deployment at real scale (thousands
of concurrently tracked vessels, sub-minute refresh), consider:

- Moving ingestion to a dedicated worker (e.g. Kafka/Redis stream) writing to
  PostgreSQL/PostGIS rather than recomputing in-process on each Streamlit rerun
- Running the ML models as a separate scheduled/batch service with results
  cached in the database rather than fit-on-the-fly per session
- A real identity provider for authentication instead of the simulated role
  selector in `utils/security.py`
- MLflow (or similar) for real model versioning or A/B testing, referenced by
  the `model_versions` table already present in the schema

## Testing

```bash
python3 tests/test_pipeline.py
```

Covers the simulator, validator, and all three ML models end-to-end.

## Responsible AI

See the in-app **Responsible AI / Model Governance** page for the full
fairness, explainability, and human-in-the-loop documentation. In short: the
Operational Risk Indicator is a behavioral/operational signal only — it never
uses flag state, country, or ownership as an input, and every score shown to
a user is accompanied by its contributing factors.
