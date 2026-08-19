# Implementation Plan

Tokyo ridership prediction + transit-equity system. Phases are ordered by
dependency: build **in-to-out** so each layer's order enforces the next
(feature interface → harness → experiments → API → frontend → deploy). Do not
reorder — the sequence is what keeps the architecture boundaries clean.

Legend: `[ ]` todo. Filenames are relative to repo root.

---

## Phase 0 — Repository scaffold

Goal: a reproducible skeleton before any modeling.

- [ ] Create package layout and empty modules — `src/tokyo_ridership/` with
      `data/`, `features/`, `models/`, `serving/`, `viz/` subpackages
- [ ] Add packaging + tooling config — `pyproject.toml` (deps, ruff, mypy)
- [ ] Add reproducibility entrypoints — `Makefile`, `dvc.yaml`, `params.yaml`
- [ ] Add central config — `config/config.yaml` (paths, feature lists, params)
- [ ] Add `.gitignore` (ignore `data/`, `models/`), and `data/{raw,interim,processed}/` dirs
- [ ] Add project guidance file — `CLAUDE.md` (already drafted)
- [ ] Stub the CI workflow — `.github/workflows/ci.yml` (lint + test)

## Phase 1 — Data loading

Goal: raw sources → clean intermediate tables. Port existing notebook logic.

- [x] GTFS parse (stops/stop_times/trips/routes, station-key unification) — `src/tokyo_ridership/data/load_gtfs.py`
- [x] MLIT ridership aggregation + kanji name matching — `src/tokyo_ridership/data/load_ridership.py`
- [x] e-Stat census loader (Shift-JIS decode, chōme filtering) — `src/tokyo_ridership/data/load_census.py`
- [x] Wire `make data` to run the three loaders — `Makefile`

## Phase 2 — Feature engineering (the shared path)

Goal: one feature-computation module used by BOTH training and serving.
This is the anti-skew boundary — design the interface as
`coordinates/context → feature vector` so a single point can be scored later.

- [ ] Route-level graph, centrality, BFS-to-Yamanote — `src/tokyo_ridership/features/network_graph.py`
- [ ] Haversine, 800 m catchments, distance-to-Yamanote — `src/tokyo_ridership/features/accessibility.py`
- [ ] **New features**: employment/daytime pop (e-Stat Economic Census), POI density (OSMnx), land-use mix (MLIT NLNI) — `src/tokyo_ridership/features/accessibility.py`
- [ ] Assemble the 474-station feature matrix — `src/tokyo_ridership/features/build_features.py`
- [ ] Unit tests for graph construction + catchment logic — `tests/test_features.py`
- [ ] Wire `make features` — `Makefile`

## Phase 3 — Evaluation harness (build before any experiment)

Goal: the fixed apparatus every model/feature-set passes through. Reusable
functions, not inline cells.

- [ ] Spatial-block fold generator (grid primary, ward folds as robustness check) — `src/tokyo_ridership/models/evaluate.py`
- [ ] Nested spatial CV (inner tuning loop + outer eval loop, same block logic) — `src/tokyo_ridership/models/evaluate.py`
- [ ] Prediction intervals (conformal via MAPIE or quantile objective) — `src/tokyo_ridership/models/evaluate.py`
- [ ] Residual diagnostics: Moran's I + LISA on out-of-fold residuals — `src/tokyo_ridership/models/evaluate.py`
- [ ] Single-call interface: `run(feature_set, model) → scores + intervals + residuals`

## Phase 4 — Model experiments + selection

Goal: run experiments through the Phase 3 harness; produce the servable model.

- [ ] Model roster: Dummy baseline, Ridge, RandomForest, LightGBM — `src/tokyo_ridership/models/train.py`
- [ ] **Endogeneity experiment**: run full / no-`daily_stop_events` / stop-events-only through the harness — `src/tokyo_ridership/models/train.py`
- [ ] Select servable model = best **no-`daily_stop_events`** model (usable for greenfield stations)
- [ ] Serialize the full pipeline (scaler + encoder + model) with a model id — `models/pipeline.joblib`
- [ ] SHAP + residual export for the map/writeup — `src/tokyo_ridership/models/explain.py`
- [ ] Wire `make train` — `Makefile`

## Phase 5 — FastAPI service

Goal: thin, typed inference service around the coordinates → prediction use case.

- [ ] Request/response Pydantic models (lat/lng + optional overrides; prediction, back-transformed interval, echoed features, assumptions, model id) — `src/tokyo_ridership/serving/schemas.py`
- [ ] Load model once at startup, inject into routes — `src/tokyo_ridership/serving/dependencies.py`
- [ ] Prediction wrapper (load pipeline, point + interval, `expm1` back-transform) — `src/tokyo_ridership/models/predict.py`
- [ ] Orchestration: coords → `build_features` → predict → assemble response (network-snap + extrapolation flag) — `src/tokyo_ridership/serving/service.py`
- [ ] Routes: `/predict`, `/health`; bounds-validation 422; CORS config — `src/tokyo_ridership/serving/api.py`
- [ ] Verify live via `/docs` with real Tokyo coordinates

## Phase 6 — Streamlit frontend (HTTP client only)

Goal: static equity map + live what-if, consuming the API over HTTP.

- [ ] Static residual/equity map of 474 stations (precomputed, no API call) — `app/streamlit_app.py`, `src/tokyo_ridership/viz/maps.py`
- [ ] Pin-drop → `requests.post` to API → render prediction + interval + assumptions — `app/streamlit_app.py`
- [ ] Override controls (e.g. hypothetical nearby jobs) → re-call API — `app/streamlit_app.py`
- [ ] Loading spinner + graceful API-error handling; `API_BASE_URL` from env — `app/streamlit_app.py`

## Phase 7 — Deployment + CI/CD

Goal: two processes deployed independently; push → test → redeploy.

- [ ] Dockerfile for the API (bundle package + `models/` + static spatial layers) — `Dockerfile`
- [ ] Finalize CI (ruff + pytest + mypy on push) — `.github/workflows/ci.yml`
- [ ] Deploy API (Render/Railway/Fly), auto-deploy on green main; verify `/docs` at public URL
- [ ] Deploy Streamlit (Community Cloud) with `API_BASE_URL` pointed at the API
- [ ] README: result + demo link first, architecture diagram, map screenshot — `README.md`
- [ ] Operational note: latency, model versioning/rollback, drift monitoring, free-tier cold-start caveat — `README.md`

---

## Deferred / out of scope (roadmap only — do not build now)

- [ ] Variogram-driven spatial-block size (defer until building the harness)
- [ ] GTFS travel-time accessibility (r5r / OpenTripPlanner) — new feature family
- [ ] Spatial econometric baseline (spreg spatial lag/error)
- [ ] Station typology clustering (k-means / HDBSCAN) for within-type residuals
- [ ] Temporal pivot (multi-year panel / OD flows) — data-gated, separate project
- [ ] Weather / events / calendar features — only meaningful for a temporal model

## Definition of done (model core, before serving)

- [ ] Feature matrix includes employment + land use
- [ ] Nested spatial CV runs with tuning and produces intervals
- [ ] Three-way endogeneity comparison run; servable model chosen
- [ ] Residuals carry uncertainty bands and are tested for spatial clustering
- [ ] Whole pipeline serialized as one artifact with a model id