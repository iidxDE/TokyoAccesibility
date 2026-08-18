# CLAUDE.md

Guidance for working in this repository. Read before writing code.

## Project

Station-level daily ridership prediction and transit-equity analysis for the
Tokyo rail network (474 stations, 23 special wards). Cross-sectional planning
model — **not** a temporal/operations forecaster. Two shippable artifacts: a
FastAPI inference service (the "what-if new station" siting tool) and a
Streamlit map frontend that consumes it over HTTP.

## Quickstart

```bash
# Install (use uv; falls back to pip -e .)
uv sync                      # or: pip install -e ".[dev]"

# Rebuild data → features → model (reproducible pipeline)
make data                    # parse GTFS, MLIT ridership, e-Stat census
make features                # assemble the 474-station feature matrix
make train                   # nested spatial CV, select + serialize pipeline

# Run the two processes (separate terminals)
uvicorn tokyo_ridership.serving.api:app --reload --port 8000
streamlit run app/streamlit_app.py        # expects API_BASE_URL in env

# Test / lint / format
pytest                       # or: make test
ruff check .                 # lint
ruff format .                # format
mypy src/                    # type-check
```

## Architecture rules (do not violate)

1. **Logic lives in `src/`; notebooks only narrate.** No modeling or feature
   logic in notebooks or in the Streamlit app.
2. **One feature-computation path, shared by training and serving.** The API
   MUST compute features by calling the same `src/features/build_features.py`
   used to build the training matrix. Re-implementing feature logic in the
   serving layer causes train-serve skew — never do it.
3. **Streamlit is an HTTP client of the API, never an importer of the model.**
   The frontend calls the API with `requests`; it must not
   `import` from `src/models/`. This service boundary is deliberate.
4. **Serve the no-`daily_stop_events` model.** A hypothetical new station has no
   schedule, so the servable planning model excludes the endogenous
   service-frequency feature. Stamp the model identifier into every response.
5. **The API layer stays thin.** Routes validate and format only; orchestration
   lives in `serving/service.py`, prediction in `models/predict.py`.
6. **Load the model once at startup**, hold it in memory, reuse across requests.
   Never reload per request.
7. **The service is stateless.** Each request carries all its own input; no
   per-user state between calls.
8. **Preprocessing is one serialized `Pipeline`** (scaler + encoder + model via
   joblib), fit inside each CV fold. Never fit scalers on full data before
   splitting.
9. **Both CV loops are spatially blocked.** Nested spatial CV: inner tuning loop
   and outer evaluation loop use the same spatial-block fold logic. No random
   splits anywhere.

## Serving contract

- Request: `latitude`, `longitude` (validated against Tokyo bounds), optional
  `overrides` for hypothetical scenarios.
- Response: `prediction` and `interval` **back-transformed** to passengers/day
  (model predicts `ln(1+y)`; serving owns `expm1`). Interval is asymmetric —
  that is correct, do not force symmetry. Include the confidence `level`, the
  echoed computed `features`, an `assumptions` block (network-snap,
  extrapolation flag, overrides applied), and the `model` identifier.
- Out-of-bounds / malformed input → 422 (Pydantic). Valid-but-extrapolating
  input → 200 with a warning in `assumptions`, not an error.
- Provide a `/health` endpoint.

## File structure

```
tokyo-ridership/
├── README.md                 # result + demo link first, then setup
├── pyproject.toml            # deps + package metadata
├── Makefile                  # data / features / train / test targets
├── dvc.yaml, params.yaml     # reproducible pipeline + hyperparameters
├── .github/workflows/ci.yml  # lint + test on push
├── config/config.yaml        # paths, feature lists, model params
├── data/{raw,interim,processed}/   # gitignored, DVC-tracked
├── src/tokyo_ridership/
│   ├── data/                 # load_gtfs, load_ridership, load_census
│   ├── features/             # network_graph, accessibility, build_features
│   ├── models/               # train, evaluate, explain, predict
│   ├── serving/              # api, schemas, service, dependencies
│   └── viz/                  # residual + prediction maps
├── models/                   # serialized pipeline (joblib)
├── notebooks/                # ONE narrative notebook; no core logic
├── app/streamlit_app.py      # thin HTTP client of the API
├── tests/                    # unit tests on graph + catchment logic
└── reports/{figures,paper.pdf}
```

## Coding style

- Python, `ruff` for lint + format, `mypy` for types on `src/`.
- Type-hint public functions; validate all external input with Pydantic.
- Config and URLs (esp. `API_BASE_URL`) come from env/config — never hardcoded.
- No secrets in code. Configure CORS on the API for the Streamlit origin.
- Unit-test the non-obvious logic: route-graph construction and the 800 m
  catchment computation at minimum.
- Keep dependencies pinned; the pipeline must regenerate end-to-end from raw
  data.

## Scope guardrails

- Do not add weather, events, or calendar features — meaningless for an annual
  cross-sectional snapshot. They belong only to a temporal model, which is out
  of scope.
- Do not over-engineer deployment (no Kubernetes/Airflow). Target: Dockerized
  API on Render/Railway/Fly, Streamlit on Community Cloud, auto-deploy on green
  CI.