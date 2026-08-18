# Tokyo Station Ridership Prediction

Predicting station-level **daily ridership** across Tokyo's rail network (474
stations, 23 special wards) from transit-network **connectivity**, **geography**,
and **demographics** — then reframing transit equity as an analysis of
prediction residuals.

Two shippable artifacts (planned): a **FastAPI** inference service (the "what-if
new station" siting tool) and a **Streamlit** map frontend that consumes it over
HTTP.

> **Status:** Phase 0 — repository scaffold. The pipeline, models, API, and
> frontend are built out over Phases 1–7. See
> [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the phase ordering and
> [CLAUDE.md](CLAUDE.md) for the architecture rules. A result-first README with
> the demo link lands in Phase 7.

## Quickstart

```bash
# Install the package + all optional groups into your environment.
python -m pip install -e ".[dev,serving,app,models,features]"
# (Windows venv) prefix commands with the interpreter, e.g.:
#   make test PYTHON=.venv/Scripts/python.exe

# Reproducible pipeline (implemented incrementally across phases)
make data        # Phase 1: parse GTFS, MLIT ridership, e-Stat census
make features    # Phase 2: assemble the 474-station feature matrix
make train       # Phase 4: nested spatial CV, select + serialize the pipeline

# Quality gates (same as CI)
make lint        # ruff check
make format      # ruff format
make typecheck   # mypy src
make test        # pytest
```

`make` targets accept `PYTHON=<path>` to pick an interpreter; each also
corresponds to a DVC stage (`dvc repro` runs the whole pipeline).

## Layout

```
src/tokyo_ridership/   data | features | models | serving | viz
config/config.yaml     paths, source filenames, feature lists, serving bounds
params.yaml            swept hyperparameters + CV/interval protocol (DVC-tracked)
dvc.yaml, Makefile     reproducible pipeline entrypoints
data/{raw,interim,processed}/   gitignored payloads, DVC-tracked
tests/                 unit tests (graph construction + 800 m catchment in Phase 2)
app/streamlit_app.py   HTTP client of the API (never imports the model)
docs/                  the source paper (RidershipProject.pdf)
```

## Data

Raw ridership and census sources are large and are **not committed** (they are
gitignored and DVC-tracked). Acquisition instructions live in
[OVERVIEW.md](OVERVIEW.md); the prototype pipeline scripts are under `data/`.
