# Reproducible pipeline + dev tasks. Override the interpreter if needed:
#   make train PYTHON=.venv/Scripts/python.exe    (Windows)
#   make train PYTHON=.venv/bin/python            (POSIX)
PYTHON ?= python

.PHONY: help install data features train test lint format typecheck serve app clean

help:
	@echo "Targets: install data features train test lint format typecheck serve app clean"

install:
	$(PYTHON) -m pip install -e ".[dev,serving,app,models,features]"

# Pipeline (Phase 1 -> 4). Each target is also a DVC stage; `dvc repro` runs all.
data:
	$(PYTHON) -m tokyo_ridership.data.load_gtfs
	$(PYTHON) -m tokyo_ridership.data.load_ridership
	$(PYTHON) -m tokyo_ridership.data.load_census

features:
	$(PYTHON) -m tokyo_ridership.features.build_features

train:
	$(PYTHON) -m tokyo_ridership.models.train

# Quality gates (match CI).
test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

typecheck:
	$(PYTHON) -m mypy src

# Serving (Phase 5) + frontend (Phase 6).
serve:
	$(PYTHON) -m uvicorn tokyo_ridership.serving.api:app --reload --port 8000

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

clean:
	rm -rf data/interim/* data/processed/* models/*.joblib
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
