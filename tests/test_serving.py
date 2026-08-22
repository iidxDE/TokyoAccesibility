"""API-level tests for the inference service (Phase 5, spec §6).

Driven through ``fastapi.testclient.TestClient``. Data-gated: the lifespan loads
the real bundle + spatial layers at startup, so the whole module skips when those
gitignored artifacts are absent (e.g. a fresh worktree or CI without the data).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from tokyo_ridership.config import (  # noqa: E402
    interim_path,
    load_config,
    processed_path,
    raw_path,
)

# A coordinate near Shinjuku (central, in every layer).
_CENTRAL = {"latitude": 35.6896, "longitude": 139.7006}
# In serving_bounds but west of the 23 special wards.
_OUTSIDE_WARDS = {"latitude": 35.70, "longitude": 139.52}


def _artifacts_present(cfg: dict) -> bool:
    interim = cfg["interim"]
    required = [
        Path(cfg["paths"]["models"]) / "pipeline.joblib",
        processed_path(cfg, cfg["processed"]["station_features"]),
        interim_path(cfg, interim["gtfs_stops"]),
        interim_path(cfg, interim["gtfs_trips"]),
        interim_path(cfg, interim["gtfs_stop_times"]),
        interim_path(cfg, interim["census_chome"]),
        interim_path(cfg, interim["employment"]),
        interim_path(cfg, interim["landuse_mesh"]),
        raw_path(cfg, cfg["sources"]["wards_geojson"]),
    ]
    return all(Path(p).exists() for p in required)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    if not _artifacts_present(load_config()):
        pytest.skip("serving artifacts (bundle + data layers) not present")
    # Imported here so the module still collects when fastapi is missing.
    from tokyo_ridership.serving.api import app

    with TestClient(app) as test_client:  # enters the startup lifespan
        yield test_client


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_id"]


def test_predict_central(client: TestClient) -> None:
    resp = client.post("/predict", json=_CENTRAL)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] > 0
    assert body["interval"]["lower"] <= body["prediction"] <= body["interval"]["upper"]
    assert body["model"]  # model id stamped (rule 4)
    assert set(body["features"]) >= {"ward_jp", "station_mode", "n_routes"}


def test_out_of_bounds_returns_422(client: TestClient) -> None:
    resp = client.post("/predict", json={"latitude": 10.0, "longitude": 139.7})
    assert resp.status_code == 422


def test_unknown_override_returns_422(client: TestClient) -> None:
    resp = client.post("/predict", json={**_CENTRAL, "overrides": {"not_a_feature": 1}})
    assert resp.status_code == 422


def test_override_past_p99_flags_extrapolation(client: TestClient) -> None:
    resp = client.post(
        "/predict", json={**_CENTRAL, "overrides": {"employment_800m": 1e12}}
    )
    assert resp.status_code == 200
    extrap = resp.json()["assumptions"]["extrapolation"]
    assert extrap["flag"] is True
    assert "employment_800m" in extrap["features"]


def test_outside_wards_is_flagged_not_errored(client: TestClient) -> None:
    resp = client.post("/predict", json=_OUTSIDE_WARDS)
    assert resp.status_code == 200
    assert resp.json()["assumptions"]["ward_outside_23"] is True
