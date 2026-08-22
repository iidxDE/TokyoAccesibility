"""Startup wiring: load the model once, inject into routes (Phase 5, spec §5.4).

The servable pipeline **and** all static spatial layers load a single time at
startup into ``app.state`` and are reused across requests (CLAUDE.md rule 6).
Nothing mutates ``app.state`` after startup, so concurrent requests read it
without locking (rule 7). ``get_state`` is the FastAPI dependency the routes and
service use to receive the loaded objects.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request

from tokyo_ridership.config import (
    interim_path,
    load_config,
    processed_path,
    raw_path,
)

BUNDLE_FILENAME = "pipeline.joblib"


@dataclass(frozen=True)
class AppState:
    """Everything loaded once at startup, held read-only for the process life."""

    cfg: dict[str, Any]
    bundle: dict[str, Any]
    stops: pd.DataFrame
    trips: pd.DataFrame
    stop_times: pd.DataFrame
    training_matrix: pd.DataFrame
    census: gpd.GeoDataFrame
    employment: gpd.GeoDataFrame
    landuse_mesh: gpd.GeoDataFrame
    wards: gpd.GeoDataFrame
    yamanote_route: str
    radius_m: float
    # {feature: (p1, p99)} over the training matrix — the extrapolation band.
    pct_bands: dict[str, tuple[float, float]] = field(default_factory=dict)


def _percentile_bands(
    matrix: pd.DataFrame, numeric_features: list[str]
) -> dict[str, tuple[float, float]]:
    """1st/99th percentile of each numeric feature (NaN-skipping, spec §2.5)."""
    bands: dict[str, tuple[float, float]] = {}
    for feat in numeric_features:
        col = matrix[feat].to_numpy(dtype="float64")
        if np.isnan(col).all():
            continue
        p1, p99 = np.nanpercentile(col, [1.0, 99.0])
        bands[feat] = (float(p1), float(p99))
    return bands


def load_state(cfg: dict[str, Any] | None = None) -> AppState:
    """Load the bundle + static layers + percentile bands into an ``AppState``.

    Fails fast if the serialized bundle is missing — never serve a broken model
    (spec §7.10).
    """
    cfg = cfg if cfg is not None else load_config()

    bundle_path = Path(cfg["paths"]["models"]) / BUNDLE_FILENAME
    if not bundle_path.exists():
        raise RuntimeError(
            f"model bundle not found at {bundle_path}; run `make train` first"
        )
    bundle: dict[str, Any] = joblib.load(bundle_path)

    interim = cfg["interim"]
    stops = pd.read_parquet(interim_path(cfg, interim["gtfs_stops"]))
    trips = pd.read_parquet(interim_path(cfg, interim["gtfs_trips"]))
    stop_times = pd.read_parquet(interim_path(cfg, interim["gtfs_stop_times"]))
    training_matrix = pd.read_parquet(
        processed_path(cfg, cfg["processed"]["station_features"])
    )
    census = gpd.read_parquet(interim_path(cfg, interim["census_chome"]))
    employment = gpd.read_parquet(interim_path(cfg, interim["employment"]))
    landuse_mesh = gpd.read_parquet(interim_path(cfg, interim["landuse_mesh"]))
    # Read the raw ward geojson once; assign_ward accepts a preloaded layer (§5.0).
    wards = gpd.read_file(raw_path(cfg, cfg["sources"]["wards_geojson"]))

    return AppState(
        cfg=cfg,
        bundle=bundle,
        stops=stops,
        trips=trips,
        stop_times=stop_times,
        training_matrix=training_matrix,
        census=census,
        employment=employment,
        landuse_mesh=landuse_mesh,
        wards=wards,
        yamanote_route=cfg["yamanote_route_id"],
        radius_m=float(cfg["catchment_radius_m"]),
        pct_bands=_percentile_bands(training_matrix, bundle["numeric_features"]),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan: load everything once on startup, hold for reuse."""
    app.state.app_state = load_state()
    yield


def get_state(request: Request) -> AppState:
    """Dependency: return the ``AppState`` loaded at startup."""
    state: AppState = request.app.state.app_state
    return state
