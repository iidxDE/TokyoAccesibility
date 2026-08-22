"""Anti-skew parity test for the single-point feature path (Phase 5, spec §6).

The most important test in Phase 5: feed a known station's own coordinates
through ``build_single_point_features`` and assert the result reproduces that
station's row in the training matrix. This proves training and serving compute
the same numbers (CLAUDE.md rule 2). Data-gated — skips when the gitignored
artifacts are absent (e.g. a fresh worktree or CI without the data pull).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pytest

from tokyo_ridership.config import (
    interim_path,
    load_config,
    processed_path,
    raw_path,
)
from tokyo_ridership.features.build_features import build_single_point_features

# The six features computed from the coordinate (not snapped) — the parity target.
_COORD_FEATURES = [
    "km_to_yamanote",
    "pop_800m_catchment",
    "employment_800m",
    "landuse_built_frac",
    "landuse_mix",
    "ward_jp",
]


@pytest.fixture(scope="module")
def inputs() -> dict[str, Any]:
    """Load the real artifacts, or skip if any is missing."""
    cfg = load_config()
    interim = cfg["interim"]
    paths = {
        "matrix": processed_path(cfg, cfg["processed"]["station_features"]),
        "bundle": Path(cfg["paths"]["models"]) / "pipeline.joblib",
        "stops": interim_path(cfg, interim["gtfs_stops"]),
        "trips": interim_path(cfg, interim["gtfs_trips"]),
        "stop_times": interim_path(cfg, interim["gtfs_stop_times"]),
        "census": interim_path(cfg, interim["census_chome"]),
        "employment": interim_path(cfg, interim["employment"]),
        "landuse_mesh": interim_path(cfg, interim["landuse_mesh"]),
        "wards": raw_path(cfg, cfg["sources"]["wards_geojson"]),
    }
    for name, path in paths.items():
        if not Path(path).exists():
            pytest.skip(f"missing data artifact ({name}): {path}")

    return {
        "cfg": cfg,
        "training_matrix": pd.read_parquet(paths["matrix"]),
        "bundle": joblib.load(paths["bundle"]),
        "stops": pd.read_parquet(paths["stops"]),
        "trips": pd.read_parquet(paths["trips"]),
        "stop_times": pd.read_parquet(paths["stop_times"]),
        "census": gpd.read_parquet(paths["census"]),
        "employment": gpd.read_parquet(paths["employment"]),
        "landuse_mesh": gpd.read_parquet(paths["landuse_mesh"]),
        "wards": gpd.read_file(paths["wards"]),
        "yamanote_route": cfg["yamanote_route_id"],
        "radius_m": float(cfg["catchment_radius_m"]),
    }


def _pick_station(matrix: pd.DataFrame) -> pd.Series:
    """A central hub with all coordinate-derived features present (clean parity)."""
    complete = matrix.dropna(subset=_COORD_FEATURES)
    if complete.empty:
        pytest.skip("no station has all coordinate-derived features populated")
    return complete.sort_values("n_routes", ascending=False).iloc[0]


def _call(inputs: dict[str, Any], station: pd.Series, **kw: Any) -> Any:
    return build_single_point_features(
        float(station["stop_lat"]),
        float(station["stop_lon"]),
        stops=inputs["stops"],
        trips=inputs["trips"],
        stop_times=inputs["stop_times"],
        census=inputs["census"],
        employment=inputs["employment"],
        landuse_mesh=inputs["landuse_mesh"],
        wards=inputs["wards"],
        training_matrix=inputs["training_matrix"],
        bundle=inputs["bundle"],
        yamanote_route=inputs["yamanote_route"],
        radius_m=inputs["radius_m"],
        **kw,
    )


def test_columns_are_exactly_the_bundle_features(inputs: dict[str, Any]) -> None:
    station = _pick_station(inputs["training_matrix"])
    row, _ = _call(inputs, station)
    expected = (
        inputs["bundle"]["numeric_features"] + inputs["bundle"]["categorical_features"]
    )
    assert list(row.columns) == expected


def test_snap_selects_the_same_station(inputs: dict[str, Any]) -> None:
    station = _pick_station(inputs["training_matrix"])
    _, meta = _call(inputs, station)
    assert meta["snapped_station"] == station["station_key"]
    assert meta["snap_distance_m"] < 1.0  # querying on the station itself


def test_coordinate_features_reproduce_training_row(inputs: dict[str, Any]) -> None:
    station = _pick_station(inputs["training_matrix"])
    row, _ = _call(inputs, station)
    got = row.iloc[0]
    # ward is categorical — exact match.
    assert got["ward_jp"] == station["ward_jp"]
    # numeric coordinate-derived features reproduce the training values.
    for feat in [f for f in _COORD_FEATURES if f != "ward_jp"]:
        assert np.isclose(
            float(got[feat]), float(station[feat]), rtol=1e-6, atol=1e-6
        ), feat


def test_snapped_connectivity_matches_training_row(inputs: dict[str, Any]) -> None:
    station = _pick_station(inputs["training_matrix"])
    row, _ = _call(inputs, station)
    got = row.iloc[0]
    assert got["station_mode"] == station["station_mode"]
    for feat in [
        "n_routes",
        "mean_route_degree_cent",
        "mean_route_btw_cent",
        "max_route_btw_cent",
    ]:
        assert np.isclose(float(got[feat]), float(station[feat]), rtol=1e-9), feat


def test_override_replaces_only_its_target(inputs: dict[str, Any]) -> None:
    station = _pick_station(inputs["training_matrix"])
    base, _ = _call(inputs, station)
    over, meta = _call(inputs, station, overrides={"employment_800m": 999_999.0})
    assert over.iloc[0]["employment_800m"] == 999_999.0
    assert meta["overridden_features"] == ["employment_800m"]
    # every other column is unchanged.
    for col in base.columns:
        if col == "employment_800m":
            continue
        a, b = base.iloc[0][col], over.iloc[0][col]
        if isinstance(a, float) and np.isnan(a):
            assert isinstance(b, float) and np.isnan(b), col
        else:
            assert a == b, col
