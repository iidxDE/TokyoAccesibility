"""Assemble the station feature matrix — the shared path (Phase 2).

THE anti train-serve-skew boundary (CLAUDE.md rule 2). Consumes the Phase 1
interim tables and produces one row per station with connectivity, service,
geography, demography, ward, and target columns.

The computation is split into per-concern functions (connectivity, distance,
catchment, ward) that operate on station points + prebuilt spatial layers, so
the serving layer can reuse the same functions to score a single hypothetical
point (with a network-snap for connectivity) rather than re-implementing feature
logic. Serving that single-point path lands in Phase 5.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from tokyo_ridership.config import (
    interim_path,
    load_config,
    processed_path,
    raw_path,
)
from tokyo_ridership.features import accessibility, landuse, network_graph

# Connectivity features snapped from the nearest existing station — these are the
# training-time values, so the snap is skew-free by construction (spec §2.1, §5.1).
_CONNECTIVITY_SNAP = [
    "n_routes",
    "min_transfers_yamanote",
    "mean_route_degree_cent",
    "mean_route_btw_cent",
    "max_route_btw_cent",
]
_QUERY_KEY = "__query__"


def _station_base(stops: pd.DataFrame) -> pd.DataFrame:
    """One row per station: key (Japanese), English label, representative coords."""
    return (
        stops.dropna(subset=["station_key"])
        .groupby("station_key")
        .agg(
            label_en=("label_en", "first"),
            stop_lat=("stop_lat", "first"),
            stop_lon=("stop_lon", "first"),
        )
        .reset_index()
    )


def build_feature_matrix(cfg: dict[str, Any]) -> pd.DataFrame:
    """Assemble the full station feature matrix from Phase 1 intermediates."""
    interim = cfg["interim"]
    stops = pd.read_parquet(interim_path(cfg, interim["gtfs_stops"]))
    trips = pd.read_parquet(interim_path(cfg, interim["gtfs_trips"]))
    stop_times = pd.read_parquet(interim_path(cfg, interim["gtfs_stop_times"]))
    routes = pd.read_parquet(interim_path(cfg, interim["gtfs_routes"]))
    census = gpd.read_parquet(interim_path(cfg, interim["census_chome"]))
    employment = gpd.read_parquet(interim_path(cfg, interim["employment"]))
    landuse_mesh = gpd.read_parquet(interim_path(cfg, interim["landuse_mesh"]))
    ridership = pd.read_parquet(interim_path(cfg, interim["ridership"]))

    yamanote = cfg["yamanote_route_id"]
    radius = cfg["catchment_radius_m"]
    wards_geojson = raw_path(cfg, cfg["sources"]["wards_geojson"])

    base = _station_base(stops)
    connectivity = network_graph.station_connectivity(
        stops, trips, stop_times, yamanote
    )
    mode = network_graph.station_modes(stops, trips, stop_times, routes)
    distance = accessibility.km_to_yamanote(stops, trips, stop_times, yamanote)
    catchment = accessibility.pop_catchment(stops, census, radius)
    employment_catchment = accessibility.catchment_sum(
        stops, employment, radius, "employment", "employment_800m"
    )
    landuse_feats = landuse.landuse_features(stops, landuse_mesh, radius)
    ward = accessibility.assign_ward(stops, wards_geojson)

    matrix = (
        base.merge(connectivity, on="station_key", how="left")
        .merge(mode, on="station_key", how="left")
        .merge(distance, on="station_key", how="left")
        .merge(catchment, on="station_key", how="left")
        .merge(employment_catchment, on="station_key", how="left")
        .merge(landuse_feats, on="station_key", how="left")
        .merge(ward, on="station_key", how="left")
        .merge(ridership, on="station_key", how="left")
    )
    return matrix


def _query_stops_frame(lat: float, lon: float) -> pd.DataFrame:
    """A one-row ``stops``-shaped frame for the query point.

    This is the input shape every per-station feature function expects, so the
    serving layer reuses the exact training-time functions (CLAUDE.md rule 2).
    """
    return pd.DataFrame(
        {
            "stop_id": [_QUERY_KEY],
            "station_key": [_QUERY_KEY],
            "stop_lat": [lat],
            "stop_lon": [lon],
        }
    )


def _query_scalar(df: pd.DataFrame, col: str) -> Any:
    """Pull the query row's value of ``col`` from a per-station feature frame."""
    return df.loc[df["station_key"] == _QUERY_KEY, col].iloc[0]


def _snap_nearest(
    lat: float,
    lon: float,
    training_matrix: pd.DataFrame,
    connectivity_cols: list[str] = _CONNECTIVITY_SNAP,
    mode_col: str = "station_mode",
) -> dict[str, Any]:
    """Snap connectivity + mode from the geographically nearest existing station.

    The snapped values are read straight from the training matrix row, so they
    are identical to what training saw (skew-free). A NaN connectivity value
    (isolated route / unreachable from Yamanote) is left NaN — the pipeline's
    imputer fills it downstream (spec §5.1, §7.5).
    """
    lats = training_matrix["stop_lat"].to_numpy(dtype="float64")
    lons = training_matrix["stop_lon"].to_numpy(dtype="float64")
    dist_km = accessibility.haversine_km(lat, lon, lats, lons)
    idx = int(np.argmin(dist_km))
    row = training_matrix.iloc[idx]

    snapped: dict[str, Any] = {
        "station_key": row["station_key"],
        "snap_distance_m": float(dist_km[idx] * 1000.0),
        mode_col: row[mode_col],
    }
    for col in connectivity_cols:
        snapped[col] = row[col]
    return snapped


def build_single_point_features(
    lat: float,
    lon: float,
    *,
    stops: pd.DataFrame,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    census: gpd.GeoDataFrame,
    employment: gpd.GeoDataFrame,
    landuse_mesh: gpd.GeoDataFrame,
    wards: gpd.GeoDataFrame,
    training_matrix: pd.DataFrame,
    bundle: dict[str, Any],
    yamanote_route: str,
    radius_m: float,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute the servable feature vector for one coordinate (spec §5.1).

    The anti-skew boundary: all feature logic stays in ``features/``; the serving
    layer only calls this. Returns ``(feature_row, meta)`` where ``feature_row``
    is a one-row DataFrame whose columns are exactly the bundle's
    ``numeric_features + categorical_features`` (order and names taken from the
    bundle so the row can never drift from what the pipeline expects).
    """
    query = _query_stops_frame(lat, lon)

    # 1. Coordinate-derived six. The catchment / land-use / ward functions are
    # per-station independent, so the one-row query is correct and cheap.
    pop = accessibility.pop_catchment(query, census, radius_m)
    emp = accessibility.catchment_sum(
        query, employment, radius_m, "employment", "employment_800m"
    )
    lu = landuse.landuse_features(query, landuse_mesh, radius_m)
    ward = accessibility.assign_ward(query, wards)
    # km_to_yamanote is NOT independent: it derives the Yamanote anchor from the
    # frame passed in, so append the query to the real stops and take its row.
    km = accessibility.km_to_yamanote(
        pd.concat([stops, query], ignore_index=True), trips, stop_times, yamanote_route
    )

    # 2. Connectivity five + mode via the network snap.
    snapped = _snap_nearest(lat, lon, training_matrix)

    # 3. Assemble the row from names read off the bundle (no drift, no extras).
    values: dict[str, Any] = {
        "km_to_yamanote": _query_scalar(km, "km_to_yamanote"),
        "pop_800m_catchment": _query_scalar(pop, "pop_800m_catchment"),
        "employment_800m": _query_scalar(emp, "employment_800m"),
        "landuse_built_frac": _query_scalar(lu, "landuse_built_frac"),
        "landuse_mix": _query_scalar(lu, "landuse_mix"),
        "ward_jp": _query_scalar(ward, "ward_jp"),
        "station_mode": snapped["station_mode"],
        **{col: snapped[col] for col in _CONNECTIVITY_SNAP},
    }

    feature_cols = bundle["numeric_features"] + bundle["categorical_features"]
    mode_overridden = False
    overridden: list[str] = []

    # 4. Apply overrides last, after computation and snap.
    if overrides:
        for key, val in overrides.items():
            if val is None:
                continue
            if key not in feature_cols:  # pydantic already forbids this; guard anyway
                raise KeyError(f"unknown override feature: {key!r}")
            values[key] = val
            overridden.append(key)
            if key == "station_mode":
                mode_overridden = True

    feature_row = pd.DataFrame([{col: values[col] for col in feature_cols}])[
        feature_cols
    ]

    ward_jp = values["ward_jp"]
    meta: dict[str, Any] = {
        "snapped_station": snapped["station_key"],
        "snap_distance_m": snapped["snap_distance_m"],
        "station_mode": values["station_mode"],
        "station_mode_source": (
            "override" if mode_overridden else f"snapped:{snapped['station_key']}"
        ),
        "overridden_features": overridden,
        "ward_jp": ward_jp,
        "ward_outside": bool(ward_jp == "Outside-23"),
    }
    return feature_row, meta


def main() -> None:
    cfg = load_config()
    print("[build_features] assembling station feature matrix...")
    matrix = build_feature_matrix(cfg)

    out = processed_path(cfg, cfg["processed"]["station_features"])
    out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out, index=False)

    n = len(matrix)
    with_target = int(matrix["ridership_2019"].notna().sum())
    print(
        f"[build_features] wrote {n} stations x {matrix.shape[1]} columns "
        f"({with_target} with ridership target) -> {out}"
    )


if __name__ == "__main__":
    main()
