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

from typing import Any

import geopandas as gpd
import pandas as pd

from tokyo_ridership.config import (
    interim_path,
    load_config,
    processed_path,
    raw_path,
)
from tokyo_ridership.features import accessibility, network_graph


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
    census = gpd.read_parquet(interim_path(cfg, interim["census_chome"]))
    ridership = pd.read_parquet(interim_path(cfg, interim["ridership"]))

    yamanote = cfg["yamanote_route_id"]
    radius = cfg["catchment_radius_m"]
    wards_geojson = raw_path(cfg, cfg["sources"]["wards_geojson"])

    base = _station_base(stops)
    connectivity = network_graph.station_connectivity(
        stops, trips, stop_times, yamanote
    )
    distance = accessibility.km_to_yamanote(stops, trips, stop_times, yamanote)
    catchment = accessibility.pop_catchment(stops, census, radius)
    ward = accessibility.assign_ward(stops, wards_geojson)

    matrix = (
        base.merge(connectivity, on="station_key", how="left")
        .merge(distance, on="station_key", how="left")
        .merge(catchment, on="station_key", how="left")
        .merge(ward, on="station_key", how="left")
        .merge(ridership, on="station_key", how="left")
    )
    return matrix


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
