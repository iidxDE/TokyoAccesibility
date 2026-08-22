"""Haversine, 800 m catchments, distance-to-Yamanote, ward assignment (Phase 2).

Ports ``data/feature_build.py`` geographic/demographic logic:
``distance_to_yamanote``, the 800 m population catchment from
``population_features``, and the stop->ward spatial join from ``assign_wards``.

The post-paper employment catchment reuses :func:`catchment_sum`; land-use
composition lives in ``features/landuse.py``. POI density is still deferred.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.validation import make_valid

# JGD2011 / Japan Plane Rectangular CS IX — metres, for buffering around Tokyo.
METRIC_CRS = "EPSG:6691"
EARTH_RADIUS_KM = 6371.0


def haversine_km(
    lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
    """Great-circle distance (km) from one point to arrays of points.

    Shared by :func:`km_to_yamanote` (conceptually) and the serving-layer
    network-snap so both measure distance identically.
    """
    lat0r, lon0r = np.radians(lat0), np.radians(lon0)
    latr, lonr = np.radians(lats), np.radians(lons)
    dlat = latr - lat0r
    dlon = lonr - lon0r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0r) * np.cos(latr) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def station_points(stops: pd.DataFrame) -> pd.DataFrame:
    """One representative (lat, lon) per ``station_key`` (first occurrence)."""
    return (
        stops.dropna(subset=["station_key"])
        .groupby("station_key")[["stop_lat", "stop_lon"]]
        .first()
        .reset_index()
    )


def _station_geodataframe(stations: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["stop_lon"], stations["stop_lat"]),
        crs="EPSG:4326",
    )


def km_to_yamanote(
    stops: pd.DataFrame,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    yamanote_route: str,
) -> pd.DataFrame:
    """Great-circle distance (km) from each station to the nearest Yamanote stop."""
    yama_trips = set(trips.loc[trips["route_id"] == yamanote_route, "trip_id"])
    yama_stop_ids = stop_times.loc[
        stop_times["trip_id"].isin(yama_trips), "stop_id"
    ].unique()
    yama_pts = stops.loc[
        stops["stop_id"].isin(yama_stop_ids), ["stop_lat", "stop_lon"]
    ].to_numpy()

    stations = station_points(stops)
    if len(yama_pts) == 0:
        stations["km_to_yamanote"] = np.nan
        return stations[["station_key", "km_to_yamanote"]]

    yama_lat = np.radians(yama_pts[:, 0])
    yama_lon = np.radians(yama_pts[:, 1])
    lat = np.radians(stations["stop_lat"].to_numpy())[:, None]
    lon = np.radians(stations["stop_lon"].to_numpy())[:, None]

    dlat = yama_lat[None, :] - lat
    dlon = yama_lon[None, :] - lon
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat) * np.cos(yama_lat[None, :]) * np.sin(dlon / 2) ** 2
    )
    dist = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    stations["km_to_yamanote"] = dist.min(axis=1)
    return stations[["station_key", "km_to_yamanote"]]


def catchment_sum(
    stops: pd.DataFrame,
    polygons: gpd.GeoDataFrame,
    radius_m: float,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Sum ``value_col`` over polygons intersecting each station's radius buffer."""
    stations = _station_geodataframe(station_points(stops)).to_crs(METRIC_CRS)
    stations["geometry"] = stations.buffer(radius_m)

    layer = polygons.to_crs(METRIC_CRS)[[value_col, "geometry"]]
    joined = gpd.sjoin(stations, layer, how="left", predicate="intersects")
    return joined.groupby("station_key")[value_col].sum().rename(out_col).reset_index()


def pop_catchment(
    stops: pd.DataFrame, census_chome: gpd.GeoDataFrame, radius_m: float
) -> pd.DataFrame:
    """Total population of chome intersecting each station's ``radius_m`` buffer."""
    return catchment_sum(
        stops, census_chome, radius_m, "population", "pop_800m_catchment"
    )


def assign_ward(
    stops: pd.DataFrame,
    wards: str | Path | gpd.GeoDataFrame,
    outside_label: str = "Outside-23",
) -> pd.DataFrame:
    """Spatially join stations to the 23 special-ward polygons.

    ``wards`` is either a path to the ward geojson (read here, as before) or an
    already-loaded GeoDataFrame — the serving layer loads it once at startup and
    passes it in (spec §5.0). Stations outside the 23 wards get ``outside_label``
    (they remain in the dataset with an explicit out-of-ward marker, matching the
    reference matrix).
    """
    if isinstance(wards, gpd.GeoDataFrame):
        wards = wards.copy()  # never mutate the caller's cached layer
    else:
        wards = gpd.read_file(wards)
    wards = wards.to_crs("EPSG:4326")
    wards["geometry"] = wards.geometry.apply(make_valid)
    if "ward_jp" not in wards.columns:
        wards["ward_jp"] = wards["N03_004"]

    stations = _station_geodataframe(station_points(stops))
    joined = gpd.sjoin(
        stations, wards[["ward_jp", "geometry"]], how="left", predicate="within"
    )
    joined = joined.drop_duplicates(subset="station_key")
    joined["ward_jp"] = joined["ward_jp"].fillna(outside_label)
    return joined[["station_key", "ward_jp"]].reset_index(drop=True)
