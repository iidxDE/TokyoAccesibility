"""GTFS parse + station-key unification (Phase 1).

Ports ``data/feature_build.py::load_gtfs`` and the bbox pre-filter in
``data/filter_gtfs.py`` into the shared package. Reads the raw Kanto-wide GTFS,
filters stops to the Tokyo bounding box, cascades that filter to
``stop_times`` (chunked) and ``trips``, derives a unified ``station_key`` (and a
Japanese ``jp_name`` for ridership matching) from the bilingual ``stop_name``
field, and writes clean intermediate tables to ``data/interim``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tokyo_ridership.config import interim_path, load_config, raw_path

# stop_times is Kanto-wide (~100M+ rows across the region); read it in chunks.
CHUNK_SIZE = 500_000


def station_key(stop_name: str | float) -> str | None:
    """English portion of a bilingual ``"<Kanji> <English>"`` stop name.

    Collapses per-line records (e.g. ``Yamanote.Tokyo`` and ``Tokyo``) under one
    key. Falls back to the whole name when there is no space separator.
    """
    if pd.isna(stop_name):
        return None
    parts = str(stop_name).split(" ", 1)
    return parts[1].strip() if len(parts) > 1 else parts[0].strip()


def jp_name(stop_name: str | float) -> str | None:
    """Japanese (kanji) portion of a bilingual stop name."""
    if pd.isna(stop_name):
        return None
    return str(stop_name).split(" ", 1)[0].strip()


def filter_stops_to_bbox(stops: pd.DataFrame, bbox: dict[str, float]) -> pd.DataFrame:
    """Keep only stops whose coordinates fall inside the Tokyo bounding box."""
    lat = stops["stop_lat"].astype(float)
    lon = stops["stop_lon"].astype(float)
    mask = lat.between(bbox["min_lat"], bbox["max_lat"]) & lon.between(
        bbox["min_lon"], bbox["max_lon"]
    )
    return stops.loc[mask].copy()


def _keep_stop_ids(stops: pd.DataFrame) -> set[str]:
    keep = set(stops["stop_id"].astype(str))
    if "parent_station" in stops.columns:
        parents = stops["parent_station"].dropna().astype(str)
        keep.update(parents[parents != ""].tolist())
    return keep


def _filter_stop_times(
    path: Path, keep_ids: set[str], chunk_size: int = CHUNK_SIZE
) -> tuple[pd.DataFrame, set[str]]:
    """Chunked read of stop_times, keeping rows at the retained stops."""
    frames: list[pd.DataFrame] = []
    trip_ids: set[str] = set()
    for chunk in pd.read_csv(path, dtype=str, chunksize=chunk_size):
        hit = chunk[chunk["stop_id"].isin(keep_ids)]
        if not hit.empty:
            frames.append(hit)
            trip_ids.update(hit["trip_id"].unique().tolist())
    stop_times = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["trip_id", "stop_id", "stop_sequence"])
    )
    return stop_times, trip_ids


def load_clean_gtfs(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Return Tokyo-filtered, key-unified GTFS tables.

    Keys: ``stops`` (stop_id, station_key, jp_name, coords), ``stop_times``
    (trip_id, stop_id, stop_sequence), ``trips`` (trip_id, route_id), ``routes``.
    """
    src = cfg["sources"]["gtfs"]

    stops = pd.read_csv(raw_path(cfg, src["stops"]), dtype=str)
    stops = filter_stops_to_bbox(stops, cfg["bbox"])
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)
    stops["station_key"] = stops["stop_name"].map(station_key)
    stops["jp_name"] = stops["stop_name"].map(jp_name)

    keep_ids = _keep_stop_ids(stops)
    stop_times, trip_ids = _filter_stop_times(
        raw_path(cfg, src["stop_times"]), keep_ids
    )

    trips = pd.read_csv(raw_path(cfg, src["trips"]), dtype=str)
    trips = trips[trips["trip_id"].isin(trip_ids)].copy()

    routes = pd.read_csv(raw_path(cfg, src["routes"]), dtype=str)

    stop_cols = ["stop_id", "station_key", "jp_name", "stop_lat", "stop_lon"]
    if "parent_station" in stops.columns:
        stop_cols.append("parent_station")
    st_cols = [c for c in ("trip_id", "stop_id", "stop_sequence") if c in stop_times]
    stop_times_out = stop_times[st_cols].reset_index(drop=True)
    if "stop_sequence" in stop_times_out.columns:
        # stop_sequence is a clean GTFS integer; store it typed rather than str.
        stop_times_out["stop_sequence"] = pd.to_numeric(
            stop_times_out["stop_sequence"]
        ).astype("int32")

    return {
        "stops": stops[stop_cols].reset_index(drop=True),
        "stop_times": stop_times_out,
        "trips": trips[["trip_id", "route_id"]].reset_index(drop=True),
        "routes": routes.reset_index(drop=True),
    }


def main() -> None:
    cfg = load_config()
    print("[load_gtfs] filtering + parsing raw GTFS...")
    tables = load_clean_gtfs(cfg)

    interim = cfg["interim"]
    Path(cfg["paths"]["data_interim"]).mkdir(parents=True, exist_ok=True)
    tables["stops"].to_parquet(interim_path(cfg, interim["gtfs_stops"]), index=False)
    tables["stop_times"].to_parquet(
        interim_path(cfg, interim["gtfs_stop_times"]), index=False
    )
    tables["trips"].to_parquet(interim_path(cfg, interim["gtfs_trips"]), index=False)
    tables["routes"].to_parquet(interim_path(cfg, interim["gtfs_routes"]), index=False)

    n_stations = tables["stops"]["station_key"].nunique()
    print(
        f"[load_gtfs] wrote {len(tables['stops'])} stops "
        f"({n_stations} unique station_key), "
        f"{len(tables['stop_times'])} stop_times, "
        f"{len(tables['trips'])} trips, {len(tables['routes'])} routes"
    )


if __name__ == "__main__":
    main()
