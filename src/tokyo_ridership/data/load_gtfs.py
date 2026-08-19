"""GTFS parse + station-key unification (Phase 1).

Ports ``data/feature_build.py::load_gtfs`` and the bbox pre-filter in
``data/filter_gtfs.py`` into the shared package. Reads the raw Kanto-wide GTFS,
filters stops to the Tokyo bounding box, cascades that filter to
``stop_times`` (chunked) and ``trips``, derives the canonical ``station_key``
(the Japanese station name) plus a romanized ``label_en`` from the bilingual
``stop_name`` field, and writes clean intermediate tables to ``data/interim``.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from tokyo_ridership.config import interim_path, load_config, raw_path

# stop_times is Kanto-wide (~100M+ rows across the region); read it in chunks.
CHUNK_SIZE = 500_000

# 〈alt-name〉 parentheticals used by the feed, e.g. 明治神宮前〈原宿〉.
_ANGLE_BRACKETS = re.compile(r"〈[^〉]*〉")


def normalize_station_name(name: str | float) -> str | None:
    """Canonicalize a Japanese station name for keying + cross-source matching.

    Applies NFKC (unifies half/full-width and compatibility kanji), drops
    ``〈alt-name〉`` parentheticals, strips all whitespace, and folds the
    place-name small ``ヶ`` into ``ケ``. These are the variants that otherwise
    split one physical station across sources (e.g. ``市ヶ谷`` vs ``市ケ谷``) or
    break the ridership name match (``霞ケ関`` vs ``霞ヶ関``, ``明治神宮前〈原宿〉``
    vs ``明治神宮前``).
    """
    if pd.isna(name):
        return None
    s = unicodedata.normalize("NFKC", str(name))
    s = _ANGLE_BRACKETS.sub("", s)
    s = re.sub(r"[\s　]+", "", s)
    s = s.replace("ヶ", "ケ")
    return s or None


def japanese_name(stop_name: str | float) -> str | None:
    """Canonical station key: the normalized Japanese portion of the name.

    Keying on the Japanese name (rather than the English romanization) unifies
    per-line records robustly: the feed romanizes some hubs inconsistently (e.g.
    ``東京 Tōkyō`` vs ``東京 Tokyo``), which would split one physical station into
    two English keys and corrupt its connectivity/service features. The result is
    run through :func:`normalize_station_name` so Japanese-side variants (ヶ/ケ,
    〈alt-name〉, width) collapse too.
    """
    if pd.isna(stop_name):
        return None
    return normalize_station_name(str(stop_name).split(" ", 1)[0])


def english_label(stop_name: str | float) -> str | None:
    """Romanized English portion, diacritics stripped (``Tōkyō`` -> ``Tokyo``).

    A human-readable display label; ``None`` when the name carries no English
    part. Canonicalized per station in :func:`load_clean_gtfs`.
    """
    if pd.isna(stop_name):
        return None
    parts = str(stop_name).split(" ", 1)
    if len(parts) < 2:
        return None
    eng = parts[1].strip()
    return "".join(
        c for c in unicodedata.normalize("NFKD", eng) if not unicodedata.combining(c)
    )


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

    Keys: ``stops`` (stop_id, station_key, label_en, coords), ``stop_times``
    (trip_id, stop_id, stop_sequence), ``trips`` (trip_id, route_id), ``routes``.
    """
    src = cfg["sources"]["gtfs"]

    stops = pd.read_csv(raw_path(cfg, src["stops"]), dtype=str)
    stops = filter_stops_to_bbox(stops, cfg["bbox"])
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)
    stops["station_key"] = stops["stop_name"].map(japanese_name)
    stops["label_en"] = stops["stop_name"].map(english_label)
    # One canonical English label per station (the most common romanization).
    canonical = (
        stops.dropna(subset=["station_key", "label_en"])
        .groupby("station_key")["label_en"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0])
    )
    stops["label_en"] = stops["station_key"].map(canonical)

    keep_ids = _keep_stop_ids(stops)
    stop_times, trip_ids = _filter_stop_times(
        raw_path(cfg, src["stop_times"]), keep_ids
    )

    trips = pd.read_csv(raw_path(cfg, src["trips"]), dtype=str)
    trips = trips[trips["trip_id"].isin(trip_ids)].copy()

    routes = pd.read_csv(raw_path(cfg, src["routes"]), dtype=str)

    stop_cols = ["stop_id", "station_key", "label_en", "stop_lat", "stop_lon"]
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
