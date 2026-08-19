"""MLIT ridership aggregation + kanji name matching (Phase 1).

Ports ``data/feature_build.py::load_ridership`` and
``match_ridership_to_stations``. Reads the MLIT S12-20 (FY2019) ridership
shapefile, sums figures per station across operators, and matches them to GTFS
stations on the normalized Japanese station name (see
:func:`normalize_station_name`). Writes a per-station target table
(``station_key`` -> ``ridership_2019``, ``log_ridership``).
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from tokyo_ridership.config import interim_path, load_config, raw_path
from tokyo_ridership.data.load_gtfs import normalize_station_name

# MLIT S12-20 field mapping: name / operator / line / FY2019 daily passengers.
RIDERSHIP_COLS = {
    "S12_001": "station_name_jp",
    "S12_002": "operator",
    "S12_003": "line_name",
    "S12_041": "ridership_2019",
}


def load_ridership(cfg: dict[str, Any]) -> pd.DataFrame:
    """Aggregate MLIT FY2019 ridership per normalized Japanese station name.

    Grouping on the normalized name (rather than the raw ``S12_001``) sums
    operators correctly even when they spell the station differently (ヶ/ケ, etc).
    """
    shp = raw_path(cfg, cfg["sources"]["mlit_ridership_shp"])
    rs = gpd.read_file(shp, encoding="cp932").rename(columns=RIDERSHIP_COLS)
    rs["ridership_2019"] = pd.to_numeric(rs["ridership_2019"], errors="coerce")
    rs = rs[rs["ridership_2019"] > 0].dropna(subset=["ridership_2019"])
    rs["station_jp_norm"] = rs["station_name_jp"].map(normalize_station_name)
    return rs.groupby("station_jp_norm", as_index=False).agg(
        ridership_2019=("ridership_2019", "sum"),
        n_operators=("operator", "nunique"),
    )


def match_to_stations(cfg: dict[str, Any], ridership: pd.DataFrame) -> pd.DataFrame:
    """Match aggregated ridership to GTFS stations by the (Japanese) station_key.

    ``station_key`` is the canonical Japanese station name, so it matches the
    MLIT Japanese station name directly (whitespace-stripped).
    """
    stops = pd.read_parquet(interim_path(cfg, cfg["interim"]["gtfs_stops"]))
    station_keys = stops["station_key"].dropna().unique()

    # station_key is already normalized (load_gtfs); MLIT is keyed on the same.
    lookup = ridership.set_index("station_jp_norm")["ridership_2019"].to_dict()

    rows: list[dict[str, Any]] = []
    for sk in station_keys:
        val = lookup.get(sk)
        rows.append(
            {
                "station_key": sk,
                "ridership_2019": val,
                "log_ridership": float(np.log1p(val))
                if val is not None and val > 0
                else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    print("[load_ridership] loading + aggregating MLIT ridership...")
    ridership = load_ridership(cfg)
    matched = match_to_stations(cfg, ridership)
    matched.to_parquet(interim_path(cfg, cfg["interim"]["ridership"]), index=False)

    n = int(matched["ridership_2019"].notna().sum())
    total = len(matched)
    pct = 100 * n / total if total else 0
    print(
        f"[load_ridership] matched ridership for {n}/{total} stations "
        f"({pct:.0f}%) from {len(ridership)} MLIT records"
    )


if __name__ == "__main__":
    main()
