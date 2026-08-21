"""Economic Census 2021 employment loader (Phase 4).

Reads the Economic Census small-area **boundary** shapefile (e-Stat, 経済センサス
‐活動調査 2021, 小地域), which carries workplace employment (``JUGYOSHA``) and
establishments (``JIGYOSHO``) per chome polygon. Because the geometry and the
counts ship together, employment attaches to its own chome geometry directly —
no cross-census code matching (the Economic and Population Censuses use
incompatible chome codes). Writes a metric-CRS geoparquet for the 800 m
employment catchment in ``features/build_features.py``.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

from tokyo_ridership.config import interim_path, load_config, raw_path
from tokyo_ridership.features.accessibility import METRIC_CRS

# 特別区部 total employees (reconciliation anchor for the 23 wards).
SPECIAL_WARDS_EMPLOYMENT = 8_493_109


def load_employment(cfg: dict[str, Any]) -> gpd.GeoDataFrame:
    """Chome polygons carrying ``employment`` + ``establishments`` (metric CRS)."""
    path = raw_path(cfg, cfg["sources"]["economic_census_boundary"])
    gdf = gpd.read_file(path)
    gdf["employment"] = pd.to_numeric(gdf["JUGYOSHA"], errors="coerce").fillna(0.0)
    gdf["establishments"] = pd.to_numeric(gdf["JIGYOSHO"], errors="coerce").fillna(0.0)
    return gdf[["KEY_CODE", "CITY", "employment", "establishments", "geometry"]].to_crs(
        METRIC_CRS
    )


def main() -> None:
    cfg = load_config()
    print("[load_employment] loading Economic Census boundary + employment...")
    gdf = load_employment(cfg)
    gdf.to_parquet(interim_path(cfg, cfg["interim"]["employment"]), index=False)

    ward_total = float(
        gdf.loc[gdf["CITY"].astype(int).between(101, 123), "employment"].sum()
    )
    diff_pct = 100 * (ward_total - SPECIAL_WARDS_EMPLOYMENT) / SPECIAL_WARDS_EMPLOYMENT
    print(
        f"[load_employment] wrote {len(gdf)} chome polygons "
        f"({int((gdf['employment'] > 0).sum())} with employment > 0)"
    )
    print(
        f"  23-ward sum = {ward_total:,.0f}  vs 特別区部 anchor = "
        f"{SPECIAL_WARDS_EMPLOYMENT:,.0f}  (diff {diff_pct:+.2f}%)"
    )


if __name__ == "__main__":
    main()
