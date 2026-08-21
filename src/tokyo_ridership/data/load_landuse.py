"""MLIT L03-b land-use mesh loader (Phase 4).

Reads the 5339 first-order-mesh land-use polygons, clips to the Tokyo bounding
box (dropping the large forest/sea tail), collapses the raw land-use code to five
classes, and writes a metric-CRS geoparquet for the per-station land-use overlay
in ``features/landuse.py``.

Source note: the shipped ``.shp`` lacks its ``.dbf`` (no attributes), so we read
the self-contained ``.geojson``; its class field is Japanese (``土地利用種別``).
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd

from tokyo_ridership.config import interim_path, load_config, raw_path
from tokyo_ridership.features.accessibility import METRIC_CRS

# Class field name in the geojson (falls back to the shapefile's L03b_002).
_FIELD_CANDIDATES = ("土地利用種別", "L03b_002")

# Official L03-b-21 land-use codes collapsed to 5 classes. Verified against the
# code distribution and the MLIT LandUseCd codelist; this corrects the draft
# spec mapping (1100 rivers/lakes -> water, 1000 other-land -> other,
# 1600 golf course -> green). Unlisted codes fall through to "other".
_LANDUSE_CLASS = {
    "0700": "built",
    "0901": "transport",
    "0902": "transport",
    "0100": "green",
    "0200": "green",
    "0500": "green",
    "0600": "green",
    "1600": "green",
    "1100": "water",
    "1400": "water",
    "1500": "water",
    "1000": "other",
}
_CLASSES = ("built", "transport", "green", "water", "other")


def _class_field(gdf: gpd.GeoDataFrame) -> str:
    for field in _FIELD_CANDIDATES:
        if field in gdf.columns:
            return field
    raise KeyError(f"no land-use class field found; looked for {_FIELD_CANDIDATES}")


def load_landuse(cfg: dict[str, Any]) -> gpd.GeoDataFrame:
    """Tokyo-clipped land-use cells with a collapsed ``landuse_class`` (metric CRS)."""
    path = raw_path(cfg, cfg["sources"]["landuse_mesh"])
    b = cfg["bbox"]
    # bbox spatial filter at read time (data CRS is geographic lon/lat) so we
    # never materialize the full 630k-cell mesh.
    gdf = gpd.read_file(
        path, bbox=(b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"])
    ).to_crs(METRIC_CRS)

    field = _class_field(gdf)
    gdf["landuse_class"] = gdf[field].map(_LANDUSE_CLASS).fillna("other")
    return gdf[["landuse_class", "geometry"]].reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    print("[load_landuse] loading + clipping land-use mesh to Tokyo bbox...")
    gdf = load_landuse(cfg)
    gdf.to_parquet(interim_path(cfg, cfg["interim"]["landuse_mesh"]), index=False)
    dist = gdf["landuse_class"].value_counts().to_dict()
    print(f"[load_landuse] wrote {len(gdf)} cells | class distribution: {dist}")


if __name__ == "__main__":
    main()
