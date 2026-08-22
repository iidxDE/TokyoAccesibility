"""e-Stat census loader (Phase 1).

Ports the data-intake half of ``data/feature_build.py::population_features``:
Shift-JIS (cp932) decode, drop the metadata row embedded after the header, and
filter to the chome (district) aggregation level (``HYOSYO == "4"``) to avoid
double-counting parent rows at ward/oaza levels. Joins the chome population onto
its boundary polygons and writes a geoparquet. The 800 m catchment aggregation
around each station is a Phase 2 feature (``features/accessibility.py``).
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

from tokyo_ridership.config import interim_path, load_config, raw_path

# e-Stat 2020 census: total population of the chome.
POP_COL = "T001081001"
CHOME_LEVEL = "4"  # HYOSYO code for the finest (chome) aggregation level
CHOME_KEY_LEN = 11  # chome KEY_CODEs are 11 digits; shorter = parent aggregates


def load_population(cfg: dict[str, Any]) -> pd.DataFrame:
    """Return chome-level population as ``KEY_CODE`` -> ``population``."""
    path = raw_path(cfg, cfg["sources"]["census_population"])
    df = pd.read_csv(
        path,
        encoding="cp932",
        dtype=str,
        skiprows=[1],  # drop the metadata row embedded after the header
        na_values=["-", "*", "X"],
    )
    if "HYOSYO" in df.columns:
        df = df[df["HYOSYO"] == CHOME_LEVEL].copy()

    col = POP_COL if POP_COL in df.columns else None
    if col is None:
        col = next((c for c in df.columns if c.startswith("T001")), None)
    if col is None:
        raise ValueError("no population column (T001*) found in census file")

    df["population"] = pd.to_numeric(df[col], errors="coerce")
    return df[["KEY_CODE", "population"]].dropna()


def load_boundaries(cfg: dict[str, Any]) -> gpd.GeoDataFrame:
    """Return chome boundary polygons keyed by ``KEY_CODE`` (WGS84).

    The shapefile also ships coarser parent polygons (2- and 9-digit KEY_CODEs
    for prefecture/oaza levels, covering areas outside the chome-subdivided core
    whose population is only reported at HYOSYO=2). They carry no chome
    population, overlap no populated chome, and would only add mixed-granularity
    rows, so we keep chome granularity only.
    """
    path = raw_path(cfg, cfg["sources"]["census_boundaries"])
    bounds = gpd.read_file(path).to_crs("EPSG:4326")
    bounds["KEY_CODE"] = bounds["KEY_CODE"].astype(str)
    bounds = bounds[bounds["KEY_CODE"].str.len() == CHOME_KEY_LEN]
    return bounds[["KEY_CODE", "geometry"]].reset_index(drop=True)


def build_chome(cfg: dict[str, Any]) -> gpd.GeoDataFrame:
    """Join chome population onto boundary polygons."""
    pop = load_population(cfg)
    bounds = load_boundaries(cfg)
    chome = bounds.merge(pop, on="KEY_CODE", how="left")
    chome["population"] = chome["population"].fillna(0.0)

    # Multipart chome (islands, river-split parcels) arrive as several rows
    # sharing one KEY_CODE and one population. Dissolve to a single geometry
    # carrying population once, so the Phase 2 catchment join can't double-count.
    chome = chome.dissolve(by="KEY_CODE", aggfunc={"population": "first"}).reset_index()
    return chome


def main() -> None:
    cfg = load_config()
    print("[load_census] loading census population + boundaries...")
    chome = build_chome(cfg)
    chome.to_parquet(interim_path(cfg, cfg["interim"]["census_chome"]), index=False)

    n = len(chome)
    with_pop = int(chome["population"].notna().sum())
    print(
        f"[load_census] wrote {n} chome polygons "
        f"({with_pop} with population, {n - with_pop} suppressed/zero, "
        f"total pop {int(chome['population'].sum()):,})"
    )


if __name__ == "__main__":
    main()
