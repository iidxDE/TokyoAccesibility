"""Land-use composition features (Phase 4).

Per-station land-use mix within the catchment buffer, from the collapsed L03-b
mesh: the built-up fraction and the Shannon entropy of the five-class shares (a
diversity / mixed-use signal). Shares are area-weighted by intersection area so
boundary cells don't bias the fractions.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from tokyo_ridership.data.load_landuse import _CLASSES
from tokyo_ridership.features.accessibility import (
    METRIC_CRS,
    _station_geodataframe,
    station_points,
)


def landuse_features(
    stops: pd.DataFrame, landuse_mesh: gpd.GeoDataFrame, radius_m: float
) -> pd.DataFrame:
    """Return ``station_key``, ``landuse_built_frac``, ``landuse_mix``.

    ``landuse_mix`` is Shannon entropy in nats over the five class shares (raw,
    not normalized by ln 5); a single-class catchment -> 0. Stations whose buffer
    intersects no mesh cell get NaN (handled by the pipeline's median imputer).
    """
    pts = station_points(stops)
    buffers = _station_geodataframe(pts).to_crs(METRIC_CRS)
    buffers["geometry"] = buffers.buffer(radius_m)

    mesh = landuse_mesh.to_crs(METRIC_CRS)[["landuse_class", "geometry"]]
    inter = gpd.overlay(buffers[["station_key", "geometry"]], mesh, how="intersection")
    inter["area"] = inter.geometry.area

    # area per (station, class) -> shares over the five classes
    area = (
        inter.groupby(["station_key", "landuse_class"])["area"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=list(_CLASSES), fill_value=0.0)
    )
    shares = area.div(area.sum(axis=1), axis=0)

    p = shares.to_numpy()
    safe = np.where(p > 0, p, 1.0)  # avoid log(0); 0·ln0 := 0 below
    entropy = -np.where(p > 0, p * np.log(safe), 0.0).sum(axis=1)
    features = pd.DataFrame(
        {
            "station_key": area.index,
            "landuse_built_frac": shares["built"].to_numpy(),
            "landuse_mix": entropy,
        }
    )

    # left-join onto the full station set so no-intersection stations stay NaN
    return pd.DataFrame({"station_key": pts["station_key"]}).merge(
        features, on="station_key", how="left"
    )
