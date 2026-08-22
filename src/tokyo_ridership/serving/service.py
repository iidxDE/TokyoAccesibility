"""Orchestration layer (Phase 5, spec §5.5).

coords -> ``features/build_features`` (the shared path) -> ``models/predict`` ->
assemble the response. Keeps the API routes thin (CLAUDE.md rule 5): all
orchestration lives here, prediction lives in ``models/predict``. This layer owns
the extrapolation check (it holds the percentile bands) and the JSON-safe echo of
the computed features.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tokyo_ridership.features.build_features import build_single_point_features
from tokyo_ridership.models.predict import predict
from tokyo_ridership.serving.dependencies import AppState
from tokyo_ridership.serving.schemas import (
    Assumptions,
    Extrapolation,
    ExtrapolationDetail,
    FeatureOverrides,
    Interval,
    PredictResponse,
)

_OUTSIDE_WARD = "Outside-23"


def _jsonable(value: Any) -> float | str | None:
    """Convert a feature cell to a JSON-safe scalar (numpy -> python, NaN -> None)."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _extrapolation(
    feature_row: pd.DataFrame,
    numeric_features: list[str],
    pct_bands: dict[str, tuple[float, float]],
) -> Extrapolation:
    """Flag numeric features whose post-override value falls outside p1–p99.

    NaN values are treated as in-band — the pipeline's imputer fills them, so they
    are not extrapolation (spec §5.5 step 3).
    """
    row = feature_row.iloc[0]
    detail: list[ExtrapolationDetail] = []
    for feat in numeric_features:
        band = pct_bands.get(feat)
        if band is None:
            continue
        raw = row[feat]
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            continue
        value = float(raw)
        if math.isnan(value):
            continue
        p1, p99 = band
        if value < p1 or value > p99:
            detail.append(ExtrapolationDetail(name=feat, value=value, p1=p1, p99=p99))
    return Extrapolation(
        flag=len(detail) > 0,
        features=[d.name for d in detail],
        detail=detail,
    )


def predict_ridership(
    lat: float,
    lon: float,
    overrides: FeatureOverrides | None,
    state: AppState,
) -> PredictResponse:
    """Score one coordinate and assemble the full ``PredictResponse``."""
    override_map = (
        overrides.model_dump(exclude_none=True) if overrides is not None else None
    )

    feature_row, meta = build_single_point_features(
        lat,
        lon,
        stops=state.stops,
        trips=state.trips,
        stop_times=state.stop_times,
        census=state.census,
        employment=state.employment,
        landuse_mesh=state.landuse_mesh,
        wards=state.wards,
        training_matrix=state.training_matrix,
        bundle=state.bundle,
        yamanote_route=state.yamanote_route,
        radius_m=state.radius_m,
        overrides=override_map,
    )

    point = predict(state.bundle, feature_row)
    extrapolation = _extrapolation(
        feature_row, state.bundle["numeric_features"], state.pct_bands
    )

    features = {col: _jsonable(feature_row.iloc[0][col]) for col in feature_row.columns}
    assumptions = Assumptions(
        snapped_station=str(meta["snapped_station"]),
        snap_distance_m=float(meta["snap_distance_m"]),
        station_mode=str(meta["station_mode"]),
        station_mode_source=str(meta["station_mode_source"]),
        overridden_features=list(meta["overridden_features"]),
        ward_outside_23=bool(meta["ward_jp"] == _OUTSIDE_WARD),
        extrapolation=extrapolation,
    )

    return PredictResponse(
        prediction=point["prediction"],
        interval=Interval(
            lower=point["lower"], upper=point["upper"], level=point["level"]
        ),
        features=features,
        assumptions=assumptions,
        model=str(state.bundle["model_id"]),
    )
