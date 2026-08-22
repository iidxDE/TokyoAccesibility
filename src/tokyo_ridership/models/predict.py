"""Prediction wrapper (Phase 5, spec §5.2).

Runs the serialized pipeline on a one-row feature frame and returns a point
prediction plus an asymmetric interval, back-transformed to passengers/day via
``expm1`` (the model predicts ``ln(1+y)``; serving owns the inverse per
CLAUDE.md). ``lower_offset`` is the ``alpha/2`` residual quantile (negative),
``upper_offset`` the ``1-alpha/2`` quantile (positive) — the interval is
asymmetric and stays that way.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def predict(bundle: dict[str, Any], feature_row: pd.DataFrame) -> dict[str, float]:
    """Point prediction + conformal interval, in passengers/day.

    ``feature_row`` is a one-row DataFrame carrying exactly the bundle's
    ``numeric_features + categorical_features`` (the pipeline selects columns by
    name, so order is irrelevant but every named feature must be present).
    ``expm1`` can dip below zero for very small predictions, so the point and
    lower bound are clipped at the ridership floor of 0 (spec §7.8).
    """
    y_log = float(bundle["pipeline"].predict(feature_row)[0])
    offsets = bundle["interval_offsets"]

    point = max(float(np.expm1(y_log)), 0.0)
    lower = max(float(np.expm1(y_log + offsets["lower_offset"])), 0.0)
    upper = float(np.expm1(y_log + offsets["upper_offset"]))
    return {
        "prediction": point,
        "lower": lower,
        "upper": upper,
        "level": float(offsets["level"]),
    }
