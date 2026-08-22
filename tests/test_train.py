"""Unit tests for the Phase 4 modeling core (registry, variants, selection)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from tokyo_ridership.models import train
from tokyo_ridership.models.registry import build_estimator


def test_feature_set_variants() -> None:
    features = {
        "connectivity": ["a"],
        "service": ["daily_stop_events"],
        "geography": ["b"],
        "demographics": ["c"],
        "landuse": ["d"],
        "categorical": ["ward_jp", "station_mode"],
    }
    variants = train.feature_sets(features)
    assert "daily_stop_events" in variants["full"][0]
    assert "daily_stop_events" not in variants["no_daily_stop_events"][0]
    assert variants["no_daily_stop_events"][1] == ["ward_jp", "station_mode"]
    assert variants["stop_events_only"] == (["daily_stop_events"], [])


def test_model_selection_rule() -> None:
    table = pd.DataFrame(
        [
            # wrong variant, best score -> must NOT be chosen (rule 4)
            {"model": "lgbm", "variant": "full", "r2_mean": 0.90, "mae_log_mean": 0.40},
            {
                "model": "rf",
                "variant": "no_daily_stop_events",
                "r2_mean": 0.70,
                "mae_log_mean": 0.55,
            },
            # tie on r2, lower mae -> wins the tie-break
            {
                "model": "ridge",
                "variant": "no_daily_stop_events",
                "r2_mean": 0.70,
                "mae_log_mean": 0.50,
            },
        ]
    )
    selection = {
        "servable_variant": "no_daily_stop_events",
        "primary_metric": "r2_mean",
        "tie_breaker": "mae_log_mean",
    }
    winner = train.select_servable(table, selection)
    assert winner["model"] == "ridge"


def test_registry_builds_estimators() -> None:
    for name in ("dummy", "ridge", "random_forest", "gradient_boosting"):
        assert build_estimator(name, seed=42, fixed={}) is not None
    try:
        import lightgbm  # noqa: F401

        assert build_estimator("lightgbm", 42, {}) is not None
    except ImportError:
        pass


def test_train_smoke_cvresult() -> None:
    # closes the harness single-call path; needs the [models] extras
    pytest.importorskip("esda")
    pytest.importorskip("libpysal")
    from tokyo_ridership.models import evaluate

    rng = np.random.default_rng(0)
    n = 40
    lat = 35.60 + (np.arange(n) % 8) * 0.02
    lon = 139.60 + (np.arange(n) // 8) * 0.02
    x1 = rng.normal(size=n)
    matrix = pd.DataFrame(
        {
            "station_key": [f"s{i}" for i in range(n)],
            "stop_lat": lat,
            "stop_lon": lon,
            "ward_jp": rng.choice(["A", "B"], size=n),
            "x1": x1,
            "log_ridership": 3.0 * x1 + 1.0 + rng.normal(scale=0.1, size=n),
        }
    )
    result = evaluate.run(
        matrix,
        numeric_features=["x1"],
        categorical_features=["ward_jp"],
        target="log_ridership",
        model=LinearRegression(),
        n_splits=4,
        block_size_km=5.0,
    )
    assert np.isfinite(result.scores["r2_mean"])
    assert 0.0 <= result.interval["coverage"] <= 1.0
    assert "I" in result.moran
