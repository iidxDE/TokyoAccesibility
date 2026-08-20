"""Unit tests for the evaluation harness: spatial blocking is the load-bearing part.

``moran_lisa``/``run`` need the optional [models] deps (libpysal/esda) and are
exercised by the pipeline smoke run, not here — these cover the pure logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from tokyo_ridership.models import evaluate


def test_spatial_blocks_grid_groups_by_cell() -> None:
    coords = pd.DataFrame(
        {
            "stop_lat": [35.700, 35.7005, 35.800],
            "stop_lon": [139.700, 139.7005, 139.850],
        }
    )
    blocks = evaluate.spatial_blocks(coords, block_size_km=5.0, strategy="grid")
    assert blocks[0] == blocks[1]  # ~50 m apart -> same 5 km cell
    assert blocks[0] != blocks[2]  # kilometres apart -> different cell


def test_spatial_blocks_ward_passthrough() -> None:
    coords = pd.DataFrame({"stop_lat": [35.7, 35.7], "stop_lon": [139.7, 139.7]})
    ward = pd.Series(["A", "B"])
    blocks = evaluate.spatial_blocks(coords, 5.0, strategy="ward", ward=ward)
    assert list(blocks) == ["A", "B"]


def test_spatial_folds_no_block_leakage_and_partition() -> None:
    blocks = np.array([f"b{i % 10}" for i in range(100)], dtype=object)
    folds = evaluate.spatial_folds(blocks, n_splits=5)

    seen_test: list[np.ndarray] = []
    for tr, te in folds:
        # the whole point of spatial blocking: no block appears in both sides
        assert set(blocks[tr]).isdisjoint(set(blocks[te]))
        seen_test.append(te)
    # every row is a test point exactly once
    combined = np.concatenate(seen_test)
    assert sorted(combined.tolist()) == list(range(100))


def test_conformal_offsets_and_coverage() -> None:
    residuals = np.linspace(-1.0, 1.0, 101)
    lo, hi = evaluate.conformal_offsets(residuals, level=0.8)
    assert lo < 0 < hi
    assert 0.75 <= evaluate._coverage(residuals, lo, hi) <= 0.85


def test_metrics_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = evaluate._metrics(y, y)
    assert m["r2"] == 1.0
    assert m["mae_log"] == 0.0
    assert m["rmse_orig"] < 1e-9


def test_nested_cv_predicts_every_row() -> None:
    # 100 stations on a 10x10 ~2km grid; y is a clean linear function of x1.
    rng = np.random.default_rng(0)
    n = 100
    lat = 35.60 + (np.arange(n) % 10) * 0.02
    lon = 139.60 + (np.arange(n) // 10) * 0.02
    x1 = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "stop_lat": lat,
            "stop_lon": lon,
            "x1": x1,
            "cat": rng.choice(["p", "q"], size=n),
        }
    )
    y = pd.Series(3.0 * x1 + 1.0)
    blocks = evaluate.spatial_blocks(df[["stop_lat", "stop_lon"]], 5.0, "grid")

    oof, fold_scores, best = evaluate.nested_spatial_cv(
        df[["x1", "cat"]],
        y,
        blocks,
        numeric=["x1"],
        categorical=["cat"],
        model=LinearRegression(),
        param_grid={},
        n_splits=4,
    )
    assert not np.isnan(oof).any()  # every station scored out-of-fold
    assert len(fold_scores) == len(best)
    assert fold_scores["r2"].mean() > 0.9  # clean linear signal recovers well
