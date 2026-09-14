"""Unit tests for the Equity Map logic in ``viz/maps.py``.

Covers the non-obvious pure pieces (CLAUDE.md: unit-test non-obvious logic):
the diverging colour mapping + clip, the |residual| radius scaling, the LISA
quadrant labelling, the clusters-only filter, and the ward rollup. No file I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tokyo_ridership.viz import maps


def _frame() -> pd.DataFrame:
    """Small in-memory residual frame shaped like ``load_residual_layer`` output."""
    df = pd.DataFrame(
        {
            "station_key": ["A", "B", "C", "D"],
            "label_en": ["A-en", "B-en", "C-en", "D-en"],
            "ward_jp": ["新宿区", "新宿区", "港区", "港区"],
            "y_true": [10.0, 8.0, 6.0, 9.0],
            "y_pred": [9.0, 8.5, 7.0, 8.0],
            "residual": [3.0, -0.5, -3.0, 0.0],
            "lisa_p": [0.01, 0.20, 0.04, 0.50],
            "quadrant": [1, 2, 3, 4],
            "stop_lat": [35.69, 35.70, 35.66, 35.65],
            "stop_lon": [139.70, 139.71, 139.75, 139.74],
        }
    )
    df["ridership_actual"] = np.expm1(df["y_true"])
    df["ridership_pred"] = np.expm1(df["y_pred"])
    df["quadrant_label"] = df["quadrant"].map(maps._quadrant_label)
    df["fill_color"] = maps._residual_rgba(df["residual"])
    df["radius_m"] = maps._radius_from_abs_residual(df["residual"])
    return df


def test_quadrant_label_maps_all_codes() -> None:
    assert [maps._quadrant_label(q) for q in (1, 2, 3, 4)] == ["HH", "LH", "LL", "HL"]
    assert maps._quadrant_label(None) == "n/a"
    assert maps._quadrant_label(99) == "n/a"


def test_residual_rgba_sign_and_midpoint() -> None:
    # RdBu_r: negative -> blue (B > R), positive -> red (R > B), zero -> ~neutral.
    blue, neutral, red = maps._residual_rgba([-2.0, 0.0, 2.0])
    assert blue[2] > blue[0]  # blue dominates
    assert red[0] > red[2]  # red dominates
    assert abs(neutral[0] - neutral[2]) <= 20  # near-symmetric midpoint


def test_residual_rgba_clips_beyond_range() -> None:
    # Values past +-RESIDUAL_CLIP map to the same colour as the clip endpoints.
    inside_hi, outside_hi = maps._residual_rgba([2.0, 10.0])
    inside_lo, outside_lo = maps._residual_rgba([-2.0, -10.0])
    assert inside_hi == outside_hi
    assert inside_lo == outside_lo


def test_radius_scales_with_abs_residual() -> None:
    r = maps._radius_from_abs_residual([0.0, -3.0, 1.5])
    # |0| -> min, |3| is the largest magnitude -> max, middle in between.
    assert r[0] == pytest.approx(maps.RADIUS_MIN_M)
    assert r[1] == pytest.approx(maps.RADIUS_MAX_M)
    assert maps.RADIUS_MIN_M < r[2] < maps.RADIUS_MAX_M


def test_radius_all_zero_returns_min() -> None:
    r = maps._radius_from_abs_residual([0.0, 0.0])
    assert np.all(r == maps.RADIUS_MIN_M)


def test_ward_residual_summary_aggregates_and_sorts() -> None:
    summary = maps.ward_residual_summary(_frame()).set_index("ward_jp")
    assert list(summary.index) == ["港区", "新宿区"]  # -1.5 mean sorts before 1.25
    assert summary.loc["港区", "mean_residual"] == pytest.approx(-1.5)
    assert summary.loc["新宿区", "n_stations"] == 2


def test_ward_summary_drops_missing_ward() -> None:
    df = _frame()
    df.loc[0, "ward_jp"] = None
    summary = maps.ward_residual_summary(df)
    assert summary["n_stations"].sum() == 3


def test_build_deck_all_stations() -> None:
    pydeck = pytest.importorskip("pydeck")
    deck = maps.build_residual_deck(_frame(), clusters_only=False)
    assert isinstance(deck, pydeck.Deck)
    assert deck.layers[0].id == "stations"
    assert len(deck.layers[0].data) == 4


def test_build_deck_clusters_only_filters_and_recolours() -> None:
    pytest.importorskip("pydeck")
    deck = maps.build_residual_deck(_frame(), clusters_only=True)
    data = deck.layers[0].data  # pydeck stores records as a list of dicts
    # Only lisa_p < 0.05 rows survive: A (0.01, HH) and C (0.04, LL).
    assert len(data) == 2
    assert {row["quadrant_label"] for row in data} == {"HH", "LL"}
    hh = next(row for row in data if row["quadrant_label"] == "HH")
    assert hh["fill_color"] == maps._QUADRANT_RGBA["HH"]
