"""Unit tests for the pure Phase 1 data-loading logic (no I/O)."""

from __future__ import annotations

import math

import pandas as pd

from tokyo_ridership.data.load_gtfs import (
    english_label,
    filter_stops_to_bbox,
    japanese_name,
    normalize_station_name,
)


def test_station_key_is_japanese_name() -> None:
    assert japanese_name("東京 Tokyo") == "東京"
    assert japanese_name("渋谷 Shibuya") == "渋谷"


def test_station_key_folds_kana_and_bracket_variants() -> None:
    # ヶ/ケ split and 〈alt-name〉 parentheticals must collapse to one key
    assert japanese_name("市ヶ谷 Ichigaya") == japanese_name("市ケ谷 Ichigaya")
    assert japanese_name("明治神宮前〈原宿〉 Meiji-jingumae") == "明治神宮前"


def test_japanese_name_handles_missing() -> None:
    assert japanese_name(float("nan")) is None


def test_english_label_strips_diacritics() -> None:
    # the whole point of the fix: macron variants collapse to one label
    assert english_label("東京 Tōkyō") == "Tokyo"
    assert english_label("東京 Tokyo") == "Tokyo"


def test_english_label_none_without_english() -> None:
    assert english_label("東京") is None
    assert english_label(float("nan")) is None


def test_normalize_strips_whitespace_and_folds_ke() -> None:
    # full-width (U+3000) + ASCII space stripped; small ヶ folded to ケ
    assert normalize_station_name("東 京　駅") == "東京駅"
    assert normalize_station_name("霞ヶ関") == "霞ケ関"


def test_normalize_handles_missing() -> None:
    assert normalize_station_name(float("nan")) is None


def test_filter_stops_to_bbox_keeps_only_inside() -> None:
    bbox = {"min_lat": 35.5, "max_lat": 35.85, "min_lon": 139.55, "max_lon": 139.92}
    stops = pd.DataFrame(
        {
            "stop_id": ["a", "b", "c"],
            "stop_lat": [35.68, 35.90, 35.70],  # b is north of the box
            "stop_lon": [139.76, 139.70, 139.30],  # c is west of the box
        }
    )
    kept = filter_stops_to_bbox(stops, bbox)
    assert kept["stop_id"].tolist() == ["a"]


def test_filter_stops_to_bbox_accepts_string_coords() -> None:
    bbox = {"min_lat": 35.5, "max_lat": 35.85, "min_lon": 139.55, "max_lon": 139.92}
    stops = pd.DataFrame(
        {"stop_id": ["a"], "stop_lat": ["35.68"], "stop_lon": ["139.76"]}
    )
    assert len(filter_stops_to_bbox(stops, bbox)) == 1


def test_log_ridership_is_log1p() -> None:
    # sanity: the target transform used by the ridership loader
    assert math.isclose(math.log1p(216163.0), 12.283792, rel_tol=1e-6)
