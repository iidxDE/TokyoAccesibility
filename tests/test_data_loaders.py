"""Unit tests for the pure Phase 1 data-loading logic (no I/O)."""

from __future__ import annotations

import math

import pandas as pd

from tokyo_ridership.data.load_gtfs import (
    filter_stops_to_bbox,
    jp_name,
    station_key,
)
from tokyo_ridership.data.load_ridership import _strip_ws


def test_station_key_takes_english_portion() -> None:
    assert station_key("東京 Tokyo") == "Tokyo"
    assert station_key("渋谷 Shibuya") == "Shibuya"


def test_station_key_falls_back_to_whole_name() -> None:
    assert station_key("Tokyo") == "Tokyo"


def test_station_key_handles_missing() -> None:
    assert station_key(float("nan")) is None


def test_jp_name_takes_kanji_portion() -> None:
    assert jp_name("東京 Tokyo") == "東京"
    assert jp_name(float("nan")) is None


def test_strip_ws_removes_ascii_and_fullwidth_space() -> None:
    # full-width space (U+3000) and ASCII space both stripped
    assert _strip_ws("東 京　駅") == "東京駅"


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
