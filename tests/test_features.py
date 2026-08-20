"""Unit tests for the non-obvious feature logic: route graph + 800 m catchment."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from tokyo_ridership.features.accessibility import km_to_yamanote, pop_catchment
from tokyo_ridership.features.network_graph import (
    build_route_graph,
    map_routes_stations,
    station_connectivity,
    station_modes,
)

# Toy network: routes R1, R2, R3.
#   station A served by R1 + R2  -> R1-R2 share A (edge)
#   station B served by R1       (R1 also stops at B)
#   station C served by R3       (isolated route)
_STOPS = pd.DataFrame(
    {
        "stop_id": ["a", "b", "c"],
        "station_key": ["A", "B", "C"],
        "stop_lat": [35.70, 35.71, 35.60],
        "stop_lon": [139.70, 139.71, 139.60],
    }
)
_TRIPS = pd.DataFrame({"trip_id": ["t1", "t2", "t3"], "route_id": ["R1", "R2", "R3"]})
_STOP_TIMES = pd.DataFrame(
    {
        "trip_id": ["t1", "t1", "t2", "t3"],
        "stop_id": ["a", "b", "a", "c"],
        "stop_sequence": [0, 1, 0, 0],
    }
)


def test_build_route_graph_edges_and_membership() -> None:
    st = map_routes_stations(_STOPS, _TRIPS, _STOP_TIMES)
    graph, station_to_routes = build_route_graph(st)

    assert station_to_routes["A"] == {"R1", "R2"}
    assert station_to_routes["B"] == {"R1"}
    assert station_to_routes["C"] == {"R3"}
    # R1 and R2 meet at station A -> adjacent; R3 shares nothing -> isolated.
    assert graph.has_edge("R1", "R2")
    assert not graph.has_edge("R1", "R3")
    assert graph.degree("R3") == 0


def test_station_modes_primary_by_priority() -> None:
    # R1 heavy rail, R2/R3 tram. Station A (rail+tram) -> rail; C (tram only) -> tram.
    routes = pd.DataFrame(
        {"route_id": ["R1", "R2", "R3"], "route_type": ["2", "0", "0"]}
    )
    modes = station_modes(_STOPS, _TRIPS, _STOP_TIMES, routes).set_index("station_key")
    assert modes.loc["A", "station_mode"] == "rail"  # heaviest mode wins
    assert modes.loc["B", "station_mode"] == "rail"
    assert modes.loc["C", "station_mode"] == "tram"  # tram-only stays tram


def test_station_connectivity_counts() -> None:
    conn = station_connectivity(_STOPS, _TRIPS, _STOP_TIMES, yamanote_route="R1")
    by_station = conn.set_index("station_key")

    assert by_station.loc["A", "n_routes"] == 2
    assert by_station.loc["B", "n_routes"] == 1
    # daily_stop_events = number of stop_time rows at the station
    assert by_station.loc["A", "daily_stop_events"] == 2  # rows (t1,a) and (t2,a)
    assert by_station.loc["B", "daily_stop_events"] == 1
    # BFS from R1: A on R1 -> 0 transfers; C only on R3 (unreachable) -> NaN
    assert by_station.loc["A", "min_transfers_yamanote"] == 0
    assert pd.isna(by_station.loc["C", "min_transfers_yamanote"])


def test_pop_catchment_sums_only_intersecting_chome() -> None:
    stops = pd.DataFrame(
        {
            "stop_id": ["s"],
            "station_key": ["S"],
            "stop_lat": [35.70],
            "stop_lon": [139.70],
        }
    )
    # near polygon (~within 800 m) counts; far polygon (~11 km away) does not.
    census = gpd.GeoDataFrame(
        {"population": [100.0, 999.0]},
        geometry=[
            box(139.695, 35.695, 139.705, 35.705),  # straddles the station
            box(139.80, 35.80, 139.81, 35.81),  # far NE
        ],
        crs="EPSG:4326",
    )
    out = pop_catchment(stops, census, radius_m=800).set_index("station_key")
    assert out.loc["S", "pop_800m_catchment"] == 100.0


def test_km_to_yamanote_distance() -> None:
    # station Y is on the Yamanote route; station X is ~1 deg-min east of it.
    stops = pd.DataFrame(
        {
            "stop_id": ["y", "x"],
            "station_key": ["Y", "X"],
            "stop_lat": [35.70, 35.70],
            "stop_lon": [139.70, 139.70 + 0.02],  # ~1.8 km east at this latitude
        }
    )
    trips = pd.DataFrame({"trip_id": ["ty"], "route_id": ["Yama"]})
    stop_times = pd.DataFrame(
        {"trip_id": ["ty"], "stop_id": ["y"], "stop_sequence": [0]}
    )
    out = km_to_yamanote(stops, trips, stop_times, yamanote_route="Yama").set_index(
        "station_key"
    )
    assert out.loc["Y", "km_to_yamanote"] == 0.0
    assert 1.5 < out.loc["X", "km_to_yamanote"] < 2.1
