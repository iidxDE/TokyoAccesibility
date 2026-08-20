"""Route-level graph, centrality, BFS-to-Yamanote (Phase 2).

Ports ``data/feature_build.py::build_connectivity_features``. Builds an
undirected graph whose nodes are GTFS routes, with an edge between two routes
whenever they share at least one physical station (encoding transfer structure).
From it we derive, per station: route count, the minimum route-to-route transfer
distance to the Yamanote Line (BFS), aggregated degree/betweenness centrality of
the routes serving it, and the daily scheduled stop-event count. Also exposes the
station's primary transport mode (:func:`station_modes`) from GTFS ``route_type``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

# GTFS route_type -> coarse transport mode. The paper's features are mode-blind,
# so a tram stop and a Metro station with the same catchment/network position look
# identical, yet their ridership differs by 1-2 orders of magnitude (the Toden
# streetcar over-prediction). ``station_mode`` restores that distinction.
_ROUTE_TYPE_MODE = {0: "tram", 1: "subway", 2: "rail", 12: "monorail"}
# Priority (heaviest ridership-driving mode first) resolves mixed-mode stations:
# a tram+rail interchange is a rail station; a tram-only stop stays "tram".
_MODE_PRIORITY = {"rail": 0, "subway": 1, "monorail": 2, "tram": 3, "other": 4}


def map_routes_stations(
    stops: pd.DataFrame, trips: pd.DataFrame, stop_times: pd.DataFrame
) -> pd.DataFrame:
    """Return stop_times annotated with ``route_id`` + ``station_key``.

    Rows without a resolvable route or station are dropped, mirroring the
    prototype so downstream counts (incl. daily_stop_events) match exactly.
    """
    stop_to_station = dict(zip(stops["stop_id"], stops["station_key"], strict=False))
    trip_to_route = dict(zip(trips["trip_id"], trips["route_id"], strict=False))
    st = stop_times.copy()
    st["route_id"] = st["trip_id"].map(trip_to_route)
    st["station_key"] = st["stop_id"].map(stop_to_station)
    return st.dropna(subset=["route_id", "station_key"])


def build_route_graph(st: pd.DataFrame) -> tuple[nx.Graph, dict[str, set[str]]]:
    """Build the route graph and the station -> serving-routes mapping.

    ``st`` is the output of :func:`map_routes_stations`. Nodes are routes; an
    edge joins two routes that share at least one station.
    """
    route_station = st[["route_id", "station_key"]].drop_duplicates()
    station_to_routes: dict[str, set[str]] = (
        route_station.groupby("station_key")["route_id"].apply(set).to_dict()
    )

    graph = nx.Graph()
    graph.add_nodes_from(route_station["route_id"].unique())
    for routes in station_to_routes.values():
        rs = list(routes)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                graph.add_edge(rs[i], rs[j])
    return graph, station_to_routes


def station_connectivity(
    stops: pd.DataFrame,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    yamanote_route: str,
) -> pd.DataFrame:
    """Per-station connectivity + service-intensity features.

    Columns: ``station_key``, ``n_routes``, ``min_transfers_yamanote``,
    ``mean_route_degree_cent``, ``mean_route_btw_cent``, ``max_route_btw_cent``,
    ``daily_stop_events``.
    """
    st = map_routes_stations(stops, trips, stop_times)
    graph, station_to_routes = build_route_graph(st)

    route_dist: dict[str, int] = (
        nx.single_source_shortest_path_length(graph, yamanote_route)
        if yamanote_route in graph
        else {}
    )
    deg_cent = nx.degree_centrality(graph)
    btw_cent = nx.betweenness_centrality(graph)
    daily = st.groupby("station_key").size()

    rows: list[dict[str, object]] = []
    for station, routes in station_to_routes.items():
        dists = [route_dist[r] for r in routes if r in route_dist]
        rows.append(
            {
                "station_key": station,
                "n_routes": len(routes),
                "min_transfers_yamanote": min(dists) if dists else None,
                "mean_route_degree_cent": float(
                    np.mean([deg_cent.get(r, 0.0) for r in routes])
                ),
                "mean_route_btw_cent": float(
                    np.mean([btw_cent.get(r, 0.0) for r in routes])
                ),
                "max_route_btw_cent": float(max(btw_cent.get(r, 0.0) for r in routes)),
                "daily_stop_events": int(daily.get(station, 0)),
            }
        )
    return pd.DataFrame(rows)


def station_modes(
    stops: pd.DataFrame,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    routes: pd.DataFrame,
) -> pd.DataFrame:
    """Primary transport mode per station from GTFS ``route_type``.

    Each station is labelled by the heaviest mode among the routes serving it
    (:data:`_MODE_PRIORITY`), so tram-only stops (Toden/Enoden/Setagaya) separate
    from the heavy-rail network. Columns: ``station_key``, ``station_mode``.
    """
    st = map_routes_stations(stops, trips, stop_times)
    route_type = pd.to_numeric(routes["route_type"], errors="coerce")
    route_to_mode = dict(
        zip(routes["route_id"], route_type.map(_ROUTE_TYPE_MODE), strict=True)
    )
    st = st.assign(mode=st["route_id"].map(route_to_mode).fillna("other"))

    def _primary(modes: set[str]) -> str:
        return min(modes, key=lambda mode: _MODE_PRIORITY.get(mode, 99))

    return (
        st.groupby("station_key")["mode"]
        .apply(lambda s: _primary(set(s)))
        .reset_index(name="station_mode")
    )
