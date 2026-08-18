"""Route-level graph, centrality, BFS-to-Yamanote (Phase 2).

Ports ``data/feature_build.py::build_connectivity_features``: builds the
undirected route graph (nodes = GTFS routes, edge = share >=1 station),
computes degree/betweenness centrality, and BFS transfer distance to the
Yamanote Line. Unit-tested per CLAUDE.md.
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover - implemented in Phase 2
    raise NotImplementedError("network_graph is implemented in Phase 2")


if __name__ == "__main__":
    main()
