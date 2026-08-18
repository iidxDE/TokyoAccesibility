"""MLIT ridership aggregation + kanji name matching (Phase 1).

Ports ``data/feature_build.py::load_ridership`` and
``match_ridership_to_stations``: aggregates the S12-20 FY2019 records per
station across operators and matches them to GTFS stations on the kanji
component of ``stop_name``.
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover - implemented in Phase 1
    raise NotImplementedError("load_ridership is implemented in Phase 1")


if __name__ == "__main__":
    main()
