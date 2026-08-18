"""GTFS parse + station-key unification (Phase 1).

Ports ``data/feature_build.py::load_gtfs`` (and the bbox pre-filter in
``data/filter_gtfs.py``) into the shared package: reads stops/stop_times/
trips/routes, derives a unified ``station_key`` from the bilingual
``stop_name`` field, and writes a clean intermediate table.
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover - implemented in Phase 1
    raise NotImplementedError("load_gtfs is implemented in Phase 1")


if __name__ == "__main__":
    main()
