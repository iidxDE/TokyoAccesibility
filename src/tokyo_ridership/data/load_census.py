"""e-Stat census loader (Phase 1).

Ports ``data/feature_build.py::population_features`` data intake: Shift-JIS
decode, drop the embedded metadata row after the header, and filter to the
chome (district) aggregation level to avoid double-counting parent rows.
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover - implemented in Phase 1
    raise NotImplementedError("load_census is implemented in Phase 1")


if __name__ == "__main__":
    main()
