"""Model experiments + selection (Phase 4).

Runs the roster (Dummy, Ridge, RandomForest, LightGBM) through the Phase 3
harness, runs the endogeneity experiment (full / no-``daily_stop_events`` /
stop-events-only), selects the servable model = best **no-``daily_stop_events``**
model (usable for greenfield stations), and serializes the full pipeline
(scaler + encoder + model) with a model id to ``models/pipeline.joblib``.
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover - implemented in Phase 4
    raise NotImplementedError("train is implemented in Phase 4")


if __name__ == "__main__":
    main()
