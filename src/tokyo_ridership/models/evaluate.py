"""Spatially-blocked evaluation harness (Phase 3).

The fixed apparatus every model/feature-set passes through. Reusable functions,
not inline cells:

- spatial-block fold generator (grid primary, ward folds as robustness check),
- nested spatial CV (inner tuning + outer eval loop, same block logic),
- prediction intervals (conformal via MAPIE or a quantile objective),
- residual diagnostics: Moran's I + LISA on out-of-fold residuals.

Single-call interface:
    run(feature_set, model) -> scores + intervals + residuals
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover - implemented in Phase 3
    raise NotImplementedError("evaluate is implemented in Phase 3")


if __name__ == "__main__":
    main()
