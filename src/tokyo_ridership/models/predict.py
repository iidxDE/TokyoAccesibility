"""Prediction wrapper (Phase 5).

Loads the serialized pipeline and returns a point prediction plus an asymmetric
interval, back-transformed to passengers/day via ``expm1`` (the model predicts
``ln(1+y)``; serving owns the inverse transform per CLAUDE.md).
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover - implemented in Phase 5
    raise NotImplementedError("predict is implemented in Phase 5")


if __name__ == "__main__":
    main()
