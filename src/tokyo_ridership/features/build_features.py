"""Assemble the 474-station feature matrix — the shared path (Phase 2).

THE anti train-serve-skew boundary. The interface is designed as
``coordinates/context -> feature vector`` so a single hypothetical point can be
scored at serving time using the exact same code that builds the training
matrix. The serving layer MUST call this module; it must never re-implement
feature logic (CLAUDE.md rule 2).
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover - implemented in Phase 2
    raise NotImplementedError("build_features is implemented in Phase 2")


if __name__ == "__main__":
    main()
