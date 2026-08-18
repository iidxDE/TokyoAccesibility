"""Phase 0 smoke tests: the package and its subpackages import cleanly.

Real unit tests (route-graph construction, 800 m catchment) land in
``tests/test_features.py`` in Phase 2 per CLAUDE.md.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "tokyo_ridership",
        "tokyo_ridership.data",
        "tokyo_ridership.features",
        "tokyo_ridership.models",
        "tokyo_ridership.serving",
        "tokyo_ridership.viz",
    ],
)
def test_subpackages_import(module: str) -> None:
    assert importlib.import_module(module) is not None


def test_version_present() -> None:
    import tokyo_ridership

    assert isinstance(tokyo_ridership.__version__, str)
