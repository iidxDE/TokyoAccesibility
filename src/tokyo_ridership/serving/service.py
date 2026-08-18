"""Orchestration layer (Phase 5).

coords -> ``features/build_features`` (the shared path) -> ``models/predict`` ->
assemble response (network-snap + extrapolation flag). Keeps the API routes
thin: all orchestration lives here, prediction lives in ``models/predict``.
"""

from __future__ import annotations
