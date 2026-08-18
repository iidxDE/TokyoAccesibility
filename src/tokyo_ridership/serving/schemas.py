"""Pydantic request/response models for the inference API (Phase 5).

Request: ``latitude``, ``longitude`` (validated against Tokyo bounds), optional
``overrides``. Response: ``prediction`` and asymmetric ``interval``
back-transformed to passengers/day, confidence ``level``, echoed computed
``features``, an ``assumptions`` block (network-snap, extrapolation flag,
overrides applied), and the ``model`` identifier.
"""

from __future__ import annotations
