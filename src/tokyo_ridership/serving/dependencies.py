"""Startup wiring: load the model once, inject into routes (Phase 5).

The servable pipeline is loaded a single time at startup and held in memory for
reuse across requests (CLAUDE.md rule 6). The service is stateless.
"""

from __future__ import annotations
