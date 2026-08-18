"""Streamlit frontend (Phase 6): static equity map + live what-if.

HTTP client of the FastAPI service ONLY. This module must never import from
``src/tokyo_ridership/models`` (CLAUDE.md rule 3); it calls the API with
``requests``. ``API_BASE_URL`` comes from the environment, never hardcoded.
"""

from __future__ import annotations
