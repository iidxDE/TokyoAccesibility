"""FastAPI application: routes only (Phase 5).

Routes ``/predict`` and ``/health``. Out-of-bounds / malformed input -> 422
(Pydantic); valid-but-extrapolating input -> 200 with a warning in
``assumptions``. CORS configured for the Streamlit origin. The app object is
created here; orchestration is delegated to ``serving/service.py``.
"""

from __future__ import annotations
