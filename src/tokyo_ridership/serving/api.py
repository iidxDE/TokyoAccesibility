"""FastAPI application: routes only (Phase 5, spec §5.6).

Routes ``/predict`` and ``/health``. Out-of-bounds / malformed input -> 422
(Pydantic); valid-but-extrapolating input -> 200 with a warning in
``assumptions``. CORS origins come from the ``ALLOWED_ORIGINS`` env var (never
hardcode the deployed origin). The app is created here with the startup lifespan;
all orchestration is delegated to ``serving/service.py`` (CLAUDE.md rule 5).
"""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tokyo_ridership.serving import service
from tokyo_ridership.serving.dependencies import AppState, get_state, lifespan
from tokyo_ridership.serving.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

# Comma-separated origins; default to local Streamlit (Phase 6 dev).
_DEFAULT_ORIGINS = "http://localhost:8501"
_allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

app = FastAPI(
    title="Tokyo Ridership — greenfield siting API",
    summary="Predict daily ridership for a hypothetical new station from a coordinate.",
    lifespan=lifespan,
)

StateDep = Annotated[AppState, Depends(get_state)]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health(state: StateDep) -> HealthResponse:
    """Liveness + model-traceability check."""
    return HealthResponse(status="ok", model_id=str(state.bundle["model_id"]))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, state: StateDep) -> PredictResponse:
    """Score one coordinate. Validation (bounds, unknown overrides) is handled by
    ``PredictRequest`` -> 422; orchestration lives in the service layer."""
    return service.predict_ridership(
        request.latitude, request.longitude, request.overrides, state
    )
