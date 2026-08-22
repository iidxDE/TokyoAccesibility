"""Pydantic request/response models for the inference API (Phase 5, spec §5.3).

Request: ``latitude``, ``longitude`` (bounded against Tokyo ``serving_bounds`` so
Pydantic yields a 422), optional per-feature ``overrides``. Response:
``prediction`` and an asymmetric ``interval`` back-transformed to passengers/day,
the echoed 12-value ``features`` vector, an ``assumptions`` block (network-snap
disclosure, overrides applied, out-of-ward flag, extrapolation), and the
``model`` identifier.

Bounds come from ``config/config.yaml`` — never hardcoded (CLAUDE.md). The config
is read once at import and fed into the ``Field`` constraints.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from tokyo_ridership.config import load_config

# Tokyo coordinate bounds enforced by the serving layer (config-driven -> 422).
_BOUNDS = load_config()["serving_bounds"]


class FeatureOverrides(BaseModel):
    """Optional overrides for any of the 12 servable features (spec §2.4).

    Every field is optional; ``None`` means "use the value computed from the
    coordinate or snapped from the nearest station". The endogenous
    ``daily_stop_events`` is intentionally absent — a greenfield station has no
    schedule (CLAUDE.md rule 4). Unknown keys are rejected (422) so typos surface
    instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    # Numeric (10). ``min_transfers_yamanote`` stays nullable: unreachable
    # stations are NaN end to end (spec §3, §5.3).
    n_routes: int | None = None
    min_transfers_yamanote: float | None = None
    mean_route_degree_cent: float | None = None
    mean_route_btw_cent: float | None = None
    max_route_btw_cent: float | None = None
    km_to_yamanote: float | None = None
    pop_800m_catchment: float | None = None
    employment_800m: float | None = None
    landuse_built_frac: float | None = None
    landuse_mix: float | None = None
    # Categorical (2).
    ward_jp: str | None = None
    station_mode: str | None = None


class PredictRequest(BaseModel):
    """A single point to score: coordinates plus optional feature overrides."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"latitude": 35.6896, "longitude": 139.7006}  # near Shinjuku
        }
    )

    latitude: float = Field(
        ...,
        ge=_BOUNDS["min_lat"],
        le=_BOUNDS["max_lat"],
        description="Latitude in decimal degrees (WGS84), within Tokyo bounds.",
    )
    longitude: float = Field(
        ...,
        ge=_BOUNDS["min_lon"],
        le=_BOUNDS["max_lon"],
        description="Longitude in decimal degrees (WGS84), within Tokyo bounds.",
    )
    overrides: FeatureOverrides | None = Field(
        default=None, description="Optional hypothetical feature overrides."
    )


class Interval(BaseModel):
    """Prediction interval in passengers/day. Asymmetric by design — do not force
    symmetry (CLAUDE.md serving contract)."""

    lower: float = Field(..., description="Lower bound (passengers/day).")
    upper: float = Field(..., description="Upper bound (passengers/day).")
    level: float = Field(..., description="Confidence level, e.g. 0.9.")


class ExtrapolationDetail(BaseModel):
    """One numeric feature whose (post-override) value fell outside its p1–p99
    training band."""

    name: str
    value: float
    p1: float
    p99: float


class Extrapolation(BaseModel):
    """Whether the request extrapolates beyond the training support (spec §2.5)."""

    flag: bool = Field(..., description="True when any numeric feature is out of band.")
    features: list[str] = Field(
        default_factory=list, description="Numeric features outside their p1–p99 band."
    )
    detail: list[ExtrapolationDetail] = Field(
        default_factory=list, description="Per-feature value with its p1/p99 bounds."
    )


class Assumptions(BaseModel):
    """What the service assumed to produce the prediction (network-snap disclosure,
    overrides, out-of-ward flag, extrapolation)."""

    snapped_station: str = Field(
        ..., description="Nearest existing station whose connectivity/mode was used."
    )
    snap_distance_m: float = Field(
        ..., description="Distance from the query point to the snapped station (m)."
    )
    station_mode: str = Field(..., description="Transport mode assumed for the point.")
    station_mode_source: str = Field(
        ..., description='Provenance of the mode: "snapped:<key>" or "override".'
    )
    overridden_features: list[str] = Field(
        default_factory=list, description="Feature names the caller overrode."
    )
    ward_outside_23: bool = Field(
        ..., description="True when the point falls outside the 23 special wards."
    )
    extrapolation: Extrapolation


class PredictResponse(BaseModel):
    """Full prediction payload returned by ``/predict``."""

    prediction: float = Field(
        ..., description="Point prediction, passengers/day (expm1 back-transformed)."
    )
    interval: Interval
    # The echoed 12-value feature vector. NaN is serialized as null for
    # JSON-safety (spec §5.5 step 5), so values may be None.
    features: dict[str, float | str | None] = Field(
        ..., description="The 12 servable features actually fed to the model."
    )
    assumptions: Assumptions
    model: str = Field(
        ..., description="Servable model identifier (stamped per rule 4)."
    )


class HealthResponse(BaseModel):
    """Liveness + model-traceability payload for ``/health``."""

    status: str = Field(..., description='"ok" when the service is ready.')
    model_id: str = Field(..., description="Identifier of the loaded servable model.")
