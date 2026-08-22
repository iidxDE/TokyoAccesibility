"""Unit tests for the serving Pydantic contract (Phase 5, spec §5.3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tokyo_ridership.serving import schemas
from tokyo_ridership.serving.schemas import (
    Assumptions,
    Extrapolation,
    Interval,
    PredictRequest,
    PredictResponse,
)

_BOUNDS = schemas._BOUNDS

# A point safely inside the Tokyo serving bounds (near Shinjuku).
_TOKYO = {"latitude": 35.6896, "longitude": 139.7006}


def test_in_bounds_request_validates() -> None:
    req = PredictRequest(**_TOKYO)
    assert req.overrides is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("latitude", _BOUNDS["min_lat"] - 0.01),
        ("latitude", _BOUNDS["max_lat"] + 0.01),
        ("longitude", _BOUNDS["min_lon"] - 0.01),
        ("longitude", _BOUNDS["max_lon"] + 0.01),
    ],
)
def test_out_of_bounds_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        PredictRequest(**{**_TOKYO, field: value})


def test_bounds_edges_inclusive() -> None:
    corner = {"latitude": _BOUNDS["min_lat"], "longitude": _BOUNDS["max_lon"]}
    assert PredictRequest(**corner).latitude == _BOUNDS["min_lat"]


def test_known_override_accepted() -> None:
    req = PredictRequest(**_TOKYO, overrides={"employment_800m": 12000.0})
    assert req.overrides is not None
    assert req.overrides.employment_800m == 12000.0


def test_unknown_override_key_rejected() -> None:
    # daily_stop_events is deliberately excluded from the servable set (rule 4).
    with pytest.raises(ValidationError):
        PredictRequest(**_TOKYO, overrides={"daily_stop_events": 500})


def _sample_response() -> PredictResponse:
    return PredictResponse(
        prediction=42000.0,
        interval=Interval(lower=31000.0, upper=58000.0, level=0.9),
        features={
            "n_routes": 3.0,
            "min_transfers_yamanote": 0.0,
            "mean_route_degree_cent": 0.4,
            "mean_route_btw_cent": 0.1,
            "max_route_btw_cent": 0.3,
            "km_to_yamanote": 0.0,
            "pop_800m_catchment": 54000.0,
            "employment_800m": 120000.0,
            "landuse_built_frac": 0.8,
            "landuse_mix": None,  # NaN echoed as null (spec §5.5)
            "ward_jp": "新宿区",
            "station_mode": "rail",
        },
        assumptions=Assumptions(
            snapped_station="新宿",
            snap_distance_m=15.2,
            station_mode="rail",
            station_mode_source="snapped:新宿",
            overridden_features=[],
            ward_outside_23=False,
            extrapolation=Extrapolation(flag=False),
        ),
        model="rf_nodse_deadbeef",
    )


def test_response_round_trips() -> None:
    resp = _sample_response()
    reparsed = PredictResponse.model_validate(resp.model_dump())
    assert reparsed == resp
    # Interval stays asymmetric — no symmetry is imposed.
    assert (reparsed.interval.upper - reparsed.prediction) != (
        reparsed.prediction - reparsed.interval.lower
    )
    # A NaN-origin feature is echoed as null.
    assert reparsed.features["landuse_mix"] is None
