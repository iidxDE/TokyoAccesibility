"""Thin HTTP client for the FastAPI inference service (Phase 6, Page 2).

The Streamlit app is an HTTP client of the API ONLY (CLAUDE.md rule 3): this
module talks to the service with ``requests`` and returns typed results the UI can
branch on. It never imports from ``src/tokyo_ridership/models`` and never runs the
model in-process. ``base_url`` is passed in by the caller (resolved from the
environment in ``streamlit_app.py``); nothing is hardcoded here.

Kept free of Streamlit imports so the request/response logic stays unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

# The five valid transport modes accepted by the service for ``station_mode``.
MODES = ["rail", "subway", "monorail", "agt", "tram"]


@dataclass(frozen=True)
class Feature:
    """UI metadata for one of the 12 servable features."""

    key: str
    label: str
    dtype: str  # "int" | "float" | "str"
    tier: str  # "primary" | "advanced" | "display"
    origin: str  # "coord" | "snapped"


# Ordered feature metadata. ``tier`` drives the override panel: ``primary`` levers
# are shown by default, ``advanced`` (network position) sit behind an expander, and
# ``display`` features are echoed but not offered as override controls (spec §UI).
# ``origin`` explains the default: coordinate-derived vs snapped from the nearest
# existing station.
FEATURES: list[Feature] = [
    Feature("employment_800m", "Employment (800 m)", "float", "primary", "coord"),
    Feature("pop_800m_catchment", "Population (800 m)", "float", "primary", "coord"),
    Feature("landuse_built_frac", "Built-up fraction", "float", "primary", "coord"),
    Feature("landuse_mix", "Land-use mix", "float", "primary", "coord"),
    Feature("station_mode", "Station mode", "str", "primary", "snapped"),
    Feature("n_routes", "Routes served", "int", "advanced", "snapped"),
    Feature(
        "min_transfers_yamanote",
        "Min transfers to Yamanote",
        "float",
        "advanced",
        "snapped",
    ),
    Feature(
        "mean_route_degree_cent", "Mean route degree", "float", "advanced", "snapped"
    ),
    Feature(
        "mean_route_btw_cent", "Mean route betweenness", "float", "advanced", "snapped"
    ),
    Feature(
        "max_route_btw_cent", "Max route betweenness", "float", "advanced", "snapped"
    ),
    Feature("km_to_yamanote", "Distance to Yamanote (km)", "float", "display", "coord"),
    Feature("ward_jp", "Ward", "str", "display", "coord"),
]

# Feature keys by tier / lookup, derived once from FEATURES.
FEATURES_BY_KEY: dict[str, Feature] = {f.key: f for f in FEATURES}
PRIMARY_KEYS = [f.key for f in FEATURES if f.tier == "primary"]
ADVANCED_KEYS = [f.key for f in FEATURES if f.tier == "advanced"]


@dataclass(frozen=True)
class PredictOutcome:
    """A typed ``/predict`` result the UI branches on.

    ``status`` is one of: ``ok`` (``data`` holds the 200 payload),
    ``validation_error`` (``message`` is a humanised 422), ``unreachable`` (the
    service could not be reached), or ``error`` (an unexpected server response;
    ``message`` describes it).
    """

    status: str
    data: dict[str, Any] | None = None
    message: str | None = None


def health(base_url: str, *, timeout: float = 3.0) -> dict[str, Any] | None:
    """GET ``/health``. Returns the parsed payload, or ``None`` if unreachable."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
    except requests.RequestException:
        return None
    if not resp.ok:
        return None
    try:
        return dict(resp.json())
    except ValueError:
        return None


def collect_overrides(
    values: dict[str, Any], enabled: dict[str, bool]
) -> dict[str, Any]:
    """Build the ``overrides`` object from only the features the user overrode.

    Keeps a key only when it is enabled and its value is not ``None``; coerces
    ``n_routes`` to ``int``; preserves the exact server key names (unknown keys
    would 422). Pure — no I/O.
    """
    out: dict[str, Any] = {}
    for key, value in values.items():
        if not enabled.get(key) or value is None:
            continue
        if key == "n_routes":
            out[key] = int(value)
        else:
            out[key] = value
    return out


def humanize_validation_error(detail: Any) -> str:
    """Turn a FastAPI 422 body into plain language (never dump raw Pydantic)."""
    if isinstance(detail, str):
        return detail
    if not isinstance(detail, list):
        return "The request was rejected as invalid."
    parts: list[str] = []
    for item in detail:
        if not isinstance(item, dict):
            continue
        loc = [str(p) for p in item.get("loc", []) if p != "body"]
        field = ".".join(loc) if loc else "input"
        msg = item.get("msg", "is invalid")
        parts.append(f"{field}: {msg}")
    return "; ".join(parts) if parts else "The request was rejected as invalid."


def predict(
    base_url: str,
    lat: float,
    lon: float,
    overrides: dict[str, Any] | None,
    *,
    timeout: float = 10.0,
) -> PredictOutcome:
    """POST ``/predict``. Maps the response to a typed :class:`PredictOutcome`."""
    payload: dict[str, Any] = {"latitude": lat, "longitude": lon}
    if overrides:
        payload["overrides"] = overrides
    try:
        resp = requests.post(f"{base_url}/predict", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return PredictOutcome(status="unreachable", message=str(exc))

    if resp.status_code == 200:
        return PredictOutcome(status="ok", data=dict(resp.json()))
    if resp.status_code == 422:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = None
        return PredictOutcome(
            status="validation_error", message=humanize_validation_error(detail)
        )
    return PredictOutcome(
        status="error",
        message=f"Unexpected response from the service (HTTP {resp.status_code}).",
    )
