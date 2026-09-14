"""Unit tests for the Page 2 HTTP client (``app/api_client.py``).

Pure / mocked — no network. Covers the override assembly (only-changed-features,
int coercion, exact key names), the /predict response branching, the 422
humaniser, and the candidate deck builder.
"""

from __future__ import annotations

import api_client
import pytest
import requests


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._payload


def test_collect_overrides_keeps_only_enabled_non_null() -> None:
    values = {
        "employment_800m": 120000.0,
        "pop_800m_catchment": 38000.0,
        "landuse_mix": None,
        "station_mode": "subway",
    }
    enabled = {
        "employment_800m": True,
        "pop_800m_catchment": False,  # disabled -> dropped
        "landuse_mix": True,  # None -> dropped
        "station_mode": True,
    }
    out = api_client.collect_overrides(values, enabled)
    assert out == {"employment_800m": 120000.0, "station_mode": "subway"}


def test_collect_overrides_coerces_n_routes_to_int() -> None:
    out = api_client.collect_overrides({"n_routes": 4.0}, {"n_routes": True})
    assert out == {"n_routes": 4}
    assert isinstance(out["n_routes"], int)


def test_collect_overrides_empty_when_nothing_enabled() -> None:
    assert api_client.collect_overrides({"employment_800m": 1.0}, {}) == {}


def test_predict_ok(monkeypatch) -> None:
    payload = {"prediction": 48213.7, "interval": {"lower": 1, "upper": 2}}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(200, payload))
    out = api_client.predict("http://x", 35.68, 139.70, None)
    assert out.status == "ok"
    assert out.data == payload


def test_predict_validation_error_is_humanised(monkeypatch) -> None:
    body = {
        "detail": [
            {"loc": ["body", "longitude"], "msg": "Input should be less than 139.95"}
        ]
    }
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(422, body))
    out = api_client.predict("http://x", 35.68, 200.0, None)
    assert out.status == "validation_error"
    assert out.message is not None
    assert "longitude" in out.message
    assert "detail" not in out.message  # not the raw Pydantic dump


def test_predict_unreachable_on_request_exception(monkeypatch) -> None:
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    out = api_client.predict("http://x", 35.68, 139.70, None)
    assert out.status == "unreachable"


def test_predict_unexpected_status_is_error(monkeypatch) -> None:
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(500, {}))
    out = api_client.predict("http://x", 35.68, 139.70, None)
    assert out.status == "error"
    assert "500" in (out.message or "")


def test_predict_includes_overrides_in_payload(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json, timeout):  # noqa: A002 - mirrors requests kw name
        captured["json"] = json
        return _FakeResponse(200, {"prediction": 1.0})

    monkeypatch.setattr(requests, "post", fake_post)
    api_client.predict("http://x", 35.68, 139.70, {"employment_800m": 5.0})
    assert captured["json"]["overrides"] == {"employment_800m": 5.0}
    # No overrides -> key omitted entirely.
    api_client.predict("http://x", 35.68, 139.70, None)
    assert "overrides" not in captured["json"]


def test_health_returns_none_on_error(monkeypatch) -> None:
    def boom(*a, **k):
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", boom)
    assert api_client.health("http://x") is None


def test_health_returns_payload(monkeypatch) -> None:
    body = {"status": "ok", "model_id": "m1"}
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(200, body))
    assert api_client.health("http://x") == body


def test_humanize_validation_error_variants() -> None:
    assert api_client.humanize_validation_error("plain string") == "plain string"
    msg = api_client.humanize_validation_error(
        [{"loc": ["body", "latitude"], "msg": "too big"}]
    )
    assert msg == "latitude: too big"
    assert api_client.humanize_validation_error(None)


def test_feature_metadata_is_complete() -> None:
    # The 12 servable features, split into the spec's override tiers.
    assert len(api_client.FEATURES) == 12
    assert set(api_client.PRIMARY_KEYS) == {
        "employment_800m",
        "pop_800m_catchment",
        "landuse_built_frac",
        "landuse_mix",
        "station_mode",
    }
    assert set(api_client.ADVANCED_KEYS) == {
        "n_routes",
        "min_transfers_yamanote",
        "mean_route_degree_cent",
        "mean_route_btw_cent",
        "max_route_btw_cent",
    }


def test_build_candidate_deck_with_and_without_overlay() -> None:
    pydeck = pytest.importorskip("pydeck")
    from tokyo_ridership.viz import maps

    deck = maps.build_candidate_deck(35.68, 139.70)
    assert isinstance(deck, pydeck.Deck)
    assert [layer.id for layer in deck.layers] == ["candidate"]

    import numpy as np
    import pandas as pd

    residual_df = pd.DataFrame(
        {
            "station_key": ["A"],
            "label_en": ["A-en"],
            "ward_jp": ["新宿区"],
            "residual": [1.0],
            "stop_lat": [35.7],
            "stop_lon": [139.7],
            "ridership_actual": [np.expm1(10.0)],
            "ridership_pred": [np.expm1(9.0)],
            "radius_m": [200.0],
            "fill_color": [[1, 2, 3, 4]],
        }
    )
    deck2 = maps.build_candidate_deck(35.68, 139.70, residual_df)
    assert [layer.id for layer in deck2.layers] == ["stations", "candidate"]
