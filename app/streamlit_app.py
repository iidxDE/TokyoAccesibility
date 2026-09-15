"""Streamlit frontend (Phase 6): static equity map + live what-if.

HTTP client of the FastAPI service ONLY. This module must never import from
``src/tokyo_ridership/models`` (CLAUDE.md rule 3); it calls the API with
``requests``. ``API_BASE_URL`` comes from the environment, never hardcoded.

Two tabs share one setup block (config load, ``API_BASE_URL`` resolution, the
``/health`` chip, cached loaders). Page 1 (Equity Map) reads a precomputed
parquet directly and works with the API stopped; Page 2 (live what-if) is the
API client and is stubbed here.
"""

from __future__ import annotations

import os

import api_client
import pandas as pd
import streamlit as st

from tokyo_ridership.config import load_config
from tokyo_ridership.viz import maps

# --- Shared setup (defined once, reused by both tabs) ------------------------

st.set_page_config(page_title="Tokyo Rail Accessibility", layout="wide")

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


@st.cache_data(show_spinner=False)
def _load_config() -> dict:
    return load_config()


@st.cache_data(show_spinner="Loading station residuals…")
def _load_residuals():
    """Read + enrich the OOF residual parquet once (no API dependency)."""
    return maps.load_residual_layer(_load_config())


@st.cache_data(ttl=15, show_spinner=False)
def _cached_health(base_url: str):
    """Poll ``/health`` at most every 15 s (shared chip; only Page 2 depends on it)."""
    return api_client.health(base_url)


def _health_chip() -> None:
    """Render the shared API-health chip. Page 1 works even when the API is down."""
    info = _cached_health(API_BASE_URL)
    if info:
        model_id = info.get("model_id", "?")
        st.caption(f"🟢 Connected · model `{model_id}` · {API_BASE_URL}")
    else:
        st.caption(f"🔴 API unreachable · {API_BASE_URL} (Equity Map still works)")


# --- Page 2 helpers (What-if siting) -----------------------------------------


def _fmt_value(v) -> str:
    """Human-friendly display for a feature value.

    Always returns a string, which also keeps the audit table's ``value`` column
    single-typed — mixing floats/ints/strings makes Streamlit's Arrow conversion
    log an ``ArrowInvalid`` on every predict. ``bool`` is checked before ``int``
    (it is a subclass) although the servable features carry none.
    """
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:,.3f}".rstrip("0").rstrip(".")
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _run_whatif() -> None:
    """Read coords + override state from session, POST /predict, store the outcome."""
    lat = st.session_state["whatif_lat"]
    lon = st.session_state["whatif_lon"]
    keys = api_client.PRIMARY_KEYS + api_client.ADVANCED_KEYS
    values = {k: st.session_state.get(f"ovr_val_{k}") for k in keys}
    enabled = {k: bool(st.session_state.get(f"ovr_on_{k}")) for k in keys}
    overrides = api_client.collect_overrides(values, enabled)
    with st.spinner("Scoring…"):
        st.session_state["whatif_result"] = api_client.predict(
            API_BASE_URL, lat, lon, overrides
        )


def _render_whatif_result(data: dict) -> None:
    """Render prediction, asymmetric interval, assumptions, extrapolation, features."""
    interval = data["interval"]
    assumptions = data["assumptions"]

    st.metric("Predicted ridership", f"{round(data['prediction']):,} /day")
    level = round(interval["level"] * 100)
    st.caption(
        f"{level}% interval (asymmetric): "
        f"{round(interval['lower']):,} … {round(interval['upper']):,} passengers/day"
    )
    st.caption(f"model `{data.get('model', '?')}`")

    st.markdown("**Assumptions** — the defaults you may want to challenge")
    overridden = assumptions.get("overridden_features") or []
    st.markdown(
        f"- Connectivity & mode taken from **{assumptions['snapped_station']}**, "
        f"{round(assumptions['snap_distance_m'])} m away\n"
        f"- Station mode: **{assumptions['station_mode']}** "
        f"({assumptions['station_mode_source']})\n"
        f"- Overridden features: {', '.join(overridden) if overridden else 'none'}"
    )
    if assumptions.get("ward_outside_23"):
        st.warning("This point is outside the 23 special wards (ward = Outside-23).")

    extrapolation = assumptions.get("extrapolation") or {}
    if extrapolation.get("flag"):
        names = ", ".join(extrapolation.get("features", [])) or "some features"
        st.warning(
            f"Extrapolating beyond the training range for: {names}. "
            "The prediction is less reliable here."
        )
        detail = extrapolation.get("detail") or []
        if detail:
            st.dataframe(pd.DataFrame(detail), hide_index=True)

    with st.expander("Feature vector (audit trail)"):
        rows = [
            {"feature": f.label, "value": _fmt_value(data["features"].get(f.key))}
            for f in api_client.FEATURES
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True)


def _override_control(feat: api_client.Feature, default) -> None:
    """One override widget: a toggle that reveals an input seeded from the default."""
    if not st.checkbox(feat.label, key=f"ovr_on_{feat.key}"):
        st.caption(f"using default: {_fmt_value(default)}")
        return
    vkey = f"ovr_val_{feat.key}"
    if feat.key == "station_mode":
        seed = default if default in api_client.MODES else api_client.MODES[0]
        st.session_state.setdefault(vkey, seed)
        st.selectbox("mode", api_client.MODES, key=vkey, label_visibility="collapsed")
    elif feat.dtype == "int":
        seed = int(default) if isinstance(default, (int, float)) else 0
        st.session_state.setdefault(vkey, seed)
        st.number_input("value", step=1, key=vkey, label_visibility="collapsed")
    else:
        seed = float(default) if isinstance(default, (int, float)) else 0.0
        st.session_state.setdefault(vkey, seed)
        st.number_input("value", key=vkey, label_visibility="collapsed")


def _render_override_panel(features: dict) -> None:
    """Primary + advanced override controls, paired with the assumptions block."""
    st.markdown("**Override assumptions and re-predict**")
    for key in api_client.PRIMARY_KEYS:
        _override_control(api_client.FEATURES_BY_KEY[key], features.get(key))
    with st.expander("Advanced — network position"):
        for key in api_client.ADVANCED_KEYS:
            _override_control(api_client.FEATURES_BY_KEY[key], features.get(key))
    col_go, col_reset = st.columns(2)
    if col_go.button("Re-predict", type="primary", key="whatif_repredict"):
        _run_whatif()
        st.rerun()
    if col_reset.button("Reset overrides", key="whatif_reset"):
        for f in api_client.FEATURES:
            st.session_state.pop(f"ovr_on_{f.key}", None)
            st.session_state.pop(f"ovr_val_{f.key}", None)
        st.rerun()


# --- Page shell --------------------------------------------------------------

st.title("Tokyo Rail Accessibility")
_health_chip()

tab_equity, tab_whatif = st.tabs(["Equity Map", "What-if Siting"])


# --- Page 1: Equity / Residual Map -------------------------------------------


def _legend() -> None:
    st.markdown(
        """
        <div style="margin:0.25rem 0 0.5rem 0;">
          <div style="height:14px;border-radius:3px;
               background:linear-gradient(to right,#0571b0,#f7f7f7,#ca0020);"></div>
          <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
            <span>quieter than predicted</span>
            <span>as predicted</span>
            <span>busier than predicted</span>
          </div>
        </div>
        <div style="font-size:0.75rem;color:#888;">
          Colour = signed log residual, clipped to ±2. Marker size = miss magnitude
          (|residual|).
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_equity:
    st.subheader("Where the structural model over- and under-serves")
    st.write(
        "Each station coloured by how far its actual ridership diverges from the "
        "structural model's prediction — the transit-equity view."
    )

    try:
        df = _load_residuals()
    except FileNotFoundError:
        st.error(
            "Missing `data/processed/oof_residuals.parquet`. Build it with "
            "`make train` (the training step writes the out-of-fold residuals "
            "this map reads)."
        )
    else:
        clusters_only = (
            st.radio(
                "View",
                ["All stations", "Significant clusters only"],
                horizontal=True,
                help="Clusters view keeps only LISA-significant stations (p < 0.05), "
                "recoloured by quadrant: HH/LL are coherent clusters, LH/HL outliers.",
            )
            == "Significant clusters only"
        )

        _legend()

        if clusters_only and not maps.significant_mask(df).any():
            st.info(
                f"No stations reach LISA significance "
                f"(p < {maps.SIGNIFICANCE_P}) in this dataset."
            )
        else:
            st.pydeck_chart(maps.build_residual_deck(df, clusters_only=clusters_only))

        with st.expander("Mean residual by ward"):
            chart = maps.ward_residual_chart(df)
            if chart is None:
                st.caption("No ward labels available for the rollup.")
            else:
                st.altair_chart(chart, use_container_width=True)


# --- Page 2: Live what-if siting (API client) --------------------------------

with tab_whatif:
    st.subheader("What-if: predict ridership for a new station")
    st.write(
        "Drop a candidate coordinate to get predicted daily ridership with an "
        "uncertainty interval, read the assumptions the service made, then override "
        "the questionable ones and re-predict."
    )

    bounds = _load_config()["serving_bounds"]

    # Apply a pending map-click selection *before* the coordinate widgets exist
    # (Streamlit forbids mutating a widget's state after it is instantiated).
    pending = st.session_state.pop("whatif_pending_coords", None)
    if pending is not None:
        st.session_state["whatif_lat"], st.session_state["whatif_lon"] = pending
    st.session_state.setdefault("whatif_lat", 35.6896)
    st.session_state.setdefault("whatif_lon", 139.7006)

    col_lat, col_lon = st.columns(2)
    lat = col_lat.number_input(
        "Latitude",
        key="whatif_lat",
        format="%.4f",
        help=f"Tokyo bounds: {bounds['min_lat']}–{bounds['max_lat']}",
    )
    lon = col_lon.number_input(
        "Longitude",
        key="whatif_lon",
        format="%.4f",
        help=f"Tokyo bounds: {bounds['min_lon']}–{bounds['max_lon']}",
    )

    in_bounds = (
        bounds["min_lat"] <= lat <= bounds["max_lat"]
        and bounds["min_lon"] <= lon <= bounds["max_lon"]
    )
    if not in_bounds:
        st.warning(
            f"Coordinate is outside Tokyo serving bounds "
            f"(lat {bounds['min_lat']}–{bounds['max_lat']}, "
            f"lon {bounds['min_lon']}–{bounds['max_lon']}). Adjust before predicting."
        )

    # Map preview + optional residual overlay for context and click-to-prefill.
    try:
        overlay_df = _load_residuals()
    except FileNotFoundError:
        overlay_df = None
    if overlay_df is not None:
        st.caption("Tip: click an existing station to prefill its coordinates.")
    event = st.pydeck_chart(
        maps.build_candidate_deck(lat, lon, overlay_df),
        on_select="rerun",
        selection_mode="single-object",
        key="whatif_map",
    )
    selection = getattr(event, "selection", None)
    picked = (selection or {}).get("objects", {}).get("stations") if selection else None
    if picked:
        row = picked[0]
        plat, plon = row.get("stop_lat"), row.get("stop_lon")
        if (
            plat is not None
            and plon is not None
            and (round(plat, 6) != round(lat, 6) or round(plon, 6) != round(lon, 6))
        ):
            st.session_state["whatif_pending_coords"] = (float(plat), float(plon))
            st.rerun()

    if st.button(
        "Predict", type="primary", disabled=not in_bounds, key="whatif_predict"
    ):
        _run_whatif()

    # Result block (priority order), covering all seven states.
    outcome = st.session_state.get("whatif_result")
    if outcome is None:
        st.info("Enter a coordinate above and press **Predict** to score a site.")
    elif outcome.status == "ok" and outcome.data is not None:
        _render_whatif_result(outcome.data)
        _render_override_panel(outcome.data["features"])
    elif outcome.status == "validation_error":
        st.error(f"The service rejected the request — {outcome.message}")
    else:  # "unreachable" or "error"
        st.error(
            f"Could not get a prediction from the service at {API_BASE_URL}. "
            "The Equity Map tab still works while the API is down."
        )
