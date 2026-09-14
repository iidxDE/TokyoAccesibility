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

import requests
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


def _health_chip() -> None:
    """Render an API-health chip. Page 1 never needs the API; Page 2 does."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if resp.ok:
            st.caption(f"🟢 API online · {API_BASE_URL}")
            return
        st.caption(f"🟠 API responded {resp.status_code} · {API_BASE_URL}")
    except requests.RequestException:
        st.caption(f"🔴 API offline · {API_BASE_URL} (Equity Map still works)")


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


# --- Page 2: Live what-if siting (API client) — implemented separately -------

with tab_whatif:
    st.subheader("What-if: predict ridership for a new station")
    st.info("Coming in the next Phase 6 step — this tab calls the FastAPI service.")
