"""Residual + prediction maps for the equity writeup and frontend (Phase 6).

Page 1 of the Streamlit app (the Equity Map) is a thin caller of this module:
all map-construction logic — colour mapping, radius scaling, tooltip assembly,
the LISA cluster filter, and the ward rollup — lives here (CLAUDE.md: no map
logic in the app). These functions are pure and Streamlit-free so they stay
unit-testable and honour the layering rule.

The Equity Map reads ``data/processed/oof_residuals.parquet`` directly; it makes
no API calls and never imports from ``models``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import colormaps

from tokyo_ridership.config import processed_path

if TYPE_CHECKING:
    import altair
    import pydeck

# Diverging colour is the signed *log* residual, clipped to this symmetric range.
# Red (RdBu_r high end) = busier than predicted; blue = quieter than predicted.
RESIDUAL_CLIP = 2.0

# Radius (metres) scaled from |residual|; the layer's pixel clamps keep the
# smallest misses visible and stop the largest from swamping the map.
RADIUS_MIN_M = 120.0
RADIUS_MAX_M = 900.0

# Per-point alpha grows with |residual|: faint = model is right, solid = miss.
ALPHA_MIN = 45
ALPHA_MAX = 230

# LISA quadrant integer -> readable label (1=HH 2=LH 3=LL 4=HL).
_QUADRANT_LABELS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}

# Categorical palette for the clusters-only view. HH/LL are the spatially
# coherent clusters worth highlighting; LH/HL are spatial outliers, greyed.
_QUADRANT_RGBA = {
    "HH": [202, 0, 32, 220],  # busy cluster (red)
    "LL": [5, 113, 176, 220],  # quiet cluster (blue)
    "LH": [160, 160, 160, 140],  # outlier (grey)
    "HL": [160, 160, 160, 140],  # outlier (grey)
}

# LISA pseudo p-value below which a station counts as a significant cluster.
# Single source of truth for the deck filter and the app's empty-state check.
SIGNIFICANCE_P = 0.05


def significant_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of LISA-significant stations (``lisa_p < SIGNIFICANCE_P``)."""
    return df["lisa_p"] < SIGNIFICANCE_P


def _quadrant_label(q: Any) -> str:
    """Map a LISA quadrant integer to its HH/LH/LL/HL label."""
    try:
        return _QUADRANT_LABELS.get(int(q), "n/a")
    except (TypeError, ValueError):
        return "n/a"


def _residual_rgba(residuals: pd.Series | np.ndarray) -> list[list[int]]:
    """RdBu_r colour for each signed log residual, clipped to ``+-RESIDUAL_CLIP``.

    The alpha channel grows with ``|residual|`` (``ALPHA_MIN``..``ALPHA_MAX``) so
    near-zero "as predicted" stations recede and the misses read loudest.
    Returns ``[r, g, b, a]`` int lists suitable for pydeck ``get_fill_color``.
    """
    values = np.asarray(residuals, dtype=float)
    norm = mpl.colors.Normalize(vmin=-RESIDUAL_CLIP, vmax=RESIDUAL_CLIP, clip=True)
    cmap = colormaps["RdBu_r"]
    rgba = (cmap(norm(values)) * 255).round().astype(int)  # (n, 4) ints in [0, 255]

    mag = np.clip(np.abs(values) / RESIDUAL_CLIP, 0.0, 1.0)
    rgba[:, 3] = (ALPHA_MIN + mag * (ALPHA_MAX - ALPHA_MIN)).round().astype(int)
    return rgba.tolist()


def _radius_from_abs_residual(residuals: pd.Series | np.ndarray) -> np.ndarray:
    """Scale ``|residual|`` into ``[RADIUS_MIN_M, RADIUS_MAX_M]`` metres.

    Normalised by the fixed ``RESIDUAL_CLIP`` (not the data max) so size shares
    one anchor with colour and stays comparable across the all/clusters views; a
    ``sqrt`` lifts the mid-range so moderate misses stay visible.
    """
    mag = np.abs(np.asarray(residuals, dtype=float))
    frac = np.sqrt(np.clip(mag / RESIDUAL_CLIP, 0.0, 1.0))
    return RADIUS_MIN_M + frac * (RADIUS_MAX_M - RADIUS_MIN_M)


def load_residual_layer(config: dict[str, Any]) -> pd.DataFrame:
    """Load the OOF residual parquet and enrich it for the Equity Map.

    Reads ``oof_residuals.parquet`` (log-scale ``y_true``/``y_pred``/``residual``
    plus LISA columns), left-joins ``label_en`` and ``ward_jp`` from
    ``station_features.parquet`` on ``station_key`` (falling back to
    ``station_key`` when a row fails to join), back-transforms ridership with
    ``expm1``, and adds the RGBA colour, radius, and readable quadrant columns.

    Pure: returns a DataFrame. Raises ``FileNotFoundError`` if the parquet is
    absent so the app can render a clear "run ``make train``" message.
    """
    residual_path = processed_path(config, "oof_residuals.parquet")
    if not residual_path.exists():
        raise FileNotFoundError(str(residual_path))
    df = pd.read_parquet(residual_path)

    # Join English name + ward for the tooltip; keep working if the file/rows miss.
    features_name = config["processed"]["station_features"]
    features_path = processed_path(config, features_name)
    if features_path.exists():
        meta = pd.read_parquet(
            features_path, columns=["station_key", "label_en", "ward_jp"]
        )
        df = df.merge(meta, on="station_key", how="left")
    else:
        df["label_en"] = pd.NA
        df["ward_jp"] = pd.NA

    # Fallback for unmatched rows: show the Japanese key as the name.
    df["label_en"] = df["label_en"].fillna(df["station_key"])

    # Back-transform to passengers/day; never surface the log-scale numbers.
    df["ridership_actual"] = np.expm1(df["y_true"])
    df["ridership_pred"] = np.expm1(df["y_pred"])

    df["quadrant_label"] = df["quadrant"].map(_quadrant_label)
    df["fill_color"] = _residual_rgba(df["residual"])
    df["radius_m"] = _radius_from_abs_residual(df["residual"])
    return df


_TOOLTIP = {
    "html": (
        "<b>{label_en}</b> <span style='color:#aaa'>{station_key}</span><br/>"
        "Ward: {ward_jp}<br/>"
        "Actual: {ridership_actual_fmt} /day<br/>"
        "Predicted: {ridership_pred_fmt} /day<br/>"
        "Residual (log): {residual_fmt}<br/>"
        "LISA cluster: {quadrant_label}"
    ),
    "style": {"backgroundColor": "#1b1b1b", "color": "#fafafa", "fontSize": "0.8rem"},
}


def _with_tooltip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Attach preformatted tooltip strings (pydeck templating can't format numbers)."""
    out = df.copy()
    out["ridership_actual_fmt"] = (
        out["ridership_actual"]
        .round()
        .astype("Int64")
        .map(lambda v: f"{v:,}" if pd.notna(v) else "n/a")
    )
    out["ridership_pred_fmt"] = (
        out["ridership_pred"]
        .round()
        .astype("Int64")
        .map(lambda v: f"{v:,}" if pd.notna(v) else "n/a")
    )
    out["residual_fmt"] = out["residual"].map(
        lambda v: f"{v:+.2f}" if pd.notna(v) else "n/a"
    )
    return out


def build_residual_deck(df: pd.DataFrame, *, clusters_only: bool) -> pydeck.Deck:
    """Assemble the equity ScatterplotLayer, view, and tooltip.

    ``clusters_only`` filters to significant LISA clusters (``lisa_p < 0.05``)
    and recolours by quadrant (HH/LL emphasised, LH/HL greyed as outliers).
    The default (``False``) shows all stations coloured by signed residual.

    ``pydeck`` (the ``app`` extra) is imported lazily so the pure helpers and
    ``load_residual_layer`` stay usable without the frontend dependency.
    """
    import pydeck

    data = df
    if clusters_only:
        data = df[significant_mask(df)].copy()
        data["fill_color"] = data["quadrant_label"].map(_QUADRANT_RGBA)

    data = _with_tooltip_strings(data)

    layer = pydeck.Layer(
        "ScatterplotLayer",
        id="stations",
        data=data,
        get_position=["stop_lon", "stop_lat"],
        get_fill_color="fill_color",
        get_radius="radius_m",
        radius_min_pixels=2,
        radius_max_pixels=28,
        pickable=True,
        opacity=1.0,  # per-point alpha (in fill_color) carries the fade
        stroked=False,
    )
    view_state = pydeck.ViewState(latitude=35.69, longitude=139.75, zoom=10.5)
    return pydeck.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=None,
        tooltip=_TOOLTIP,
    )


def ward_residual_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean residual + station count per ward, sorted by mean residual.

    Mirrors the notebook's ward table for the optional rollup. Rows without a
    ``ward_jp`` are dropped.
    """
    valid = df.dropna(subset=["ward_jp"])
    summary = (
        valid.groupby("ward_jp")
        .agg(mean_residual=("residual", "mean"), n_stations=("residual", "size"))
        .reset_index()
        .sort_values("mean_residual")
        .reset_index(drop=True)
    )
    return summary


def ward_residual_chart(df: pd.DataFrame) -> altair.Chart | None:
    """Horizontal bar of mean residual per ward, ranked by value.

    ``st.bar_chart`` sorts a nominal axis by codepoint and discards frame order,
    so the ranking is drawn explicitly with Altair (``sort`` pinned to the
    value). Returns ``None`` when no ward labels are available. ``altair`` (ships
    with Streamlit, in the ``app`` extra) is imported lazily like ``pydeck``.
    """
    import altair as alt

    summary = ward_residual_summary(df)
    if summary.empty:
        return None
    return (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X("mean_residual:Q", title="mean residual (log)"),
            y=alt.Y(
                "ward_jp:N",
                title=None,
                sort=alt.EncodingSortField("mean_residual", order="ascending"),
            ),
            tooltip=["ward_jp", "mean_residual", "n_stations"],
        )
    )


# Bright marker for the What-if candidate point (distinct from the residual ramp).
_CANDIDATE_RGBA = [255, 214, 10, 235]


def build_candidate_deck(
    lat: float,
    lon: float,
    residual_df: pd.DataFrame | None = None,
    *,
    view_zoom: float = 12.0,
) -> pydeck.Deck:
    """Map preview for Page 2: a candidate marker over an optional residual layer.

    When ``residual_df`` (the Page-1 ``load_residual_layer`` frame) is provided, the
    existing stations are drawn underneath as a pickable ``id="stations"`` layer so
    the app can prefill the coordinate inputs from an ``on_select`` pick; the
    candidate marker sits on top. The view centres on the candidate. ``pydeck`` is
    imported lazily so the module stays importable without the ``app`` extra.
    """
    import pydeck

    layers = []
    if residual_df is not None and not residual_df.empty:
        stations = _with_tooltip_strings(residual_df)
        layers.append(
            pydeck.Layer(
                "ScatterplotLayer",
                id="stations",
                data=stations,
                get_position=["stop_lon", "stop_lat"],
                get_fill_color="fill_color",
                get_radius="radius_m",
                radius_min_pixels=2,
                radius_max_pixels=20,
                pickable=True,
                opacity=1.0,
                stroked=False,
            )
        )

    candidate = pd.DataFrame({"lon": [lon], "lat": [lat]})
    layers.append(
        pydeck.Layer(
            "ScatterplotLayer",
            id="candidate",
            data=candidate,
            get_position=["lon", "lat"],
            get_fill_color=_CANDIDATE_RGBA,
            get_radius=180,
            radius_min_pixels=7,
            radius_max_pixels=16,
            pickable=False,
            stroked=True,
            get_line_color=[20, 20, 20, 255],
            line_width_min_pixels=2,
        )
    )

    view_state = pydeck.ViewState(latitude=lat, longitude=lon, zoom=view_zoom)
    return pydeck.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=None,
        tooltip=_TOOLTIP,
    )
