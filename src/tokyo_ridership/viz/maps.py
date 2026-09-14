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

from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import pydeck
from matplotlib import colormaps

from tokyo_ridership.config import processed_path

# Diverging colour is the signed *log* residual, clipped to this symmetric range.
# Red (RdBu_r high end) = busier than predicted; blue = quieter than predicted.
RESIDUAL_CLIP = 2.0

# Radius (metres) scaled from |residual|; the layer's pixel clamps keep the
# smallest misses visible and stop the largest from swamping the map.
RADIUS_MIN_M = 120.0
RADIUS_MAX_M = 900.0

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

_SIGNIFICANCE = 0.05


def _quadrant_label(q: Any) -> str:
    """Map a LISA quadrant integer to its HH/LH/LL/HL label."""
    try:
        return _QUADRANT_LABELS.get(int(q), "n/a")
    except (TypeError, ValueError):
        return "n/a"


def _residual_rgba(residuals: pd.Series | np.ndarray) -> list[list[int]]:
    """RdBu_r colour for each signed log residual, clipped to ``+-RESIDUAL_CLIP``.

    Returns ``[r, g, b, a]`` int lists suitable for pydeck ``get_fill_color``.
    """
    norm = mpl.colors.Normalize(vmin=-RESIDUAL_CLIP, vmax=RESIDUAL_CLIP, clip=True)
    cmap = colormaps["RdBu_r"]
    rgba = cmap(norm(np.asarray(residuals, dtype=float)))  # (n, 4) floats in [0, 1]
    return (rgba * 255).round().astype(int).tolist()


def _radius_from_abs_residual(residuals: pd.Series | np.ndarray) -> np.ndarray:
    """Scale ``|residual|`` linearly into ``[RADIUS_MIN_M, RADIUS_MAX_M]`` metres."""
    mag = np.abs(np.asarray(residuals, dtype=float))
    top = float(np.nanmax(mag)) if mag.size else 0.0
    if top <= 0:
        return np.full(mag.shape, RADIUS_MIN_M)
    frac = np.clip(mag / top, 0.0, 1.0)
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
    """
    data = df
    if clusters_only:
        data = df[df["lisa_p"] < _SIGNIFICANCE].copy()
        data["fill_color"] = data["quadrant_label"].map(_QUADRANT_RGBA)

    data = _with_tooltip_strings(data)

    layer = pydeck.Layer(
        "ScatterplotLayer",
        id="stations",
        data=data,
        get_position=["stop_lon", "stop_lat"],
        get_fill_color="fill_color",
        get_radius="radius_m",
        radius_min_pixels=3,
        radius_max_pixels=28,
        pickable=True,
        opacity=0.85,
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
