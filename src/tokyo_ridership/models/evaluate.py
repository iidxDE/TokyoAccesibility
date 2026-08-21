"""Spatially-blocked evaluation harness (Phase 3).

The fixed apparatus every model / feature-set passes through. Reusable
functions, not inline cells:

- spatial-block fold generator (grid primary, ward folds as robustness check),
- nested spatial CV (inner tuning loop + outer eval loop, same block logic),
- split-conformal prediction intervals from out-of-fold residuals,
- residual diagnostics: Moran's I + LISA on the out-of-fold residuals.

Both CV loops are spatially blocked and share one block-assignment path — there
are no random splits anywhere (CLAUDE.md rule 9). Preprocessing is one Pipeline
(impute + scale + one-hot + model) fit inside every fold (rule 8).

Single-call interface: :func:`run` -> scores + intervals + residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Local equirectangular scale factors (km per degree) near Tokyo.
_KM_PER_DEG_LAT = 110.574
_KM_PER_DEG_LON = 111.320

Fold = tuple[np.ndarray, np.ndarray]


# --------------------------------------------------------------------------- #
# Spatial blocking
# --------------------------------------------------------------------------- #
def spatial_blocks(
    coords: pd.DataFrame,
    block_size_km: float,
    strategy: str = "grid",
    ward: pd.Series | None = None,
) -> np.ndarray:
    """Assign each station a spatial-block label.

    ``grid``: square cells of ``block_size_km`` in a local equirectangular
    projection. ``ward``: the categorical ward assignment (robustness check).
    """
    if strategy == "ward":
        if ward is None:
            raise ValueError("strategy='ward' requires the ward series")
        return ward.to_numpy(dtype=object)
    if strategy != "grid":
        raise ValueError(f"unknown blocking strategy: {strategy!r}")

    lat = coords["stop_lat"].to_numpy()
    lon = coords["stop_lon"].to_numpy()
    x_km = (lon - lon.mean()) * _KM_PER_DEG_LON * np.cos(np.radians(lat.mean()))
    y_km = (lat - lat.mean()) * _KM_PER_DEG_LAT
    col = np.floor(x_km / block_size_km).astype(int)
    row = np.floor(y_km / block_size_km).astype(int)
    return np.array([f"{r}_{c}" for r, c in zip(row, col, strict=True)], dtype=object)


def spatial_folds(blocks: np.ndarray, n_splits: int) -> list[Fold]:
    """Group blocks into folds so no block is split across train/test.

    Uses ``GroupKFold`` (deterministic, no shuffling); caps ``n_splits`` at the
    number of distinct blocks.
    """
    n_groups = len(np.unique(blocks))
    n = min(n_splits, n_groups)
    if n < 2:
        raise ValueError("need at least 2 spatial blocks to form folds")
    gkf = GroupKFold(n_splits=n)
    idx = np.arange(len(blocks))
    return [(tr, te) for tr, te in gkf.split(idx, groups=blocks)]


# --------------------------------------------------------------------------- #
# Pipeline + metrics
# --------------------------------------------------------------------------- #
def build_pipeline(
    numeric: list[str], categorical: list[str], model: BaseEstimator
) -> Pipeline:
    """One preprocessing+model Pipeline (impute+scale numeric, one-hot categorical)."""
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    transformers: list[tuple[str, Any, list[str]]] = [("num", numeric_pipe, numeric)]
    # OneHotEncoder.fit errors on zero columns; omit the cat block when empty
    # (the stop_events_only variant has no categorical features).
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        )
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("pre", pre), ("model", model)])


def _metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    """R2 + MAE on the log target; RMSE back-transformed to passengers/day."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "r2": float(r2_score(y_true_log, y_pred_log)),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmse_orig": float(root_mean_squared_error(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Nested spatial CV
# --------------------------------------------------------------------------- #
def _tune(
    X: pd.DataFrame,
    y: pd.Series,
    numeric: list[str],
    categorical: list[str],
    model: BaseEstimator,
    param_grid: dict[str, list[Any]],
    inner_folds: list[Fold],
) -> dict[str, Any]:
    """Grid-search hyperparameters over the inner spatial folds (score = log R2)."""
    combos = list(ParameterGrid(param_grid)) if param_grid else [{}]
    if len(combos) == 1:
        return combos[0]

    best_score, best = -np.inf, combos[0]
    for params in combos:
        scores = []
        for tr, te in inner_folds:
            pipe = build_pipeline(
                numeric, categorical, clone(model).set_params(**params)
            )
            pipe.fit(X.iloc[tr], y.iloc[tr])
            scores.append(r2_score(y.iloc[te], pipe.predict(X.iloc[te])))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best = mean_score, params
    return best


def nested_spatial_cv(
    X: pd.DataFrame,
    y: pd.Series,
    blocks: np.ndarray,
    *,
    numeric: list[str],
    categorical: list[str],
    model: BaseEstimator,
    param_grid: dict[str, list[Any]],
    n_splits: int,
) -> tuple[np.ndarray, pd.DataFrame, list[dict[str, Any]]]:
    """Outer eval loop + inner tuning loop, both on the same spatial blocks.

    Returns out-of-fold predictions (log scale), per-fold scores, and the tuned
    params chosen in each outer fold.
    """
    outer = spatial_folds(blocks, n_splits)
    oof = np.full(len(y), np.nan)
    fold_rows: list[dict[str, Any]] = []
    best_params: list[dict[str, Any]] = []

    for k, (tr, te) in enumerate(outer):
        inner = spatial_folds(blocks[tr], n_splits)
        best = _tune(
            X.iloc[tr], y.iloc[tr], numeric, categorical, model, param_grid, inner
        )
        pipe = build_pipeline(numeric, categorical, clone(model).set_params(**best))
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        oof[te] = pred

        row = _metrics(y.iloc[te].to_numpy(), pred)
        row.update(fold=k, n_test=int(len(te)))
        fold_rows.append(row)
        best_params.append(best)

    return oof, pd.DataFrame(fold_rows), best_params


# --------------------------------------------------------------------------- #
# Intervals + residual diagnostics
# --------------------------------------------------------------------------- #
def conformal_offsets(residuals: np.ndarray, level: float) -> tuple[float, float]:
    """Split-conformal (log-scale) offsets from OOF residuals for a given level."""
    alpha = 1.0 - level
    lo = float(np.quantile(residuals, alpha / 2))
    hi = float(np.quantile(residuals, 1 - alpha / 2))
    return lo, hi


def _coverage(residuals: np.ndarray, lo: float, hi: float) -> float:
    return float(np.mean((residuals >= lo) & (residuals <= hi)))


def moran_lisa(
    residuals: np.ndarray,
    coords: pd.DataFrame,
    k: int = 8,
    permutations: int = 999,
    seed: int = 42,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Global Moran's I and local LISA on residuals (KNN spatial weights)."""
    from esda.moran import Moran, Moran_Local
    from libpysal.weights import KNN

    np.random.seed(seed)  # reproducible permutation inference (not a data split)
    points = coords[["stop_lon", "stop_lat"]].to_numpy()
    weights = KNN.from_array(points, k=k)
    weights.transform = "r"

    global_moran = Moran(residuals, weights, permutations=permutations)
    local = Moran_Local(residuals, weights, permutations=permutations)
    moran = {
        "I": float(global_moran.I),
        "p_value": float(global_moran.p_sim),
        "z_score": float(global_moran.z_sim),
    }
    lisa = pd.DataFrame(
        {
            "lisa_I": local.Is,
            "lisa_p": local.p_sim,
            "quadrant": local.q,  # 1=HH 2=LH 3=LL 4=HL
        }
    )
    return moran, lisa


# --------------------------------------------------------------------------- #
# Single-call interface
# --------------------------------------------------------------------------- #
@dataclass
class CVResult:
    scores: dict[str, float]
    fold_scores: pd.DataFrame
    oof: pd.DataFrame
    interval: dict[str, float]
    moran: dict[str, float]
    lisa: pd.DataFrame
    best_params: list[dict[str, Any]]


def _aggregate(fold_scores: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in ("r2", "mae_log", "rmse_orig"):
        out[f"{col}_mean"] = float(fold_scores[col].mean())
        out[f"{col}_std"] = float(fold_scores[col].std(ddof=1))
    return out


def run(
    matrix: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    target: str,
    model: BaseEstimator,
    param_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 8,
    block_size_km: float = 5.0,
    strategy: str = "grid",
    confidence_level: float = 0.9,
    knn_k: int = 8,
    seed: int = 42,
) -> CVResult:
    """Run one (feature-set, model) through the harness end-to-end.

    ``matrix`` must carry the features, ``target``, ``station_key``, ``ward_jp``,
    and ``stop_lat``/``stop_lon``. Rows with a missing target are dropped.
    """
    df = matrix.dropna(subset=[target]).reset_index(drop=True)
    features = numeric_features + categorical_features
    X = df[features]
    y = df[target]
    coords = df[["stop_lat", "stop_lon"]]
    ward = df.get("ward_jp")
    blocks = spatial_blocks(coords, block_size_km, strategy, ward)

    oof_pred, fold_scores, best_params = nested_spatial_cv(
        X,
        y,
        blocks,
        numeric=numeric_features,
        categorical=categorical_features,
        model=model,
        param_grid=param_grid or {},
        n_splits=n_splits,
    )

    residuals = (y.to_numpy() - oof_pred).astype(float)
    lo, hi = conformal_offsets(residuals, confidence_level)
    interval = {
        "level": confidence_level,
        "lower_offset": lo,
        "upper_offset": hi,
        "coverage": _coverage(residuals, lo, hi),
    }
    moran, lisa = moran_lisa(residuals, coords, k=knn_k, seed=seed)

    oof = pd.DataFrame(
        {
            "station_key": df["station_key"],
            "block": blocks,
            "y_true": y.to_numpy(),
            "y_pred": oof_pred,
            "residual": residuals,
        }
    )
    oof = pd.concat([oof, lisa], axis=1)

    return CVResult(
        scores=_aggregate(fold_scores),
        fold_scores=fold_scores,
        oof=oof,
        interval=interval,
        moran=moran,
        lisa=lisa,
        best_params=best_params,
    )
