"""Model experiments + selection (Phase 4).

Runs the model roster through the Phase 3 harness under three endogeneity
variants (full / no-``daily_stop_events`` / stop-events-only) on one identical
set of spatial blocks, selects the best **no-``daily_stop_events``** model (the
servable, greenfield-usable set — CLAUDE.md rule 4), refits it on all labelled
stations, and serializes it as one bundle with a model id and conformal interval
offsets. Also exports the comparison metrics and the OOF residuals for the map.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone

from tokyo_ridership.config import load_config, processed_path
from tokyo_ridership.models import evaluate
from tokyo_ridership.models.registry import ModelSpec, model_specs

_SLUG = {
    "lightgbm": "lgbm",
    "random_forest": "rf",
    "gradient_boosting": "gbm",
    "ridge": "ridge",
    "dummy": "dummy",
}
_VARIANT_ABBREV = {
    "full": "full",
    "no_daily_stop_events": "no-dse",
    "stop_events_only": "dse-only",
}


def load_params(path: str | Path = "params.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        params: dict[str, Any] = yaml.safe_load(fh)
    return params


def feature_sets(features: dict[str, Any]) -> dict[str, tuple[list[str], list[str]]]:
    """The three endogeneity feature-set variants, built from config groups."""
    numeric_full = (
        features["connectivity"]
        + features["service"]
        + features["geography"]
        + features["demographics"]
        + features["landuse"]
    )
    categorical = features["categorical"]
    dse = "daily_stop_events"
    return {
        "full": (numeric_full, categorical),
        "no_daily_stop_events": ([c for c in numeric_full if c != dse], categorical),
        "stop_events_only": ([dse], []),
    }


def _cell_scores(
    df: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    spec: ModelSpec,
    blocks: np.ndarray,
    n_splits: int,
) -> dict[str, float]:
    _, fold_scores, _ = evaluate.nested_spatial_cv(
        df[numeric + categorical],
        df[target],
        blocks,
        numeric=numeric,
        categorical=categorical,
        model=spec.estimator,
        param_grid=spec.grid,
        n_splits=n_splits,
    )
    return evaluate._aggregate(fold_scores)


def run_experiments(
    df: pd.DataFrame,
    target: str,
    variants: dict[str, tuple[list[str], list[str]]],
    specs: list[ModelSpec],
    blocks: np.ndarray,
    n_splits: int,
) -> pd.DataFrame:
    """Model x variant comparison table on identical folds (Dummy computed once)."""
    rows: list[dict[str, Any]] = []
    dummy_scores: dict[str, float] | None = None
    for spec in specs:
        for variant, (numeric, categorical) in variants.items():
            if spec.name == "dummy":  # variant-invariant floor
                if dummy_scores is None:
                    dummy_scores = _cell_scores(
                        df, target, numeric, categorical, spec, blocks, n_splits
                    )
                scores = dummy_scores
            else:
                scores = _cell_scores(
                    df, target, numeric, categorical, spec, blocks, n_splits
                )
            print(
                f"  [{spec.name:16s} | {variant:20s}] "
                f"r2={scores['r2_mean']:.3f} mae={scores['mae_log_mean']:.3f}",
                flush=True,
            )
            rows.append({"model": spec.name, "variant": variant, **scores})
    return pd.DataFrame(rows)


def select_servable(table: pd.DataFrame, selection: dict[str, Any]) -> pd.Series:
    """Best model within the servable (no-dse) variant only; never full/dse-only."""
    candidates = table[table["variant"] == selection["servable_variant"]]
    ordered = candidates.sort_values(
        [selection["primary_metric"], selection["tie_breaker"]],
        ascending=[False, True],
    )
    return ordered.iloc[0]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_id(
    model: str,
    variant: str,
    best_params: dict[str, Any],
    features: list[str],
    matrix_hash: str,
    seed: int,
) -> str:
    payload = json.dumps(
        {
            "best_params": best_params,
            "features": sorted(features),
            "matrix_hash": matrix_hash,
            "seed": seed,
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"{_SLUG.get(model, model)}_{_VARIANT_ABBREV.get(variant, variant)}_{digest}"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def main() -> None:
    cfg = load_config()
    params = load_params()
    seed = int(params["seed"])
    target = cfg["target"]["transformed"]
    scv = params["spatial_cv"]
    n_splits, block_km, strategy = (
        int(scv["n_splits"]),
        float(scv["grid_size_km"]),
        scv["strategy"],
    )
    level = float(params["intervals"]["confidence_level"])

    matrix_path = processed_path(cfg, cfg["processed"]["station_features"])
    df = pd.read_parquet(matrix_path).dropna(subset=[target]).reset_index(drop=True)
    variants = feature_sets(cfg["features"])
    specs = model_specs(params, seed)

    # Step 2 — spatial blocks ONCE (identical folds across every cell).
    ward = df["ward_jp"] if strategy == "ward" else None
    blocks = evaluate.spatial_blocks(
        df[["stop_lat", "stop_lon"]], block_km, strategy, ward
    )

    # Step 3 — scores-only comparison grid.
    print(f"[train] running {len(specs)} models x {len(variants)} variants...")
    table = run_experiments(df, target, variants, specs, blocks, n_splits)
    print(
        table[
            ["model", "variant", "r2_mean", "r2_std", "mae_log_mean", "rmse_orig_mean"]
        ]
        .round(4)
        .to_string(index=False)
    )

    # Step 4 — select the servable model (no_daily_stop_events variant only).
    winner_row = select_servable(table, params["model_selection"])
    winner = next(s for s in specs if s.name == winner_row["model"])
    servable_variant = params["model_selection"]["servable_variant"]
    numeric, categorical = variants[servable_variant]
    print(f"[train] servable model = {winner.name} ({servable_variant})")

    # Step 5 — full run on the winner: OOF + conformal offsets + Moran/LISA.
    result = evaluate.run(
        df,
        numeric_features=numeric,
        categorical_features=categorical,
        target=target,
        model=winner.estimator,
        param_grid=winner.grid,
        n_splits=n_splits,
        block_size_km=block_km,
        strategy=strategy,
        confidence_level=level,
        seed=seed,
    )

    # Step 6 — final tuning + refit on ALL labelled stations (same block logic).
    x_all, y_all = df[numeric + categorical], df[target]
    folds = evaluate.spatial_folds(blocks, n_splits)
    best_params = evaluate._tune(
        x_all, y_all, numeric, categorical, winner.estimator, winner.grid, folds
    )
    pipeline = evaluate.build_pipeline(
        numeric, categorical, clone(winner.estimator).set_params(**best_params)
    )
    pipeline.fit(x_all, y_all)

    # Step 7 — serialize the artifact bundle.
    matrix_hash = _sha256_file(matrix_path)
    model_id = _model_id(
        winner.name,
        servable_variant,
        best_params,
        numeric + categorical,
        matrix_hash,
        seed,
    )
    bundle = {
        "model_id": model_id,
        "pipeline": pipeline,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target": target,
        "transform": "log1p",
        "interval_offsets": {
            "level": result.interval["level"],
            "lower_offset": result.interval["lower_offset"],
            "upper_offset": result.interval["upper_offset"],
        },
        "training_summary": {
            "variant": servable_variant,
            "model": winner.name,
            "best_params": best_params,
            "n_stations": int(len(df)),
            "cv_scores": {
                "r2_mean": float(winner_row["r2_mean"]),
                "r2_std": float(winner_row["r2_std"]),
                "mae_log_mean": float(winner_row["mae_log_mean"]),
                "rmse_orig_mean": float(winner_row["rmse_orig_mean"]),
            },
            "moran": result.moran,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature_matrix_hash": matrix_hash,
        },
    }
    models_dir = Path(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, models_dir / "pipeline.joblib")

    # Step 8 — metrics.json + OOF residuals export.
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = _jsonable(
        {
            "model_id": model_id,
            "servable_variant": servable_variant,
            "comparison": table.to_dict("records"),
            "winner": bundle["training_summary"]["cv_scores"],
            "interval": result.interval,
            "moran": result.moran,
        }
    )
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    oof = result.oof.merge(
        df[["station_key", "stop_lat", "stop_lon"]], on="station_key", how="left"
    )
    oof.to_parquet(processed_path(cfg, "oof_residuals.parquet"), index=False)

    print(f"[train] model_id={model_id}")
    print(
        "[train] wrote models/pipeline.joblib, reports/metrics.json, "
        "data/processed/oof_residuals.parquet"
    )


if __name__ == "__main__":
    main()
