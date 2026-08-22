"""SHAP explanations for the servable model (Phase 4).

Explains the SAME fitted pipeline that is served: transforms the servable feature
set through the pipeline's fitted preprocessor, runs the SHAP explainer matched
to the model type, and writes per-station SHAP values (expanded one-hot names)
plus a mean-|SHAP| summary for the writeup. SHAP is imported lazily so the rest
of the package loads without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tokyo_ridership.config import load_config, processed_path

_TREE_MODELS = {
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "LGBMRegressor",
}


def _explainer(model: Any, background: np.ndarray) -> Any:
    import shap

    name = type(model).__name__
    if name in _TREE_MODELS:
        return shap.TreeExplainer(model)
    if name == "Ridge":
        return shap.LinearExplainer(model, background)
    return shap.KernelExplainer(model.predict, shap.sample(background, 50))


def main() -> None:
    cfg = load_config()
    bundle = joblib.load(Path(cfg["paths"]["models"]) / "pipeline.joblib")
    pipeline = bundle["pipeline"]
    numeric = bundle["numeric_features"]
    categorical = bundle["categorical_features"]
    target = bundle["target"]

    matrix = processed_path(cfg, cfg["processed"]["station_features"])
    df = pd.read_parquet(matrix).dropna(subset=[target]).reset_index(drop=True)

    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]
    x_trans = pre.transform(df[numeric + categorical])
    if hasattr(x_trans, "toarray"):  # OneHotEncoder yields a sparse matrix
        x_trans = x_trans.toarray()
    x_trans = np.asarray(x_trans, dtype=float)
    feature_names = list(pre.get_feature_names_out())

    shap_values = np.asarray(_explainer(model, x_trans).shap_values(x_trans))

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.insert(0, "station_key", df["station_key"].to_numpy())
    shap_df.to_parquet(processed_path(cfg, "shap_values.parquet"), index=False)

    mean_abs = {
        name: float(np.abs(shap_values[:, i]).mean())
        for i, name in enumerate(feature_names)
    }
    mean_abs = dict(sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True))
    reports = Path(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "shap_summary.json").write_text(json.dumps(mean_abs, indent=2))

    print(
        f"[explain] wrote shap_values.parquet + shap_summary.json "
        f"(top: {list(mean_abs)[:3]})"
    )


if __name__ == "__main__":
    main()
