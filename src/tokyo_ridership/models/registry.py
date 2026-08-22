"""Model registry (Phase 4).

Maps an ``estimator`` name to a constructed sklearn estimator (seed threaded,
``fixed`` params applied) and pairs it with its hyperparameter grid. LightGBM is
imported lazily so the module loads without the ``models`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


@dataclass
class ModelSpec:
    name: str
    estimator: BaseEstimator
    grid: dict[str, list[Any]]


def build_estimator(
    name: str, seed: int, fixed: dict[str, Any] | None = None
) -> BaseEstimator:
    """Construct one estimator by name with the seed + fixed params applied."""
    fixed = dict(fixed or {})
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Ridge(**fixed)  # default solver has no random_state
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **fixed)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=seed, **fixed)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1, **fixed)
    raise ValueError(f"unknown estimator: {name!r}")


def model_specs(params: dict[str, Any], seed: int) -> list[ModelSpec]:
    """Build a :class:`ModelSpec` for every entry in ``params['models']``."""
    specs: list[ModelSpec] = []
    for name, cfg in params["models"].items():
        estimator = build_estimator(cfg["estimator"], seed, cfg.get("fixed"))
        specs.append(
            ModelSpec(name=name, estimator=estimator, grid=cfg.get("grid", {}))
        )
    return specs
