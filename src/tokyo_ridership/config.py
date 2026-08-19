"""Configuration loading and path resolution.

Config and paths come from ``config/config.yaml`` (CLAUDE.md: never hardcode
paths/URLs). Loaders resolve inputs under ``paths.data_raw`` and write clean
intermediate tables under ``paths.data_interim``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the central YAML config into a plain dict."""
    with Path(path).open(encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    return cfg


def raw_path(cfg: dict[str, Any], *parts: str) -> Path:
    """Resolve a path under ``paths.data_raw``."""
    return Path(cfg["paths"]["data_raw"], *parts)


def interim_path(cfg: dict[str, Any], *parts: str) -> Path:
    """Resolve a path under ``paths.data_interim``."""
    return Path(cfg["paths"]["data_interim"], *parts)


def processed_path(cfg: dict[str, Any], *parts: str) -> Path:
    """Resolve a path under ``paths.data_processed``."""
    return Path(cfg["paths"]["data_processed"], *parts)
