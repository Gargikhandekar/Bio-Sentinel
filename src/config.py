"""Shared configuration and paths for BIO-SENTINEL."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "default.yaml"
CONFIG_ENV_VAR = "BIO_SENTINEL_CONFIG"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the YAML config. BIO_SENTINEL_CONFIG overrides the default file (used by tests)."""
    cfg_path = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    with cfg_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def models_dir(cfg: dict[str, Any]) -> Path:
    return project_path(cfg.get("paths", {}).get("models_dir", "models/trained"))


def results_dir(cfg: dict[str, Any]) -> Path:
    return project_path(cfg.get("paths", {}).get("results_dir", "results"))
