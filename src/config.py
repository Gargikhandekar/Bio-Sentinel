"""Shared configuration and paths for BIO-SENTINEL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "default.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)
