"""Run the whole pipeline once on synthetic data in a temp dir; every test reads from there.

BIO_SENTINEL_CONFIG points the code (including the API) at a temp config, so tests
never touch models/, results/ or the SQLite file in the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from src.config import CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH
from tests.synthetic import make_synthetic_brfss


@pytest.fixture(scope="session")
def test_config(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("bio_sentinel")
    data = root / "synthetic.parquet"
    make_synthetic_brfss().to_parquet(data, index=False)
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["dataset"]["processed_path"] = str(data)
    cfg["paths"] = {"models_dir": str(root / "models"), "results_dir": str(root / "results")}
    cfg["database"]["url"] = f"sqlite:///{root / 'test.db'}"
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture(scope="session", autouse=True)
def _use_test_config(test_config):
    previous = os.environ.get(CONFIG_ENV_VAR)
    os.environ[CONFIG_ENV_VAR] = str(test_config)
    yield
    if previous is None:
        os.environ.pop(CONFIG_ENV_VAR, None)
    else:
        os.environ[CONFIG_ENV_VAR] = previous


@pytest.fixture(scope="session")
def trained(test_config):
    """Full experiment run + summary + figures, as scripts/reproduce.py does."""
    from backend.services.inference import get_engine
    from src.config import load_config, results_dir
    from src.evaluation.experiments import run_all_experiments
    from src.evaluation.report import make_figures, write_summary

    payload = run_all_experiments(test_config)
    out = results_dir(load_config(test_config))
    write_summary(payload, out / "metrics")
    make_figures(payload, out)
    get_engine.cache_clear()  # the API must pick up the freshly trained artifacts
    return payload
