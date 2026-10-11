"""Shared FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.config import load_config
from src.persistence.models import make_session_factory


@lru_cache
def get_settings() -> dict:
    return load_config()


def get_db_session():
    cfg = get_settings()
    factory = make_session_factory(cfg["database"]["url"])
    session = factory()
    try:
        yield session
    finally:
        session.close()


def results_path() -> Path:
    from src.config import project_path

    return project_path("results", "experiments", "latest.json")
