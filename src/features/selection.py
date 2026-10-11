"""No unsupervised test-set feature selection. Keep the configured column set."""

from __future__ import annotations

from src.data.preprocessing import FEATURE_COLUMNS


def selected_features() -> list[str]:
    return list(FEATURE_COLUMNS)
