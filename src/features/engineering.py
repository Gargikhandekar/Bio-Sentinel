"""Lightweight feature helpers. Selection is configuration-driven, not test-set driven."""

from __future__ import annotations

import pandas as pd

from src.data.preprocessing import FEATURE_COLUMNS


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, FEATURE_COLUMNS].copy()
