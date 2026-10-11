"""Schema validation for the processed BRFSS analytic table."""

from __future__ import annotations

import pandas as pd

from src.data.loader import LABELS

REQUIRED = [
    "respondent_id",
    "age_group",
    "sex",
    "bmi",
    "diabetes",
    "hypertension",
    "cvd",
]


def validate_processed(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [col for col in REQUIRED if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors
    if df[list(LABELS)].isna().any().any():
        errors.append("Label columns contain missing values after complete-case filtering.")
    if not set(df[list(LABELS)].stack().unique()).issubset({0, 1}):
        errors.append("Labels must be binary {0,1}.")
    if df["respondent_id"].duplicated().any():
        errors.append("Duplicate respondent IDs detected.")
    if df["bmi"].dropna().lt(10).any() or df["bmi"].dropna().gt(100).any():
        errors.append("BMI values outside a plausible 10–100 range.")
    return errors
