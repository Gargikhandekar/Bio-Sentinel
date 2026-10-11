"""Leakage-safe preprocessing with ColumnTransformer fitted on training data only."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "age_group",
    "bmi",
    "gen_health",
    "phys_unhealthy_days",
    "ment_unhealthy_days",
    "education",
    "income",
]
CATEGORICAL = [
    "sex",
    "race_group",
    "smoker",
    "phys_activity",
    "fruits_daily",
    "veggies_daily",
    "difficulty_walking",
    "healthcare_coverage",
    "binge_drinking",
    "high_cholesterol",
    "chol_check",
]
FEATURE_COLUMNS = NUMERIC + CATEGORICAL


@dataclass
class PreparedSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_val: pd.DataFrame
    y_test: pd.DataFrame
    preprocessor: ColumnTransformer


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL),
        ]
    )


def transform_split(prepared: PreparedSplit) -> tuple:
    x_train = prepared.preprocessor.fit_transform(prepared.X_train)
    x_val = prepared.preprocessor.transform(prepared.X_val)
    x_test = prepared.preprocessor.transform(prepared.X_test)
    return x_train, x_val, x_test


def save_preprocessor(preprocessor: ColumnTransformer, path) -> None:
    joblib.dump(preprocessor, path)


def load_preprocessor(path) -> ColumnTransformer:
    return joblib.load(path)
