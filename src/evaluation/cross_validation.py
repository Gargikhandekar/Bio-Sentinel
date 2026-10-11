"""Train/validation/test splitting with multi-label strata."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import LABELS
from src.data.preprocessing import FEATURE_COLUMNS, PreparedSplit, build_preprocessor
from src.features.engineering import feature_frame


def stratified_label_key(y: pd.DataFrame) -> pd.Series:
    return y[list(LABELS)].astype(str).agg("".join, axis=1)


def make_splits(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
    max_rows: int | None = None,
) -> PreparedSplit:
    if max_rows and len(df) > max_rows:
        strata = stratified_label_key(df[list(LABELS)])
        df, _ = train_test_split(df, train_size=max_rows, stratify=strata, random_state=seed)
        df = df.reset_index(drop=True)
    y = df[list(LABELS)]
    x = feature_frame(df)
    strata = stratified_label_key(y)
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=test_size, stratify=strata, random_state=seed
    )
    strata_tv = stratified_label_key(y_trainval)
    relative_val = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=relative_val, stratify=strata_tv, random_state=seed
    )
    preprocessor = build_preprocessor()
    return PreparedSplit(
        X_train=x_train.reset_index(drop=True),
        X_val=x_val.reset_index(drop=True),
        X_test=x_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        preprocessor=preprocessor,
    )


HOLDOUT_PARTS = ("competence", "calibration", "conformal")


def split_holdout(y_val: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Split the validation holdout into three disjoint, label-stratified thirds.

    Each fitting step after model training gets its own data: ACWE competence and
    reliability weights, the isotonic calibrator, and the conformal quantile. Reusing
    one set for calibration and conformal scores would make the scores optimistic
    and void the coverage guarantee.
    """
    idx = np.arange(len(y_val))
    strata = stratified_label_key(y_val)
    competence, rest = train_test_split(idx, train_size=1 / 3, stratify=strata, random_state=seed)
    calibration, conformal = train_test_split(
        rest, train_size=0.5, stratify=strata.iloc[rest], random_state=seed
    )
    return {"competence": np.sort(competence), "calibration": np.sort(calibration), "conformal": np.sort(conformal)}
