"""Synthetic stand-in for the processed BRFSS table (same columns and codes), for tests only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_brfss(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = {
        "age_group": 13, "sex": 2, "race_group": 5, "education": 4, "income": 7, "gen_health": 5,
        "smoker": 4, "phys_activity": 2, "fruits_daily": 2, "veggies_daily": 2, "difficulty_walking": 2,
        "healthcare_coverage": 2, "binge_drinking": 2, "high_cholesterol": 2, "chol_check": 3,
    }
    df = pd.DataFrame({name: rng.integers(1, top + 1, n).astype(float) for name, top in codes.items()})
    df.insert(0, "respondent_id", [f"1-{i}" for i in range(n)])
    df.insert(1, "state", rng.integers(1, 56, n).astype(float))
    df["bmi"] = rng.normal(28, 6, n).clip(12, 70)
    df["phys_unhealthy_days"] = rng.integers(0, 31, n).astype(float)
    df["ment_unhealthy_days"] = rng.integers(0, 31, n).astype(float)
    for col in ("income", "bmi"):
        df.loc[rng.random(n) < 0.05, col] = np.nan

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    risk = 0.25 * df["age_group"] + 0.12 * (df["bmi"].fillna(28) - 28) + 0.5 * df["gen_health"]
    df["diabetes"] = (rng.random(n) < sigmoid(risk - 5.0)).astype(int)
    df["hypertension"] = (rng.random(n) < sigmoid(risk - 3.2 + 0.8 * df["diabetes"])).astype(int)
    df["cvd"] = (rng.random(n) < sigmoid(risk - 5.5 + 0.6 * df["hypertension"] + 0.5 * (df["sex"] == 1))).astype(int)
    return df
