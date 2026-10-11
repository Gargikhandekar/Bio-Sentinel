"""Official CDC BRFSS 2021 loader with documented target recoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadstat

from src.config import load_config, project_path
from src.utils.logging import get_logger

LOGGER = get_logger("bio_sentinel.data")

RAW_COLUMNS = [
    "SEQNO",
    "_STATE",
    "DIABETE4",
    "BPHIGH6",
    "_MICHD",
    "CVDINFR4",
    "CVDCRHD4",
    "CVDSTRK3",
    "_AGEG5YR",
    "_SEX",
    "_RACEGR3",
    "_EDUCAG",
    "_INCOMG1",
    "_BMI5",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "_SMOKER3",
    "_TOTINDA",
    "_FRTLT1A",
    "_VEGLT1A",
    "DIFFWALK",
    "_HLTHPLN",
    "_RFBING5",
    "TOLDHI3",
    "CHOLCHK3",
]

LABELS = ("diabetes", "hypertension", "cvd")


def recode_diabetes(value: float) -> float:
    if value == 1:
        return 1.0
    if value in (2, 3, 4):
        return 0.0
    return np.nan


def recode_hypertension(value: float) -> float:
    if value == 1:
        return 1.0
    if value in (2, 3, 4):
        return 0.0
    return np.nan


def recode_cvd(value: float) -> float:
    if value == 1:
        return 1.0
    if value == 2:
        return 0.0
    return np.nan


def _missing_codes(series: pd.Series, codes: tuple[float, ...]) -> pd.Series:
    return series.replace(list(codes), np.nan)


def recode_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["respondent_id"] = (
        df["_STATE"].astype("int64").astype(str)
        + "-"
        + df["SEQNO"].astype("int64").astype(str)
    )
    out["state"] = df["_STATE"]
    out["age_group"] = _missing_codes(df["_AGEG5YR"], (14,))
    out["sex"] = df["_SEX"]
    out["race_group"] = _missing_codes(df["_RACEGR3"], (9,))
    out["education"] = _missing_codes(df["_EDUCAG"], (9,))
    out["income"] = _missing_codes(df["_INCOMG1"], (9,))
    out["bmi"] = df["_BMI5"] / 100.0
    out["gen_health"] = _missing_codes(df["GENHLTH"], (7, 9))
    phys = df["PHYSHLTH"].replace({88: 0, 77: np.nan, 99: np.nan})
    ment = df["MENTHLTH"].replace({88: 0, 77: np.nan, 99: np.nan})
    out["phys_unhealthy_days"] = phys
    out["ment_unhealthy_days"] = ment
    out["smoker"] = _missing_codes(df["_SMOKER3"], (9,))
    out["phys_activity"] = _missing_codes(df["_TOTINDA"], (9,))
    out["fruits_daily"] = _missing_codes(df["_FRTLT1A"], (9,))
    out["veggies_daily"] = _missing_codes(df["_VEGLT1A"], (9,))
    out["difficulty_walking"] = _missing_codes(df["DIFFWALK"], (7, 9))
    out["healthcare_coverage"] = _missing_codes(df["_HLTHPLN"], (9,))
    out["binge_drinking"] = _missing_codes(df["_RFBING5"], (9,))
    out["high_cholesterol"] = _missing_codes(df["TOLDHI3"], (7, 9))
    out["chol_check"] = _missing_codes(df["CHOLCHK3"], (7, 9))
    out["diabetes"] = df["DIABETE4"].map(recode_diabetes)
    out["hypertension"] = df["BPHIGH6"].map(recode_hypertension)
    out["cvd"] = df["_MICHD"].map(recode_cvd)
    return out


def load_raw_brfss(raw_path: Path | None = None) -> pd.DataFrame:
    path = raw_path or project_path("data", "raw", "LLCP2021.XPT")
    if not path.exists():
        raise FileNotFoundError(
            f"BRFSS 2021 file not found at {path}. Place LLCP2021.XPT in data/raw/."
        )
    LOGGER.info("Reading BRFSS XPT from %s", path)
    df, _meta = pyreadstat.read_xport(str(path), usecols=RAW_COLUMNS)
    LOGGER.info("Loaded %s rows", f"{len(df):,}")
    return df


def build_processed_frame(raw: pd.DataFrame) -> pd.DataFrame:
    framed = recode_features(raw)
    complete = framed[list(LABELS)].notna().all(axis=1)
    processed = framed.loc[complete].copy()
    for label in LABELS:
        processed[label] = processed[label].astype(int)
    LOGGER.info("Complete-case rows: %s", f"{len(processed):,}")
    return processed.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: Path | None = None) -> Path:
    dest = path or project_path("data", "processed", "brfss_2021_multilabel.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    LOGGER.info("Wrote processed dataset to %s", dest)
    return dest


def load_processed(path: Path | None = None) -> pd.DataFrame:
    dest = path or project_path("data", "processed", "brfss_2021_multilabel.parquet")
    if not dest.exists():
        raw = load_raw_brfss()
        processed = build_processed_frame(raw)
        save_processed(processed, dest)
        return processed
    return pd.read_parquet(dest)


def dataset_summary(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    summary = {
        "name": cfg["dataset"]["name"],
        "version": cfg["dataset"]["version"],
        "n_rows": int(len(df)),
        "n_features": int(df.drop(columns=["respondent_id", "state", *LABELS], errors="ignore").shape[1]),
        "label_prevalence": {label: float(df[label].mean()) for label in LABELS},
        "label_positives": {label: int(df[label].sum()) for label in LABELS},
        "missingness": df.isna().mean().round(6).to_dict(),
        "cooccurrence_corr": df[list(LABELS)].corr().round(4).to_dict(),
        "florida_note": cfg["dataset"]["missing_geography"],
    }
    return summary
