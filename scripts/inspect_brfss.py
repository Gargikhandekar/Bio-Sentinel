"""Inspect official CDC BRFSS 2021 LLCP public-use file (SAS XPORT)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyreadstat

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "LLCP2021.XPT"

COLUMNS = [
    "SEQNO",
    "_STATE",
    "DIABETE4",
    "BPHIGH6",
    "_MICHD",
    "CVDINFR4",
    "CVDCRHD4",
    "CVDSTRK3",
    "BPMEDS",
    "DIABAGE3",
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


def recode_diabetes(v: float) -> float:
    if v == 1:
        return 1.0
    if v in (2, 3, 4):
        return 0.0
    return np.nan


def recode_hypertension(v: float) -> float:
    if v == 1:
        return 1.0
    if v in (2, 3, 4):
        return 0.0
    return np.nan


def recode_cvd(v: float) -> float:
    if v == 1:
        return 1.0
    if v == 2:
        return 0.0
    return np.nan


def main() -> None:
    print(f"Reading {RAW} ({RAW.stat().st_size / 1e9:.2f} GB)")
    df, meta = pyreadstat.read_xport(str(RAW), usecols=COLUMNS)
    print(f"rows={len(df):,} cols={df.shape[1]} file_rows_meta={getattr(meta, 'number_rows', None)}")
    print("columns:", list(df.columns))
    print("\n=== missingness (raw) ===")
    miss = df.isna().mean().sort_values(ascending=False)
    print(miss.to_string())
    print("\n=== target raw value counts ===")
    for col in ["DIABETE4", "BPHIGH6", "_MICHD"]:
        print(f"\n{col}")
        print(df[col].value_counts(dropna=False).sort_index().to_string())

    diabetes = df["DIABETE4"].map(recode_diabetes)
    htn = df["BPHIGH6"].map(recode_hypertension)
    cvd = df["_MICHD"].map(recode_cvd)
    labels = {
        "diabetes": diabetes,
        "hypertension": htn,
        "cvd": cvd,
    }
    print("\n=== recoded target prevalence (including incomplete rows) ===")
    for name, s in labels.items():
        valid = s.dropna()
        print(
            f"{name}: n={len(valid):,} missing={s.isna().sum():,} "
            f"positive={int(valid.sum()):,} rate={valid.mean():.4f}"
        )

    complete = diabetes.notna() & htn.notna() & cvd.notna()
    print(f"\ncomplete-case (all 3 labels defined): {int(complete.sum()):,} / {len(df):,}")
    sub = df.loc[complete].copy()
    sub["diabetes"] = diabetes[complete].astype(int)
    sub["hypertension"] = htn[complete].astype(int)
    sub["cvd"] = cvd[complete].astype(int)
    print("\n=== complete-case label rates ===")
    for name in ["diabetes", "hypertension", "cvd"]:
        print(f"{name}: {sub[name].mean():.4f}  n_pos={sub[name].sum():,}")
    print("\n=== co-occurrence (complete-case) ===")
    print(sub[["diabetes", "hypertension", "cvd"]].corr().round(4).to_string())
    combo = (
        sub["diabetes"].astype(str)
        + sub["hypertension"].astype(str)
        + sub["cvd"].astype(str)
    )
    print("\nlabel pattern D/H/C counts:")
    print(combo.value_counts().to_string())
    print("\n=== potential leakage columns overlap with labels ===")
    print("BPMEDS missingness:", df["BPMEDS"].isna().mean())
    print("DIABAGE3 missingness:", df["DIABAGE3"].isna().mean())
    print("CVDSTRK3 value counts:")
    print(df["CVDSTRK3"].value_counts(dropna=False).head(10).to_string())


if __name__ == "__main__":
    main()
