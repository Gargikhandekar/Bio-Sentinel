# BIO-SENTINEL

**Adaptive Computational Intelligence for Chronic Disease Risk Prognostics**

Research prototype for multi-label chronic-disease *risk* modeling on the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2021 public-use file. It is **not** a diagnostic device and must not be used for clinical decision-making.

## Project overview

BIO-SENTINEL investigates whether adaptive, context-weighted ensembles, online learning, calibration, and conformal uncertainty improve robustness of Type 2 diabetes, hypertension, and CVD *predicted risk* when the data distribution is deliberately shifted.

## Research motivation

Static classifiers trained on survey snapshots assume the feature/label relationship is stable. Public-health data are cross-sectional, imbalanced, and entangled across related chronic conditions. The system tests adaptive machinery on that setting without pretending the stream is a longitudinal cohort.

## Research gap

See `research/research_gap.md`. The work is an integration and experimental protocol, not a new ML algorithm. ACWE (Adaptive Context-Weighted Ensemble) is an internal name for a convex combination of existing competence, recency, and reliability scores.

## Research questions

RQ1–RQ6 are listed in `research/research_questions.md` (including calibration/uncertainty quality, not only AUROC).

## Contributions

- Leakage-safe multi-label BRFSS 2021 analytic extract (labels excluded from predictors).
- End-to-end pipeline: baselines, LightGBM, modern tabular (TabNet or CPU MLP), River online learners, ADWIN, ACWE, isotonic calibration, split conformal prediction, SHAP, ablation, bootstrap CIs.
- FastAPI + React research console that displays **measured** artifacts or “Experiment not yet executed”.

## Architecture

```
BRFSS 2021 XPT → processed parquet → sklearn pipeline → model artifacts
                                              ↓
                                    SQLite + results JSON
                                              ↓
                                    FastAPI  ←→  React dashboard
```

## Dataset

- **Source:** CDC BRFSS 2021 LLCP public-use file (438,693 records).
- **Official page:** https://www.cdc.gov/brfss/annual_data/annual_2021.html
- **Florida** is not in the 2021 national file.
- **Targets:** binary diabetes (`DIABETE4==1`), hypertension (`BPHIGH6==1`), CVD (`_MICHD==1`). Pregnancy-only and borderline codes are negatives. Complete-case n = 431,843.
- If this environment cannot download from cdc.gov, retrieve the same `LLCP2021XPT.zip` from a CDC website snapshot (Internet Archive) and keep the provenance note in `config/default.yaml`.

## Methodology

See `research/methodology.md` and `research/leakage_audit.md`.

## Models

Baselines: logistic regression, random forest, linear SVM, LightGBM. Modern tabular: TabNet if PyTorch is present, otherwise a documented CPU MLP. Online: River logistic regression and passive-aggressive (predict-then-update).

## Dynamic ensemble

ACWE is a named integration of local competence × recency × reliability. It is not claimed as a new algorithm.

## Results

From the current laptop run on a stratified 40k subsample (diabetes, held-out test):

| Method | ROC-AUC | Brier | ECE |
| --- | --- | --- | --- |
| LightGBM | 0.817 | 0.165 | 0.209 |
| Soft vote | 0.821 | 0.159 | 0.215 |
| ACWE raw | 0.821 | 0.157 | 0.212 |
| ACWE calibrated | 0.820 | 0.095 | 0.008 |

Calibration, not dynamic weighting, is the large reliability change. See `results/experiments/latest.json`.

## Ablation

Generated automatically in the JSON and on `/research/ablation`. Removing calibration worsens Brier; AUROC stays similar.

## Limitations

See `research/limitations.md`.

## Installation

```bash
python -m pip install -r requirements.txt
cd frontend
npm install
```

Place `LLCP2021.XPT` in `data/raw/` if it is not already present.

## Running the backend

From the repository root:

```bash
python scripts/prepare_data.py
python scripts/run_experiments.py
set PYTHONPATH=.
uvicorn backend.main:app --reload --port 8000
```

PowerShell:

```powershell
$env:PYTHONPATH="."
uvicorn backend.main:app --reload --port 8000
```

## Running the frontend

```bash
cd frontend
npm run dev
```

Open http://localhost:5173

Optional containerized stack (not required on a laptop):

```bash
docker compose up
```

## Running experiments

```bash
python scripts/run_experiments.py
```

Laptop profile uses `training.max_rows: 40000` in `config/default.yaml`. Increase it for a fuller run.

## API

- `GET /api/health`
- `GET /api/datasets`
- `GET /api/models`
- `POST /api/predict`
- `POST /api/explain`
- `GET /api/experiments`
- `POST /api/experiments/run`
- `GET /api/drift`
- `GET /api/research/metrics`
- `GET /api/research/ablation`
- `GET /api/research/calibration`

## Testing

```bash
pytest
cd frontend
npm test
```

## Limitations

See `research/limitations.md`.

## License

MIT. BRFSS data remain CDC public-use survey data; cite CDC BRFSS 2021.
