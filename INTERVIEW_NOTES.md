# Interview notes: BIO-SENTINEL

A walkthrough I can give in about 5 minutes, then the reasoning behind each choice.

## 1. One-paragraph pitch

I predict three co-occurring chronic conditions (type 2 diabetes, hypertension, CVD) from the CDC BRFSS 2021 survey (431,843 respondents with all three labels). The question was whether "adaptive" machinery (a dynamic ensemble, online learners, drift detection) makes risk estimates more reliable than a static model, and how calibration and conformal prediction compare. The honest answer: **calibration drove nearly all of the reliability gain.** The dynamic ensemble ties a soft vote, and a calibrated LightGBM matches it. Finding and fixing the evaluation flaws mattered more than the model choice.

## 2. End-to-end walkthrough (follow the code)

1. **Data** (`src/data/loader.py`). Read the SAS XPT with `pyreadstat`, keeping only the columns I need. Recode targets: `DIABETE4==1` → diabetes, `BPHIGH6==1` → hypertension, `_MICHD==1` → CVD. Pregnancy-only and borderline answers become 0; "don't know/refused" become NaN. Keep rows with all three labels. Survey missing codes (7, 9, 77, 99) become NaN, and "88 = none" becomes 0 days.
2. **Leakage audit** (`src/data/leakage.py`). The feature list must not contain the labels, their skip-pattern proxies (`BPMEDS` is asked only of hypertensives, `DIABAGE3` only of diabetics) or stroke. If it does, the run aborts.
3. **Splits** (`src/evaluation/cross_validation.py`). A stratified 40k subsample, then 60/20/20 train/validation/test, stratified on the 3-bit label pattern (8 strata), so all label combinations keep their ratios. The validation set is split again into three disjoint thirds: *competence*, *calibration*, *conformal*.
4. **Preprocessing** (`src/data/preprocessing.py`). A `ColumnTransformer` with median impute + standardize for ordinal/numeric features and mode impute + one-hot for categorical ones. It is fit on train only.
5. **Members** (`src/models/`). Logistic regression, random forest, linear SVM (sigmoid-calibrated by CV), LightGBM, and an MLP (TabNet if installed). One independent model per target. A classifier chain is the multi-label baseline.
6. **ACWE** (`src/ensemble/`, `acwe_stream` in `experiments.py`). Each member's weight = local competence (accuracy of its kNN neighbours in the competence third) × recent accuracy (an EMA updated prequentially) × reliability (validation AUC). The weights are normalized to sum to 1, and the members' probabilities are combined with them.
7. **Calibration** (`src/uncertainty/calibration.py`). Isotonic regression fitted on the calibration third, one calibrator per score being calibrated, outputs clipped to [0.001, 0.999].
8. **Conformal** (`src/uncertainty/conformal.py`). Split conformal on the conformal third. The score is 1 − p(true class), with the finite-sample quantile ⌈(n+1)(1−α)⌉/n. A prediction set ⊆ {0, 1}. Coverage is measured on test, overall and per class.
9. **Evaluation** (`src/evaluation/`). ROC-AUC, PR-AUC, Brier, 15-bin ECE, bootstrap CIs, a paired bootstrap AUC test, McNemar, and an ablation where every variant is calibrated.
10. **Explainability** (`src/explainability/`). TreeSHAP on the LightGBM member, globally for the plots and per request in the API.
11. **Online/drift** (`src/models/online.py`, `src/drift/`). A stream built from the validation and test rows, with a synthetic BMI shift plus 8% label noise injected halfway. River learners are scored prequentially, and ADWIN watches the static model's error.
12. **Serving** (`backend/`). FastAPI loads each target's preprocessor, members, competence model, calibrator and conformal quantile. `/api/predict` returns the calibrated risk, the prediction set, the weights and the SHAP contributions for that input. SQLite logs predictions. React shows it.
13. **Reproduce** (`scripts/reproduce.py`). One command, seed 42, byte-identical summary files on rerun. CI runs the whole pipeline on synthetic data.

## 3. Why each modelling choice

- **ACWE (dynamic ensemble).** Different models can be competent in different regions of feature space (dynamic ensemble selection, e.g. KNORA / DES literature). Combining *local* competence with *recent* accuracy and *global* reliability lets the ensemble adapt per patient and over time. I name it explicitly as an integration of existing ideas, not a new algorithm. **Result:** no gain. The members are highly correlated (all ≈ 0.81–0.83 AUC on the same 18 features), so there is little complementary competence to exploit.
- **Isotonic calibration.** Risk tools need probabilities you can read as frequencies ("20% means 1 in 5"). I trained with `class_weight="balanced"` to help minority-class recall, but that inflates probabilities (raw ECE ≈ 0.21). Isotonic is non-parametric and monotone, so it keeps the ranking (AUC barely moves) and fixes the scale. With thousands of calibration rows it beats Platt scaling, which assumes a sigmoid shape. **Result:** ECE 0.21 → 0.009–0.016, Brier −39% to −52% on diabetes and CVD.
- **Split conformal prediction.** A distribution-free, finite-sample guarantee: P(y ∈ set) ≥ 1 − α if calibration and test data are exchangeable. It needs no model assumptions, which suits an ensemble. A two-element set {0, 1} is an honest "uncertain" flag. **Result:** marginal coverage holds (0.90–0.92), but per-class coverage for the rare classes is poor (see limitations).
- **Online learners + ADWIN.** The question was whether incremental learners recover faster than a frozen model after a distribution shift. River gives true `learn_one` models. ADWIN detects change in the error stream with a bound on false alarms. Scoring is prequential (test-then-train), so no label is used before its prediction. **Result:** no benefit over a majority-class predictor. BRFSS is cross-sectional, so the "stream" is simulated.
- **Classifier chain / co-occurrence prior.** The diseases are correlated (diabetes–hypertension r ≈ 0.27), so I tested whether feeding in predicted other diseases helps. It doesn't beat independent models. Using *true* other-disease labels as features would be leakage at prediction time.
- **LightGBM as the main tree model.** Strong default for tabular data, handles nonlinearity and interactions, fast, and exact TreeSHAP is available.

## 4. What the leakage / evaluation audit found (and fixed)

1. **ACWE reliability weights were test-set AUCs.** Test labels leaked into test predictions. Fixed: validation AUC.
2. **One validation set fitted competence, isotonic and conformal.** Conformal scores computed on the points the isotonic map was fit to are optimistic, so the coverage guarantee no longer formally held. Fixed: three disjoint thirds.
3. **The calibrator was fit on the soft-vote mean but applied to the ACWE output.** Fixed: each score has its own calibrator.
4. **The ablation was confounded.** The "full" model was calibrated and the ablated ones weren't, and two rows were copies of other rows. Fixed: each variant is calibrated; the fake rows are gone.
5. **The best baseline for significance tests was chosen on test.** Fixed: chosen on validation.
6. **A bug, not leakage:** the same model objects were re-fitted for each label, so the "diabetes" SHAP plot and the drift experiment's static model were actually the CVD models. The old diabetes SHAP ranking was identical to CVD's.
7. **Misleading claim:** "online beats static (0.83 vs 0.71)". Both should be compared with the majority-class baseline (0.84). The static model looked bad because of balanced weights at a 0.5 threshold.

**Did the numbers change?** Discrimination didn't (diabetes calibrated ACWE ROC-AUC 0.820 → 0.819). Calibrated ECE went from 0.008 to 0.016, because the calibrator now has a third of the data. That is a more honest number. The original README numbers reproduce exactly from the original code, so the change comes from the fixes, not from noise.

**What was already right:** the preprocessing is fit on train only (now tested), the labels and skip-pattern proxies are excluded, and the split is stratified on the joint label pattern. Each row is a unique respondent, so no person appears in two splits.

## 5. Limitations (say these before they're asked)

- Self-reported, **cross-sectional** labels: this models *prevalent* disease, not future risk. Some features are plausibly consequences of a diagnosis (`chol_check`, `gen_health`, `difficulty_walking`). That is reverse causation, not label leakage, but it inflates apparent "prediction".
- No survey weights (`_LLCPWT`), so metrics describe the sample, not the US population. Florida is missing. The numbers come from a 40k subsample.
- Conformal coverage is marginal. For CVD (8% prevalence) the sets are almost always {0} and positive coverage is 5%. **Fix:** Mondrian (class-conditional) conformal.
- Isotonic is a step function: it ties scores, which slightly lowers PR-AUC, and it needs enough calibration data per region.
- Drift is simulated, and TabNet falls back to an MLP without `pytorch-tabnet`.
- No external validation, e.g. on BRFSS 2022 or another cohort.

## 6. Ten likely questions, short answers

1. **Why ROC-AUC *and* PR-AUC?** ROC-AUC ignores prevalence. PR-AUC shows how hard the rare classes are (CVD PR-AUC 0.27 vs prevalence 0.08; hypertension 0.70 vs 0.39). Report both, plus Brier/ECE for probability quality.
2. **AUC barely changed after calibration, so why bother?** Calibration is monotone, so ranking (AUC) is preserved. What changes is whether "30%" means 30%. For a risk tool that is the property users act on. Brier fell 39–52% on diabetes and CVD.
3. **How do you know there's no leakage?** An automated feature audit, preprocessing fit on train (with a unit test), disjoint holdout thirds (tested), prequential updates (tested: flipping row i's label leaves predictions ≤ i unchanged), and nothing after training fitted on test. I found and fixed four places where that wasn't true.
4. **How did you handle class imbalance?** A stratified split on the joint label pattern, `class_weight="balanced"` for training, then isotonic calibration to undo the probability distortion. The metrics are threshold-free (AUC, PR-AUC, Brier); accuracy is reported only against a majority-class baseline.
5. **What is conformal prediction guaranteeing here?** Marginal coverage ≥ 90% under exchangeability. It does *not* guarantee per-class coverage, and with 8% prevalence the cheapest way to reach 90% is always predicting {0}. That is exactly what happened for CVD. Mondrian conformal fixes it.
6. **Why didn't the dynamic ensemble help?** Its members are highly correlated and trained on the same 18 features, so local competence barely differs between them. Dynamic selection helps when members are diverse and good in different regions. Reporting a null result is better than overselling it.
7. **Why isotonic over Platt?** Isotonic is non-parametric and handles the non-sigmoid distortion from class weighting. It needs more data, which we have (2,667 rows). Platt is the safer choice for small calibration sets.
8. **How is it reproducible?** One command, a fixed seed in the config, deterministic splits, seeded models, summary values rounded so reruns are byte-identical, and CI running the whole pipeline on synthetic data. The data source and zip SHA-256 are documented.
9. **What would you do next?** Mondrian conformal; use survey weights; temporal validation (train 2021, test 2022 BRFSS) for real drift; restrict to features measured before diagnosis; drop the ensemble in favour of calibrated LightGBM as the simpler model of equal quality.
10. **What does SHAP tell you, and what doesn't it?** Age and general health rank top-2 for all three targets, followed by BMI and high cholesterol, which is clinically plausible. But SHAP explains *the model's* association, not causation. `chol_check=2` raising diabetes risk reflects monitoring after diagnosis, not a risk factor.
