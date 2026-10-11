# Leakage audit

## Passed checks

- Label columns are not in `FEATURE_COLUMNS`.
- Skip-pattern variables BPMEDS (asked only if hypertensive) and DIABAGE3 (asked only if diabetic) are excluded. Inspection: BPMEDS missingness 0.608, DIABAGE3 missingness 0.869.
- Stroke (CVDSTRK3) excluded as a cardiovascular event adjacent to the CVD label.
- Preprocessor is fitted on training data only (tested: imputer medians equal training medians).
- The validation holdout is split into three disjoint label-stratified thirds: ACWE competence and
  reliability weights, isotonic calibration, conformal quantile. Nothing downstream of training is
  fitted on test data (tested).
- ACWE recency is prequential: row *i* is scored before its label updates the recency state (tested).

## Findings from the October 2026 audit (fixed)

| Issue | Effect | Fix |
| --- | --- | --- |
| ACWE reliability weights were the members' **test** ROC-AUC | test labels influenced test predictions | weights from the competence third of validation |
| One validation set fitted competence, isotonic and conformal | conformal scores computed on points the calibrator was fitted to, so the coverage guarantee no longer held | three disjoint thirds |
| Calibrator fitted on soft-vote output, applied to ACWE output | calibrating a different score | each variant has its own calibrator |
| Best baseline for paired tests chosen on test | selection on test | chosen on validation |
| Estimator objects reused across labels | "diabetes" SHAP and the drift static reference were the CVD models | fresh instances per label |

Effect on reported diabetes numbers: ROC-AUC unchanged (0.820 → 0.819); calibrated ECE 0.008 → 0.016
because the calibrator now gets a third of the holdout instead of all of it.

## Remaining threats

- Self-report of cholesterol may still correlate with CVD care-seeking.
- General health and unhealthy-days can be downstream of disease (reverse causation). They are retained as health-status covariates and discussed as a limitation.
- The sequential stream is a permutation of a cross-section plus synthetic shift, not a time series.
