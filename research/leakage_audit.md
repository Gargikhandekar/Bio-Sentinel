# Leakage audit

## Passed checks

- Label columns are not in `FEATURE_COLUMNS`.
- Skip-pattern variables BPMEDS (asked only if hypertensive) and DIABAGE3 (asked only if diabetic) are excluded. Inspection: BPMEDS missingness 0.608, DIABAGE3 missingness 0.869.
- Stroke (CVDSTRK3) excluded as a cardiovascular event adjacent to the CVD label.
- Preprocessor is fitted on training data only.
- Isotonic calibration is fitted on validation probabilities only.
- Conformal quantile is fitted on validation scores only.

## Remaining threats

- Self-report of cholesterol may still correlate with CVD care-seeking.
- General health and unhealthy-days can be downstream of disease (reverse causation). They are retained as health-status covariates and discussed as a limitation.
- The sequential stream is a permutation of a cross-section plus synthetic shift, not a time series.
