# Methodology

## Targets

Self-reported diagnosed conditions in CDC BRFSS 2021. These are not incident laboratory diagnoses.

Complete-case n after requiring all three labels: 431,843 / 438,693.

Observed complete-case prevalence:

- diabetes 0.1306
- hypertension 0.3917
- cvd 0.0813

Pairwise Pearson correlations: diabetes–hypertension 0.27, hypertension–cvd 0.21, diabetes–cvd 0.18.

## Features

Shared representation: age group, sex, race group, education, income, BMI, general health, unhealthy days, smoking, activity, fruit/vegetable intake, walking difficulty, coverage, binge drinking, cholesterol awareness/check.

Excluded: DIABETE4, DIABAGE3, BPHIGH6, BPMEDS, _RFHYPE6, CVDINFR4, CVDCRHD4, _MICHD, CVDSTRK3.

## Split

IID stratified train / validation / test. All transformers fitted on training data only. Calibration and conformal quantiles fitted on validation predictions only.

Laptop profile uses a stratified subsample (`training.max_rows`) for wall-clock feasibility. The processed parquet retains the full complete-case table.

## Models

Baselines: logistic regression, random forest, linear SVM with Platt calibration, LightGBM.

Modern tabular: TabNet when PyTorch is present; otherwise a documented MLP substitute.

Dependency: classifier chain with order hypertension → diabetes → CVD. This is a predictive dependence device, not a causal graph.

Online: River logistic regression, PA, Hoeffding tree; prequential predict-then-update.

Ensemble: ACWE
`score_i(x) = local_competence_i(x) × recent_performance_i × reliability_i`
weights normalized to a convex combination of member probabilities.

Drift: ADWIN on the error stream. Synthetic numerical shift and label-noise after the stream midpoint. This is not clinical concept drift.

Uncertainty: split conformal over {0,1} at level 1 − α.
