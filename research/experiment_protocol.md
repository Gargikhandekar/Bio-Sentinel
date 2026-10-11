# Experiment protocol

1. Build processed parquet from LLCP2021.XPT.
2. Leakage audit on the configured feature list.
3. Stratified subsample (laptop) then 60/20/20 train/val/test by 3-bit label pattern; validation split into competence / calibration / conformal thirds.
4. Fit ColumnTransformer on training features only.
5. Experiment 1: baselines per label.
6. Experiment 2: modern tabular model per label.
7. Experiment 3: classifier chain vs independent models.
8. Experiment 4: static vs online prequential accuracy on a constructed stream.
9. Experiment 5: inject synthetic shift; record ADWIN events and rolling accuracy.
10. Experiment 6 (every target): ACWE vs members, soft vote, global weights, LightGBM + isotonic; each variant calibrated on the calibration third; conformal coverage measured on test.
11. Ablation of ACWE components, each variant with its own calibrator.
12. Bootstrap CI on ACWE AUC; McNemar and paired bootstrap AUC vs the best baseline chosen on validation.
13. Persist JSON + SQLite; `scripts/reproduce.py` writes `results/metrics/summary.*` and figures.

Do not evaluate calibration or feature selection on the test set.
Do not update online models before the current prediction.
