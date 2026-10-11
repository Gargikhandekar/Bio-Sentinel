# Experiment protocol

1. Build processed parquet from LLCP2021.XPT.
2. Leakage audit on the configured feature list.
3. Stratified subsample (laptop) then 60/20/20 train/val/test by 3-bit label pattern.
4. Fit ColumnTransformer on training features only.
5. Experiment 1: baselines per label.
6. Experiment 2: modern tabular model per label.
7. Experiment 3: classifier chain vs independent models.
8. Experiment 4: static vs online prequential accuracy on a constructed stream.
9. Experiment 5: inject synthetic shift; record ADWIN events and rolling accuracy.
10. Experiment 6: ACWE vs best member, soft vote, global weights; isotonic calibration; conformal sets.
11. Ablation of ACWE components.
12. Bootstrap CI on ACWE AUC; McNemar vs best baseline; paired AUC difference.
13. Persist JSON + SQLite.

Do not evaluate calibration or feature selection on the test set.
Do not update online models before the current prediction.
