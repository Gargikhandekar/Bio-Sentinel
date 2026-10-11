# Research gap

## Claimed contribution

BIO-SENTINEL does not introduce a new optimizer, tree, or neural architecture. The contribution is a **leakage-safe, disease-dependency-aware experimental framework** that:

1. predicts three co-occurring self-reported chronic conditions from a shared BRFSS 2021 feature set that **excludes** the labels and their skip-pattern proxies,
2. compares independent classifiers, a classifier chain, static boosting, incremental learners, and a context-weighted ensemble (ACWE) built from existing DES/competence ideas,
3. evaluates probability calibration and split conformal prediction sets,
4. stresses the system with **documented synthetic** feature and label-noise shifts on a constructed stream, not with claimed clinical drift.

## Gap relative to prior work

- Dynamic ensemble selection and online drift methods exist, but are rarely evaluated together on multi-label BRFSS targets with an explicit leakage audit.
- BRFSS diabetes papers often keep hypertension and CVD as features, which is invalid if those conditions are also outcomes.
- Calibration and conformal prediction are usually omitted from chronic-disease dashboard demos.

## What would falsify the contribution

## Empirical snapshot (BRFSS 2021 laptop subsample)

Held-out diabetes predicted-risk metrics from `results/experiments/latest.json` (not fabricated):

- LightGBM ROC-AUC 0.817, Brier 0.165, ECE 0.209
- ACWE raw ROC-AUC 0.821, Brier 0.157, ECE 0.212
- ACWE + isotonic (validation-only) ROC-AUC 0.820, Brier 0.095, ECE 0.008
- Soft vote ROC-AUC 0.821 (discrimination essentially tied with ACWE)
- Dependency-aware blend ROC-AUC 0.818
- Online logreg / PA prequential accuracy 0.826 / 0.838 vs static reference 0.713 on the **simulated** stream

Interpretation: on this extract, calibration (RQ3/RQ6) is the component with a clear reliability gain. Dynamic weighting does not show a large AUROC advantage over soft voting. That is a valid research outcome and must not be oversold.

