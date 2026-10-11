# Research questions

- RQ1: Does adaptive online learning improve robustness under simulated distribution shift compared with a frozen static reference model?
- RQ2: Does disease-dependency-aware, context-aware dynamic model selection outperform a fixed ensemble, soft voting, and strong individual models?
- RQ3: Does probability calibration improve reliability of predicted probabilities (Brier, ECE, reliability diagrams)?
- RQ4: Can split conformal prediction sets flag observations whose labels are not uniquely covered at the target error rate?
- RQ5: Which ACWE components (dynamic weights, local competence, online recency, calibration) change predictive metrics in ablation?
- RQ6: Does the adaptive framework improve calibration and uncertainty quality, not only discrimination (F1 / AUROC)?

Primary labels (binary, same respondent):

- diabetes: DIABETE4 = 1 vs {2,3,4}; 7/9 dropped
- hypertension: BPHIGH6 = 1 vs {2,3,4}; 7/9 dropped
- cvd: _MICHD = 1 vs 2
