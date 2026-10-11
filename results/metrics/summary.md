| Target | Model | ROC-AUC | PR-AUC | Brier | ECE |
| --- | --- | --- | --- | --- | --- |
| diabetes | LightGBM | 0.817 | 0.377 | 0.165 | 0.209 |
| diabetes | LightGBM + isotonic | 0.815 | 0.360 | 0.097 | 0.015 |
| diabetes | Soft vote | 0.821 | 0.388 | 0.159 | 0.214 |
| diabetes | ACWE raw | 0.821 | 0.387 | 0.157 | 0.212 |
| diabetes | ACWE + isotonic | 0.819 | 0.362 | 0.096 | 0.016 |
| hypertension | LightGBM | 0.806 | 0.711 | 0.180 | 0.068 |
| hypertension | LightGBM + isotonic | 0.803 | 0.691 | 0.175 | 0.020 |
| hypertension | Soft vote | 0.808 | 0.713 | 0.178 | 0.069 |
| hypertension | ACWE raw | 0.808 | 0.713 | 0.178 | 0.069 |
| hypertension | ACWE + isotonic | 0.806 | 0.696 | 0.174 | 0.016 |
| cvd | LightGBM | 0.820 | 0.273 | 0.145 | 0.213 |
| cvd | LightGBM + isotonic | 0.815 | 0.248 | 0.067 | 0.007 |
| cvd | Soft vote | 0.832 | 0.287 | 0.141 | 0.224 |
| cvd | ACWE raw | 0.831 | 0.286 | 0.137 | 0.219 |
| cvd | ACWE + isotonic | 0.826 | 0.268 | 0.066 | 0.009 |

| Target | Prevalence | Conformal target | Coverage | Coverage (positives) | Coverage (negatives) | Avg set size |
| --- | --- | --- | --- | --- | --- | --- |
| diabetes | 0.131 | 0.90 | 0.924 | 0.433 | 0.997 | 1.14 |
| hypertension | 0.392 | 0.90 | 0.899 | 0.828 | 0.945 | 1.39 |
| cvd | 0.081 | 0.90 | 0.918 | 0.048 | 0.995 | 1.00 |
