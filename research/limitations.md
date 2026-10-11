# Limitations and threats to validity

- BRFSS outcomes are self-reported diagnosed conditions, not clinically adjudicated incident disease.
- Florida is missing from the 2021 national public-use file.
- Cross-sectional survey: no true longitudinal drift. Simulated shift is an experimental intervention.
- Laptop compute profile trains on a stratified subsample; point estimates can differ from a full-file run.
- TabNet may be substituted by an MLP if PyTorch is unavailable.
- Classifier-chain order is a modeling choice, not a causal claim.
- Survey weights (`_LLCPWT`) are not used in the learning objective; estimates are not population-weighted prevalence models.
- Split conformal coverage is marginal. With 8–13% prevalence the sets are almost always {0}; positive-class coverage is 0.05 (CVD) and 0.43 (diabetes). Mondrian (class-conditional) conformal would address this.
- Some predictors (cholesterol check, general health, walking difficulty) are plausibly downstream of a diagnosis; the models learn association with prevalent disease, not incident risk.
- `class_weight="balanced"` distorts raw probabilities; only calibrated outputs should be read as risk.
- This software is a research prototype and is not a medical device.
