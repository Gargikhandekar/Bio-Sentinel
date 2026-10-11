# Literature review

This review distinguishes existing methods from the BIO-SENTINEL integration. It does not claim a new learning algorithm.

## Data source

Centers for Disease Control and Prevention. Behavioral Risk Factor Surveillance System Survey Data, 2021. Atlanta, Georgia: U.S. Department of Health and Human Services, CDC.
Official documentation: https://www.cdc.gov/brfss/annual_data/annual_2021.html
Public-use file: 438,693 records, 304 variables, 49 states + DC and participating territories. Florida is missing from the 2021 national file.

## Tabular prediction

- Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. AAAI.
- Chen, T., & Guestrin, C. (2016). XGBoost. KDD.
- Ke, G., et al. (2017). LightGBM. NeurIPS.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS.

## Multi-label learning

- Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier Chains for Multi-label Classification. Machine Learning.

## Dynamic ensembles and competence

- Ko, A. H. R., Sabourin, R., & Britto Jr, A. S. (2008). From Dynamic Classifier Selection to Dynamic Ensemble Selection (KNORA).
- Cruz, R. M. O., Sabourin, R., & Cavalcanti, G. D. C. META-DES and later dynamic ensemble selection surveys.
- Kolter, J. Z., & Maloof, M. A. (2007). Dynamic Weighted Majority.

## Online learning and drift

- Bifet, A., & Gavaldà, R. (2007). Learning from Time-Changing Data with Adaptive Windowing (ADWIN).
- Gama, J., et al. Concept drift surveys and DDM/EDDM.
- Gomes, H. M., et al. Adaptive Random Forest for evolving data streams.
- River library: Montiel, J., et al. River: machine learning for streaming data in Python.

## Calibration and uncertainty

- Platt, J. (1999). Probabilistic outputs for support vector machines.
- Zadrozny, B., & Elkan, C. Transforming classifier scores into accurate multiclass probability estimates (isotonic).
- Niculescu-Mizil, A., & Caruana, R. Predicting good probabilities with supervised learning.
- Vovk, V., Gammerman, A., & Shafer, G. Algorithmic Learning in a Random World (conformal prediction).
- Angelopoulos, A. N., & Bates, S. A Gentle Introduction to Conformal Prediction.

## Related applied BRFSS ML papers

Many papers train static classifiers on BRFSS diabetes or heart-disease indicators. They typically:
- treat one disease as an isolated target,
- use comorbidity variables such as HighBP or HeartDiseaseorAttack as predictors,
- evaluate under an IID split,
- and do not combine online learning, local competence, calibration, and conformal uncertainty in one protocol.

## Closest existing methods (not claimed as invented here)

- Multi-label chronic disease prediction with classifier chains (Read et al., 2011) and many BRFSS single-disease papers.
- Dynamic ensemble selection / KNORA / META-DES (Ko et al.; Cruz et al.).
- Dynamic Weighted Majority and Learn++.NSE for drifting streams.
- ADWIN (Bifet) and related error-rate detectors.
- Isotonic/Platt calibration and split conformal prediction.

## How this project differs

BIO-SENTINEL combines a **leakage-safe three-label BRFSS 2021 extract** (hypertension and CVD are outcomes, not diabetes features) with:

1. independent vs classifier-chain vs co-occurrence-prior ensembles,
2. local-competence context weights (ACWE as an integration name only),
3. validation-only calibration + conformal sets,
4. a **documented synthetic** sequential shift protocol.

ACWE is not a new algorithm. If it does not beat soft voting and the best member on Brier/ECE as well as AUROC, that negative result is the scientific finding.

