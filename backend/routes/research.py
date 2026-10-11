from fastapi import APIRouter

from backend.services.inference import load_results

router = APIRouter(tags=["research"])


@router.get("/api/research/metrics")
def research_metrics():
    results = load_results()
    if not results:
        return {"status": "Experiment not yet executed"}
    return {
        "metrics": results.get("metrics"),
        "statistics": results.get("statistics"),
        "split_sizes": results.get("split_sizes"),
        "explainability": results.get("explainability"),
        "permutation_importance": results.get("permutation_importance"),
        "feature_space": results.get("feature_space"),
        "dataset": results.get("dataset"),
        "disclaimer": results.get("disclaimer"),
    }


@router.get("/api/research/ablation")
def research_ablation():
    results = load_results()
    if not results:
        return {"status": "Experiment not yet executed", "ablation": []}
    return {"ablation": results.get("ablation", [])}


@router.get("/api/research/calibration")
def research_calibration():
    results = load_results()
    if not results:
        return {"status": "Experiment not yet executed"}
    return results.get("calibration", {})
