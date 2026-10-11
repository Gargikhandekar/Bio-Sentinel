from fastapi import APIRouter

from backend.services.inference import load_results

router = APIRouter(tags=["models"])


@router.get("/api/models")
def list_models():
    results = load_results()
    if not results:
        return {"status": "Awaiting training run", "models": []}
    rows = []
    for label, models in results.get("metrics", {}).items():
        for name, metrics in models.items():
            rows.append({"id": f"{label}:{name}", "label": label, "name": name, **{k: metrics.get(k) for k in ["accuracy", "f1", "roc_auc", "pr_auc", "brier", "ece"]}})
    return {"models": rows}


@router.get("/api/models/{model_id}")
def get_model(model_id: str):
    results = load_results()
    if not results:
        return {"status": "Awaiting training run"}
    label, name = model_id.split(":", 1)
    metrics = results["metrics"].get(label, {}).get(name)
    if metrics is None:
        return {"status": "unknown model"}
    return {"id": model_id, "metrics": metrics}
