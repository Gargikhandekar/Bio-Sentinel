from fastapi import APIRouter, HTTPException

from backend.schemas.common import PredictRequest
from backend.services.inference import get_engine, load_results

router = APIRouter(tags=["explainability"])


@router.post("/api/explain")
def explain(payload: PredictRequest):
    engine = get_engine()
    try:
        result = engine.predict(payload.features.model_dump())
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Unable to generate prediction. The model service is temporarily unavailable. Please retry.",
        ) from exc
    results = load_results() or {}
    return {
        "header": "WHY DID THE SYSTEM DECIDE THIS?",
        "disclaimer": "Model contribution does not imply causation. Model contribution ≠ causal medical effect.",
        "local": result["explanation"].get("local", []),
        "global": (results.get("explainability") or {}).get("global", []),
        "feature_space": results.get("feature_space"),
        "prediction": result,
    }
