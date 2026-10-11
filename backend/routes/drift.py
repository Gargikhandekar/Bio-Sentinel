from fastapi import APIRouter

from backend.services.inference import load_results

router = APIRouter(tags=["drift"])


@router.get("/api/drift")
def drift_overview():
    results = load_results()
    if not results:
        return {"status": "Experiment not yet executed"}
    return results.get("online", {})


@router.get("/api/drift/events")
def drift_events():
    results = load_results()
    if not results:
        return {"events": [], "status": "Experiment not yet executed"}
    return {"events": results.get("online", {}).get("drift_events", [])}
