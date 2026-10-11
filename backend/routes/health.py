from fastapi import APIRouter

from backend.services.inference import get_engine

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health():
    engine = get_engine()
    services = engine.status()
    overall = "operational" if all(v == "OPERATIONAL" for v in services.values()) else "degraded"
    return {"status": overall, "services": services}
