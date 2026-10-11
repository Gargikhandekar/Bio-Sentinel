from fastapi import APIRouter, HTTPException

from backend.schemas.common import PredictRequest
from backend.services.inference import DISCLAIMER, get_engine
from src.persistence.models import PredictionRecord, make_session_factory
from src.config import load_config

router = APIRouter(tags=["prediction"])


@router.post("/api/predict")
def predict(payload: PredictRequest):
    engine = get_engine()
    try:
        result = engine.predict(payload.features.model_dump())
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Unable to generate prediction. The model service is temporarily unavailable. Please retry.",
        ) from exc
    result["disclaimer"] = DISCLAIMER
    try:
        Session = make_session_factory(load_config()["database"]["url"])
        with Session() as session:
            session.add(
                PredictionRecord(
                    dataset="brfss_2021",
                    model="acwe",
                    prediction=result["prediction"],
                    probability=result["raw_probability"],
                    calibrated_probability=result["calibrated_probability"],
                    uncertainty=result["uncertainty"],
                )
            )
            session.commit()
    except Exception:
        pass
    return result
