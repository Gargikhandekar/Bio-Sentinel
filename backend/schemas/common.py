"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str]


class FeatureVector(BaseModel):
    age_group: Optional[float] = Field(None, description="BRFSS _AGEG5YR 1–13 (18–24 … 80+)")
    sex: Optional[float] = Field(None, description="1 = male, 2 = female")
    race_group: Optional[float] = None
    education: Optional[float] = None
    income: Optional[float] = None
    bmi: Optional[float] = Field(None, ge=10, le=80)
    gen_health: Optional[float] = Field(None, description="1 excellent … 5 poor")
    phys_unhealthy_days: Optional[float] = Field(None, ge=0, le=30)
    ment_unhealthy_days: Optional[float] = Field(None, ge=0, le=30)
    smoker: Optional[float] = None
    phys_activity: Optional[float] = None
    fruits_daily: Optional[float] = None
    veggies_daily: Optional[float] = None
    difficulty_walking: Optional[float] = None
    healthcare_coverage: Optional[float] = None
    binge_drinking: Optional[float] = None
    high_cholesterol: Optional[float] = None
    chol_check: Optional[float] = None


Target = Literal["diabetes", "hypertension", "cvd"]


class PredictRequest(BaseModel):
    features: FeatureVector
    label: Target = "diabetes"


class PredictResponse(BaseModel):
    label: Target
    prediction: int
    raw_probability: float
    calibrated_probability: float
    uncertainty: str
    prediction_set: list[int]
    model_predictions: dict[str, float]
    model_weights: dict[str, float]
    explanation: dict[str, Any]
    disclaimer: str


class ExperimentRunRequest(BaseModel):
    confirm: bool = False
