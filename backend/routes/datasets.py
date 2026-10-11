from fastapi import APIRouter

from backend.services.inference import load_results
from src.config import load_config
from src.data.loader import load_processed, dataset_summary

router = APIRouter(tags=["datasets"])


@router.get("/api/datasets")
def list_datasets():
    cfg = load_config()
    return [
        {
            "id": "brfss_2021",
            "name": cfg["dataset"]["name"],
            "version": cfg["dataset"]["version"],
            "source": cfg["dataset"]["source"],
            "official_url": cfg["dataset"]["official_url"],
        }
    ]


@router.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    cfg = load_config()
    try:
        df = load_processed()
        summary = dataset_summary(df, cfg)
    except Exception as exc:
        summary = {"error": str(exc)}
    return {"id": dataset_id, **cfg["dataset"], "summary": summary}
