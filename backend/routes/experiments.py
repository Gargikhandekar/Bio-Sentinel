from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.schemas.common import ExperimentRunRequest
from backend.services.inference import load_results

router = APIRouter(tags=["experiments"])


def _run():
    from src.evaluation.experiments import run_all_experiments
    from src.persistence.results import persist_experiment_payload
    from src.config import load_config

    payload = run_all_experiments()
    persist_experiment_payload(load_config()["database"]["url"], payload)


@router.get("/api/experiments")
def list_experiments():
    results = load_results()
    if not results:
        return {"status": "Experiment not yet executed", "experiments": []}
    return {
        "status": results.get("status"),
        "runtime_sec": results.get("runtime_sec"),
        "split_sizes": results.get("split_sizes"),
        "experiments": [
            "baseline_comparison",
            "modern_tabular",
            "static_vs_online",
            "drift_robustness",
            "dynamic_ensemble",
            "ablation",
        ],
    }


@router.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    results = load_results()
    if not results:
        return {"status": "Experiment not yet executed"}
    return {"id": experiment_id, "metrics": results.get("metrics"), "online": results.get("online")}


@router.post("/api/experiments/run")
def run_experiment(payload: ExperimentRunRequest, background: BackgroundTasks):
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to start a training run.")
    background.add_task(_run)
    return {"status": "started", "message": "Experiment running in the background."}
