from src.evaluation.experiments import run_all_experiments
from src.config import load_config

if __name__ == "__main__":
    payload = run_all_experiments()
    try:
        from src.persistence.results import persist_experiment_payload

        persist_experiment_payload(load_config()["database"]["url"], payload)
    except Exception as exc:
        print("sqlite persist skipped:", exc)
    print("status", payload["status"], "runtime_sec", payload["runtime_sec"])
