"""Run the full experiment matrix and persist the payload to SQLite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.evaluation.experiments import run_all_experiments  # noqa: E402

if __name__ == "__main__":
    payload = run_all_experiments()
    try:
        from src.persistence.results import persist_experiment_payload

        persist_experiment_payload(load_config()["database"]["url"], payload)
    except Exception as exc:
        print("sqlite persist skipped:", exc)
    print("status", payload["status"], "runtime_sec", payload["runtime_sec"])
