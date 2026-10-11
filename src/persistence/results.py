"""Helpers to persist experiment JSON into SQLite."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.persistence.models import (
    DatasetRecord,
    DriftEventRecord,
    ExperimentRecord,
    ExperimentResultRecord,
    ModelRecord,
    make_session_factory,
)


def persist_experiment_payload(db_url: str, payload: dict[str, Any]) -> None:
    Session = make_session_factory(db_url)
    with Session() as session:
        session.add(
            DatasetRecord(
                name=payload["dataset"]["name"],
                version=str(payload["dataset"]["version"]),
                description=payload["dataset"].get("florida_note"),
            )
        )
        experiment = ExperimentRecord(
            name="full_research_matrix",
            type="offline",
            status=payload.get("status", "completed"),
            configuration=payload.get("split_sizes"),
            completed_at=datetime.now(timezone.utc),
        )
        session.add(experiment)
        session.flush()
        for label, models in payload.get("metrics", {}).items():
            for model_name, metrics in models.items():
                session.add(
                    ModelRecord(
                        name=f"{label}:{model_name}",
                        type=model_name,
                        training_status="trained",
                        metrics=metrics,
                    )
                )
                for metric_name in ["roc_auc", "f1", "brier", "ece"]:
                    if metric_name in metrics and metrics[metric_name] is not None:
                        session.add(
                            ExperimentResultRecord(
                                experiment_id=experiment.id,
                                model=f"{label}:{model_name}",
                                metric=metric_name,
                                value=metrics[metric_name],
                            )
                        )
        for event in payload.get("online", {}).get("drift_events", []):
            session.add(
                DriftEventRecord(
                    detector=event.get("detector", "ADWIN"),
                    stream_position=int(event.get("stream_position", 0)),
                    affected_features="bmi,gen_health",
                    severity=event.get("severity", "detected"),
                )
            )
        session.commit()
