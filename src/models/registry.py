"""Simple in-memory model registry for trained artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelRecord:
    name: str
    kind: str
    label: str
    artifact_path: str | None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


class ModelRegistry:
    def __init__(self) -> None:
        self._items: dict[str, ModelRecord] = {}

    def add(self, record: ModelRecord) -> None:
        self._items[f"{record.label}:{record.name}"] = record

    def get(self, key: str) -> ModelRecord:
        return self._items[key]

    def all(self) -> list[ModelRecord]:
        return list(self._items.values())
