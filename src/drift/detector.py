"""Concept-drift detectors used on prequential error streams."""

from __future__ import annotations

from dataclasses import dataclass

from river.drift import ADWIN, PageHinkley


@dataclass
class DriftEvent:
    detector: str
    stream_position: int
    severity: str
    note: str


class DriftMonitor:
    def __init__(self, method: str = "ADWIN", delta: float = 0.002) -> None:
        self.method = method.upper()
        if self.method == "PAGE-HINKLEY":
            self.detector = PageHinkley()
        else:
            self.detector = ADWIN(delta=delta)
            self.method = "ADWIN"
        self.events: list[DriftEvent] = []

    def update(self, error: float, position: int) -> bool:
        self.detector.update(error)
        drifted = bool(getattr(self.detector, "drift_detected", False))
        if drifted:
            self.events.append(
                DriftEvent(
                    detector=self.method,
                    stream_position=position,
                    severity="detected",
                    note="Error-rate change on a simulated stream, not clinical drift.",
                )
            )
        return drifted
