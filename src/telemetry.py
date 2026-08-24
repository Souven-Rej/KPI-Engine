from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TelemetryEvent:
    """Structured event for runtime observability."""
    event_name: str
    stage: str
    latency_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_name": self.event_name,
            "stage": self.stage,
            "latency_seconds": round(self.latency_seconds, 4),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class RuntimeTelemetry:
    """Simple in-memory and file-based telemetry logger for signature/latency capture."""

    def __init__(self, log_dir: str | Path | None = None) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else Path("data")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[TelemetryEvent] = []

    def record(self, event_name: str, stage: str, latency_seconds: float = 0.0, **metadata: Any) -> TelemetryEvent:
        event = TelemetryEvent(
            event_name=event_name,
            stage=stage,
            latency_seconds=latency_seconds,
            metadata=dict(metadata),
        )
        self.events.append(event)
        try:
            log_path = self.log_dir / "runtime_telemetry.jsonl"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict()) + "\n")
        except Exception:
            pass
        return event

    def summary(self) -> dict[str, Any]:
        if not self.events:
            return {"total_events": 0, "avg_latency_seconds": 0.0}
        latencies = [event.latency_seconds for event in self.events]
        return {
            "total_events": len(self.events),
            "avg_latency_seconds": round(sum(latencies) / len(latencies), 4),
            "max_latency_seconds": round(max(latencies), 4),
        }
