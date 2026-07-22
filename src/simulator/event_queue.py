from __future__ import annotations

import heapq
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    WORKFLOW_ARRIVAL = "workflow_arrival"
    TASK_FINISH = "task_finish"


class SimEvent:
    def __init__(
        self,
        time: float,
        sequence: int,
        event_type: EventType,
        payload: Optional[dict[str, Any]] = None,
    ):
        self.time = time
        self.sequence = sequence
        self.event_type = event_type
        self.payload = payload or {}

    def __lt__(self, other: "SimEvent") -> bool:
        return (self.time, self.sequence) < (other.time, other.sequence)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimEvent):
            return False
        return (self.time, self.sequence) == (other.time, other.sequence)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[SimEvent] = []
        self._sequence = 0

    def push(self, time: float, event_type: EventType, **payload: Any) -> None:
        heapq.heappush(
            self._heap,
            SimEvent(float(time), self._sequence, event_type, dict(payload)),
        )
        self._sequence += 1

    def peek(self) -> Optional[SimEvent]:
        return self._heap[0] if self._heap else None

    def pop(self) -> Optional[SimEvent]:
        return heapq.heappop(self._heap) if self._heap else None

    def empty(self) -> bool:
        return not self._heap

    def __len__(self) -> int:
        return len(self._heap)
