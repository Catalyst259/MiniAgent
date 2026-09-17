"""Lightweight event objects emitted by the orchestrator.

The CLI renders these; nothing else consumes them.  Keeping them as plain
Pydantic models avoids a "complex event bus" while still giving the terminal a
clean separation between *running* the agent and *showing* it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "task_start",
    "iteration",
    "context",
    "token_guard",
    "compact",
    "assistant_text",
    "assistant_message",
    "assistant_reasoning",
    "tool_call",
    "tool_start",
    "tool_result",
    "skill_load",
    "delegate_start",
    "delegate_end",
    "memory_write",
    "memory_recall",
    "terminate",
    "error",
]


class Event(BaseModel):
    type: EventType
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"[{self.type}] {self.message}"


__all__ = ["Event", "EventType"]
