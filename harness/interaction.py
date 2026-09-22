"""User-interaction seam shared by the agent loop and its front ends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class Choice:
    """One stable value the model can offer to the user."""

    value: str
    label: str
    description: str = ""

    @classmethod
    def from_payload(cls, payload: Any, index: int) -> "Choice":
        if isinstance(payload, str):
            label = payload.strip()
            return cls(value=label or str(index + 1), label=label or f"Option {index + 1}")
        if not isinstance(payload, dict):
            raise ValueError(f"option {index + 1} must be a string or object")
        label = str(payload.get("label") or "").strip()
        if not label:
            raise ValueError(f"option {index + 1} needs a label")
        value = str(payload.get("value") or label).strip()
        description = str(payload.get("description") or "").strip()
        return cls(value=value, label=label, description=description)


class InteractionCancelled(RuntimeError):
    """The user explicitly dismissed a pending question."""


class InteractionUnavailable(RuntimeError):
    """The current front end has no way to ask a human."""


class InteractionProvider(Protocol):
    """The only interface the agent loop needs for human choices."""

    available: bool

    async def choose(
        self,
        question: str,
        options: Sequence[Choice],
        *,
        title: str = "Choose an option",
        detail: str = "",
    ) -> Choice: ...


class UnavailableInteractionProvider:
    """Fail-fast adapter for batch runs and isolated subagents."""

    available = False

    def __init__(self, reason: str = "no interactive user-input UI is attached") -> None:
        self.reason = reason

    async def choose(
        self,
        question: str,
        options: Sequence[Choice],
        *,
        title: str = "Choose an option",
        detail: str = "",
    ) -> Choice:
        raise InteractionUnavailable(self.reason)


def choices_from_payload(payload: Any) -> list[Choice]:
    """Validate and normalize the model-facing option list."""

    if not isinstance(payload, list):
        raise ValueError("options must be a list")
    choices = [Choice.from_payload(item, index) for index, item in enumerate(payload)]
    if not 2 <= len(choices) <= 4:
        raise ValueError("request_user_input requires between 2 and 4 options")
    values = [choice.value for choice in choices]
    if len(values) != len(set(values)):
        raise ValueError("option values must be unique")
    return choices


__all__ = [
    "Choice",
    "InteractionCancelled",
    "InteractionProvider",
    "InteractionUnavailable",
    "UnavailableInteractionProvider",
    "choices_from_payload",
]
