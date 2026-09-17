"""Harness exception hierarchy."""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for every error raised by the harness."""


class ConfigError(HarnessError):
    """Invalid or missing configuration."""


class ModelError(HarnessError):
    """The model gateway could not produce a normalized response."""


class ToolError(HarnessError):
    """A tool failed in a way the runtime must surface to the model."""


class ToolNotFound(ToolError):
    """The model asked for a tool that is not registered."""


class ToolPermissionError(ToolError):
    """The tool exists but is not allowed in the current context."""


class PatchError(ToolError):
    """An ``apply_patch`` payload could not be parsed or applied."""


class SkillNotFound(HarnessError):
    """A requested skill does not exist on disk."""


class SubAgentNotFound(HarnessError):
    """A requested subagent does not exist on disk."""


class TerminationSignal(HarnessError):
    """Raised internally to stop the agent loop early."""

    def __init__(self, reason: str, final_answer: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.final_answer = final_answer


__all__ = [
    "HarnessError",
    "ConfigError",
    "ModelError",
    "ToolError",
    "ToolNotFound",
    "ToolPermissionError",
    "PatchError",
    "SkillNotFound",
    "SubAgentNotFound",
    "TerminationSignal",
]
