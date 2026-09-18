"""AgentEvent: the single boundary between the agent runtime and the CLI.

The UI consumes these events and never touches LangGraph, MCP, the model SDK or a
subprocess itself; the runtime emits them and never renders anything.  See
``CLI_Design.md`` sections 24 and 32.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class TurnStarted:
    prompt: str = ""


@dataclass
class AssistantStarted:
    """One assistant message begins; ``message_id`` identifies its cell."""

    iteration: int = 0
    message_id: str = ""


@dataclass
class AssistantDelta:
    """A raw Markdown fragment.  Never re-render a delta on its own."""

    text: str = ""
    message_id: str = ""


@dataclass
class AssistantFinished:
    """The message identified by ``message_id`` is complete; never a new one."""

    text: str = ""
    reasoning: str | None = None
    message_id: str = ""


@dataclass
class ToolStarted:
    call_id: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutput:
    call_id: str
    text: str


@dataclass
class ToolFinished:
    call_id: str
    tool: str = ""
    ok: bool = True
    text: str = ""
    duration_ms: int = 0


@dataclass
class ToolFailed:
    call_id: str
    tool: str = ""
    error: str = ""


@dataclass
class SkillLoaded:
    name: str
    ok: bool = True


@dataclass
class SubAgentStarted:
    agent: str
    task: str = ""


@dataclass
class SubAgentOutput:
    agent: str
    text: str = ""


@dataclass
class SubAgentFinished:
    agent: str
    ok: bool = True
    summary: str = ""
    iterations: int = 0


@dataclass
class ContextUsage:
    """Bookkeeping event: token budget after the context was built."""

    tokens: int = 0
    budget: int = 0
    should_compact: bool = False


@dataclass
class Compacted:
    folded: int = 0


@dataclass
class ApprovalRequested:
    """A permission question is open (the question itself lives in the provider)."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionDecided:
    """The permission layer reached a verdict for one tool call."""

    call_id: str = ""
    tool: str = ""
    permission: str = ""
    reason: str = ""
    source: str = ""
    approval: str | None = None

    @property
    def allowed(self) -> bool:
        return self.permission == "allow"


@dataclass
class ErrorEvent:
    message: str
    fatal: bool = False


@dataclass
class TurnFinished:
    status: str = ""
    final_answer: str = ""
    iterations: int = 0


AgentEvent = Union[
    TurnStarted,
    AssistantStarted,
    AssistantDelta,
    AssistantFinished,
    ToolStarted,
    ToolOutput,
    ToolFinished,
    ToolFailed,
    SkillLoaded,
    SubAgentStarted,
    SubAgentOutput,
    SubAgentFinished,
    ContextUsage,
    Compacted,
    ApprovalRequested,
    PermissionDecided,
    ErrorEvent,
    TurnFinished,
]

EVENT_TYPES: tuple[type, ...] = (
    TurnStarted,
    AssistantStarted,
    AssistantDelta,
    AssistantFinished,
    ToolStarted,
    ToolOutput,
    ToolFinished,
    ToolFailed,
    SkillLoaded,
    SubAgentStarted,
    SubAgentOutput,
    SubAgentFinished,
    ContextUsage,
    Compacted,
    ApprovalRequested,
    ErrorEvent,
    TurnFinished,
)


def event_name(event: object) -> str:
    return type(event).__name__


__all__ = [name for name in globals() if not name.startswith("_")]
