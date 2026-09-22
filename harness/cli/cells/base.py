"""Typed state records kept in CLI history.

Cells own lifecycle data only.  Their display meaning lives behind
``presentation.present_cell`` so every output adapter uses the same decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

from harness.cli.sanitize import safe_text


class ToolStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class BaseCell:
    """Common cell metadata."""

    title: str = ""
    preview_lines: int = 0

@dataclass
class UserCell(BaseCell):
    """The prompt the user submitted."""

    text: str = ""

@dataclass
class AssistantCell(BaseCell):
    """One assistant message: grown by deltas, completed by a final event.

    Identity is the cell itself, not the event: streaming deltas *update* this
    cell and the completion event only marks it done.  Appending a new cell per
    event is what produced the "answer rendered twice" bug.
    """

    source: str = ""
    reasoning: str | None = None
    message_id: str = ""
    complete: bool = False

    @property
    def expand_key(self) -> str:
        return reasoning_key(self)

    # ---------------------------------------------------------------- lifecycle
    def append_delta(self, delta: str) -> None:
        self.source += delta

    def complete_with(self, text: str | None = None) -> None:
        if text is not None:
            self.source = text
        self.complete = True

    @property
    def empty(self) -> bool:
        return not self.source.strip() and not (self.reasoning or "").strip()

@dataclass
class ToolCell(BaseCell):
    """A tool invocation; updates in place while it is the active cell."""

    call_id: str = ""
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    status: ToolStatus = ToolStatus.RUNNING
    duration_ms: int = 0
    error: str = ""
    max_preview_lines: int = 8

    # ------------------------------------------------------------------ mutation
    def append(self, text: str) -> None:
        self.output += text

    def finish(self, ok: bool, text: str = "", duration_ms: int = 0) -> None:
        if text:
            self.output = text
        self.status = ToolStatus.DONE if ok else ToolStatus.FAILED
        self.duration_ms = duration_ms

    def fail(self, error: str) -> None:
        self.error = error
        self.status = ToolStatus.FAILED

    def body_lines(self) -> list[str]:
        """Every sanitized output line (no display limit applied)."""

        body = safe_text(self.error or self.output).rstrip("\n")
        return body.splitlines() if body else []

    def preview_body(self, max_lines: int | None = None) -> tuple[list[str], int]:
        """Return the visible body slice and the number of hidden lines.

        ``max_lines=None`` means "use this cell's configured preview limit" (the
        rich/plain renderer's default); pass an explicit number to override it,
        including a large one for a truly expanded view.
        """

        lines = self.body_lines()
        if not lines:
            return [], 0
        limit = self.max_preview_lines if max_lines is None else max_lines
        if limit <= 0 or limit >= len(lines):
            return lines, 0
        return lines[:limit], len(lines) - limit

    # ------------------------------------------------------------------ rendering
    @property
    def glyph(self) -> str:
        # running and done must differ by *shape*, not only by colour
        return {
            ToolStatus.RUNNING: "◐",
            ToolStatus.DONE: "●",
            ToolStatus.FAILED: "✗",
        }[self.status]

@dataclass
class SubAgentCell(BaseCell):
    agent: str = ""
    task: str = ""
    summary: str = ""
    ok: bool = True
    iterations: int = 0
    max_preview_lines: int = 12
    status: ToolStatus = ToolStatus.RUNNING

    @property
    def expand_key(self) -> str:
        return f"subagent:{self.agent}:{id(self)}"

@dataclass
class ErrorCell(BaseCell):
    message: str = ""
    fatal: bool = False

@dataclass
class InfoCell(BaseCell):
    message: str = ""
    style: str = "dim"

@dataclass
class SkillCell(BaseCell):
    name: str = ""
    ok: bool = True

def format_arguments(arguments: dict[str, Any], limit: int = 110) -> str:
    """One-line JSON preview of a tool call's arguments."""

    import json

    try:
        rendered = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        rendered = str(arguments)
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


def reasoning_key(cell: AssistantCell) -> str:
    """Stable key for "expand this cell's chain of thought"."""

    return f"reasoning:{cell.message_id or id(cell)}"


def expand_key(cell: Any) -> str:
    """Key used by ``AppState.expanded_tool_ids`` for any expandable cell."""

    if isinstance(cell, ToolCell):
        return cell.call_id
    if isinstance(cell, AssistantCell):
        return reasoning_key(cell)
    if isinstance(cell, SubAgentCell):
        return cell.expand_key
    return f"{type(cell).__name__}:{id(cell)}"


#: backwards-compatible private alias (older call sites)
_format_arguments = format_arguments


HistoryCell: TypeAlias = (
    UserCell | AssistantCell | ToolCell | SubAgentCell | ErrorCell | InfoCell | SkillCell
)


__all__ = [
    "ToolStatus",
    "HistoryCell",
    "BaseCell",
    "UserCell",
    "AssistantCell",
    "ToolCell",
    "SubAgentCell",
    "ErrorCell",
    "InfoCell",
    "SkillCell",
    "expand_key",
    "format_arguments",
    "reasoning_key",
]
