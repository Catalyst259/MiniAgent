"""HistoryCell: the unit of CLI history.

History is a list of typed cells, not a list of strings, so the renderer never
grows into a chain of ``elif event.type == ...``.  Each cell exposes
``render(console)`` and may mutate itself while it is the active cell.
See ``CLI_Design.md`` sections 26 and 27.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from harness.cli.sanitize import safe_text


class ToolStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@runtime_checkable
class ConsoleLike(Protocol):
    """The slice of ``rich.console.Console`` the cells are allowed to use."""

    def print(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class HistoryCell(Protocol):
    def render(self, console: ConsoleLike) -> None: ...


@dataclass
class BaseCell:
    """Common behaviour: optional title, optional collapsed preview."""

    title: str = ""
    preview_lines: int = 0

    def render(self, console: ConsoleLike) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    @staticmethod
    def preview(text: str, lines: int) -> tuple[str, int]:
        """Return the first ``lines`` lines plus the number of hidden lines."""

        if lines <= 0:
            return text, 0
        split = text.splitlines()
        if len(split) <= lines:
            return text, 0
        return "\n".join(split[:lines]), len(split) - lines


@dataclass
class UserCell(BaseCell):
    """The prompt the user submitted."""

    text: str = ""

    def render(self, console: ConsoleLike) -> None:
        from rich.text import Text

        line = Text()
        line.append("› ", style="bold green")
        line.append(self.text)
        console.print(line)


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

    def render(self, console: ConsoleLike) -> None:
        if self.reasoning:
            console.print(f"[dim italic]{safe_text(self.reasoning).strip()}[/dim italic]")
        if not self.source.strip():
            return
        try:
            from rich.markdown import Markdown

            console.print(Markdown(self.source))
        except Exception:  # pragma: no cover - markdown is best effort
            console.print(safe_text(self.source))


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

    def preview_body(self, max_lines: int | None = None) -> tuple[list[str], int]:
        """Return the sanitized visible body and the number of hidden lines."""

        body = safe_text(self.error or self.output).rstrip("\n")
        if not body:
            return [], 0
        shown, hidden = self.preview(body, self.max_preview_lines if max_lines is None else max_lines)
        return shown.splitlines(), hidden

    # ------------------------------------------------------------------ rendering
    @property
    def glyph(self) -> str:
        return {
            ToolStatus.RUNNING: "●",
            ToolStatus.DONE: "●",
            ToolStatus.FAILED: "✗",
        }[self.status]

    @property
    def style(self) -> str:
        return {
            ToolStatus.RUNNING: "yellow",
            ToolStatus.DONE: "green",
            ToolStatus.FAILED: "bold red",
        }[self.status]

    def header_key(self) -> str:
        """Identity of one rendering of this call.

        Includes the status so that the "running" announcement and the finished
        result are two distinct renders (the result carries the body), while a
        duplicated event for the *same* state renders nothing twice.
        """

        # the call id is what makes two identical-looking calls distinct;
        # the status makes the "running" announcement and the finished result
        # two separate renders instead of one suppressed duplicate
        return f"{self.call_id}|{self.tool}|{self.status.value}"

    def render(self, console: ConsoleLike) -> None:
        from rich.text import Text

        header = Text()
        header.append(f"{self.glyph} ", style=self.style)
        header.append(self.tool, style=f"bold {self.style}")
        if self.arguments:
            header.append(f"  {_format_arguments(self.arguments)}", style="dim")
        if self.status is ToolStatus.DONE and self.duration_ms:
            header.append(f"  ({self.duration_ms} ms)", style="dim")
        elif self.status is ToolStatus.RUNNING:
            header.append("  …", style="dim")
        console.print(header)

        # While the call is still running only the announced header is shown;
        # the body (and any error) appears once, when the call finishes.
        if self.status is ToolStatus.RUNNING and not self.error:
            return

        lines, hidden = self.preview_body()
        for line in lines:
            console.print(f"  [dim]│[/dim] {line}", highlight=False)
        if hidden:
            console.print(f"  [dim]│ … {hidden} more line(s)[/dim]")


@dataclass
class SubAgentCell(BaseCell):
    agent: str = ""
    task: str = ""
    summary: str = ""
    ok: bool = True
    iterations: int = 0
    max_preview_lines: int = 12

    def render(self, console: ConsoleLike) -> None:
        from rich.text import Text

        header = Text()
        header.append("⇢ " if self.ok else "✗ ", style="blue" if self.ok else "bold red")
        header.append(f"{self.agent}", style="bold blue")
        header.append(f"  {self.task.splitlines()[0][:80] if self.task else ''}", style="dim")
        if self.iterations:
            header.append(f"  ({self.iterations} iterations)", style="dim")
        console.print(header)
        if self.summary:
            shown, hidden = self.preview(safe_text(self.summary), self.max_preview_lines)
            for line in shown.splitlines():
                console.print(f"  [dim]│[/dim] {line}", highlight=False)
            if hidden:
                console.print(f"  [dim]│ … {hidden} more line(s)[/dim]")


@dataclass
class ErrorCell(BaseCell):
    message: str = ""
    fatal: bool = False

    def render(self, console: ConsoleLike) -> None:
        style = "bold red" if self.fatal else "red"
        console.print(f"[{style}]!! {self.message}[/{style}]")


@dataclass
class InfoCell(BaseCell):
    message: str = ""
    style: str = "dim"

    def render(self, console: ConsoleLike) -> None:
        console.print(f"[{self.style}]{self.message}[/{self.style}]")


@dataclass
class SkillCell(BaseCell):
    name: str = ""
    ok: bool = True

    def render(self, console: ConsoleLike) -> None:
        if self.ok:
            console.print(f"[magenta]◆ skill loaded: {self.name}[/magenta]")
        else:
            console.print(f"[red]◆ skill failed: {self.name}[/red]")


def _format_arguments(arguments: dict[str, Any], limit: int = 110) -> str:
    import json

    try:
        rendered = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        rendered = str(arguments)
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


__all__ = [
    "ToolStatus",
    "HistoryCell",
    "ConsoleLike",
    "BaseCell",
    "UserCell",
    "AssistantCell",
    "ToolCell",
    "SubAgentCell",
    "ErrorCell",
    "InfoCell",
    "SkillCell",
]
