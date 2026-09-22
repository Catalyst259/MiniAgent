"""One semantic presentation for every CLI history cell.

Cells hold state.  This module decides what that state means on screen.  Rich
and prompt_toolkit are adapters at the seam; neither decides preview limits,
headers, expansion, live tails or error wording independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from harness.cli.cells.base import (
    AssistantCell,
    ErrorCell,
    HistoryCell,
    InfoCell,
    SkillCell,
    SubAgentCell,
    ToolCell,
    ToolStatus,
    UserCell,
    expand_key,
    format_arguments,
)
from harness.cli.sanitize import clip_to_width, display_width, safe_text

MAX_EXPANDED_LINES = 400


@dataclass(frozen=True)
class Span:
    text: str
    role: str = "text"


@dataclass(frozen=True)
class Line:
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class MarkdownBlock:
    source: str


DisplayBlock: TypeAlias = Line | MarkdownBlock


@dataclass(frozen=True)
class CellPresentation:
    blocks: tuple[DisplayBlock, ...] = ()


@dataclass(frozen=True)
class PresentationOptions:
    """Display policy supplied by an output adapter.

    Width and expansion are viewport facts.  Preview counts are front-end
    policy: a scrollable TUI can start compact, while line-based output cannot
    be expanded after it is printed.
    """

    width: int = 0
    expanded_keys: frozenset[str] = field(default_factory=frozenset)
    final: bool = False
    tool_preview_lines: int | None = None
    subagent_preview_lines: int | None = None
    reasoning_preview_lines: int | None = None
    running_tail_lines: int = 0
    reasoning_header: bool = False
    expand_hint: bool = False
    streaming_tail_plain: bool = False
    max_expanded_lines: int = MAX_EXPANDED_LINES


def present_cell(
    cell: HistoryCell,
    options: PresentationOptions | None = None,
) -> CellPresentation:
    """Return the complete semantic display of ``cell`` for one viewport."""

    view = options or PresentationOptions()
    if isinstance(cell, UserCell):
        return CellPresentation(tuple(_prefixed_lines("› ", cell.text, "user")))
    if isinstance(cell, AssistantCell):
        return CellPresentation(tuple(_assistant_blocks(cell, view)))
    if isinstance(cell, ToolCell):
        return CellPresentation(tuple(_tool_lines(cell, view)))
    if isinstance(cell, SubAgentCell):
        return CellPresentation(tuple(_subagent_lines(cell, view)))
    if isinstance(cell, SkillCell):
        role = "skill" if cell.ok else "error"
        result = "loaded" if cell.ok else "failed"
        return CellPresentation((line(f"◆ skill {result}: {safe_text(cell.name)}", role),))
    if isinstance(cell, ErrorCell):
        role = "error-fatal" if cell.fatal else "error"
        return CellPresentation(tuple(_prefixed_lines("!! ", cell.message, role)))
    if isinstance(cell, InfoCell):
        return CellPresentation(tuple(_prefixed_lines("", cell.message, "info")))
    return CellPresentation((line(safe_text(str(cell)), "text"),))


def presentation_key(cell: HistoryCell) -> str:
    """Identity of one printable state, used to suppress duplicate events."""

    if isinstance(cell, ToolCell):
        identity = cell.call_id or str(id(cell))
        return f"tool:{identity}:{cell.tool}:{cell.status.value}"
    if isinstance(cell, SubAgentCell):
        return f"subagent:{id(cell)}:{cell.status.value}"
    return f"cell:{id(cell)}"


def line(text: str, role: str = "text") -> Line:
    return Line((Span(text, role),))


def _assistant_blocks(cell: AssistantCell, view: PresentationOptions) -> list[DisplayBlock]:
    blocks: list[DisplayBlock] = []
    if view.final:
        blocks.append(line("── final answer ──", "final-marker"))

    reasoning = safe_text(cell.reasoning or "").strip()
    if reasoning:
        if view.reasoning_header:
            blocks.append(line("∴ thinking", "reasoning"))
        reasoning_lines = reasoning.splitlines()
        expanded = expand_key(cell) in view.expanded_keys
        limit = _limit(
            expanded,
            view.reasoning_preview_lines,
            view.max_expanded_lines,
            len(reasoning_lines),
        )
        for value in reasoning_lines[:limit]:
            prefix = "  " if view.reasoning_header else ""
            blocks.append(line(prefix + _clip(value, view.width - len(prefix)), "reasoning"))
        hidden = len(reasoning_lines) - limit
        if hidden:
            hint = _hint(view, expanded)
            blocks.append(line(f"  … {hidden} more line(s){hint}", "reasoning"))

    source = safe_text(cell.source)
    if not source:
        return blocks
    if not view.streaming_tail_plain or cell.complete or "\n" not in source.strip():
        blocks.append(MarkdownBlock(source))
        return blocks
    head, _, tail = source.rpartition("\n")
    if head:
        blocks.append(MarkdownBlock(head + "\n"))
    if tail:
        blocks.append(line(tail, "assistant"))
    return blocks


def _tool_lines(cell: ToolCell, view: PresentationOptions) -> list[Line]:
    status_role = {
        ToolStatus.RUNNING: "tool-running",
        ToolStatus.DONE: "tool-done",
        ToolStatus.FAILED: "tool-failed",
    }[cell.status]
    spans = [Span(f"{cell.glyph} ", status_role), Span(safe_text(cell.tool), status_role + "-name")]
    if cell.arguments:
        argument = format_arguments(cell.arguments)
        if view.width:
            argument = _clip(argument, view.width - 6)
        spans.append(Span(f"  {argument}", "dim"))
    if cell.status is ToolStatus.RUNNING:
        spans.append(Span("  …", "dim"))
    elif cell.duration_ms:
        spans.append(Span(f"  ({cell.duration_ms} ms)", "dim"))
    lines = [Line(tuple(spans))]

    body = cell.body_lines()
    if cell.status is ToolStatus.RUNNING and not cell.error:
        if view.running_tail_lines:
            visible = [value for value in body if value.strip()][-view.running_tail_lines :]
            lines.extend(_body_line(value, view.width) for value in visible)
        return lines

    expanded = bool(cell.call_id and cell.call_id in view.expanded_keys)
    default_limit = cell.max_preview_lines if view.tool_preview_lines is None else view.tool_preview_lines
    limit = _limit(expanded, default_limit, view.max_expanded_lines, len(body))
    lines.extend(_body_line(value, view.width) for value in body[:limit])
    hidden = len(body) - limit
    if hidden:
        lines.append(_body_line(f"… {hidden} more line(s){_hint(view, expanded)}", view.width))
    return lines


def _subagent_lines(cell: SubAgentCell, view: PresentationOptions) -> list[Line]:
    role = "subagent" if cell.ok else "error"
    glyph = "⇢" if cell.ok else "✗"
    spans = [Span(f"{glyph} ", role), Span(safe_text(cell.agent), "subagent-name" if cell.ok else role)]
    task = safe_text((cell.task or "").splitlines()[0] if cell.task else "")
    if task:
        available = max(10, view.width - sum(display_width(part.text) for part in spans) - 20)
        spans.append(Span(f"  {_clip(task, available) if view.width else task[:80]}", "dim"))
    if cell.iterations:
        spans.append(Span(f"  ({cell.iterations} iterations)", "dim"))
    if cell.status is ToolStatus.RUNNING:
        spans.append(Span("  …", "dim"))
    lines = [Line(tuple(spans))]

    summary = safe_text(cell.summary).rstrip("\n")
    if not summary:
        return lines
    body = summary.splitlines()
    expanded = expand_key(cell) in view.expanded_keys
    default_limit = (
        cell.max_preview_lines
        if view.subagent_preview_lines is None
        else view.subagent_preview_lines
    )
    limit = _limit(expanded, default_limit, view.max_expanded_lines, len(body))
    lines.extend(_body_line(value, view.width) for value in body[:limit])
    hidden = len(body) - limit
    if hidden:
        lines.append(_body_line(f"… {hidden} more line(s){_hint(view, expanded)}", view.width))
    return lines


def _prefixed_lines(prefix: str, text: str, role: str) -> list[Line]:
    values = safe_text(text).split("\n") if text else [""]
    return [
        line((prefix if index == 0 else " " * len(prefix)) + value, role)
        for index, value in enumerate(values)
    ]


def _body_line(value: str, width: int) -> Line:
    shown = _clip(value, width - 4) if width else value
    return Line((Span("  │ ", "body-prefix"), Span(shown, "tool-body")))


def _limit(expanded: bool, collapsed: int | None, maximum: int, length: int) -> int:
    if expanded:
        return min(length, maximum)
    if collapsed is None or collapsed <= 0:
        return length
    return min(length, collapsed)


def _hint(view: PresentationOptions, expanded: bool) -> str:
    return "" if expanded or not view.expand_hint else "  (Ctrl+O 展开)"


def _clip(value: str, width: int) -> str:
    if width <= 1 or display_width(value) <= width:
        return value
    return clip_to_width(value, width - 1) + "…"


__all__ = [
    "CellPresentation",
    "DisplayBlock",
    "Line",
    "MarkdownBlock",
    "MAX_EXPANDED_LINES",
    "PresentationOptions",
    "Span",
    "present_cell",
    "presentation_key",
]
