"""Prompt-toolkit transcript control for the interactive CLI."""

from __future__ import annotations

import io
from typing import Any

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.screen import Point
from rich.console import Console
from rich.markdown import Markdown

from harness.cli.cells import (
    AssistantCell,
    ErrorCell,
    HistoryCell,
    InfoCell,
    SkillCell,
    SubAgentCell,
    ToolCell,
    ToolStatus,
    UserCell,
)
from harness.cli.sanitize import safe_text


class TranscriptControl(FormattedTextControl):
    """Render committed and active cells inside the prompt application."""

    def __init__(self, state: Any, final_cell=None) -> None:
        self.state = state
        self.final_cell = final_cell
        super().__init__(
            self._get_fragments,
            focusable=False,
            show_cursor=False,
            get_cursor_position=self._cursor_position,
        )

    def _cursor_position(self) -> Point:
        """Keep the transcript viewport anchored to its newest line."""

        line_count = sum(text.count("\n") for _style, text in self._get_fragments())
        scroll = getattr(self.state, "transcript_scroll", None)
        return Point(x=0, y=max(0, line_count - 1 if scroll is None else scroll))

    @property
    def line_count(self) -> int:
        return sum(text.count("\n") for _style, text in self._get_fragments())

    def _get_fragments(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = []
        for cell in self.state.history_cells:
            fragments.extend(self._cell_fragments(cell))
        activity = getattr(self.state, "activity", "")
        if activity:
            fragments.append(("class:activity", f"· {safe_text(activity)}\n"))
        if not fragments:
            fragments.append(("class:transcript-dim", ""))
        return fragments

    def _cell_fragments(self, cell: HistoryCell) -> StyleAndTextTuples:
        if isinstance(cell, UserCell):
            return [("class:user", f"› {safe_text(cell.text)}\n")]
        if isinstance(cell, AssistantCell):
            parts: StyleAndTextTuples = []
            if self.final_cell is not None and cell is self.final_cell():
                parts.append(("class:final-marker", "── final answer ──\n"))
            if cell.reasoning:
                parts.append(("class:reasoning", f"{safe_text(cell.reasoning).strip()}\n"))
            if cell.source:
                parts.append(("class:assistant", f"{_render_markdown(cell.source)}\n"))
            return parts
        if isinstance(cell, ToolCell):
            style = {
                ToolStatus.RUNNING: "class:tool-running",
                ToolStatus.DONE: "class:tool-done",
                ToolStatus.FAILED: "class:tool-failed",
            }[cell.status]
            suffix = "  …" if cell.status is ToolStatus.RUNNING else ""
            parts = [(style, f"{cell.glyph} {cell.tool}{suffix}\n")]
            expanded = cell.call_id in getattr(self.state, "expanded_tool_ids", set())
            lines, hidden = cell.preview_body(None if expanded else 3)
            if cell.status is not ToolStatus.RUNNING:
                parts.extend(("class:tool-body", f"  │ {line}\n") for line in lines)
                if hidden:
                    parts.append(("class:tool-body", f"  │ … {hidden} more line(s)\n"))
            return parts
        if isinstance(cell, SubAgentCell):
            return [("class:subagent", f"⇢ {cell.agent}  {safe_text(cell.summary or cell.task)}\n")]
        if isinstance(cell, SkillCell):
            style = "class:skill" if cell.ok else "class:error"
            return [(style, f"◆ skill loaded: {safe_text(cell.name)}\n")]
        if isinstance(cell, ErrorCell):
            return [("class:error", f"!! {safe_text(cell.message)}\n")]
        if isinstance(cell, InfoCell):
            return [("class:info", f"{safe_text(cell.message)}\n")]
        return [("class:transcript", f"{safe_text(str(cell))}\n")]


__all__ = ["TranscriptControl"]


def _render_markdown(source: str) -> str:
    """Render assistant Markdown to readable terminal text for the UI control."""

    output = io.StringIO()
    console = Console(file=output, width=120, color_system=None, highlight=False)
    console.print(Markdown(safe_text(source)))
    return output.getvalue().rstrip("\n")