"""Prompt-toolkit transcript control for the interactive CLI.

The transcript is a *scrollable* view: :class:`TranscriptPane` (a
``ScrollablePane``) owns the viewport while :class:`TranscriptControl` renders
the cells.  Two rules make the view behave like a terminal transcript rather
than a fake cursor:

* the pane never auto-scrolls (the focused window lives outside it), so the
  scroll position is ours alone — no per-frame clamping that fights the user;
* while the view sits at the bottom it keeps following the newest line, and the
  moment the user scrolls up it stays where it was put.
"""

from __future__ import annotations

import io
import re
from typing import Any, Callable

from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.layout import ScrollablePane
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
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
from harness.cli.cells.base import expand_key, format_arguments
from harness.cli.render.theme import build_markdown_console
from harness.cli.sanitize import clip_to_width, display_width, safe_text

#: lines shown for a collapsed tool body / reasoning block
COLLAPSED_TOOL_LINES = 3
COLLAPSED_SUBAGENT_LINES = 3
COLLAPSED_REASONING_LINES = 6
#: lines of the *tail* shown while a tool is still running
RUNNING_TAIL_LINES = 3
#: hard cap for one expanded block: a 100k-line file must not freeze the UI
MAX_EXPANDED_LINES = 400

#: OSC sequences (hyperlinks, window titles) that prompt_toolkit's ANSI parser
#: would otherwise leave behind as literal text.
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


class TranscriptPane(ScrollablePane):
    """Scrollable transcript viewport with page scrolling and tail following."""

    def __init__(self, content, **kwargs: Any) -> None:
        kwargs.setdefault("show_scrollbar", True)
        kwargs.setdefault("display_arrows", False)
        super().__init__(content, **kwargs)
        self.window_height = 0
        self.content_height = 0
        self._following = True
        # only take a column for the scrollbar when there is something to scroll
        from prompt_toolkit.filters import Condition

        self.show_scrollbar = Condition(
            lambda: self.content_height > self.window_height
        )

    # ------------------------------------------------------------------ render
    def write_to_screen(  # noqa: PLR0913 - prompt_toolkit's signature
        self,
        screen,
        mouse_handlers,
        write_position,
        parent_style: str,
        erase_bg: bool,
        z_index: int | None,
    ) -> None:
        # Heights are only known during a render; remember them so key presses
        # can page by exactly one screen.
        self.window_height = write_position.height
        virtual_width = write_position.width - (1 if self.show_scrollbar() else 0)
        self.content_height = max(
            write_position.height,
            min(
                self.content.preferred_height(virtual_width, self.max_available_height).preferred,
                self.max_available_height,
            ),
        )
        self._apply_scroll()
        super().write_to_screen(
            screen, mouse_handlers, write_position, parent_style, erase_bg, z_index
        )

    def _apply_scroll(self) -> None:
        limit = self.max_scroll
        if self._following:
            self.vertical_scroll = limit
        elif self.vertical_scroll > limit:
            self.vertical_scroll = limit
        self._following = self.vertical_scroll >= limit

    # --------------------------------------------------------------- scrolling
    @property
    def max_scroll(self) -> int:
        return max(0, self.content_height - max(1, self.window_height))

    @property
    def at_bottom(self) -> bool:
        return self.vertical_scroll >= self.max_scroll

    @property
    def scrolled_up_by(self) -> int:
        return max(0, self.max_scroll - self.vertical_scroll)

    def page(self, direction: int) -> None:
        """Scroll one screen up (``-1``) or down (``+1``)."""

        step = max(1, self.window_height - 1)
        self.scroll_lines(direction * step)

    def scroll_lines(self, lines: int) -> None:
        self.vertical_scroll = max(0, min(self.max_scroll, self.vertical_scroll + lines))
        self._following = self.at_bottom

    def to_top(self) -> None:
        self.vertical_scroll = 0
        self._following = False

    def to_bottom(self) -> None:
        self.vertical_scroll = self.max_scroll
        self._following = True


class TranscriptControl(FormattedTextControl):
    """Render committed and active cells inside the prompt application."""

    def __init__(self, state: Any, final_cell: Callable[[], Any] | None = None) -> None:
        self.state = state
        self.final_cell = final_cell
        self.pane: TranscriptPane | None = None
        #: width of the last render pass (used to clip long preview lines)
        self.width = 80
        self._markdown_cache: dict[tuple[int, int, int], StyleAndTextTuples] = {}
        super().__init__(self._get_fragments, focusable=False, show_cursor=False)

    # ------------------------------------------------------------------ render
    def create_content(self, width: int, height: int | None):  # noqa: ANN201
        if width > 0:
            self.width = width
        return super().create_content(width, height)

    def mouse_handler(self, mouse_event: MouseEvent):
        """Wheel scrolling: three lines per notch, like a terminal."""

        if mouse_event.event_type is MouseEventType.SCROLL_UP:
            self._scroll_by(-3)
            return None
        if mouse_event.event_type is MouseEventType.SCROLL_DOWN:
            self._scroll_by(3)
            return None
        return NotImplemented

    def _scroll_by(self, lines: int) -> None:
        """``lines < 0`` moves towards older content, ``> 0`` towards the tail."""

        if self.pane is not None:
            self.pane.scroll_lines(lines)

    # ---------------------------------------------------------------- fragments
    def _get_fragments(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = []
        cells = list(getattr(self.state, "history_cells", []))
        for cell in cells:
            fragments.extend(self._cell_fragments(cell))
        # A running tool / delegated subagent is not committed to history yet;
        # render it here or the user sees nothing until it finishes.
        active = getattr(self.state, "active_cell", None)
        if active is not None and not any(active is cell for cell in cells):
            fragments.extend(self._cell_fragments(active))
        activity = getattr(self.state, "activity", "")
        if activity:
            fragments.append(("class:activity", f"· {safe_text(activity)}\n"))
        if not fragments:
            fragments.append(("class:transcript-dim", ""))
        return fragments

    @property
    def line_count(self) -> int:
        """Logical lines currently rendered (used by tests and hints)."""

        fragments = self._get_fragments()
        if not fragments:
            return 0
        return sum(text.count("\n") for _style, text in fragments) or 1

    def _expanded(self) -> set[str]:
        return getattr(self.state, "expanded_tool_ids", set())

    def _cell_fragments(self, cell: HistoryCell) -> StyleAndTextTuples:
        if isinstance(cell, UserCell):
            return _prefixed_lines("› ", safe_text(cell.text), "class:user")
        if isinstance(cell, AssistantCell):
            return self._assistant_fragments(cell)
        if isinstance(cell, ToolCell):
            return self._tool_fragments(cell)
        if isinstance(cell, SubAgentCell):
            return self._subagent_fragments(cell)
        if isinstance(cell, SkillCell):
            if cell.ok:
                return [("class:skill", f"◆ skill loaded: {safe_text(cell.name)}\n")]
            return [("class:error", f"◆ skill failed: {safe_text(cell.name)}\n")]
        if isinstance(cell, ErrorCell):
            style = "class:error-fatal" if cell.fatal else "class:error"
            return _prefixed_lines("!! ", safe_text(cell.message), style)
        if isinstance(cell, InfoCell):
            return _prefixed_lines("", safe_text(cell.message), "class:info")
        return [("class:transcript", f"{safe_text(str(cell))}\n")]

    # --------------------------------------------------------------- assistants
    def _assistant_fragments(self, cell: AssistantCell) -> StyleAndTextTuples:
        parts: StyleAndTextTuples = []
        if self.final_cell is not None and cell is self.final_cell():
            parts.append(("class:final-marker", "── final answer ──\n"))
        if cell.reasoning:
            parts.extend(self._reasoning_fragments(cell))
        if cell.source:
            parts.extend(self._source_fragments(cell))
        return parts

    def _source_fragments(self, cell: AssistantCell) -> StyleAndTextTuples:
        """Markdown for complete lines; the mutable tail stays plain text.

        Re-rendering half a Markdown document on every delta is what made an
        unterminated code fence swallow the rest of the answer.
        """

        source = cell.source
        if cell.complete or "\n" not in source.strip():
            return self._markdown(source)
        head, _, tail = source.rpartition("\n")
        parts = self._markdown(head + "\n") if head else []
        if tail:
            parts.append(("class:assistant", f"{safe_text(tail)}\n"))
        return parts

    def _reasoning_fragments(self, cell: AssistantCell) -> StyleAndTextTuples:
        text = safe_text(cell.reasoning or "").strip()
        if not text:
            return []
        lines = text.splitlines()
        expanded = expand_key(cell) in self._expanded()
        limit = MAX_EXPANDED_LINES if expanded else COLLAPSED_REASONING_LINES
        shown = lines[:limit]
        parts: StyleAndTextTuples = [("class:reasoning", "∴ thinking\n")]
        for line in shown:
            parts.append(("class:reasoning", f"  {_clip(line, self.width - 4)}\n"))
        if len(lines) > limit:
            parts.append(
                (
                    "class:reasoning",
                    f"  … {len(lines) - limit} more line(s) — Ctrl+O 折叠/展开\n",
                )
            )
        return parts

    # -------------------------------------------------------------------- tools
    def _tool_fragments(self, cell: ToolCell) -> StyleAndTextTuples:
        style = {
            ToolStatus.RUNNING: "class:tool-running",
            ToolStatus.DONE: "class:tool-done",
            ToolStatus.FAILED: "class:tool-failed",
        }[cell.status]
        header = f"{cell.glyph} {safe_text(cell.tool)}"
        if cell.arguments:
            header += f"  {_clip(format_arguments(cell.arguments), self.width - 6)}"
        if cell.status is ToolStatus.RUNNING:
            header += "  …"
        elif cell.duration_ms:
            header += f"  ({cell.duration_ms} ms)"
        parts: StyleAndTextTuples = [(style, header + "\n")]

        if cell.status is ToolStatus.RUNNING and not cell.error:
            # live feedback: the tail of what the tool has produced so far
            lines = [line for line in cell.body_lines() if line.strip()][-RUNNING_TAIL_LINES:]
            for line in lines:
                parts.append(("class:tool-body", f"  │ {_clip(line, self.width - 4)}\n"))
            return parts

        expanded = cell.call_id in self._expanded() if cell.call_id else False
        limit = MAX_EXPANDED_LINES if expanded else COLLAPSED_TOOL_LINES
        lines, hidden = cell.preview_body(limit)
        parts.extend(
            ("class:tool-body", f"  │ {_clip(line, self.width - 4)}\n") for line in lines
        )
        if hidden:
            hint = "" if expanded else "  (Ctrl+O 展开)"
            parts.append(("class:tool-body", f"  │ … {hidden} more line(s){hint}\n"))
        return parts

    # ---------------------------------------------------------------- subagents
    def _subagent_fragments(self, cell: SubAgentCell) -> StyleAndTextTuples:
        style = "class:subagent" if cell.ok else "class:error"
        header = f"{'⇢' if cell.ok else '✗'} {safe_text(cell.agent)}"
        task = (cell.task or "").splitlines()[0] if cell.task else ""
        if task:
            header += f"  {_clip(task, max(10, self.width - len(header) - 20))}"
        if cell.iterations:
            header += f"  ({cell.iterations} iterations)"
        if cell.status is ToolStatus.RUNNING:
            header += "  …"
        parts: StyleAndTextTuples = [(style, header + "\n")]
        if cell.summary:
            expanded = expand_key(cell) in self._expanded()
            limit = MAX_EXPANDED_LINES if expanded else COLLAPSED_SUBAGENT_LINES
            lines, hidden = cell.preview(cell.summary, limit)  # type: ignore[arg-type]
            for line in safe_text(lines).splitlines():
                parts.append(("class:tool-body", f"  │ {_clip(line, self.width - 4)}\n"))
            if hidden:
                hint = "" if expanded else "  (Ctrl+O 展开)"
                parts.append(("class:tool-body", f"  │ … {hidden} more line(s){hint}\n"))
        return parts

    # ---------------------------------------------------------------- markdown
    def _markdown(self, source: str) -> StyleAndTextTuples:
        key = (hash(source), len(source), self.width)
        cached = self._markdown_cache.get(key)
        if cached is not None:
            return cached
        fragments = _render_markdown(source, width=self.width)
        if len(self._markdown_cache) > 64:
            self._markdown_cache.clear()
        self._markdown_cache[key] = fragments
        return fragments


__all__ = ["TranscriptControl", "TranscriptPane"]


# --------------------------------------------------------------------- helpers
def _prefixed_lines(prefix: str, text: str, style: str) -> StyleAndTextTuples:
    """Render user/error/info text with the prefix on the first line only."""

    lines = text.split("\n") if text else [""]
    parts: StyleAndTextTuples = []
    for index, line in enumerate(lines):
        marker = prefix if index == 0 else " " * len(prefix)
        parts.append((style, f"{marker}{line}\n"))
    return parts


def _clip(line: str, width: int) -> str:
    """Clip one line to the available columns so a long line is one row."""

    if width <= 1 or display_width(line) <= width:
        return line
    return clip_to_width(line, width - 1) + "…"


def _escape_angle_brackets(source: str) -> str:
    """Escape ``<``/``>`` outside code so rich's Markdown keeps them.

    rich parses ``<value>`` as an HTML tag (and ``<https://…>`` as an autolink)
    and silently drops the text; ``\\<`` renders as a literal ``<``.  Inside
    code spans/fences the backslash would show up verbatim, so those are left
    alone.
    """

    lines: list[str] = []
    in_fence = False
    fence = ""
    for line in source.split("\n"):
        stripped = line.lstrip()
        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fence = True
            fence = stripped[:3]
            lines.append(line)
            continue
        if in_fence:
            if stripped.startswith(fence):
                in_fence = False
            lines.append(line)
            continue
        out: list[str] = []
        in_code = False
        # a leading ">" is a blockquote marker, not text to escape
        head = len(line) - len(line.lstrip())
        quote_end = head
        while quote_end < len(line) and line[quote_end] == ">":
            quote_end += 1
        for index, char in enumerate(line):
            if char == "`":
                in_code = not in_code
                out.append(char)
            elif char in "<>" and not in_code and index >= quote_end:
                out.append("\\" + char)
            else:
                out.append(char)
        lines.append("".join(out))
    return "\n".join(lines)


def _render_markdown(source: str, *, width: int = 100) -> StyleAndTextTuples:
    """Render assistant Markdown to terminal fragments at the real width."""

    output = io.StringIO()
    console = build_markdown_console(
        file=output,
        width=max(20, width),
        force_terminal=True,
        color_system="standard",
    )
    console.print(Markdown(_escape_angle_brackets(safe_text(source))))
    # OSC 8 hyperlinks (rich emits them for links) are not understood by
    # prompt_toolkit's ANSI parser: strip them or their payload shows as text.
    rendered = _OSC_RE.sub("", output.getvalue())
    rendered = re.sub(r"[ \t]+(?=\n)", "", rendered)
    return to_formatted_text(ANSI(rendered))
