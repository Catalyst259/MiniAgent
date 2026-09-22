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

from harness.cli.cells import HistoryCell
from harness.cli.presentation import (
    CellPresentation,
    Line,
    MarkdownBlock,
    PresentationOptions,
    present_cell,
)
from harness.cli.render.theme import build_markdown_console
from harness.cli.sanitize import safe_text

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
        else:
            # A trailing newline creates a real extra UIContent row in
            # prompt_toolkit.  Cells use newlines *between* rows, but the final
            # row must not terminate with one or the composer is separated from
            # the transcript by a phantom blank line.
            last = fragments[-1]
            if last[1].endswith("\n"):
                fragments[-1] = (last[0], last[1][:-1], *last[2:])
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
        presentation = present_cell(
            cell,
            PresentationOptions(
                width=self.width,
                expanded_keys=frozenset(self._expanded()),
                final=self.final_cell is not None and cell is self.final_cell(),
                tool_preview_lines=3,
                subagent_preview_lines=3,
                reasoning_preview_lines=6,
                running_tail_lines=3,
                reasoning_header=True,
                expand_hint=True,
                streaming_tail_plain=True,
            ),
        )
        return _to_fragments(presentation, self._markdown)

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
_ROLE_STYLES = {
    "user": "class:user",
    "assistant": "class:assistant",
    "reasoning": "class:reasoning",
    "final-marker": "class:final-marker",
    "tool-running": "class:tool-running",
    "tool-running-name": "class:tool-running bold",
    "tool-done": "class:tool-done",
    "tool-done-name": "class:tool-done bold",
    "tool-failed": "class:tool-failed",
    "tool-failed-name": "class:tool-failed bold",
    "body-prefix": "class:tool-body",
    "tool-body": "class:tool-body",
    "subagent": "class:subagent",
    "subagent-name": "class:subagent bold",
    "skill": "class:skill",
    "info": "class:info",
    "error": "class:error",
    "error-fatal": "class:error-fatal",
    "dim": "class:transcript-dim",
    "text": "class:transcript",
}


def _to_fragments(
    presentation: CellPresentation,
    markdown: Callable[[str], StyleAndTextTuples],
) -> StyleAndTextTuples:
    parts: StyleAndTextTuples = []
    for block in presentation.blocks:
        if isinstance(block, MarkdownBlock):
            parts.extend(markdown(block.source))
            continue
        if not isinstance(block, Line):  # pragma: no cover - closed union
            continue
        if not block.spans:
            parts.append(("class:transcript", "\n"))
            continue
        for index, span in enumerate(block.spans):
            ending = "\n" if index == len(block.spans) - 1 else ""
            parts.append((_ROLE_STYLES.get(span.role, "class:transcript"), span.text + ending))
    return parts


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
