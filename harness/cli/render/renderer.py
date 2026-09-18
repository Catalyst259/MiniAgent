"""Renderer: cells, the live tail, and the transcript.

The renderer is the only component allowed to touch the terminal.  It keeps the
list of cells it has already committed so history is never printed twice, and it
tracks the mutable tail of the assistant stream separately
(``CLI_Design.md`` sections 27, 38).
"""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from harness.cli.cells.base import AssistantCell, HistoryCell, InfoCell, ToolCell, UserCell
from harness.cli.render.theme import DEFAULT_THEME, Theme
from harness.cli.sanitize import clip_to_width, safe_text
from harness.cli.streaming.assistant_stream import AssistantStream


@dataclass
class Renderer:
    """Consumes cells; never calls a tool, a model or LangGraph."""

    console: Any = None
    theme: Theme = DEFAULT_THEME
    stream: AssistantStream = field(default_factory=AssistantStream)
    #: where the live region is written (defaults to stdout at construction)
    output: Any = None
    #: ``None`` = auto (live tail only on a real terminal)
    use_live_tail: bool | None = None
    _committed: int = 0
    _tail_open: bool = False
    _last_tail: str = ""
    _drawn_committed: int = 0
    _last_tail_at: float = 0.0
    _printed_cells: list = field(default_factory=list)
    _last_tool_line: str = ""
    _pending_tail: str = ""
    #: rows of raw streamed preview currently on screen (they are replaced by
    #: the canonical rendering, never printed twice)
    _streamed_lines: int = 0
    #: minimum seconds between two in-place tail redraws
    tail_interval: float = 0.09

    def __post_init__(self) -> None:
        if self.output is None:
            self.output = sys.stdout
        if self.console is None:
            from harness.cli.render.theme import build_markdown_console

            self.console = build_markdown_console()
        if self.use_live_tail is None:
            self.use_live_tail = bool(getattr(self.output, "isatty", lambda: False)())

    # ------------------------------------------------------------------ terminal
    @property
    def width(self) -> int:
        return shutil.get_terminal_size((100, 24)).columns

    def banner(self) -> None:
        from rich.console import Console

        console = self.console if isinstance(self.console, Console) else self.console
        console.print(
            f"[{self.theme.banner}]MiniAgent[/{self.theme.banner}] "
            "[dim]· LangGraph + ModelGateway + MCP + Qdrant[/dim]"
        )
        console.print("[dim]type / for commands, ! for a shell command, Ctrl+C to interrupt[/dim]")

    def bind_output(self, output: Any) -> None:
        """Bind terminal writes to the same stream as the active UI driver."""

        self.output = output
        try:
            from rich.console import Console

            if isinstance(self.console, Console):
                self.console.file = output
        except Exception:  # pragma: no cover - custom console/test double
            pass

    # -------------------------------------------------------------- cell history
    def render_cell(self, cell: HistoryCell) -> None:
        """Render one cell.

        A tool announces itself while running and prints its result when it
        finishes, so it renders once per *state*.  Every other cell renders once
        per object, whatever the call site.
        """

        if isinstance(cell, ToolCell):
            self._close_tail()
            signature = cell.header_key()
            if signature and signature == self._last_tool_line:
                return  # duplicate event for the same state
            self._last_tool_line = signature
        elif any(cell is printed for printed in self._printed_cells):
            return
        cell.render(self.console)
        self._printed_cells.append(cell)

    def flush(self, cells: list[HistoryCell]) -> None:
        """Print every cell that has not been printed yet."""

        while self._committed < len(cells):
            self.render_cell(cells[self._committed])
            self._committed += 1

    def info(self, message: str) -> None:
        self._close_tail()
        InfoCell(message=message).render(self.console)

    def error(self, message: str, *, fatal: bool = False) -> None:
        from harness.cli.cells.base import ErrorCell

        self._close_tail()
        ErrorCell(message=message, fatal=fatal).render(self.console)

    def rule(self, title: str = "") -> None:
        self._close_tail()
        rule = getattr(self.console, "rule", None)
        if callable(rule):
            rule(title)

    def print(self, *args, **kwargs) -> None:
        self.console.print(*args, **kwargs)

    # -------------------------------------------------------------- live tail
    def begin_stream(self) -> None:
        self.stream.reset()
        self._tail_open = False
        self._last_tail = ""
        self._pending_tail = ""
        self._drawn_committed = 0
        self._streamed_lines = 0

    def push_delta(self, delta: str) -> None:
        """Feed a raw Markdown delta.

        Complete lines are written once, each on its own line, so the answer
        never runs together; only the still-mutable last line is redrawn in
        place.  Nothing is rendered as Markdown until the stream ends.
        """

        self.stream.append(delta)
        if not self.use_live_tail:
            return

        committed = self.stream.committed_source
        if len(committed) > self._drawn_committed:
            fresh = committed[self._drawn_committed :]
            self._drawn_committed = len(committed)
            self._write_committed(fresh)

        tail = self.stream.pending_source.split("\n")[-1]
        if tail == self._last_tail:
            return
        now = time.monotonic()
        if now - self._last_tail_at < self.tail_interval:
            # remember it so end_stream() can draw the final state
            self._pending_tail = tail
            return
        self._last_tail = tail
        self._last_tail_at = now
        self._write_tail(tail)

    def _write_committed(self, chunk: str) -> None:
        """Write finished lines (they never change again).

        The in-place tail row is erased first, otherwise the finished line would
        be concatenated with the preview that is already on screen.  Lines are
        clipped to one row each so the preview can be erased exactly once the
        canonical Markdown rendering replaces it.
        """

        lines = [
            clip_to_width(safe_text(line), max(1, self.width - 4))
            for line in chunk.splitlines()
        ]
        if not lines:
            return
        try:
            prefix = "\r\x1b[2K" if self._tail_open else ""
            self.output.write(prefix + "".join(f"  │ {line}\n" for line in lines))
            self.output.flush()
        except Exception:  # pragma: no cover - terminal hiccup
            return
        self._streamed_lines += len(lines)
        self._tail_open = False
        self._last_tail = ""

    def _erase_streamed(self) -> None:
        """Remove the raw streaming preview (rows) from the screen.

        The answer is rendered once, as Markdown, when the message completes;
        without this the same text showed up twice (raw preview + final block).
        """

        rows = self._streamed_lines
        self._streamed_lines = 0
        if not rows or not self.use_live_tail:
            return
        try:
            self.output.write(f"\x1b[{rows}A\x1b[J")
            self.output.flush()
        except Exception:  # pragma: no cover - terminal hiccup
            return

    def _write_tail(self, preview: str) -> None:
        text = clip_to_width(safe_text(preview), max(0, self.width - 4))
        try:
            # Carriage return + erase line: the mutable tail lives on one row.
            self.output.write(f"\r\x1b[2K  │ {text}")
            self.output.flush()
            self._tail_open = True
        except Exception:  # pragma: no cover - terminal hiccup
            self._tail_open = False

    def _close_tail(self) -> None:
        if self._tail_open:
            try:
                self.output.write("\r\x1b[2K")
                self.output.flush()
            except Exception:  # pragma: no cover
                pass
            self._tail_open = False
            self._last_tail = ""
        # whatever is rendered next is canonical: drop the raw preview rows
        self._erase_streamed()

    def end_stream(self, source: str | None = None) -> str:
        """Finalize: drop the live preview and return the canonical source."""

        if self.use_live_tail and self._pending_tail and self._pending_tail != self._last_tail:
            self._write_tail(self._pending_tail)
            self._last_tail = self._pending_tail
        self._pending_tail = ""
        final = self.stream.finish(source)
        self._close_tail()
        self._last_tool_line = ""
        return final

    def render_assistant(self, source: str, reasoning: str | None = None) -> AssistantCell:
        """Canonical render of the complete Markdown source."""

        self._close_tail()
        cell = AssistantCell(source=source, reasoning=reasoning)
        cell.render(self.console)
        return cell

    def final_answer(self, answer: str | AssistantCell, *, status: str = "") -> None:
        """Render the finished answer as an unmistakable block."""

        self._close_tail()
        cell = answer if isinstance(answer, AssistantCell) else AssistantCell(source=str(answer))
        if status and status not in ("", "final_answer"):
            self.console.print(f"[bold yellow]── final answer (stopped: {status}) ──[/bold yellow]")
        else:
            self.console.print("[bold green]── final answer ──[/bold green]")
        if not cell.source.strip():
            self.console.print("[dim](no answer)[/dim]")
        elif not any(cell is printed for printed in self._printed_cells):
            cell.render(self.console)
            self._printed_cells.append(cell)
        self.console.print()

    # ------------------------------------------------------------------ helpers
    def reset_history_tracking(self) -> None:
        self._committed = 0

    def mark_cells_rendered(self, count: int) -> None:
        """Record that the first ``count`` history cells are already on screen."""

        self._committed = max(self._committed, count)

    @property
    def committed_cells(self) -> int:
        return self._committed


def render_prompt_preview(composer_text: str, cursor: int) -> str:
    """Two-line preview used by tests and by ``--plain`` debugging."""

    return UserCell(text=composer_text).text + f"  (cursor={cursor})"


__all__ = ["Renderer", "render_prompt_preview"]
