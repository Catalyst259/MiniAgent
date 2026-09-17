"""CLI state objects.

The terminal is only a render target: ``text + cursor`` is the real data, and the
command popup is a *derived* state of the composer text.  See ``CLI_Design.md``
sections 3, 14 and 25.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harness.cli.cells.base import HistoryCell


@dataclass
class TextAreaState:
    """Editable buffer: the single source of truth for user input."""

    text: str = ""
    cursor: int = 0

    # ------------------------------------------------------------------ editing
    def set_text(self, text: str, cursor: int | None = None) -> None:
        self.text = text
        self.cursor = len(text) if cursor is None else max(0, min(cursor, len(text)))

    def insert(self, chunk: str) -> None:
        self.text = self.text[: self.cursor] + chunk + self.text[self.cursor :]
        self.cursor += len(chunk)

    def backspace(self) -> None:
        """Delete one *grapheme* before the cursor, not one UTF-8 byte."""

        if self.cursor <= 0:
            return
        start = _previous_boundary(self.text, self.cursor)
        self.text = self.text[:start] + self.text[self.cursor :]
        self.cursor = start

    def delete(self) -> None:
        if self.cursor >= len(self.text):
            return
        end = _next_boundary(self.text, self.cursor)
        self.text = self.text[: self.cursor] + self.text[end:]

    def delete_word_backward(self) -> None:
        if self.cursor <= 0:
            return
        index = self.cursor
        while index > 0 and self.text[index - 1].isspace():
            index -= 1
        while index > 0 and not self.text[index - 1].isspace():
            index -= 1
        self.text = self.text[:index] + self.text[self.cursor :]
        self.cursor = index

    def move(self, delta: int) -> None:
        self.cursor = max(0, min(self.cursor + delta, len(self.text)))

    def move_home(self) -> None:
        self.cursor = 0

    def move_end(self) -> None:
        self.cursor = len(self.text)

    def clear(self) -> None:
        self.text = ""
        self.cursor = 0

    @property
    def before_cursor(self) -> str:
        return self.text[: self.cursor]

    @property
    def after_cursor(self) -> str:
        return self.text[self.cursor :]

    def render(self, prompt: str = "> ") -> str:
        """Plain two-line rendering of buffer + visual cursor (used by tests)."""

        return f"{prompt}{self.text}\n{' ' * (len(prompt) + self.cursor)}^"


@dataclass
class StreamState:
    """Assistant streaming: raw source plus committed/pending split."""

    source: str = ""
    committed_offset: int = 0
    stable_lines: list[str] = field(default_factory=list)
    tail_lines: list[str] = field(default_factory=list)

    @property
    def committed_source(self) -> str:
        return self.source[: self.committed_offset]

    @property
    def pending_source(self) -> str:
        return self.source[self.committed_offset :]


class CommandMatchLike(Protocol):
    command: Any
    matched_indices: list[int]
    score: int
    order: int


@dataclass
class CommandPopupState:
    """Derived state of the composer text (`/…`)."""

    visible: bool = False
    query: str = ""
    matches: list[Any] = field(default_factory=list)
    selected: int = 0
    scroll: int = 0
    window: int = 6
    dismissed: bool = False

    @property
    def selected_match(self):
        if not self.matches:
            return None
        index = max(0, min(self.selected, len(self.matches) - 1))
        return self.matches[index]

    @property
    def visible_matches(self) -> list[Any]:
        return self.matches[self.scroll : self.scroll + self.window]

    def reset(self) -> None:
        self.visible = False
        self.query = ""
        self.matches = []
        self.selected = 0
        self.scroll = 0
        self.dismissed = False

    def update(
        self,
        query: str,
        matches: list[Any],
        *,
        dismissed: bool = False,
    ) -> None:
        """Re-filter, clamp the selection and keep it inside the window."""

        changed = query != self.query or [m.command.name for m in matches] != [
            m.command.name for m in self.matches
        ]
        self.query = query
        self.matches = matches
        self.dismissed = dismissed
        if changed:
            self.selected = 0
            self.scroll = 0
        if matches:
            self.selected = max(0, min(self.selected, len(matches) - 1))
            self.scroll = max(0, min(self.scroll, max(0, len(matches) - self.window)))
            if self.selected < self.scroll:
                self.scroll = self.selected
            elif self.selected >= self.scroll + self.window:
                self.scroll = self.selected - self.window + 1
        else:
            self.selected = 0
            self.scroll = 0
        self.visible = bool(matches) and not dismissed

    def move(self, delta: int) -> None:
        if not self.matches:
            return
        self.selected = (self.selected + delta) % len(self.matches)
        if self.selected < self.scroll:
            self.scroll = self.selected
        elif self.selected >= self.scroll + self.window:
            self.scroll = self.selected - self.window + 1

    def dismiss(self) -> None:
        self.dismissed = True
        self.visible = False


@runtime_checkable
class Renderable(Protocol):
    def render(self, console: Any) -> None: ...


@dataclass
class AppState:
    """Everything the renderer needs for one frame."""

    composer: TextAreaState = field(default_factory=TextAreaState)
    command_popup: CommandPopupState = field(default_factory=CommandPopupState)
    history_cells: list[HistoryCell] = field(default_factory=list)
    expanded_tool_ids: set[str] = field(default_factory=set)
    activity: str = ""
    transcript_scroll: int | None = None
    active_cell: HistoryCell | None = None
    assistant_stream: StreamState | None = None
    dirty: bool = True
    on_change: Callable[[], None] | None = None
    #: the previously running cell, kept only for debugging/telemetry
    _stale_active: HistoryCell | None = None

    # ------------------------------------------------------------- history cells
    def commit(self, cell: HistoryCell | None = None) -> None:
        """Move the active cell into committed history."""

        target = cell if cell is not None else self.active_cell
        if target is None:
            return
        self.history_cells.append(target)
        if target is self.active_cell:
            self.active_cell = None
        self.touch()

    def set_active(self, cell: HistoryCell) -> None:
        """Point at the cell that is currently running.

        Deliberately does *not* commit the previous cell: the app commits a cell
        exactly once, when its completion event arrives.  Auto-committing here
        is what put the same ToolCell into history twice.
        """

        if self.active_cell is not None and self.active_cell is not cell:
            self._stale_active = self.active_cell
        self.active_cell = cell
        self.touch()

    def clear_active(self) -> None:
        self.active_cell = None
        self.touch()

    def append_cell(self, cell: HistoryCell) -> None:
        self.history_cells.append(cell)
        self.touch()

    def toggle_latest_tool(self) -> None:
        """Toggle the newest completed tool's detailed terminal preview."""

        from harness.cli.cells import ToolCell

        for cell in reversed(self.history_cells):
            if isinstance(cell, ToolCell) and cell.status.value != "running":
                if cell.call_id in self.expanded_tool_ids:
                    self.expanded_tool_ids.remove(cell.call_id)
                else:
                    self.expanded_tool_ids.add(cell.call_id)
                self.touch()
                return

    def touch(self) -> None:
        self.dirty = True
        if self.on_change is not None:
            self.on_change()


def _previous_boundary(text: str, index: int) -> int:
    """Step back one grapheme cluster (combining marks / ZWJ emoji aware)."""

    if index <= 0:
        return 0
    position = index - 1
    # consume trailing combiners
    while position > 0 and _is_combining(text[position]):
        position -= 1
    # consume a joined sequence (emoji ZWJ, variation selectors)
    if position > 0 and text[position - 1] == "\u200d":
        position -= 1
        while position > 0 and _is_combining(text[position]):
            position -= 1
        if position > 0:
            position -= 1
    return position


def _next_boundary(text: str, index: int) -> int:
    if index >= len(text):
        return len(text)
    position = index + 1
    while position < len(text) and _is_combining(text[position]):
        position += 1
    if position < len(text) and text[position] == "\u200d":
        position += 1
        while position < len(text) and _is_combining(text[position]):
            position += 1
    return position


def _is_combining(char: str) -> bool:
    import unicodedata

    return unicodedata.combining(char) != 0 or char in ("\ufe0f", "\ufe0e")


__all__ = [
    "TextAreaState",
    "StreamState",
    "CommandPopupState",
    "AppState",
    "Renderable",
]
