"""CLI state objects.

The terminal is only a render target: ``text + cursor`` is the real data, and the
command popup is a *derived* state of the composer text.  See ``CLI_Design.md``
sections 3, 14 and 25.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harness.cli.cells.base import HistoryCell


@dataclass
class TextAreaState:
    """Latest prompt_toolkit buffer snapshot used by derived UI state."""

    text: str = ""
    cursor: int = 0

    def set_text(self, text: str, cursor: int | None = None) -> None:
        self.text = text
        self.cursor = len(text) if cursor is None else max(0, min(cursor, len(text)))

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


@dataclass
class AppState:
    """Everything the renderer needs for one frame."""

    composer: TextAreaState = field(default_factory=TextAreaState)
    command_popup: CommandPopupState = field(default_factory=CommandPopupState)
    history_cells: list[HistoryCell] = field(default_factory=list)
    #: cells whose collapsed body/reasoning block is expanded (see ``expand_key``)
    expanded_tool_ids: set[str] = field(default_factory=set)
    activity: str = ""
    active_cell: HistoryCell | None = None
    on_change: Callable[[], None] | None = None
    #: the single user choice currently on screen (approval or model question)
    interaction: Any = None

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

        self.active_cell = cell
        self.touch()

    def append_cell(self, cell: HistoryCell) -> None:
        self.history_cells.append(cell)
        self.touch()

    def toggle_latest_expandable(self) -> None:
        """Expand/collapse the newest cell that has collapsible content.

        That is the newest completed tool body, delegated subagent report, or
        assistant chain of thought.  (Historically this only knew about tools;
        the key is now :func:`harness.cli.cells.base.expand_key`.)
        """

        from harness.cli.cells import AssistantCell, SubAgentCell, ToolCell
        from harness.cli.cells.base import expand_key

        for cell in reversed(self.history_cells):
            expandable = False
            if isinstance(cell, ToolCell):
                expandable = cell.status.value != "running" and bool(cell.body_lines())
            elif isinstance(cell, SubAgentCell):
                expandable = bool(cell.summary)
            elif isinstance(cell, AssistantCell):
                expandable = bool((cell.reasoning or "").strip())
            if not expandable:
                continue
            key = expand_key(cell)
            if key in self.expanded_tool_ids:
                self.expanded_tool_ids.discard(key)
            else:
                self.expanded_tool_ids.add(key)
            self.touch()
            return

    def touch(self) -> None:
        if self.on_change is not None:
            self.on_change()


__all__ = [
    "TextAreaState",
    "StreamState",
    "CommandPopupState",
    "AppState",
]
