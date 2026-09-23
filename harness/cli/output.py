"""Output adapters for a live transcript and direct console rendering."""
from __future__ import annotations

from typing import Callable

from harness.cli import events as ui
from harness.cli.cells import AssistantCell, ErrorCell, HistoryCell, InfoCell, ToolCell
from harness.cli.render.renderer import Renderer
from harness.cli.state import AppState


class ConsoleOutput:
    def __init__(self, renderer: Renderer) -> None:
        self.renderer = renderer

    def present(self, event: ui.AgentEvent, state: AppState, cell: HistoryCell | None) -> None:
        """Render a transition; AssistantStarted carries the preceding message."""
        renderer = self.renderer
        if isinstance(event, ui.AssistantStarted):
            if isinstance(cell, AssistantCell) and cell.complete and not cell.empty:
                renderer.render_cell(cell)
            renderer.begin_stream()
        elif isinstance(event, ui.AssistantDelta):
            renderer.push_delta(event.text)
        elif isinstance(event, ui.AssistantFinished):
            renderer.end_stream(cell.source if isinstance(cell, AssistantCell) else None)
        elif isinstance(event, (ui.ToolFinished, ui.ToolFailed)):
            if cell is not None:
                renderer.render_cell(cell)
                renderer.mark_cells_rendered(len(state.history_cells))
        elif isinstance(event, (ui.TurnStarted, ui.ToolStarted, ui.SubAgentStarted, ui.SubAgentFinished)):
            if cell is not None:
                renderer.render_cell(cell)
        elif isinstance(event, ui.TurnFinished):
            renderer.end_stream(cell.source if isinstance(cell, AssistantCell) else None)
            if isinstance(cell, AssistantCell) and not cell.empty:
                for history_cell in state.history_cells:
                    if history_cell is not cell and not isinstance(history_cell, ToolCell):
                        renderer.render_cell(history_cell)
                renderer.final_answer(cell, status=event.status)
                renderer.mark_cells_rendered(len(state.history_cells))

    def refresh(self, state: AppState) -> None:
        self.renderer.flush(state.history_cells)

    def notify(self, state: AppState, text: str, *, error: bool = False, line: bool = False) -> None:
        if line:
            self.renderer.print(text)
        elif error:
            self.renderer.error(text)
        else:
            self.renderer.info(text)

    def clear(self) -> None:
        self.renderer.reset_history_tracking()


class TranscriptOutput:
    def __init__(self, invalidate: Callable[[], None]) -> None:
        self.invalidate = invalidate

    def present(self, event: ui.AgentEvent, state: AppState, cell: HistoryCell | None) -> None:
        self.invalidate()

    def refresh(self, state: AppState) -> None:
        self.invalidate()

    def notify(self, state: AppState, text: str, *, error: bool = False, line: bool = False) -> None:
        state.append_cell(ErrorCell(message=text) if error else InfoCell(message=text))
        self.invalidate()

    def clear(self) -> None:
        self.invalidate()
