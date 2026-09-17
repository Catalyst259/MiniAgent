"""Tool streaming: raw log semantics, no Markdown.

Tool output is not prose, so it has its own pipeline (``CLI_Design.md`` section
23): deltas are appended verbatim to the active :class:`ToolCell`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.cli.cells.base import ToolCell, ToolStatus


@dataclass
class ToolStream:
    """Owns the currently running tool cells, keyed by call id."""

    cells: dict[str, ToolCell] = field(default_factory=dict)

    # ------------------------------------------------------------------- writing
    def start(self, call_id: str, tool: str, arguments: dict | None = None) -> ToolCell:
        cell = ToolCell(call_id=call_id, tool=tool, arguments=dict(arguments or {}))
        self.cells[call_id] = cell
        return cell

    def append(self, call_id: str, text: str) -> ToolCell | None:
        cell = self.cells.get(call_id)
        if cell is None:
            return None
        cell.append(text)
        return cell

    def finish(
        self,
        call_id: str,
        *,
        ok: bool = True,
        text: str = "",
        duration_ms: int = 0,
        error: str = "",
    ) -> ToolCell | None:
        cell = self.cells.get(call_id)
        if cell is None:
            return None
        if error:
            cell.fail(error)
        else:
            cell.finish(ok, text=text, duration_ms=duration_ms)
        self.cells.pop(call_id, None)
        return cell

    # ------------------------------------------------------------------ reading
    def running(self) -> list[ToolCell]:
        return [cell for cell in self.cells.values() if cell.status is ToolStatus.RUNNING]


__all__ = ["ToolStream"]
