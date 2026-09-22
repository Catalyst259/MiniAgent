"""Reusable test doubles for CLI tests."""

from __future__ import annotations

from typing import Any

from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput


class RecordingConsole:
    """Captures everything a cell or renderer tries to display."""

    def __init__(self) -> None:
        self.entries: list[Any] = []

    def print(self, *args: Any, **kwargs: Any) -> None:
        for arg in args:
            self.entries.append(arg)

    def info(self, content: str) -> None:
        self.entries.append(content)

    def error(self, content: str) -> None:
        self.entries.append(content)

    def text(self, content: str) -> None:
        self.entries.append(content)

    def banner(self) -> None:
        self.entries.append("banner")

    def rule(self, title: str = "") -> None:
        self.entries.append(f"rule:{title}")

    # ------------------------------------------------------------------ helpers
    def plain(self) -> str:
        return "\n".join(str(entry) for entry in self.entries)

    def rich_text(self) -> str:
        from rich.text import Text

        chunks = []
        for entry in self.entries:
            if isinstance(entry, Text):
                chunks.append(entry.plain)
            else:
                chunks.append(str(entry))
        return "\n".join(chunks)

    def types(self) -> list[str]:
        return [type(entry).__name__ for entry in self.entries]

    def clear(self) -> None:
        self.entries.clear()


class SizedDummyOutput(DummyOutput):
    """Headless terminal with a deterministic viewport size."""

    def __init__(self, *, rows: int = 24, columns: int = 80) -> None:
        self._size = Size(rows=rows, columns=columns)

    def get_size(self) -> Size:
        return self._size


def render_application_rows(app, *, rows: int = 24, columns: int = 80) -> list[str]:
    """Render the real interactive layout once and return its visible rows.

    This is the UI's external test seam: it exercises the same HSplit, controls,
    dimensions and formatted text that a terminal sees, without allocating a
    pseudo-terminal.
    """

    output = SizedDummyOutput(rows=rows, columns=columns)
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            application = app.create_application()
            application.renderer.render(application, application.layout)
            screen = application.renderer._last_screen
            return [
                "".join(screen.data_buffer[y][x].char for x in range(screen.width)).rstrip()
                for y in range(screen.height)
            ]


__all__ = ["RecordingConsole", "SizedDummyOutput", "render_application_rows"]
