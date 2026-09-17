"""Reusable test doubles for CLI tests."""

from __future__ import annotations

from typing import Any


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


__all__ = ["RecordingConsole"]
