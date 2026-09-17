"""Legacy event renderers kept for compatibility and for ``--plain`` output.

The interactive front end now uses :class:`harness.cli.render.renderer.Renderer`
(cells + live tail); these two renderers simply translate runtime events into
terminal lines and are still useful for debugging and for tests.
"""

from __future__ import annotations

import sys
from typing import Any

from harness.agent.events import Event


class PlainRenderer:
    """Dependency-free renderer (also used by tests and ``--plain``)."""

    def __call__(self, event: Event) -> None:
        line = self.format(event)
        if line:
            print(line, flush=True)

    # -- same tiny surface as RichRenderer, so the CLI never branches on type --
    def info(self, content: str) -> None:
        print(content, flush=True)

    def error(self, content: str) -> None:
        print(content, file=sys.stderr, flush=True)

    def text(self, content: str) -> None:
        print(content, flush=True)

    def table(self, title: str, rows: list[tuple[str, Any]]) -> None:
        print(f"\n{title}")
        for key, value in rows:
            print(f"  {key}: {value}")

    @staticmethod
    def format(event: Event) -> str:
        kind = event.type
        if kind == "iteration":
            return f"\n── {event.message} ──"
        if kind == "context":
            return f"   context: {event.message}"
        if kind == "token_guard":
            return f"   budget: {event.message}"
        if kind == "compact":
            return f"   compact: {event.message}"
        if kind in ("tool_call", "tool_start"):
            return f"→ {event.message}"
        if kind == "tool_result":
            return f"← {event.message}"
        if kind == "skill_load":
            return f"   skill: {event.message}"
        if kind == "delegate_start":
            return f"⇢ delegate {event.message}"
        if kind == "delegate_end":
            return f"⇠ {event.message}"
        if kind == "memory_write":
            return f"   memory: {event.message}"
        if kind == "assistant_reasoning":
            return f"   (thinking) {event.message}"
        if kind == "terminate":
            return f"   stop: {event.message}"
        if kind == "error":
            return f"!! {event.message}"
        return ""


class RichRenderer:
    """``rich`` renderer for the plain event stream."""

    QUIET = {"context", "token_guard", "memory_write", "memory_recall"}

    COLORS = {
        "iteration": "bold cyan",
        "context": "dim",
        "token_guard": "dim",
        "compact": "yellow",
        "tool_call": "bold green",
        "tool_start": "bold green",
        "tool_result": "green",
        "skill_load": "magenta",
        "delegate_start": "blue",
        "delegate_end": "blue",
        "memory_write": "cyan",
        "assistant_reasoning": "dim italic",
        "terminate": "bold yellow",
        "error": "bold red",
    }
    PREFIX = {
        "tool_call": "→ ",
        "tool_start": "→ ",
        "tool_result": "← ",
        "delegate_start": "⇢ ",
        "delegate_end": "⇠ ",
        "error": "!! ",
        "compact": "◆ ",
    }

    def __init__(self, *, verbose: bool = False) -> None:
        from harness.cli.render.theme import build_markdown_console

        self.console = build_markdown_console()
        self.verbose = verbose

    def __call__(self, event: Event) -> None:
        if event.type in ("assistant_text", "assistant_reasoning"):
            return  # printed separately, with Markdown
        if event.type in self.QUIET and not self.verbose:
            return
        style = self.COLORS.get(event.type, "")
        prefix = self.PREFIX.get(event.type, "  ")
        message = event.message
        if event.type == "iteration":
            self.console.rule(f"[bold cyan]{message}", style="cyan")
            return
        if event.type in ("tool_call", "tool_start"):
            self.console.print(f"[bold green]{prefix}{message}[/bold green]")
            return
        self.console.print(f"[{style}]{prefix}{message}[/{style}]")

    def text(self, content: str) -> None:
        self.console.print()
        self.console.print(content)
        self.console.print()

    def info(self, content: str) -> None:
        self.console.print(f"[dim]{content}[/dim]")

    def error(self, content: str) -> None:
        self.console.print(f"[bold red]{content}[/bold red]")

    def table(self, title: str, rows: list[tuple[str, Any]]) -> None:
        from rich.table import Table

        table = Table(title=title, show_header=False, box=None, pad_edge=False)
        table.add_column(style="bold")
        table.add_column()
        for key, value in rows:
            table.add_row(str(key), str(value))
        self.console.print(table)


__all__ = ["PlainRenderer", "RichRenderer"]
