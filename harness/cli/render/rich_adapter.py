"""Rich adapter for semantic cell presentations."""

from __future__ import annotations

from typing import Any

from harness.cli.presentation import CellPresentation, Line, MarkdownBlock

ROLE_STYLES = {
    "user": "bold green",
    "assistant": "",
    "reasoning": "dim italic",
    "final-marker": "bold green",
    "tool-running": "yellow",
    "tool-running-name": "bold yellow",
    "tool-done": "green",
    "tool-done-name": "bold green",
    "tool-failed": "bold red",
    "tool-failed-name": "bold red",
    "body-prefix": "dim",
    "tool-body": "",
    "subagent": "blue",
    "subagent-name": "bold blue",
    "skill": "magenta",
    "info": "dim",
    "error": "red",
    "error-fatal": "bold red",
    "dim": "dim",
    "text": "",
}


def render_presentation(console: Any, presentation: CellPresentation) -> None:
    """Write a presentation without re-interpreting its display policy."""

    from rich.text import Text

    for block in presentation.blocks:
        if isinstance(block, MarkdownBlock):
            try:
                from rich.markdown import Markdown

                console.print(Markdown(block.source))
            except Exception:  # pragma: no cover - Markdown is best effort
                console.print(block.source)
            continue
        if isinstance(block, Line):
            value = Text()
            for span in block.spans:
                value.append(span.text, style=ROLE_STYLES.get(span.role, ""))
            console.print(value)


__all__ = ["render_presentation"]
