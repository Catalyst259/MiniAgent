"""Colours and glyphs used by the CLI renderer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    prompt: str = "bold green"
    user: str = "bold green"
    assistant: str = "white"
    tool_running: str = "yellow"
    tool_done: str = "green"
    tool_failed: str = "bold red"
    subagent: str = "blue"
    skill: str = "magenta"
    info: str = "dim"
    error: str = "bold red"
    banner: str = "bold cyan"


DEFAULT_THEME = Theme()

GLYPHS = {
    "prompt": "›",
    "tool_running": "●",
    "tool_done": "●",
    "tool_failed": "✗",
    "subagent": "⇢",
    "skill": "◆",
    "error": "!!",
    "compact": "◆",
}

__all__ = ["Theme", "DEFAULT_THEME", "GLYPHS", "MARKDOWN_THEME", "build_markdown_console"]


#: Markdown styles for a terminal agent: no background fills.
#:
#: Rich's defaults paint ``markdown.code`` and ``markdown.code_block`` with
#: ``bgcolor="black"``, which shows up as a heavy full-width bar in a dark
#: terminal.  A terminal transcript should stay light: inline code and code
#: blocks get a colour, never a background panel.
MARKDOWN_THEME = {
    "markdown.code": "bold cyan",
    "markdown.code_block": "cyan",
    "markdown.h1": "bold",
    "markdown.h2": "bold",
    "markdown.h3": "bold",
    "markdown.h4": "bold",
    "markdown.h5": "bold",
    "markdown.h6": "bold",
    "markdown.block_quote": "dim",
    "markdown.item.bullet": "dim",
    "markdown.link": "underline cyan",
    "markdown.link_url": "dim underline",
}


def build_markdown_console(**kwargs):
    """A rich Console themed for a terminal agent transcript."""

    from rich.console import Console
    from rich.theme import Theme

    kwargs.setdefault("highlight", False)
    kwargs.setdefault("soft_wrap", False)
    return Console(theme=Theme(MARKDOWN_THEME), **kwargs)
