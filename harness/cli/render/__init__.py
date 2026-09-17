"""Rendering: history cells, the live tail and the terminal."""

from harness.cli.render.legacy import PlainRenderer, RichRenderer
from harness.cli.render.renderer import Renderer, render_prompt_preview
from harness.cli.render.theme import DEFAULT_THEME, GLYPHS, Theme
from harness.cli.render.transcript import TranscriptControl

__all__ = [
    "Renderer",
    "render_prompt_preview",
    "Theme",
    "DEFAULT_THEME",
    "GLYPHS",
    "PlainRenderer",
    "RichRenderer",
    "TranscriptControl",
]
