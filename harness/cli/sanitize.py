"""Terminal text sanitising.

Tool output is arbitrary text: a command may emit ANSI colour codes, and half a
code sequence rendered verbatim is exactly the "[0m" garbage you see in a
terminal that lost the ESC byte.  Everything that reaches the screen therefore
goes through :func:`strip_ansi` and :func:`safe_text`.
"""

from __future__ import annotations

import re
import unicodedata

# CSI/OSC/other escape sequences plus lone control characters.
_ANSI_RE = re.compile(
    r"""
    \x1b\[[0-?]*[ -/]*[@-~]        # CSI ... final byte
    | \x1b\][^\x07\x1b]*(?:\x07|\x1b\\)  # OSC ... BEL/ST
    | \x1b[@-Z\\-_]                # 2-byte escapes
    | \x1b.                        # any other ESC + char
    """,
    re.VERBOSE,
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (and stray ESC bytes)."""

    if not text:
        return ""
    return _ANSI_RE.sub("", text.replace("\x1b", "\x1b"))


def safe_text(text: str, *, expand_tabs: bool = True, max_chars: int | None = None) -> str:
    """Make arbitrary tool output safe to print.

    * strips ANSI/control characters,
    * normalises line endings and non-breaking spaces,
    * optionally expands tabs so a preview keeps its alignment.
    """

    if not text:
        return ""
    cleaned = strip_ansi(text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00a0", " ").replace("\u2028", "\n").replace("\u2029", "\n")
    cleaned = _CONTROL_RE.sub("", cleaned)
    if expand_tabs:
        cleaned = cleaned.expandtabs(4)
    if max_chars is not None and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "…"
    return cleaned


def display_width(text: str) -> int:
    """Terminal columns of ``text`` (wide CJK counts as two)."""

    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        east = unicodedata.east_asian_width(char)
        width += 2 if east in ("W", "F") else 1
    return width


def clip_to_width(text: str, width: int) -> str:
    """Clip one line to ``width`` terminal columns without splitting a grapheme."""

    if width <= 0:
        return ""
    out: list[str] = []
    used = 0
    for char in text:
        char_width = 0 if unicodedata.combining(char) else (
            2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        )
        if used + char_width > width:
            break
        out.append(char)
        used += char_width
    return "".join(out)


__all__ = ["strip_ansi", "safe_text", "display_width", "clip_to_width"]
