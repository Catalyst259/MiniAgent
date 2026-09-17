"""`apply_patch`: the tool that has to be *really* good.

Three input dialects are accepted, because models emit all three in practice:

1. **Codex-style envelope**

   ```text
   *** Begin Patch
   *** Update File: src/app.py
   @@ def main():
    context line
   -old line
   +new line
   *** Add File: src/new.py
   +print("hi")
   *** Delete File: src/gone.py
   *** End Patch
   ```

2. **Unified diff** (``--- a/x`` / ``+++ b/x`` / ``@@ -1,3 +1,4 @@``).

3. **Search/Replace blocks** (``<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE``).

Matching is exact first, then whitespace-insensitive, then a tolerant
sliding-window match, so a patch still applies when the model got indentation
slightly wrong.  Application is all-or-nothing per file: if any hunk of a file
fails, the file is left untouched and every failure is reported.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

from harness.agent.errors import PatchError

_BEGIN = re.compile(r"^\s*\*\*\*\s*(Begin Patch|Start Patch)\s*$", re.IGNORECASE)
_END = re.compile(r"^\s*\*\*\*\s*(End Patch)\s*$", re.IGNORECASE)
_ADD = re.compile(r"^\s*\*\*\*\s*Add File:\s*(?P<path>.+?)\s*$", re.IGNORECASE)
_DELETE = re.compile(r"^\s*\*\*\*\s*Delete File:\s*(?P<path>.+?)\s*$", re.IGNORECASE)
_UPDATE = re.compile(r"^\s*\*\*\*\s*Update File:\s*(?P<path>.+?)\s*$", re.IGNORECASE)
_MOVE_TO = re.compile(r"^\s*\*\*\*\s*Move to:\s*(?P<path>.+?)\s*$", re.IGNORECASE)
_HUNK = re.compile(r"^\s*@@(?:@)?(?P<header>.*?)(?:@@)?\s*$")
_UNIFIED_FILE = re.compile(r"^---\s+(?P<old>\S+)(?:\s+.*)?$")
_UNIFIED_NEW = re.compile(r"^\+\+\+\s+(?P<new>\S+)(?:\s+.*)?$")
_SR_SEARCH = re.compile(r"^\s*<{5,}\s*SEARCH\s*$", re.IGNORECASE)
_SR_SEP = re.compile(r"^\s*={5,}\s*$")
_SR_REPLACE = re.compile(r"^\s*>{5,}\s*REPLACE\s*$", re.IGNORECASE)


class PatchSyntaxError(PatchError):
    pass


@dataclass
class Hunk:
    old_lines: list[str]
    new_lines: list[str]
    header: str = ""
    context_before: int = 0
    exact: bool = True


@dataclass
class FilePatch:
    path: str
    kind: str  # "update" | "add" | "delete"
    hunks: list[Hunk] = field(default_factory=list)
    content: list[str] = field(default_factory=list)  # for "add"
    move_to: str | None = None
    old_path: str | None = None  # for unified diffs


@dataclass
class ApplyResult:
    path: str
    kind: str
    ok: bool
    added: int = 0
    removed: int = 0
    hunks_applied: int = 0
    fuzzy_hunks: int = 0
    error: str | None = None

    def line(self) -> str:
        if not self.ok:
            return f"  ✗ {self.path}: {self.error}"
        if self.kind == "add":
            return f"  + {self.path} (new file, {self.added} lines)"
        if self.kind == "delete":
            return f"  - {self.path} (deleted, {self.removed} lines)"
        fuzzy = f", {self.fuzzy_hunks} fuzzy" if self.fuzzy_hunks else ""
        return f"  ~ {self.path} (+{self.added}/-{self.removed}, {self.hunks_applied} hunks{fuzzy})"


# --------------------------------------------------------------------- parsing
def parse_patch(text: str) -> list[FilePatch]:
    """Parse any supported dialect into a list of :class:`FilePatch`."""

    if not text or not text.strip():
        raise PatchSyntaxError("empty patch")
    lines = text.splitlines()
    if any(_SR_SEARCH.match(line) for line in lines):
        patches = _parse_search_replace(lines)
        if patches:
            return patches
    if any(_UNIFIED_FILE.match(line) for line in lines) and any(
        _UNIFIED_NEW.match(line) for line in lines
    ):
        patches = _parse_unified(lines)
        if patches:
            return patches
    return _parse_envelope(lines)


def _strip_envelope(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    for index, line in enumerate(lines):
        if _BEGIN.match(line):
            start = index + 1
            break
    for index in range(len(lines) - 1, start - 1, -1):
        if _END.match(lines[index]):
            end = index
            break
    return lines[start:end]


def _parse_envelope(lines: list[str]) -> list[FilePatch]:
    body = _strip_envelope(lines)
    patches: list[FilePatch] = []
    current: FilePatch | None = None
    mode: str | None = None  # "hunk" | "content" | None
    hunk_buffer: list[str] = []
    content_buffer: list[str] = []
    header = ""

    def flush() -> None:
        nonlocal hunk_buffer, content_buffer, header
        if current is not None:
            if hunk_buffer or header:
                old_lines, new_lines = _split_hunk(hunk_buffer)
                current.hunks.append(Hunk(old_lines=old_lines, new_lines=new_lines, header=header))
            if content_buffer:
                current.content.extend(content_buffer)
        hunk_buffer = []
        content_buffer = []
        header = ""

    for raw in body:
        match_add = _ADD.match(raw)
        match_delete = _DELETE.match(raw)
        match_update = _UPDATE.match(raw)
        match_move = _MOVE_TO.match(raw)
        match_hunk = _HUNK.match(raw)

        if match_add or match_delete or match_update:
            flush()
            if match_add:
                current = FilePatch(path=match_add.group("path"), kind="add")
                mode = "content"
            elif match_delete:
                current = FilePatch(path=match_delete.group("path"), kind="delete")
                mode = None
            else:
                current = FilePatch(path=match_update.group("path"), kind="update")
                mode = None
            patches.append(current)
            continue

        if match_move is not None:
            if current is None:
                raise PatchSyntaxError("`*** Move to:` without a preceding `*** Update File:`")
            current.move_to = match_move.group("path")
            continue

        if match_hunk is not None:
            if hunk_buffer or header:
                old_lines, new_lines = _split_hunk(hunk_buffer)
                if current is not None:
                    current.hunks.append(
                        Hunk(old_lines=old_lines, new_lines=new_lines, header=header)
                    )
                hunk_buffer = []
            if current is None:
                raise PatchSyntaxError("hunk found before any `*** Update File:` header")
            mode = "hunk"
            header = (match_hunk.group("header") or "").strip()
            continue

        if current is None:
            if raw.strip():
                raise PatchSyntaxError(f"unexpected line outside of any file section: {raw!r}")
            continue

        if current.kind == "add":
            mode = "content"
            content_buffer.append(raw[1:] if raw[:1] == "+" else raw)
            continue

        if raw.startswith("\\"):  # "\ No newline at end of file"
            continue

        if mode != "hunk":
            if not raw.strip():
                continue
            # Tolerate a missing `@@` header: start a hunk implicitly.
            mode = "hunk"
        hunk_buffer.append(raw)

    flush()

    if not patches:
        raise PatchSyntaxError("no file sections found; expected `*** Update File:` / `*** Add File:`")
    return [p for p in patches if p.kind == "add" or p.hunks or p.kind == "delete"]


def _split_hunk(buffer: list[str]) -> tuple[list[str], list[str]]:
    old_lines: list[str] = []
    new_lines: list[str] = []
    for line in buffer:
        if not line:
            old_lines.append("")
            new_lines.append("")
            continue
        marker, rest = line[0], line[1:]
        if marker == "-":
            old_lines.append(rest)
        elif marker == "+":
            new_lines.append(rest)
        elif marker in (" ", "\t"):
            old_lines.append(rest if marker == " " else line)
            new_lines.append(rest if marker == " " else line)
        else:
            # No marker: treat the whole line as shared context.
            old_lines.append(line)
            new_lines.append(line)
    return old_lines, new_lines


def _parse_unified(lines: list[str]) -> list[FilePatch]:
    patches: list[FilePatch] = []
    index = 0
    while index < len(lines):
        old_match = _UNIFIED_FILE.match(lines[index])
        if not old_match or index + 1 >= len(lines):
            index += 1
            continue
        new_match = _UNIFIED_NEW.match(lines[index + 1])
        if not new_match or new_match.group("new") == "/dev/null" and old_match.group("old") == "/dev/null":
            index += 1
            continue
        old_path = _clean_path(old_match.group("old"))
        new_path = _clean_path(new_match.group("new"))
        kind = "update"
        path = new_path
        if old_path == "/dev/null":
            kind = "add"
            patch = FilePatch(path=new_path, kind="add")
        elif new_path == "/dev/null":
            kind = "delete"
            patch = FilePatch(path=old_path, kind="delete", old_path=old_path)
        else:
            patch = FilePatch(path=path, kind=kind, old_path=old_path)
        index += 2
        buffer: list[str] = []
        header = ""
        while index < len(lines):
            line = lines[index]
            if _UNIFIED_FILE.match(line) and index + 1 < len(lines) and _UNIFIED_NEW.match(lines[index + 1]):
                break
            if line.startswith("@@"):
                if buffer:
                    old_lines, new_lines = _split_hunk(buffer)
                    patch.hunks.append(Hunk(old_lines=old_lines, new_lines=new_lines, header=header))
                    buffer = []
                header = line.strip()
                index += 1
                continue
            if line.startswith("\\"):
                index += 1
                continue
            if line.startswith("diff --git") or line.startswith("index "):
                index += 1
                continue
            buffer.append(line)
            index += 1
        if buffer:
            old_lines, new_lines = _split_hunk(buffer)
            patch.hunks.append(Hunk(old_lines=old_lines, new_lines=new_lines, header=header))
        patches.append(patch)
    return patches


def _clean_path(path: str) -> str:
    if path in ("/dev/null", "dev/null"):
        return "/dev/null"
    cleaned = path.strip().strip('"')
    for prefix in ("a/", "b/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned


def _parse_search_replace(lines: list[str]) -> list[FilePatch]:
    """Minimal dialect for blocks without a file header.

    When a path appears in a preceding ``*** Update File:`` / ``### path`` /
    ``File: path`` line, the block is attached to it; otherwise the caller gets
    a syntax error telling it to include a file header.
    """

    patches: list[FilePatch] = []
    current: FilePatch | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        for pattern, kind in ((_UPDATE, "update"), (_DELETE, "delete"), (_ADD, "add")):
            match = pattern.match(line)
            if match:
                current = FilePatch(path=match.group("path"), kind=kind)
                patches.append(current)
                break
        else:
            loose = re.match(r"^\s*(?:###|File:)\s*(?P<path>.+?)\s*$", line, re.IGNORECASE)
            if loose:
                current = FilePatch(path=loose.group("path"), kind="update")
                patches.append(current)
        if _SR_SEARCH.match(line):
            if current is None:
                raise PatchSyntaxError(
                    "search/replace block found without a file header; add `*** Update File: <path>` first"
                )
            index += 1
            old_lines: list[str] = []
            while index < len(lines) and not _SR_SEP.match(lines[index]):
                old_lines.append(lines[index])
                index += 1
            if index >= len(lines):
                raise PatchSyntaxError("search/replace block is missing its `=======` separator")
            index += 1
            new_lines: list[str] = []
            while index < len(lines) and not _SR_REPLACE.match(lines[index]):
                new_lines.append(lines[index])
                index += 1
            if index >= len(lines):
                raise PatchSyntaxError("search/replace block is missing its `>>>>>>> REPLACE` terminator")
            current.hunks.append(Hunk(old_lines=old_lines, new_lines=new_lines, header="search/replace"))
        index += 1
    return [p for p in patches if p.hunks or p.kind != "update"]


# ------------------------------------------------------------------- matching
def _normalize(line: str) -> str:
    return " ".join(line.split())


def _find_match(haystack: list[str], needle: list[str], *, hint: int = 0) -> tuple[int, bool] | None:
    """Locate ``needle`` inside ``haystack``.

    Returns ``(start_index, exact)`` or ``None``.  Exact match wins; otherwise a
    whitespace-insensitive comparison is used, preferring the occurrence closest
    to ``hint``.
    """

    if not needle:
        return None
    size = len(needle)

    exact_hits = [
        start
        for start in range(0, len(haystack) - size + 1)
        if haystack[start : start + size] == needle
    ]
    if exact_hits:
        return min(exact_hits, key=lambda start: abs(start - hint)), True

    normalized_needle = [_normalize(line) for line in needle]
    fuzzy_hits = []
    for start in range(0, len(haystack) - size + 1):
        window = [_normalize(line) for line in haystack[start : start + size]]
        if window == normalized_needle:
            fuzzy_hits.append(start)
    if fuzzy_hits:
        return min(fuzzy_hits, key=lambda start: abs(start - hint)), False

    # Last resort: locate a contiguous block whose *content* matches after
    # ignoring empty lines (models often drop blank context lines).
    compact_needle = [n for n in normalized_needle if n]
    if len(compact_needle) >= 2:
        compact_hay = [_normalize(line) for line in haystack]
        for start in range(0, len(haystack)):
            cursor = start
            matched = 0
            for expected in compact_needle:
                while cursor < len(compact_hay) and not compact_hay[cursor]:
                    cursor += 1
                if cursor < len(compact_hay) and compact_hay[cursor] == expected:
                    matched += 1
                    cursor += 1
                else:
                    break
            if matched == len(compact_needle) and cursor - start <= size + 4:
                return start, False
    return None


def _count_delta(old: list[str], new: list[str]) -> tuple[int, int]:
    """Count real ``+``/``-`` lines inside a hunk (shared context excluded)."""

    prefix = 0
    limit = min(len(old), len(new))
    while prefix < limit and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < limit - prefix
        and old[len(old) - 1 - suffix] == new[len(new) - 1 - suffix]
    ):
        suffix += 1
    added = len(new) - prefix - suffix
    removed = len(old) - prefix - suffix
    return max(0, added), max(0, removed)


def apply_hunks(lines: list[str], hunks: list[Hunk]) -> tuple[list[str], int, int, int]:
    """Apply hunks to ``lines``; return ``(new_lines, added, removed, fuzzy)``."""

    result = list(lines)
    added = removed = fuzzy = 0
    cursor = 0
    for position, hunk in enumerate(hunks):
        if not hunk.old_lines and not hunk.new_lines:
            continue
        match = _find_match(result, hunk.old_lines, hint=cursor) if hunk.old_lines else None
        if hunk.old_lines and match is None:
            snippet = "\n".join(f"    {line}" for line in hunk.old_lines[:6])
            raise PatchError(
                f"hunk {position + 1} ({hunk.header or 'no header'}) did not match the file; "
                f"searched for:\n{snippet}"
            )
        if not hunk.old_lines:
            start = min(cursor, len(result))
        else:
            start, exact = match  # type: ignore[misc]
            if not exact:
                fuzzy += 1
        end = start + len(hunk.old_lines)
        result[start:end] = list(hunk.new_lines)
        cursor = start + len(hunk.new_lines)
        delta_added, delta_removed = _count_delta(hunk.old_lines, hunk.new_lines)
        added += delta_added
        removed += delta_removed
    return result, added, removed, fuzzy


# -------------------------------------------------------------------- applying
def apply_to_lines(
    path: Path,
    lines: list[str],
    hunks: list[Hunk],
    *,
    atomic: bool = True,
) -> list[str]:
    try:
        new_lines, _added, _removed, _fuzzy = apply_hunks(lines, hunks)
    except PatchError:
        if atomic:
            raise
        return lines
    return new_lines


def preview_diff(path: str, old_lines: list[str], new_lines: list[str], limit: int = 60) -> str:
    diff = list(
        difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="")
    )
    if len(diff) > limit:
        return "\n".join(diff[:limit] + [f"... ({len(diff) - limit} more diff lines)"])
    return "\n".join(diff)


def read_text_lines(path: Path) -> tuple[list[str], bool]:
    """Read a file into lines.  Returns ``(lines, had_trailing_newline)``."""

    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise PatchError(f"`{path}` looks like a binary file; refusing to patch it")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="surrogateescape")
    trailing = text.endswith("\n")
    lines = text.split("\n")
    if trailing:
        lines = lines[:-1]
    return lines, trailing


def write_text_lines(path: Path, lines: list[str], trailing_newline: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines)
    if trailing_newline and lines:
        body += "\n"
    path.write_text(body, encoding="utf-8", errors="surrogateescape")


__all__ = [
    "FilePatch",
    "Hunk",
    "ApplyResult",
    "PatchSyntaxError",
    "parse_patch",
    "apply_hunks",
    "apply_to_lines",
    "preview_diff",
    "read_text_lines",
    "write_text_lines",
]
