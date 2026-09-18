"""Concrete tool implementations.

Every function here is a *tool*: a plain, typed, synchronous function that
returns a string observation.  The MCP server exposes exactly these functions,
and the in-process runtime calls the same code, so there is a single source of
truth for tool behaviour.

Tool names and signatures are frozen by :data:`TOOL_SPECS`.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from harness.agent.errors import PatchError, ToolError, ToolPermissionError
from harness.tools import patch as patch_mod
from harness.tools.paths import Workspace, is_skipped_dir

TEXT_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".md", ".rst", ".txt", ".sh", ".bash", ".zsh", ".fish",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".java", ".kt", ".go", ".rs", ".rb",
    ".php", ".pl", ".lua", ".sql", ".html", ".htm", ".css", ".scss", ".xml", ".csv",
    ".tsv", ".env", ".gitignore", ".dockerignore", ".mk", ".cmake", ".proto", ".tf",
    ".vue", ".svelte", ".dart", ".swift", ".m", ".mm", ".r", ".jl", ".ipynb", ".lock",
}


def _is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    try:
        chunk = path.open("rb").read(4096)
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    try:
        chunk.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _human_size(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f}{unit}" if unit == "B" else f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}GB"


@dataclass
class ToolContext:
    """Everything a tool needs from its environment."""

    workspace: Workspace
    max_output_chars: int = 30_000
    shell_timeout: int = 120
    shell_max_output_chars: int = 20_000
    command_allowlist: tuple[str, ...] = ()
    command_denylist: tuple[str, ...] = ()


def _truncate(text: str, limit: int, note: str = "output truncated") -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return (
        text[:head]
        + f"\n... [{note}: {len(text) - limit} chars omitted] ...\n"
        + text[-tail:]
    )


# --------------------------------------------------------------------- read_file
def read_file(
    ctx: ToolContext,
    path: str,
    offset: int = 1,
    limit: int = 400,
    show_line_numbers: bool = True,
) -> str:
    """Read a file, or a slice of it, with 1-based inclusive line numbers.

    Output always starts with a header describing the file and the slice so the
    model can tell whether it has seen the whole file.
    """

    target = ctx.workspace.resolve(path, must_exist=True)
    if target.is_dir():
        raise ToolError(f"`{path}` is a directory; use list_dir instead")
    if not _is_probably_text(target):
        return f"{path}: binary file ({_human_size(target.stat().st_size)}); not shown"

    lines, trailing = patch_mod.read_text_lines(target)
    total = len(lines)
    start = max(1, int(offset))
    if limit is None or int(limit) <= 0:
        limit = total or 1
    end = min(total, start + int(limit) - 1)
    if start > total:
        return f"{path}: offset {start} is past the end of the file ({total} lines)"

    rel = ctx.workspace.relative(target)
    header = (
        f"{rel} ({total} lines, {_human_size(target.stat().st_size)}"
        + ("" if trailing or not lines else ", no trailing newline")
        + f") — showing {start}-{end}"
    )
    body_lines: list[str] = []
    width = len(str(end))
    for number in range(start, end + 1):
        text = lines[number - 1]
        if show_line_numbers:
            body_lines.append(f"{number:>{width}}\t{text}")
        else:
            body_lines.append(text)
    body = "\n".join(body_lines)
    footer = ""
    if end < total:
        footer = f"\n... {total - end} more lines (call read_file again with offset={end + 1})"
    return _truncate(f"{header}\n{body}{footer}", ctx.max_output_chars, "file slice truncated")


# --------------------------------------------------------------------- list_dir
def list_dir(ctx: ToolContext, path: str = ".", depth: int = 1, show_hidden: bool = False) -> str:
    """List one directory level (or ``depth`` levels) as an indented tree."""

    root = ctx.workspace.resolve(path, must_exist=True)
    if root.is_file():
        return f"{ctx.workspace.relative(root)} is a file ({_human_size(root.stat().st_size)})"
    depth = max(1, min(int(depth), 4))
    lines: list[str] = []
    entries = 0

    def walk(directory: Path, level: int) -> None:
        nonlocal entries
        try:
            children = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            lines.append(f"{'  ' * level}[permission denied]")
            return
        for child in children:
            if not show_hidden and child.name.startswith("."):
                continue
            if child.is_dir():
                if is_skipped_dir(child.name):
                    continue
                lines.append(f"{'  ' * level}{child.name}/")
                entries += 1
                if level + 1 < depth:
                    walk(child, level + 1)
            else:
                try:
                    size = _human_size(child.stat().st_size)
                except OSError:
                    size = "?"
                lines.append(f"{'  ' * level}{child.name}  ({size})")
                entries += 1

    walk(root, 0)
    rel = ctx.workspace.relative(root) or "."
    if not lines:
        return f"{rel}/ is empty"
    return _truncate(f"{rel}/ ({entries} entries):\n" + "\n".join(lines), ctx.max_output_chars)


# ------------------------------------------------------------------------ glob
def glob(ctx: ToolContext, pattern: str, path: str = ".", max_results: int = 200) -> str:
    """Find files whose path matches a glob pattern (``**/*.py``, ``src/*/test_*.py``).

    The pattern is always relative to the workspace: an absolute pattern would be
    a second way out of the sandbox that the ``path`` guard never sees, so it is
    resolved against the workspace root instead of the filesystem root.
    """

    base = ctx.workspace.resolve(path, must_exist=True)
    if base.is_file():
        base = base.parent
    pattern = (pattern or "**/*").strip()
    max_results = max(1, min(int(max_results), 2000))
    matches: list[str] = []
    truncated = False

    if os.path.isabs(pattern):
        # Treat "/etc/*.conf" as "etc/*.conf" inside the workspace: harmless when
        # the workspace has no such directory, and never a filesystem walk.
        stripped = pattern.lstrip("/")
        if ".." in Path(stripped).parts:
            raise ToolPermissionError(
                f"glob pattern `{pattern}` is absolute and points outside the workspace"
            )
        pattern = stripped or "**/*"
    if ".." in Path(pattern).parts:
        raise ToolPermissionError(
            f"glob pattern `{pattern}` escapes the workspace root `{ctx.workspace.root}`"
        )
    candidates: Iterable[Path] = base.glob(pattern)

    for candidate in candidates:
        if candidate.is_dir():
            continue
        parts = set(candidate.parts)
        if parts & {"__pycache__", "node_modules", ".git", ".venv"}:
            continue
        matches.append(ctx.workspace.relative(candidate))
        if len(matches) >= max_results:
            truncated = True
            break
    matches.sort()
    if not matches:
        return f"no files match `{pattern}` under {ctx.workspace.relative(base) or '.'}"
    header = f"{len(matches)} file(s) match `{pattern}`:"
    note = f"\n... more matches exist (limit {max_results}); narrow the pattern" if truncated else ""
    return _truncate(header + "\n" + "\n".join(matches) + note, ctx.max_output_chars)


# ------------------------------------------------------------------------ grep
def _inside(base: Path, candidate: Path) -> bool:
    """Whether ``candidate`` is under ``base`` *after resolving symlinks*.

    A plain string prefix test is not enough: a symlink inside the workspace
    keeps its in-workspace path while pointing anywhere, so ``grep`` would read
    (and print) files the workspace guard exists to protect.
    """

    try:
        resolved_base = base.resolve()
        resolved = candidate.resolve()
    except OSError:  # pragma: no cover - unreadable path
        return False
    if resolved_base.is_file():
        return resolved == resolved_base
    return resolved == resolved_base or resolved_base in resolved.parents


def grep(
    ctx: ToolContext,
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    ignore_case: bool = False,
    context_lines: int = 0,
    max_results: int = 100,
    fixed_string: bool = False,
) -> str:
    """Search file contents with a regular expression.

    ``glob`` filters candidate files (e.g. ``*.py``); ``context_lines`` adds
    surrounding lines to each hit.
    """

    if not pattern:
        raise ToolError("`pattern` must not be empty")
    base = ctx.workspace.resolve(path, must_exist=True)
    flags = re.IGNORECASE if ignore_case else 0
    try:
        if fixed_string:
            regex = re.compile(re.escape(pattern), flags)
        else:
            regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ToolError(f"invalid regular expression: {exc}") from exc

    max_results = max(1, min(int(max_results), 1000))
    context_lines = max(0, min(int(context_lines), 5))
    files: list[Path] = []
    if base.is_file():
        files = [base]
    else:
        for candidate in ctx.workspace.walk():
            if not _inside(base, candidate):
                continue
            if glob and not (
                fnmatch.fnmatch(candidate.name, glob) or fnmatch.fnmatch(str(candidate), glob)
            ):
                continue
            files.append(candidate)

    hits: list[str] = []
    matched_files = 0
    for file_path in files:
        if len(hits) >= max_results:
            break
        if not _is_probably_text(file_path):
            continue
        try:
            lines, _ = patch_mod.read_text_lines(file_path)
        except (OSError, PatchError):
            continue
        file_hits: list[str] = []
        for index, line in enumerate(lines):
            if regex.search(line):
                start = max(0, index - context_lines)
                end = min(len(lines), index + context_lines + 1)
                for cursor in range(start, end):
                    marker = ":" if cursor == index else "-"
                    file_hits.append(f"{cursor + 1}{marker}{lines[cursor]}")
                if context_lines:
                    file_hits.append("--")
        if file_hits:
            matched_files += 1
            rel = ctx.workspace.relative(file_path)
            hits.append(f"{rel}:\n" + "\n".join(file_hits))
            if len(hits) >= max_results:
                hits.append("... more matches exist; narrow the pattern or add `glob`")
                break

    if not hits:
        scope = ctx.workspace.relative(base) or "."
        suffix = f" in {glob}" if glob else ""
        return f"no matches for `{pattern}`{suffix} under {scope}"
    header = f"{matched_files} file(s) match `{pattern}`:"
    return _truncate(header + "\n\n" + "\n\n".join(hits), ctx.max_output_chars)


# ------------------------------------------------------------------- write_file
def write_file(ctx: ToolContext, path: str, content: str, create_dirs: bool = True) -> str:
    """Create or fully overwrite a file."""

    target = ctx.workspace.resolve(path)
    if target.exists() and target.is_dir():
        raise ToolError(f"`{path}` is a directory")
    if not create_dirs and not target.parent.exists():
        raise ToolError(f"parent directory of `{path}` does not exist")
    existed = target.exists()
    before = len(patch_mod.read_text_lines(target)[0]) if existed else 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    after = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    rel = ctx.workspace.relative(target)
    action = "overwrote" if existed else "created"
    delta = ""
    if existed:
        delta = f" ({after - before:+d} lines)"
    return f"{action} {rel} — {after} lines, {_human_size(len(content.encode('utf-8')))}{delta}"


# ------------------------------------------------------------------ apply_patch
def apply_patch(
    ctx: ToolContext,
    patch: str,
    path: str | None = None,
    dry_run: bool = False,
    atomic: bool = True,
) -> str:
    """Apply a patch to one or more files.

    ``path`` (optional) is only used as the implicit target when the patch body
    carries no file header of its own.
    """

    try:
        if path and not re.search(r"\*\*\*\s*(Update|Add|Delete) File:", patch, re.IGNORECASE):
            patch = f"*** Begin Patch\n*** Update File: {path}\n{patch}\n*** End Patch"
        file_patches = patch_mod.parse_patch(patch)
    except patch_mod.PatchSyntaxError as exc:
        raise PatchError(str(exc)) from exc

    results: list[patch_mod.ApplyResult] = []
    for file_patch in file_patches:
        results.append(_apply_one(ctx, file_patch, dry_run=dry_run, atomic=atomic))

    ok = all(result.ok for result in results)
    added = sum(result.added for result in results)
    removed = sum(result.removed for result in results)
    mode = "dry run — nothing written" if dry_run else "applied"
    header = (
        f"apply_patch {mode}: {sum(1 for r in results if r.ok)}/{len(results)} files ok, "
        f"+{added}/-{removed}"
    )
    body = "\n".join(result.line() for result in results)
    return header + "\n" + body


def _apply_one(
    ctx: ToolContext,
    file_patch: patch_mod.FilePatch,
    *,
    dry_run: bool,
    atomic: bool,
) -> patch_mod.ApplyResult:
    try:
        target = ctx.workspace.resolve(file_patch.path)
    except Exception as exc:  # permission guard etc.
        return patch_mod.ApplyResult(file_patch.path, file_patch.kind, ok=False, error=str(exc))

    if file_patch.kind == "add":
        if target.exists():
            return patch_mod.ApplyResult(
                file_patch.path, "add", ok=False, error="file already exists; use `*** Update File:`"
            )
        lines = list(file_patch.content)
        result = patch_mod.ApplyResult(file_patch.path, "add", ok=True, added=len(lines))
        if not dry_run:
            patch_mod.write_text_lines(target, lines)
        return result

    if file_patch.kind == "delete":
        if not target.exists():
            return patch_mod.ApplyResult(file_patch.path, "delete", ok=False, error="file does not exist")
        old_lines, _ = patch_mod.read_text_lines(target)
        result = patch_mod.ApplyResult(file_patch.path, "delete", ok=True, removed=len(old_lines))
        if not dry_run:
            target.unlink()
        return result

    if not target.exists():
        return patch_mod.ApplyResult(
            file_patch.path, "update", ok=False, error="file does not exist; use `*** Add File:`"
        )
    old_lines, trailing = patch_mod.read_text_lines(target)
    try:
        new_lines, added, removed, fuzzy = patch_mod.apply_hunks(old_lines, file_patch.hunks)
    except PatchError as exc:
        if not atomic:
            return patch_mod.ApplyResult(file_patch.path, "update", ok=False, error=str(exc))
        return patch_mod.ApplyResult(file_patch.path, "update", ok=False, error=str(exc))

    result = patch_mod.ApplyResult(
        file_patch.path,
        "update",
        ok=True,
        added=added,
        removed=removed,
        hunks_applied=len(file_patch.hunks),
        fuzzy_hunks=fuzzy,
    )
    if dry_run:
        return result
    patch_mod.write_text_lines(target, new_lines, trailing_newline=trailing or bool(new_lines))
    if file_patch.move_to:
        try:
            destination = ctx.workspace.resolve(file_patch.move_to)
            destination.parent.mkdir(parents=True, exist_ok=True)
            target.replace(destination)
        except Exception as exc:
            return patch_mod.ApplyResult(
                file_patch.path, "update", ok=False, error=f"patch applied but move failed: {exc}"
            )
    return result


# ------------------------------------------------------------------------ shell
_BLOCKED_SHELL_PATTERNS = (
    r"rm\s+-rf\s+/(?:\s|$)",
    r":\(\)\s*\{.*\}\s*;",  # fork bomb
    r"mkfs\.",
    r"dd\s+if=.*of=/dev/[sh]d",
    r">\s*/dev/[sh]d",
    r"chmod\s+-R\s+777\s+/(?:\s|$)",
    r"shutdown\b",
    r"reboot\b",
)


def shell(
    ctx: ToolContext,
    command: str,
    timeout: int | None = None,
    cwd: str | None = None,
    max_output_chars: int | None = None,
) -> str:
    """Run a shell command from the workspace root and capture its output.

    Intended for ``pytest``, ``python``, ``git`` and similar developer commands.
    """

    if not command or not command.strip():
        raise ToolError("`command` must not be empty")
    for pattern in _BLOCKED_SHELL_PATTERNS:
        if re.search(pattern, command):
            raise ToolError(f"refusing to run a destructive command matching /{pattern}/")
    if ctx.command_denylist and any(word in command for word in ctx.command_denylist):
        raise ToolError(f"command matches the configured denylist: {ctx.command_denylist}")
    if ctx.command_allowlist:
        first = command.strip().split()[0]
        if not any(first == allowed or first.endswith("/" + allowed) for allowed in ctx.command_allowlist):
            raise ToolError(
                f"`{first}` is not in the tool allowlist {list(ctx.command_allowlist)}"
            )

    workdir = ctx.workspace.resolve(cwd or ".", must_exist=True)
    if not workdir.is_dir():
        raise ToolError(f"cwd `{cwd}` is not a directory")
    effective_timeout = int(timeout or ctx.shell_timeout)
    limit = int(max_output_chars or ctx.shell_max_output_chars)

    started = time.monotonic()
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=effective_timeout,
            env=env,
            executable="/bin/bash",
        )
    except subprocess.TimeoutExpired as exc:
        partial_out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        partial_err = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return _truncate(
            f"$ {command}\n[timeout after {effective_timeout}s]\n"
            f"--- partial stdout ---\n{partial_out}\n--- partial stderr ---\n{partial_err}",
            limit,
            "timed-out command output truncated",
        )
    except OSError as exc:
        raise ToolError(f"could not run command: {exc}") from exc

    elapsed = time.monotonic() - started
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    parts = [f"$ {command}", f"exit_code: {completed.returncode}  ({elapsed:.1f}s)"]
    if stdout.strip():
        parts.append("--- stdout ---\n" + stdout.rstrip("\n"))
    if stderr.strip():
        parts.append("--- stderr ---\n" + stderr.rstrip("\n"))
    if not stdout.strip() and not stderr.strip():
        parts.append("(no output)")
    return _truncate("\n".join(parts), limit, "command output truncated")


# --------------------------------------------------------------------- git_diff
def git_diff(
    ctx: ToolContext,
    path: str | None = None,
    staged: bool = False,
    max_lines: int = 400,
    stat_only: bool = False,
) -> str:
    """Show the working-tree diff (``git diff``), optionally for one path."""

    args = ["git", "diff", "--no-color"]
    if staged:
        args.append("--cached")
    if stat_only:
        args.append("--stat")
    if path:
        args.extend(["--", path])
    try:
        completed = subprocess.run(
            args,
            cwd=str(ctx.workspace.root),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolError(f"git diff failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return f"git diff unavailable (exit {completed.returncode}): {detail}"
    output = completed.stdout
    if not output.strip():
        return "no changes" + (" (staged)" if staged else "")
    lines = output.splitlines()
    if len(lines) > max_lines:
        kept = lines[:max_lines]
        return "\n".join(kept) + f"\n... {len(lines) - max_lines} more diff lines (use path= or stat_only=true)"
    return _truncate(output, ctx.max_output_chars)


# ------------------------------------------------------------------ tool specs
@dataclass(frozen=True)
class ToolFn:
    name: str
    description: str
    fn: Callable[..., str]
    parameters: dict[str, Any]


def _obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


_PATH = {"type": "string", "description": "Path relative to the workspace root."}

TOOL_FUNCS: dict[str, ToolFn] = {
    "list_dir": ToolFn(
        name="list_dir",
        description="List the contents of a directory as an indented tree.",
        fn=list_dir,
        parameters=_obj(
            {
                "path": {**_PATH, "default": "."},
                "depth": {"type": "integer", "description": "How many levels to descend (1-4).", "default": 1},
                "show_hidden": {"type": "boolean", "default": False},
            }
        ),
    ),
    "glob": ToolFn(
        name="glob",
        description="Find files by glob pattern, e.g. `**/*.py` or `src/**/test_*.py`.",
        fn=glob,
        parameters=_obj(
            {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. `**/*.py`."},
                "path": {**_PATH, "default": "."},
                "max_results": {"type": "integer", "default": 200},
            },
            ["pattern"],
        ),
    ),
    "grep": ToolFn(
        name="grep",
        description="Search file contents with a regular expression and return matching lines.",
        fn=grep,
        parameters=_obj(
            {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "path": {**_PATH, "default": "."},
                "glob": {"type": "string", "description": "Only search files matching this glob."},
                "ignore_case": {"type": "boolean", "default": False},
                "context_lines": {"type": "integer", "default": 0},
                "max_results": {"type": "integer", "default": 100},
                "fixed_string": {"type": "boolean", "description": "Treat pattern as a literal string.", "default": False},
            },
            ["pattern"],
        ),
    ),
    "read_file": ToolFn(
        name="read_file",
        description=(
            "Read a file (or a line range) with line numbers. Always use offset/limit for "
            "large files instead of reading everything."
        ),
        fn=read_file,
        parameters=_obj(
            {
                "path": {**_PATH},
                "offset": {"type": "integer", "description": "First line to return (1-based).", "default": 1},
                "limit": {"type": "integer", "description": "Maximum number of lines to return.", "default": 400},
                "show_line_numbers": {"type": "boolean", "default": True},
            },
            ["path"],
        ),
    ),
    "write_file": ToolFn(
        name="write_file",
        description="Create a new file or completely overwrite an existing one.",
        fn=write_file,
        parameters=_obj(
            {
                "path": {**_PATH},
                "content": {"type": "string", "description": "Full file content."},
                "create_dirs": {"type": "boolean", "default": True},
            },
            ["path", "content"],
        ),
    ),
    "apply_patch": ToolFn(
        name="apply_patch",
        description=(
            "Edit files with a patch. Prefer this over write_file for changes. Supported forms: "
            "`*** Begin Patch` envelope with `*** Update File: p` / `*** Add File: p` / "
            "`*** Delete File: p` sections and `@@` hunks, a unified diff, or "
            "`<<<<<<< SEARCH / ======= / >>>>>>> REPLACE` blocks under a `*** Update File:` header."
        ),
        fn=apply_patch,
        parameters=_obj(
            {
                "patch": {"type": "string", "description": "The patch text."},
                "path": {"type": "string", "description": "Implicit target file when the patch has no header."},
                "dry_run": {"type": "boolean", "description": "Validate without writing.", "default": False},
                "atomic": {"type": "boolean", "default": True},
            },
            ["patch"],
        ),
    ),
    "shell": ToolFn(
        name="shell",
        description="Run a shell command (pytest, python, git, ...) from the workspace root.",
        fn=shell,
        parameters=_obj(
            {
                "command": {"type": "string", "description": "Command line to execute."},
                "timeout": {"type": "integer", "description": "Seconds before the command is killed."},
                "cwd": {"type": "string", "description": "Directory to run in (relative to workspace root)."},
                "max_output_chars": {"type": "integer"},
            },
            ["command"],
        ),
    ),
    "git_diff": ToolFn(
        name="git_diff",
        description="Show the current uncommitted changes (git diff).",
        fn=git_diff,
        parameters=_obj(
            {
                "path": {"type": "string", "description": "Limit the diff to this path."},
                "staged": {"type": "boolean", "default": False},
                "max_lines": {"type": "integer", "default": 400},
                "stat_only": {"type": "boolean", "default": False},
            }
        ),
    ),
}

ALL_TOOL_NAMES: tuple[str, ...] = tuple(TOOL_FUNCS)

READ_ONLY_TOOLS: tuple[str, ...] = ("list_dir", "glob", "grep", "read_file", "git_diff")


def call_tool(name: str, ctx: ToolContext, arguments: dict[str, Any]) -> str:
    """Dispatch one tool call by name using plain keyword arguments."""

    if name not in TOOL_FUNCS:
        raise ToolError(f"unknown tool `{name}`; available: {', '.join(ALL_TOOL_NAMES)}")
    spec = TOOL_FUNCS[name]
    allowed = set(spec.parameters.get("properties", {}))
    unknown = set(arguments) - allowed
    if unknown:
        raise ToolError(
            f"unexpected argument(s) for `{name}`: {', '.join(sorted(unknown))}; allowed: {', '.join(sorted(allowed))}"
        )
    return spec.fn(ctx, **arguments)


__all__ = [
    "ToolContext",
    "TOOL_FUNCS",
    "ALL_TOOL_NAMES",
    "READ_ONLY_TOOLS",
    "call_tool",
    "read_file",
    "list_dir",
    "glob",
    "grep",
    "write_file",
    "apply_patch",
    "shell",
    "git_diff",
]
