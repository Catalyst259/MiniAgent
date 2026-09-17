"""Workspace path resolution and the sandbox guard.

Every filesystem tool goes through :class:`Workspace`.  A tool cannot escape the
configured workspace root, not even through ``..`` or a symlink.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from harness.agent.errors import ToolPermissionError

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".harness",
    "dist",
    "build",
    ".idea",
    ".vscode",
}


def is_skipped_dir(name: str) -> bool:
    return name in _SKIP_DIRS


@dataclass
class Workspace:
    """Resolves tool paths against a root and blocks escapes."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()

    def resolve(self, path: str | None = None, *, must_exist: bool = False) -> Path:
        raw = (path or ".").strip() or "."
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # ``strict=False`` resolves the existing prefix, which still catches
        # symlinked escapes for paths that exist.
        resolved = candidate.resolve()
        if not self._inside(resolved):
            raise ToolPermissionError(
                f"path `{raw}` escapes the workspace root `{self.root}`"
            )
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"`{raw}` does not exist")
        return resolved

    def relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:  # pragma: no cover - guarded by resolve()
            return str(path)

    def _inside(self, path: Path) -> bool:
        try:
            path.relative_to(self.root)
        except ValueError:
            return False
        return True

    def walk(self, *, include_hidden: bool = False):
        """Yield files under the root, skipping noise directories."""

        for dirpath, dirnames, filenames in os.walk(self.root):
            current = Path(dirpath)
            dirnames[:] = [
                d
                for d in sorted(dirnames)
                if not is_skipped_dir(d) and (include_hidden or not d.startswith("."))
            ]
            for name in sorted(filenames):
                if not include_hidden and name.startswith("."):
                    continue
                yield current / name


__all__ = ["Workspace", "is_skipped_dir"]
