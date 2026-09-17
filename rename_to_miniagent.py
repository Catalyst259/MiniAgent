#!/usr/bin/env python3
"""One-off rename: product identity 'Agent Harness' -> 'MiniAgent'.

Deliberately conservative:
  * the Python package directory stays ``harness/`` (and so does every import),
  * variables/parameters literally named ``harness`` stay,
  * ``Agent_Harness_Design.md`` (the design spec) is never touched.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# exact phrase replacements (order matters: longest first)
PHRASES: list[tuple[str, str]] = [
    ("agent-harness", "miniagent"),
    ("Agent Harness", "MiniAgent"),
    ("agent harness", "MiniAgent"),
    ("Agent harness", "MiniAgent"),
    ("harness-tools", "miniagent-tools"),
    ("mcp:harness-tools", "mcp:miniagent-tools"),
    ("harness.yaml", "miniagent.yaml"),
    (".harness/", ".miniagent/"),
    ("HARNESS_", "MINIAGENT_"),
    ("`harness` event stream", "`MiniAgent` event stream"),
    ("every harness event", "every MiniAgent event"),
    ("the harness", "MiniAgent"),
    ("The harness", "MiniAgent"),
    ("a harness", "a MiniAgent"),
    ("the harness'", "MiniAgent's"),
]

# identifiers/paths that must never be rewritten
PROTECT = [
    "harness.",
    "harness/",
    "harness\")",
    "harness')",
    "AgentHarness",
    "HarnessConfig",
    "harness=",
    "harness,",
    "harness)",
    "harness]",
    "harness }",
    "harness.on_event",
    "harness.gateway",
    "harness._harness_factory",
    "harness.tool_runtime",
    "harness.skill_registry",
    "harness.subagent_registry",
    "harness.context_manager",
    "harness.context_builder",
    "harness.orchestrator",
    "harness.set_gateway",
    "harness.finish",
    "harness.run",
    "harness.ask",
    "harness.thread_id",
    "harness.config",
    "harness.memory",
    "harness.status",
    "harness.model_config",
    "harness.model_name",
    "harness.build",
    "harness.close",
    "harness.support",
    "harness.memory_lines",
]

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".sh", ".cfg", ".ini"}
SKIP_FILES = {"Agent_Harness_Design.md", "rename_to_miniagent.py"}


def rename_text(text: str) -> str:
    for old, new in PHRASES:
        if old not in text:
            continue
        # protect identifiers by temporarily masking them
        masked = text
        for index, token in enumerate(PROTECT):
            masked = masked.replace(token, f"\x00{index}\x00")
        masked = masked.replace(old, new)
        for index, token in enumerate(PROTECT):
            masked = masked.replace(f"\x00{index}\x00", token)
        text = masked
    return text


def main() -> int:
    changed: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in (".venv", ".git", ".harness", ".miniagent", ".pytest_cache"):
            continue
        if path.name in SKIP_FILES or "__pycache__" in relative.parts:
            continue
        original = path.read_text(encoding="utf-8")
        updated = rename_text(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(str(relative))
    print(f"rewrote {len(changed)} file(s)")
    for name in changed:
        print("  ", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
