"""Action abstraction: every tool call becomes one auditable action.

Design document section 4.  An :class:`Action` is what the permission layer
reasons about: a category, a tool name, a human-readable target, and the
metadata a rule needs (the shell program, for instance).

Two tool families never reach :class:`~harness.tools.runtime.ToolRuntime` - the
harness-native ``load_skill`` and ``delegate`` calls are dispatched directly by
the ``act`` node.  They are given action types of their own (``skill`` and
``agent``) so those paths are governed by the same evaluator instead of escaping
the permission layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.agent.dto import ToolCall
from harness.permission import shell as shell_mod

#: Action type -> the tool names that map to it.
_ACTION_TYPES: dict[str, str] = {
    "list_dir": "filesystem",
    "read_file": "filesystem",
    "glob": "filesystem",
    "grep": "filesystem",
    "write_file": "filesystem",
    "apply_patch": "filesystem",
    "git_diff": "filesystem",
    "shell": "shell",
    "load_skill": "skill",
    "delegate": "agent",
    "request_user_input": "interaction",
}

#: Tools that never modify anything; used by the read-only sandbox profile.
_READ_ONLY_TOOLS = frozenset(
    {
        "list_dir",
        "read_file",
        "glob",
        "grep",
        "git_diff",
        "load_skill",
        "delegate",
        "request_user_input",
    }
)

#: How much damage one action can do (design document section 4, ``metadata.risk``).
#: A rule may constrain on it (``{"risk": "high", "permission": "deny"}``), which is
#: the coarse safety net for tools nobody wrote a rule for - including MCP tools
#: this harness has never seen.
RISK_LEVELS = ("low", "medium", "high")

#: Tool risk by category.  Anything not listed is ``medium``: an unknown tool is
#: not assumed harmless.
_RISK_BY_TOOL: dict[str, str] = {
    "list_dir": "low",
    "read_file": "low",
    "glob": "low",
    "grep": "low",
    "git_diff": "low",
    "load_skill": "low",
    "write_file": "medium",
    "apply_patch": "medium",
    "delegate": "medium",
    "request_user_input": "low",
    "shell": "high",
}

#: Risk by action type, used when the tool name is not in the table above.
_RISK_BY_TYPE: dict[str, str] = {
    "filesystem": "medium",
    "shell": "high",
    "skill": "low",
    "agent": "medium",
    "interaction": "low",
    "network": "high",
    "tool": "medium",
}

#: Argument that names the target of each tool, in priority order.  Getting this
#: wrong is a security bug: ``grep`` with ``path='.env'`` reads that file, and
#: ``apply_patch`` names its files only inside the payload.
_TARGET_KEYS: dict[str, tuple[str, ...]] = {
    "list_dir": ("path",),
    "read_file": ("path",),
    "write_file": ("path",),
    "apply_patch": ("path", "patch"),
    "git_diff": ("path",),
    "glob": ("pattern", "path"),
    # grep reads files: a rule about ``.env`` must be able to see the file being
    # searched.  Its ``glob`` filter is also a statement about which files it will
    # open (``glob="*.pem"`` reaches a private key), so it is checked too.
    "grep": ("path", "pattern", "glob"),
    "shell": ("command",),
    "load_skill": ("name",),
    "delegate": ("agent", "task"),
    "request_user_input": ("question",),
}

_DEFAULT_TARGET_KEYS = ("path", "command", "pattern", "name", "agent", "task")


@dataclass(frozen=True)
class Action:
    """One concrete thing an agent wants to do."""

    type: str
    tool: str
    target: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def program(self) -> str:
        return str(self.metadata.get("program") or "")

    @property
    def compound(self) -> bool:
        return bool(self.metadata.get("compound"))

    @property
    def read_only(self) -> bool:
        return self.tool in _READ_ONLY_TOOLS

    @property
    def risk(self) -> str:
        """``low`` / ``medium`` / ``high`` for this action (see :data:`RISK_LEVELS`)."""

        listed = self.metadata.get("risk")
        if isinstance(listed, str) and listed in RISK_LEVELS:
            return listed
        return _RISK_BY_TOOL.get(self.tool) or _RISK_BY_TYPE.get(self.type) or "medium"

    def targets(self) -> list[str]:
        """Every path this action can touch.

        Three cases drive this, all of them "the tool does something the argument
        list does not obviously say":

        * ``apply_patch`` can rewrite *and move* several files in one call, so a
          ``deny`` rule on ``secrets/**`` must see every one of them;
        * ``grep`` names files through its ``glob`` filter as well as its ``path``;
        * ``shell`` can be told to write a file through its own options
          (``git log --output=...``), which no redirect-based check would catch.
        """

        if self.type == "shell":
            parsed = self.shell_command()
            # Writes and targeted reads are both "things the command was told to
            # touch"; the raw command line is the fallback target so rules that
            # match on the command text keep working.
            found = [*shell_mod.writing_targets(parsed.tokens), *shell_mod.reading_targets(parsed.tokens)]
            if found:
                return found
            return [self.target] if self.target else []
        if self.type != "filesystem":
            return [self.target] if self.target else []
        if self.tool != "apply_patch":
            return [value for value in (self.metadata.get(key) for key in _TARGET_KEYS.get(self.tool, ())) if value]

        paths = self.metadata.get("paths")
        if not isinstance(paths, list):
            paths = _patch_paths(self.metadata.get("patch") or "")
            self.metadata["paths"] = paths
        explicit = [value for value in [self.metadata.get("path")] if value]
        merged: list[str] = []
        for candidate in [*paths, *explicit]:
            if candidate and candidate not in merged:
                merged.append(candidate)
        return merged

    def shell_command(self) -> shell_mod.ShellCommand:
        """The parsed command line (parsed once, even though rules re-read it)."""

        parsed = self.metadata.get("_parsed")
        if not isinstance(parsed, shell_mod.ShellCommand):
            parsed = shell_mod.parse(self.target)
            self.metadata["_parsed"] = parsed
        return parsed

    def describe(self) -> str:
        """One line for the approval prompt and the transcript."""

        if self.type == "shell":
            return f"$ {self.target}"
        if self.target:
            return f"{self.tool}({self.target})"
        return self.tool

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "tool": self.tool,
            "target": self.target,
            "metadata": dict(self.metadata),
            "risk": self.risk,
        }


def action_type_for(tool: str) -> str:
    return _ACTION_TYPES.get(tool, "tool")


def from_tool_call(call: ToolCall) -> Action:
    """Convert a model-issued tool call into an :class:`Action`."""

    return from_arguments(call.name, call.arguments or {})


def from_arguments(tool: str, arguments: dict[str, Any]) -> Action:
    """Convert a tool name plus arguments into an :class:`Action`.

    Used for both model tool calls and user ``!command`` shell intents, so the
    two paths are evaluated identically.
    """

    action_type = action_type_for(tool)
    keys = _TARGET_KEYS.get(tool, _DEFAULT_TARGET_KEYS)
    target = ""
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            target = value.strip()
            break

    metadata: dict[str, Any] = {}
    if action_type == "shell":
        parsed = shell_mod.parse(str(arguments.get("command") or ""))
        metadata = {
            "program": parsed.name,
            "tokens": list(parsed.tokens),
            "compound": parsed.compound,
            "compound_reason": parsed.reason,
        }
        target = parsed.raw
    elif action_type == "filesystem":
        # Keep every path-ish argument, not just the chosen target: an
        # ``apply_patch`` carries its file list inside the payload.
        for key in keys:
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                metadata[key] = value.strip()

    return Action(type=action_type, tool=tool, target=target, metadata=metadata)


def _patch_paths(payload: str) -> list[str]:
    """Best-effort file list of an ``apply_patch`` payload.

    Includes ``*** Move to:`` destinations: a move *writes* the destination, so a
    rule protecting that path must see it.  A payload that cannot be parsed yields
    no paths rather than an exception - the patch engine rejects it later, and the
    permission layer must never crash on hostile input.
    """

    if not payload:
        return []
    from harness.tools import patch as patch_mod

    try:
        files = patch_mod.parse_patch(payload)
    except Exception:  # noqa: BLE001 - malformed patches are the tool's problem
        return []

    paths: list[str] = []
    for file in files:
        for candidate in (file.path, getattr(file, "move_to", None), getattr(file, "old_path", None)):
            if candidate and candidate not in paths:
                paths.append(candidate)
    return paths


__all__ = ["Action", "action_type_for", "from_tool_call", "from_arguments"]
