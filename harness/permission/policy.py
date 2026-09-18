"""The permission policy: sandbox capability, approval policy and rule set.

Design document section 3.  The three layers are evaluated in one fixed order:

1. **Sandbox** - the hard boundary.  Nothing an agent does can talk its way past
   this, and no user approval can either.  In MiniAgent the filesystem boundary
   is already implemented by :class:`harness.tools.paths.Workspace`; this layer
   makes it an explicit permission verdict instead of an exception.
2. **Rules** - the configured allow/ask/deny statements.
3. **Mode** - what to do when neither has an opinion (``off`` / ``ask`` / ``auto``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.permission.action import Action
from harness.permission.decision import Permission, Verdict
from harness.permission.rules import Rule, RuleMatcher, parse_rules
from harness.tools.paths import Workspace

log = logging.getLogger(__name__)

#: How to treat an action that no rule mentions.
MODES = ("off", "ask", "auto")


@dataclass
class PermissionPolicy:
    """Configured permissions plus the sandbox boundary they apply within."""

    mode: str = "off"
    default: Permission = Permission.ASK
    rules: list[Rule] = field(default_factory=list)
    workspace: Workspace | None = None
    #: ``filesystem`` actions the sandbox refuses outright, evaluated *after*
    #: the workspace check and applied to reads as well as writes.
    sandbox_deny: list[Rule] = field(default_factory=list)
    #: tool names refused before any rule is consulted
    denied_tools: tuple[str, ...] = ()
    #: when set, only read-only actions are permitted at all.  Subagents run with
    #: this on: their tool list already omits the write tools, and this is the
    #: second layer that holds even when the permission layer is otherwise off.
    read_only: bool = False
    #: The tools the active skill declares it needs.  A skill can only *narrow*
    #: what is permitted (design document section 12: ``Skill Permission <= Tool
    #: Permission``); it can never grant a tool the policy would otherwise refuse.
    active_skill_tools: tuple[str, ...] = ()
    active_skill_name: str = ""

    def __post_init__(self) -> None:
        mode = (self.mode or "off").strip().lower()
        if mode not in MODES:
            raise ValueError(f"unknown permissions.mode `{self.mode}`; expected one of {', '.join(MODES)}")
        self.mode = mode
        if mode == "auto" and self.default is Permission.ALLOW:
            # Every un-matched action runs silently.  That is not "auto-deny";
            # spelling it out keeps the operator from believing they are covered.
            log.warning(
                "permissions: mode=auto with default=allow means an action no rule "
                "mentions is ALLOWED. Use default=deny (or ask) for a closed posture."
            )

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def has_opinion(self) -> bool:
        return bool(self.rules)

    def matcher(self) -> RuleMatcher:
        return RuleMatcher(self.rules)

    def apply_skill_ceiling(self, tools: Iterable[str], name: str = "") -> None:
        """Narrow the policy to the tools one loaded skill declares.

        An empty declaration clears the ceiling: a skill that names no tools makes
        no promise about what it needs, so it must not silently lock the agent out
        of everything else.
        """

        cleaned = tuple(str(tool).strip() for tool in tools or () if str(tool).strip())
        self.active_skill_tools = cleaned
        self.active_skill_name = name if cleaned else ""

    def skill_ceiling(self) -> dict[str, Any]:
        return {"name": self.active_skill_name, "tools": list(self.active_skill_tools)}

    # ------------------------------------------------------------------- sandbox
    def ceiling_check(self, action: Action) -> Verdict | None:
        """The constraints that hold in *every* mode, or ``None``.

        ``denied_tools``, the read-only ceiling, the workspace boundary and the
        active skill's tool ceiling are properties of the context, not configured
        rules: they are enforced even when the permission layer is switched off,
        so a subagent can never write just because ``mode: off`` was chosen.
        """

        if action.tool in self.denied_tools:
            return Verdict(
                Permission.DENY,
                f"tool `{action.tool}` is disabled by configuration",
                source="sandbox",
            )

        if self.read_only and not action.read_only:
            return Verdict(
                Permission.DENY,
                f"`{action.tool}` is not a read-only action and this context is "
                "restricted to reads",
                source="sandbox",
            )

        if self.active_skill_tools and action.tool not in self.active_skill_tools:
            # Loading a skill must not be a way to reach tools its author never
            # asked for; the skill's list is a ceiling, not a permission slip.
            # ``load_skill`` itself stays available so the model can switch to
            # another skill (or read the one already active) instead of deadlocking.
            if action.tool != "load_skill":
                label = (
                    f"skill `{self.active_skill_name}`"
                    if self.active_skill_name
                    else "the active skill"
                )
                return Verdict(
                    Permission.DENY,
                    f"`{action.tool}` is not among the tools {label} declares "
                    f"({', '.join(self.active_skill_tools)})",
                    source="skill",
                )

        return self._path_check(action)

    def _path_check(self, action: Action) -> Verdict | None:
        """Inspect every path the action can reach, whatever kind of action it is.

        This deliberately covers *shell* commands too: a prefix rule approves a
        command line, but a command line can carry a path (``git log
        --output=/tmp/x`` writes outside the workspace, ``grep glob='*.pem'``
        reads a private key).  Checking only ``filesystem`` actions was the single
        largest hole in the layer, because a tool's real side effect rarely lives
        in the argument a rule matched on.
        """

        for target in action.targets():
            if not target:
                continue
            escape = self._outside(target)
            if escape:
                return Verdict(Permission.DENY, escape, source="sandbox")
        return None

    def sandbox_check(self, action: Action) -> Verdict | None:
        """The hard boundary, or ``None`` when the action is inside it.

        Returns a ``DENY`` verdict rather than raising, so the same code serves
        the model-facing tool path (which turns it into a tool error) and the
        audit path (which records it).
        """

        ceiling = self.ceiling_check(action)
        if ceiling is not None:
            return ceiling

        targets = action.targets() or ([action.target] if action.target else [])
        for target in targets:
            if not target:
                continue
            for rule in self.sandbox_deny:
                if rule.matches(action):
                    return Verdict(
                        Permission.DENY,
                        f"`{target}` is a protected path (rule: {rule.describe()})",
                        source="sandbox",
                        rule=rule,
                    )
        return None

    def _outside(self, target: str) -> str | None:
        """Why ``target`` is outside the workspace, or ``None``."""

        if self.workspace is None:
            return None
        if Path(target).expanduser().is_absolute():
            # ``Workspace.resolve`` would accept an absolute path that happens to
            # sit inside the root; anything absolute was not workspace-relative to
            # begin with, so refuse it here as well.
            return f"`{target}` is an absolute path outside the agent's workspace"
        try:
            self.workspace.resolve(target)
        except Exception as exc:  # ToolPermissionError and friends
            return f"`{target}` escapes the workspace root: {exc}"
        return None

    # -------------------------------------------------------------------- report
    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "default": self.default.value,
            "rules": len(self.rules),
            "sandbox_deny": len(self.sandbox_deny),
            "workspace": str(self.workspace.root) if self.workspace else None,
        }

    # ------------------------------------------------------------------ building
    @classmethod
    def from_config(cls, config: Any, *, workspace: Workspace | None = None) -> "PermissionPolicy":
        """Build a policy from a :class:`harness.infra.config.PermissionsConfig`."""

        permissions = getattr(config, "permissions", None)
        if permissions is None:  # pragma: no cover - defensive
            return cls(workspace=workspace)

        raw_rules: list[Any] = []
        for entry in getattr(permissions, "rules", []) or []:
            raw_rules.append(entry)
        pure = {
            Permission.ALLOW: getattr(permissions, "allow", []) or [],
            Permission.ASK: getattr(permissions, "ask", []) or [],
            Permission.DENY: getattr(permissions, "deny", []) or [],
        }
        rules: list[Rule] = []
        for permission, entries in pure.items():
            # A bare string in these lists names a *tool*, not a path.
            rules.extend(parse_rules(entries, permission=permission))
        rules.extend(parse_rules(raw_rules))

        # ``protected_paths`` holds path globs, so these rules are built directly:
        # passing a bare string through ``parse_rules`` would turn ``.env`` into a
        # tool named ".env" and protect nothing.
        protected_globs = getattr(permissions, "protected_paths", []) or []
        protected = [
            Rule(permission=Permission.DENY, target=str(glob).strip())
            for glob in protected_globs
            if str(glob).strip()
        ]

        default_raw = str(getattr(permissions, "default", "ask") or "ask").strip().lower()
        default = {
            "allow": Permission.ALLOW,
            "ask": Permission.ASK,
            "deny": Permission.DENY,
        }.get(default_raw, Permission.ASK)

        return cls(
            mode=getattr(permissions, "mode", "off"),
            default=default,
            rules=rules,
            workspace=workspace,
            sandbox_deny=protected,
            denied_tools=tuple(getattr(permissions, "denied_tools", []) or []),
        )


__all__ = ["PermissionPolicy", "MODES"]
