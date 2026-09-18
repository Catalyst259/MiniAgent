"""The permission decision flow.

Design document section 6, made concrete:

```text
Action
  |
  v
sandbox check      (hard boundary: workspace escape, protected paths)
  |
  v
configured rules   (allow / ask / deny, most restrictive wins)
  |
  v
memory             (once / session / persistent grants)
  |
  v
mode default       (off -> allow, ask -> ASK, auto -> deny)
  |
  v
ALLOW / DENY / ASK
```

Two invariants matter more than the flow itself:

* **The sandbox is checked first and cannot be overridden** - not by a rule, not
  by a remembered grant, and not by the user clicking "Always allow".
* **Memory is only consulted when the rules are silent.**  A remembered grant
  widens what is *asked*, never what is *forbidden*, so adding a safe rule or a
  deny rule always has the effect the operator expects.
"""

from __future__ import annotations

import logging

from harness.permission.action import Action
from harness.permission.decision import Permission, Verdict
from harness.permission.memory import PermissionMemory
from harness.permission.policy import PermissionPolicy

log = logging.getLogger(__name__)


class PermissionEvaluator:
    """Turns an :class:`Action` into a :class:`Verdict`."""

    def __init__(
        self,
        policy: PermissionPolicy,
        memory: PermissionMemory | None = None,
        *,
        can_prompt: bool = True,
    ) -> None:
        self.policy = policy
        self.memory = memory if memory is not None else PermissionMemory()
        #: ``False`` when nothing can answer an ``ASK`` (a subagent, a headless
        #: run).  An unanswerable question is a denial, not a hang.
        self.can_prompt = can_prompt

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, action: Action) -> Verdict:
        if not self.policy.enabled:
            # Even with the layer off, a read-only context stays read-only: the
            # ceiling is a property of the context, not of the rule set.
            ceiling = self.policy.ceiling_check(action)
            if ceiling is not None:
                return ceiling
            return Verdict(
                Permission.ALLOW,
                f"permissions are disabled (mode={self.policy.mode})",
                source="mode",
            )

        sandbox = self.policy.sandbox_check(action)
        if sandbox is not None:
            return sandbox

        rule = self.policy.matcher().match(action)
        if rule is not None:
            if rule.permission is not Permission.ASK:
                # A configured allow or deny is final: memory never widens a
                # deny, and never needs to widen an allow.
                return Verdict(
                    rule.permission,
                    f"rule {rule.describe()}",
                    source="rule",
                    rule=rule,
                )
            # The rules ask a question.  A remembered grant may already answer
            # it - that is the whole point of "Allow this session" - but only an
            # *allow* grant can; a stored deny is honoured too.
            remembered = self._from_memory(action, asking_rule=rule)
            if remembered is not None:
                return remembered
            return self._ask_or_deny(f"rule {rule.describe()}", rule=rule, source="config")

        remembered = self._from_memory(action)
        if remembered is not None:
            return remembered

        return self._default(action, "no rule matched")

    def _ask_or_deny(self, why: str, *, rule=None, source: str = "config") -> Verdict:
        """Produce the ``ASK`` verdict, or a denial when nobody can answer."""

        if self.can_prompt:
            return Verdict(Permission.ASK, why, source=source, rule=rule)
        return Verdict(
            Permission.DENY,
            f"{why}, and no approver can be reached from here "
            "(subagents and headless runs cannot prompt)",
            source="approval",
            rule=rule,
        )

    # -------------------------------------------------------------------- pieces
    def _from_memory(self, action: Action, asking_rule=None) -> Verdict | None:
        found = self.memory.lookup(action)
        if found is None:
            return None
        permission, rule, layer = found

        if permission is Permission.DENY:
            return Verdict(
                Permission.DENY,
                f"denied by a stored {layer} grant",
                source=layer,
                rule=rule,
            )

        if permission is not Permission.ALLOW:
            return None

        if layer == "once":
            # A one-shot grant is spent by the action it was granted for.
            if not self.memory.spend(action):
                return None
            return Verdict(
                Permission.ALLOW,
                "allowed once by the user",
                source="once",
                rule=rule,
                approval="once",
            )

        reason = f"allowed by a stored {layer} grant"
        if asking_rule is not None:
            reason += f", which answers `{asking_rule.describe()}`"
        return Verdict(
            Permission.ALLOW,
            reason,
            source=layer,
            rule=rule,
            approval=layer,
        )

    def _default(self, action: Action, why: str) -> Verdict:
        default = self.policy.default
        if default is Permission.ALLOW:
            return Verdict(Permission.ALLOW, f"{why}; default is allow", source="default")
        if default is Permission.DENY:
            return Verdict(Permission.DENY, f"{why}; default is deny", source="default")

        # default = ask
        if self.policy.mode == "auto":
            return self._ask_or_deny(
                f"{why} and mode=auto", source="mode"
            )
        return self._ask_or_deny(f"{why}; asking the user")

    # ------------------------------------------------------------------- grants
    def remember(self, action: Action, scope: str, rule=None) -> bool:
        """Persist a user approval in the layer named by ``scope``.

        ``scope`` is ``session`` or ``persistent``; an unknown scope returns
        ``False`` so the caller can report that nothing was remembered.  "Allow
        once" is deliberately *not* a scope: it produces a verdict without
        storing anything, so a one-shot approval can never widen later decisions.

        A rejection is not stored either: remembering "no" would silently deny an
        unrelated action later, and DENY rules belong in the config file where
        they are visible and reviewable.
        """

        if scope not in ("session", "persistent"):
            return False
        chosen = rule if rule is not None else self.grant_rule(action)
        if chosen is None:
            return False
        return self.memory.remember(chosen, scope)

    def grant_rule(self, action: Action):
        """The rule that a standing approval should store for ``action``.

        A standing grant is only sound when a rule can *reliably* match the same
        permission again:

        * ``write_file`` -> "always allow writes" is exactly what the user chose;
        * ``shell`` -> the program, never the command line, because a command line
          that chains operators can never be matched by a rule;
        * ``apply_patch`` -> nothing.  A patch is a target-scoped action: it names
          the files it rewrites, and a tool-wide grant from one patch would
          silently authorise every future patch.  Those callers fall back to
          allow-once.
        """

        from harness.permission.rules import Rule

        if action.tool == "apply_patch":
            return None

        kind = action.type
        if kind == "shell":
            program = action.program
            if not program or action.compound:
                return None
            rule = Rule(permission=Permission.ALLOW, tool=action.tool, program=program)
        elif kind in ("filesystem", "skill", "agent", "tool"):
            rule = Rule(permission=Permission.ALLOW, tool=action.tool)
        else:  # pragma: no cover - defensive
            return None

        # Never hand back a grant that would not actually match next time.
        return rule if rule.matches(action) else None

    # -------------------------------------------------------------------- report
    def describe(self) -> dict:
        return {"policy": self.policy.describe(), "memory": self.memory.status()}


__all__ = ["PermissionEvaluator"]
