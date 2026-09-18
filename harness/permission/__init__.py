"""Permission layer: one entry point for every action an agent takes.

Design document sections 1-16.  The layer is deliberately independent of the
orchestrator: it imports ``harness.agent`` (DTOs), ``harness.tools.paths`` (the
workspace boundary) and ``harness.infra.config`` (the config section) only, so
the dependency direction stays ``orchestration -> permission -> agent/tools``.

```text
ToolCall / `!command`
        |
        v
  Action.from_tool_call        harness/permission/action.py
        |
        v
  PermissionEvaluator          harness/permission/evaluator.py
        +------------------+
        | PermissionPolicy |  sandbox + rules + mode
        | PermissionMemory |  once / session / persistent
        | ApprovalProvider |  Allow once | session | always | reject
        +------------------+
        |
        v
  PermissionGate               harness/permission/gate.py
        |
        v
  allow -> execute | deny -> tool error
```

The two entry points most callers need are :func:`build_permission_stack` (wire
everything from config) and :class:`~harness.permission.gate.PermissionGate`
(decide a batch of calls).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.permission.action import Action, action_type_for, from_arguments, from_tool_call
from harness.permission.approval import (
    SCOPE_LABELS,
    SCOPES,
    Approval,
    ApprovalProvider,
    ApprovalRequest,
    AutoAllowProvider,
    AutoDenyProvider,
    ScriptedProvider,
)
from harness.permission.decision import Permission, Verdict, most_restrictive
from harness.permission.evaluator import PermissionEvaluator
from harness.permission.gate import GateResult, PermissionGate, combine_calls
from harness.permission.memory import (
    DEFAULT_PERSISTENT_PATH,
    PersistentMemory,
    PermissionMemory,
    SessionMemory,
    TemporaryMemory,
)
from harness.permission.policy import MODES, PermissionPolicy
from harness.permission.rules import Rule, RuleMatcher, parse_rules

__all__ = [
    # entry points
    "build_permission_stack",
    "PermissionStack",
    "PermissionGate",
    "GateResult",
    "combine_calls",
    # model
    "Action",
    "action_type_for",
    "from_arguments",
    "from_tool_call",
    "Permission",
    "Verdict",
    "most_restrictive",
    "Rule",
    "RuleMatcher",
    "parse_rules",
    # decision making
    "PermissionPolicy",
    "PermissionEvaluator",
    "PermissionMemory",
    "SessionMemory",
    "TemporaryMemory",
    "PersistentMemory",
    "MODES",
    "DEFAULT_PERSISTENT_PATH",
    # approval
    "Approval",
    "ApprovalProvider",
    "ApprovalRequest",
    "AutoAllowProvider",
    "AutoDenyProvider",
    "ScriptedProvider",
    "SCOPES",
    "SCOPE_LABELS",
]


@dataclass
class PermissionStack:
    """The wired permission objects, so callers can inspect and reuse them."""

    policy: PermissionPolicy
    memory: PermissionMemory
    evaluator: PermissionEvaluator
    gate: PermissionGate

    @property
    def enabled(self) -> bool:
        return self.policy.enabled

    def describe(self) -> dict[str, Any]:
        return self.evaluator.describe()


def build_permission_stack(
    config: Any,
    *,
    workspace: Any = None,
    approver: ApprovalProvider | None = None,
    emit: Any = None,
    persistent_path: str | None = None,
) -> PermissionStack:
    """Build the policy, memory, evaluator and gate from a harness config."""

    policy = PermissionPolicy.from_config(config, workspace=workspace)

    permissions = getattr(config, "permissions", None)
    store_path = persistent_path or getattr(permissions, "persistent_path", None)
    persistent_enabled = bool(getattr(permissions, "persistent", False))
    persistent: PersistentMemory | None = None
    if persistent_enabled:
        persistent = PersistentMemory(
            store_path or DEFAULT_PERSISTENT_PATH,
            workspace_root=getattr(workspace, "root", None),
        ).load()

    memory = PermissionMemory(persistent=persistent)
    evaluator = PermissionEvaluator(
        policy,
        memory,
        # A verdict of ASK is only meaningful when something can answer it.
        can_prompt=approver is None or bool(getattr(approver, "interactive", True)),
    )
    gate = PermissionGate(evaluator, approver, emit=emit)
    return PermissionStack(policy=policy, memory=memory, evaluator=evaluator, gate=gate)
