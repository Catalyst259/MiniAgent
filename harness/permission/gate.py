"""The permission gate: the single choke point every agent action passes.

Design document sections 10 and 16 ("all agent behaviour must pass one permission
entry").  The gate sits between the model and execution and does four things for
a whole batch of calls:

1. convert every tool call into an :class:`~harness.permission.action.Action`,
2. let the evaluator decide ``ALLOW`` / ``DENY`` / ``ASK``,
3. resolve the ``ASK`` cases through an :class:`ApprovalProvider` and store the
   answer in the right memory layer,
4. hand back a per-call :class:`GateResult` the execution code acts on.

Why a *batch* and not one call at a time: a single assistant turn may contain
several tool calls, and the ``act`` node runs the plain tools concurrently
(:meth:`harness.tools.runtime.ToolRuntime.run_many`).  Asking the user from
inside a concurrent worker would interleave prompts unpredictably and hold a
semaphore while a human types.  Deciding everything up front keeps the
concurrent execution path untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from harness.agent.dto import ToolCall
from harness.permission.action import Action, from_tool_call
from harness.permission.approval import ApprovalProvider, AutoDenyProvider
from harness.permission.decision import Permission, Verdict
from harness.permission.evaluator import PermissionEvaluator

log = logging.getLogger(__name__)

EventEmitter = Callable[[str, str, dict[str, Any]], Any]


@dataclass
class GateResult:
    """What should happen to one pending call."""

    call: ToolCall
    action: Action
    verdict: Verdict
    #: the rule to store if the user picks a memorising scope
    grant: Any = None

    @property
    def allowed(self) -> bool:
        return self.verdict.permission is Permission.ALLOW

    @property
    def denied(self) -> bool:
        return self.verdict.permission is Permission.DENY

    @property
    def denial_message(self) -> str:
        """The tool error the model sees for a denied call.

        Phrasing matters: it must be unmistakable that this was a permission
        refusal (so the model reports it instead of retrying) without leaking the
        internal rule syntax as if it were a user-facing instruction.
        """

        reason = self.verdict.reason
        note = self.verdict.approval
        text = f"PERMISSION DENIED: {self.action.describe()} was refused by the permission layer ({reason})."
        if self.verdict.source == "approval":
            text += " The user rejected this action."
        elif note:
            text += f" {note}"
        return text + " Do not retry; report this to the user."


def combine_calls(*groups: Sequence[ToolCall]) -> list[ToolCall]:
    """Flatten call groups, skipping blanks and duplicate ids.

    A model that emits the same ``tool_call_id`` twice would otherwise execute it
    twice; the conversation only requires one answer per id.
    """

    seen: set[str] = set()
    out: list[ToolCall] = []
    for group in groups:
        for call in group or ():
            if call is None or call.id in seen:
                continue
            seen.add(call.id)
            out.append(call)
    return out


class PermissionGate:
    """Evaluates a batch of tool calls and resolves the approvals."""

    def __init__(
        self,
        evaluator: PermissionEvaluator,
        approver: ApprovalProvider | None = None,
        *,
        emit: EventEmitter | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.approver = approver if approver is not None else AutoDenyProvider()
        self.emit = emit
        #: set for the duration of a batch whose caller knows no UI can answer
        self._unavailable = False

    @property
    def enabled(self) -> bool:
        return self.evaluator.policy.enabled

    async def check_batch(
        self,
        calls: Sequence[ToolCall],
        *,
        allow_when_unavailable: bool = True,
    ) -> list[GateResult]:
        """Decide every call of one turn, asking the user where required.

        Every call goes through :meth:`PermissionEvaluator.evaluate`, including
        when the permission layer is switched off.  Deciding here instead ("mode
        is off, allow everything") would silently lift the ceiling that survives
        ``mode: off`` - the workspace boundary, ``denied_tools`` and the read-only
        ceiling a subagent runs under.

        ``allow_when_unavailable=False`` is for a caller that knows the question
        cannot be answered (no live UI): an ``ASK`` then fails closed instead of
        waiting on a future nobody will resolve.
        """

        prepared = [(call, from_tool_call(call)) for call in calls or ()]
        if not prepared:
            return []

        if not allow_when_unavailable:
            self._unavailable = True
        begin = getattr(self.approver, "begin_batch", None)
        if callable(begin):
            begin()
        try:
            results: list[GateResult] = []
            for call, action in prepared:
                results.append(await self._check_one(call, action))
            return results
        finally:
            end = getattr(self.approver, "end_batch", None)
            if callable(end):
                end()
            self._unavailable = False

    # --------------------------------------------------------------------- single
    async def _check_one(self, call: ToolCall, action: Action) -> GateResult:
        verdict = self.evaluator.evaluate(action)
        await self._emit_action(call, action, verdict)

        if verdict.needs_approval and self._unavailable:
            # The caller said the question cannot reach anyone: refusing is the
            # only honest answer, and it must not block the turn.
            verdict = Verdict(
                Permission.DENY,
                f"{verdict.reason}, and no approval prompt can be shown right now",
                source="approval",
                rule=verdict.rule,
            )
        elif verdict.needs_approval:
            verdict = await self._ask(call, action, verdict)

        log.log(
            logging.INFO if verdict.denied else logging.DEBUG,
            "permission %s for %s (%s)",
            verdict.permission.value,
            action.describe(),
            verdict.reason,
        )
        await self._emit_decision(call, action, verdict)
        return GateResult(call=call, action=action, verdict=verdict)

    async def _ask(self, call: ToolCall, action: Action, verdict: Verdict) -> Verdict:
        """Ask the user and remember the answer.

        The provider arms the question first and the event is emitted second, so
        an answer that arrives immediately cannot be lost against a future the
        caller has not started awaiting yet.
        """

        try:
            request = self.approver.request(action, verdict.reason)
        except Exception as exc:  # noqa: BLE001 - a broken UI must not grant access
            log.exception("approval provider failed for %s", action.describe())
            return Verdict(
                Permission.DENY,
                f"the approval prompt failed: {type(exc).__name__}: {exc}",
                source="approval",
                rule=verdict.rule,
            )

        options = self._grant_options(action)
        payload = {
            "tool": action.tool,
            "id": call.id,
            "target": action.target,
            "type": action.type,
            "reason": verdict.reason,
            "rule": getattr(verdict.rule, "id", None),
            **options,
        }
        # A display provider may need the choice list (it is armed before the
        # event exists), so hand it the same payload the event carries.
        offer = getattr(self.approver, "offer", None)
        if callable(offer):
            offer(payload)
        await self._emit("permission_ask", action.describe(), payload)

        try:
            approval = await request
        except Exception as exc:  # noqa: BLE001 - a broken UI must not grant access
            log.exception("approval prompt failed for %s", action.describe())
            return Verdict(
                Permission.DENY,
                f"the approval prompt failed: {type(exc).__name__}: {exc}",
                source="approval",
                rule=verdict.rule,
            )

        scope = (approval.scope or "reject").strip().lower()
        if scope == "reject":
            reason = "the user rejected this action"
            if approval.note:
                reason += f": {approval.note}"
            return Verdict(
                Permission.DENY,
                reason,
                source="approval",
                rule=verdict.rule,
                approval=approval.note or None,
            )

        grant = self.grant_rule(action)
        if scope == "once":
            # "Allow once" means exactly this call: no standing grant is stored,
            # so the next identical call is asked about again.
            return Verdict(
                Permission.ALLOW,
                "allowed once by the user (not remembered)",
                source="approval",
                rule=verdict.rule,
                approval="once",
            )

        stored = self.evaluator.remember(action, scope, rule=grant)
        if not stored:
            # Be explicit instead of letting the user believe it was remembered:
            # the action still runs, the wider scope just does not apply.
            log.warning("could not store a %s grant for %s", scope, action.describe())
            return Verdict(
                Permission.ALLOW,
                f"allowed by the user, but the {scope} grant could not be saved",
                source="approval",
                rule=verdict.rule,
                approval="once",
            )

        label = {
            "session": "allowed for this session",
            "persistent": "always allowed (saved)",
        }.get(scope, "allowed by the user")
        return Verdict(
            Permission.ALLOW,
            label,
            source="approval",
            rule=verdict.rule,
            approval=scope,
        )

    # -------------------------------------------------------------------- helpers
    def grant_rule(self, action: Action):
        """The rule an "allow this session" / "always allow" would store, if any."""

        return self.evaluator.grant_rule(action)

    def _grant_options(self, action: Action) -> dict[str, bool]:
        """Which memorising scopes the UI should offer for this action.

        A command that chains other commands (``npm x; rm -rf ~``) can never be
        matched by a tool-wide rule, so offering "Always allow" there would be a
        lie.  The same applies to an action we cannot name a rule for.
        """

        grant = self.grant_rule(action)
        memorizable = grant is not None and getattr(grant, "memorizable", False)
        return {
            "can_session": memorizable,
            "can_persist": memorizable and self._persistence_available(),
        }

    def _persistence_available(self) -> bool:
        store = getattr(self.evaluator.memory, "persistent", None)
        return bool(store is not None and store.enabled)

    async def _emit_action(self, call: ToolCall, action: Action, verdict: Verdict) -> None:
        await self._emit(
            "tool_call",
            action.describe(),
            {
                "tool": action.tool,
                "id": call.id,
                "arguments": call.arguments,
                "action": action.to_dict(),
                "permission": verdict.permission.value,
            },
        )

    async def _emit_decision(self, call: ToolCall, action: Action, verdict: Verdict) -> None:
        await self._emit(
            "permission_decision",
            f"{verdict.permission.value}: {action.describe()}",
            {
                "tool": action.tool,
                "id": call.id,
                "action": action.to_dict(),
                "permission": verdict.permission.value,
                "reason": verdict.reason,
                "source": verdict.source,
                "approval": verdict.approval,
            },
        )

    async def _emit(self, event_type: str, message: str, data: dict[str, Any]) -> None:
        if self.emit is None:
            return
        result = self.emit(event_type, message, data)
        if hasattr(result, "__await__"):
            await result


__all__ = ["PermissionGate", "GateResult", "combine_calls", "EventEmitter"]
