"""Shell intents: ``!command`` is executed by the runtime, not by the UI.

``CLI_Design.md`` section 16: the composer only produces an intent; running the
subprocess is the agent runtime's job (here: the same sandboxed ``shell`` tool the
model uses, so output capture, timeouts and the workspace guard are identical).
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.agent.dto import ToolCall
from harness.cli import events as ui


@dataclass
class ShellIntent:
    command: str
    call_id: str = "user-shell"


async def run_shell_intent(runtime, intent: ShellIntent) -> ui.ToolFinished:
    """Execute a ``!command`` through the tool runtime and return the result."""

    call = ToolCall(name="shell", arguments={"command": intent.command}, id=intent.call_id)
    observation = await runtime.run(call)
    if observation.ok:
        return ui.ToolFinished(
            call_id=intent.call_id,
            tool="shell",
            ok=True,
            text=observation.content,
            duration_ms=observation.duration_ms,
        )
    return ui.ToolFinished(
        call_id=intent.call_id,
        tool="shell",
        ok=False,
        text=observation.error or observation.content,
        duration_ms=observation.duration_ms,
    )


async def execute_shell_intent(harness, intent: ShellIntent, *, can_answer: bool, emit) -> None:
    """Gate and execute a user shell command, publishing its progress as UI events."""
    command = intent.command
    emit(ui.TurnStarted(prompt=f"!{command}"))
    emit(ui.ToolStarted(call_id=intent.call_id, tool="shell", arguments={"command": command}))

    call = ToolCall(name="shell", arguments={"command": command}, id=intent.call_id)
    results = await harness.permission.gate.check_batch(
        [call], allow_when_unavailable=can_answer
    )
    result = results[0] if results else None
    if result is not None and not result.allowed:
        emit(ui.ToolFailed(call_id=intent.call_id, tool="shell", error=result.denial_message))
        emit(
            ui.PermissionDecided(
                call_id=intent.call_id,
                tool="shell",
                permission=result.verdict.permission.value,
                reason=result.verdict.reason,
                source=result.verdict.source,
                approval=result.verdict.approval,
            )
        )
        return

    from harness.tools.runtime import calls_already_decided

    with calls_already_decided([intent.call_id]):
        result_event = await run_shell_intent(harness.tool_runtime, intent)
    if result_event.ok:
        emit(result_event)
    else:
        emit(ui.ToolFailed(call_id=intent.call_id, tool="shell", error=result_event.text))
