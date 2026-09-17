"""Shell intents: ``!command`` is executed by the runtime, not by the UI.

``CLI_Design.md`` section 16: the composer only produces an intent; running the
subprocess is the agent runtime's job (here: the same sandboxed ``shell`` tool the
model uses, so output capture, timeouts and the workspace guard are identical).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from harness.agent.dto import ToolCall
from harness.cli import events as ui


@dataclass
class ShellIntent:
    command: str
    call_id: str = "user-shell"


async def run_shell_intent(runtime, intent: ShellIntent) -> ui.ToolFinished:
    """Execute a ``!command`` through the tool runtime and return the result."""

    call = ToolCall(name="shell", arguments={"command": intent.command})
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


async def stream_shell_intent(runtime, intent: ShellIntent, emit) -> None:
    """Emit ToolStarted/ToolFinished around a user shell command."""

    emit(ui.ToolStarted(call_id=intent.call_id, tool="shell", arguments={"command": intent.command}))
    result = await run_shell_intent(runtime, intent)
    if not result.ok:
        emit(ui.ToolFailed(call_id=intent.call_id, tool="shell", error=result.text))
    else:
        emit(result)


__all__ = ["ShellIntent", "run_shell_intent", "stream_shell_intent"]
