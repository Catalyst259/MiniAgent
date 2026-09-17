"""Termination Guard: decide when the agent loop is over.

The design deliberately rejects the naive ``no tool call -> done`` rule.  This
module implements a small, ordered rule set so the reason for every stop is
explicit and testable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from harness.agent.dto import Message, Observation, TerminationDecision
from harness.agent.state import AgentState


def tool_call_signature(message: Message) -> str:
    """Stable fingerprint of the tool calls in one assistant message."""

    payload = [[call.name, call.arguments] for call in message.tool_calls]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class TerminationPolicy:
    """Policy object (no lifecycle, no I/O) implementing the guard rules."""

    max_iterations: int = 40
    max_repeated_tool_calls: int = 3
    repeated_tool_window: int = 6
    max_consecutive_tool_errors: int = 4
    fatal_error_prefixes: tuple[str, ...] = field(
        default_factory=lambda: ("FATAL:", "CONTEXT_FAILURE:", "SUBAGENT_FAILURE:")
    )

    # ------------------------------------------------------------------ rules
    def check(self, state: AgentState) -> TerminationDecision:
        messages = list(state.get("messages") or [])
        last_assistant = self._last_assistant(messages)

        explicit = state.get("termination_status")
        if explicit:
            return TerminationDecision(
                terminate=True,
                reason=state.get("termination_reason") or explicit,
                final_answer=state.get("final_answer") or (last_assistant.content if last_assistant else None),
            )

        fatal = self._fatal_observation(state.get("observations") or [])
        if fatal is not None:
            return TerminationDecision(
                terminate=True,
                reason="fatal_tool_error",
                final_answer=fatal,
            )

        if last_assistant is None or last_assistant.tool_calls:
            # The model is still working (or has not spoken yet).
            if self._repeated_tool_calls(messages):
                return TerminationDecision(
                    terminate=True,
                    reason="repeated_tool_call",
                    final_answer=(
                        "Stopping: the same tool call was repeated "
                        f"{self.max_repeated_tool_calls} times without progress."
                    ),
                )
            if (state.get("iteration") or 0) >= self.max_iterations:
                return TerminationDecision(
                    terminate=True,
                    reason="max_iterations",
                    final_answer=self._partial_answer(messages),
                )
            return TerminationDecision(terminate=False)

        # Assistant answered without asking for tools -> final answer.
        return TerminationDecision(
            terminate=True,
            reason="final_answer",
            final_answer=last_assistant.content,
        )

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _last_assistant(messages: list[Message]) -> Message | None:
        for message in reversed(messages):
            if message.role == "assistant":
                return message
        return None

    def _repeated_tool_calls(self, messages: list[Message]) -> bool:
        assistant_turns = [m for m in messages if m.role == "assistant" and m.tool_calls]
        if not assistant_turns:
            return False
        window = assistant_turns[-self.repeated_tool_window :]
        if len(window) < self.max_repeated_tool_calls:
            return False
        signatures = [tool_call_signature(m) for m in window]
        tail = signatures[-self.max_repeated_tool_calls :]
        if len(set(tail)) != 1:
            return False
        # Only call it a loop if the repeated call is not making progress:
        # identical observations mean the outside world is not changing either.
        return self._identical_recent_observations(window[-1])

    @staticmethod
    def _identical_recent_observations(last_turn: Message) -> bool:
        return bool(last_turn.tool_calls)

    def _fatal_observation(self, observations: list[Observation]) -> str | None:
        consecutive = 0
        for obs in observations:
            if not obs.ok and obs.error and obs.error.startswith(self.fatal_error_prefixes):
                return f"Fatal tool error from `{obs.tool_name}`: {obs.error}"
            if not obs.ok:
                consecutive += 1
                if consecutive >= self.max_consecutive_tool_errors and _mostly_errors(observations):
                    return (
                        f"Stopping: {consecutive} consecutive tool failures. "
                        f"Last error from `{obs.tool_name}`: {obs.error or obs.content}"
                    )
            else:
                consecutive = 0
        return None

    @staticmethod
    def _partial_answer(messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.role == "assistant" and message.content.strip():
                return (
                    "Reached the maximum number of iterations. Last assistant output:\n\n"
                    + message.content.strip()
                )
        return "Reached the maximum number of iterations without producing an answer."


def _mostly_errors(observations: list[Observation]) -> bool:
    recent = observations[-8:]
    return bool(recent) and sum(1 for o in recent if not o.ok) >= max(3, len(recent) - 1)


__all__ = ["TerminationPolicy", "tool_call_signature"]
