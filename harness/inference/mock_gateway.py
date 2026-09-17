"""Offline gateway used by the CLI's mock mode and by the test suite.

It behaves like any other :class:`~harness.inference.gateway.ModelGateway`, so
the entire loop (context, token guard, compact, skills, tools, delegation,
termination) can be exercised without a network or an API key.  Two modes:

* **scripted** - a queue of canned responses (used by tests), and
* **scripted-per-task** - a small deterministic planner that reads the task and
  the tool observations and decides the next step.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from harness.agent.dto import Message, ModelRequest, ModelResponse, TokenUsage, ToolCall
from harness.inference.config import ModelConfig

_PATH_RE = re.compile(r"[\w./-]+\.(?:py|md|toml|yaml|yml|json|txt|cfg|ini|sh|js|ts|sql)\b")


@dataclass
class ScriptedResponse:
    """One canned model answer."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None

    def to_response(self, model: str | None = None) -> ModelResponse:
        return ModelResponse(
            text=self.text,
            reasoning=self.reasoning,
            tool_calls=list(self.tool_calls),
            finish_reason="tool_calls" if self.tool_calls else "stop",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model=model,
        )

    @staticmethod
    def tool(tool_name: str, /, **arguments: Any) -> "ScriptedResponse":
        return ScriptedResponse(tool_calls=[ToolCall(name=tool_name, arguments=arguments)])

    @staticmethod
    def say(text: str) -> "ScriptedResponse":
        return ScriptedResponse(text=text)


class MockGateway:
    """Deterministic, offline ``ModelGateway``."""

    name = "mock"

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        *,
        responses: Sequence[ScriptedResponse] | None = None,
        name: str = "mock",
        skills: dict[str, list[str]] | None = None,
        offline: bool | None = None,
    ) -> None:
        self.config = model_config or ModelConfig(provider="mock", model="mock-main")
        self.name = name
        self.responses: list[ScriptedResponse] = list(responses or [])
        self.skills = skills or {}
        self.calls: list[ModelRequest] = []
        self._cursor = 0
        if offline is None:
            offline = "mock" in (self.config.model or "").lower() or not _has_api_key(self.config)
        self.offline = bool(offline)

    # ------------------------------------------------------------------- gateway
    async def chat(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        if request.metadata.get("kind") == "compact":
            return self._compact_response(request)
        if request.metadata.get("kind") == "summary":
            return self._summary_response(request)
        if self._cursor < len(self.responses):
            response = self.responses[self._cursor]
            self._cursor += 1
        else:
            response = self._plan(request)
        result = response.to_response(request.model or self.config.model)
        on_delta = request.metadata.get("on_delta")
        if on_delta is not None and result.text:
            # simulate token streaming so the CLI streaming path is exercised offline
            for chunk in _chunk_text(result.text):
                on_delta(chunk)
        return result

    def _compact_response(self, request: ModelRequest) -> ModelResponse:
        """Compact calls always get a usable summary, never the scripted queue."""

        transcript = request.messages[-1].content if request.messages else ""
        folded = transcript.count("\nASSISTANT:") + transcript.count("\nUSER:")
        text = (
            "[mock compact summary] Folded earlier turns of the task. "
            f"Roughly {folded} conversation entries were summarised; "
            "the goal, files touched and open next step are preserved."
        )
        return ModelResponse(
            text=text,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=200, completion_tokens=60, total_tokens=260),
            model=request.model or self.config.model,
        )

    def _summary_response(self, request: ModelRequest) -> ModelResponse:
        """Summary calls get a valid (empty) memory payload unless scripted."""

        if self._cursor < len(self.responses):
            response = self.responses[self._cursor]
            self._cursor += 1
            return response.to_response(request.model or self.config.model)
        return ModelResponse(
            text='{"memories": []}',
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=200, completion_tokens=10, total_tokens=210),
            model=request.model or self.config.model,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[str]:
        response = await self.chat(request)
        if response.text:
            yield response.text

    # -------------------------------------------------------------- deterministic
    def _plan(self, request: ModelRequest) -> ScriptedResponse:
        available = {
            schema.get("function", {}).get("name")
            for schema in (request.tools or [])
            if isinstance(schema, dict)
        }
        messages = request.messages
        user_task = ""
        for message in messages:
            if message.role == "user" and not message.content.startswith("[Context compacted"):
                user_task = message.content.strip()
                break
        lowered = user_task.lower()
        steps = sum(
            1 for message in messages if message.role == "assistant" and message.tool_calls
        )

        # Without an API key the mock must not pretend to reason about the task:
        # it performs a short, honest reconnaissance and reports exactly that.
        if self.offline:
            return self._offline_step(available, user_task, steps)

        # 1. "compact ..." tasks exercise the compaction path.
        if "long" in lowered and "history" in lowered and steps > 6:
            return ScriptedResponse.say("Compaction kept the task alive; summary looks usable.")

        # 2. Delegation demonstration.
        if ("plan" in lowered or "explore" in lowered or "find" in lowered) and "delegate" in available and steps == 0:
            agent = "planner" if "plan" in lowered else "explorer"
            return ScriptedResponse.tool(
                "delegate",
                agent=agent,
                task=user_task,
                context="Workspace root only; report findings as required by your contract.",
            )

        # 3. Skill loading - driven by the keywords the skill declares.
        if steps == 0 and "load_skill" in available:
            skill = self._pick_skill(lowered)
            if skill:
                return ScriptedResponse.tool("load_skill", name=skill)

        # 4. Repository-orientation steps.
        if steps == 0 and "list_dir" in available:
            return ScriptedResponse.tool("list_dir", path=".", depth=1)

        paths = _PATH_RE.findall(user_task)
        if steps == 1 and paths and "read_file" in available:
            return ScriptedResponse.tool("read_file", path=paths[0], limit=120)

        if steps == 1 and "grep" in available and ("find" in lowered or "search" in lowered or "where" in lowered):
            keywords = [word for word in re.findall(r"[A-Za-z_]{4,}", user_task) if word.lower() not in ("find", "search", "where", "code")]
            if keywords:
                return ScriptedResponse.tool("grep", pattern=keywords[0], max_results=40)

        if steps == 1 and "shell" in available:
            return ScriptedResponse.tool("shell", command="pwd && ls")

        # 5. Report what happened, deterministically.
        summary = self._report(user_task, messages, steps)
        return ScriptedResponse.say(summary)

    def _offline_step(
        self, available: set[str], user_task: str, steps: int
    ) -> ScriptedResponse:
        """Short deterministic reconnaissance used when no API key is present."""

        if steps == 0 and "list_dir" in available:
            return ScriptedResponse.tool("list_dir", path=".", depth=1)
        paths = _PATH_RE.findall(user_task)
        if steps == 1:
            if paths and "read_file" in available:
                return ScriptedResponse.tool("read_file", path=paths[0], limit=120)
            if "grep" in available:
                keywords = [
                    word
                    for word in re.findall(r"[A-Za-z_]{5,}", user_task)
                    if word.lower() not in ("there", "which", "where", "should", "files", "file")
                ]
                if keywords:
                    return ScriptedResponse.tool("grep", pattern=keywords[0], max_results=30)
        return ScriptedResponse.say(
            "\n".join(
                [
                    "[mock model - no API key configured]",
                    f"task: {user_task.splitlines()[0][:200]}" if user_task else "task: (none)",
                    f"reconnaissance steps performed: {steps}",
                    "",
                    "This run used the offline deterministic model, so it inspected the "
                    "workspace but did not reason about the task. Configure a real model to "
                    "get actual work done:",
                    "  export DEEPSEEK_API_KEY=...        # or edit config.yaml",
                    "  ./miniagent.sh \"<your task>\"",
                ]
            )
        )

    def _pick_skill(self, lowered_task: str) -> str | None:
        """Pick the skill whose declared keywords match the task."""

        best: tuple[int, str] | None = None
        for skill, keywords in (self.skills or {}).items():
            hits = 0
            for keyword in keywords:
                if not keyword:
                    continue
                if len(keyword) <= 4:
                    if re.search(rf"\b{re.escape(keyword)}\b", lowered_task):
                        hits += 1
                elif keyword in lowered_task:
                    hits += 1
            if hits and (best is None or hits > best[0]):
                best = (hits, skill)
        return best[1] if best else None

    @staticmethod
    def _report(user_task: str, messages: list[Message], steps: int) -> str:
        observations = [message for message in messages if message.role == "tool"]
        lines = [
            "[mock model] offline deterministic run",
            f"task: {user_task.splitlines()[0][:160]}" if user_task else "task: (none)",
            f"tool calls executed: {steps}",
            f"observations received: {len(observations)}",
        ]
        for observation in observations[-4:]:
            preview = " ".join((observation.content or "").split())[:120]
            lines.append(f"- {observation.name}: {preview}")
        lines.append(f"iterations used: {steps}")
        lines.append(
            "No model API key is configured, so this answer is produced locally. "
            "Set MINIAGENT_MODELS__MAIN__API_KEY (or DEEPSEEK_API_KEY) and rerun for real reasoning."
        )
        return "\n".join(lines)


def _has_api_key(config: ModelConfig) -> bool:
    """Whether a usable key exists for the *real* provider behind this config."""

    if config.api_key and config.api_key != "EMPTY":
        return True
    if config.api_key_env and os.environ.get(config.api_key_env):
        return True
    return False



def _chunk_text(text: str, size: int = 24) -> list[str]:
    """Split text into small deltas, preferring newline boundaries."""

    chunks: list[str] = []
    buffer = ""
    for line in text.splitlines(keepends=True):
        buffer += line
        while len(buffer) >= size:
            chunks.append(buffer[:size])
            buffer = buffer[size:]
    if buffer:
        chunks.append(buffer)
    return chunks or [text]


__all__ = ["MockGateway", "ScriptedResponse"]
