"""Context Builder: the heart of the harness.

Deterministic assembly of the next model input:

```text
System Prompt
      + Relevant Memory
      + Available Tool Schemas
      + Available Skill Metadata
      + Loaded Skill Details
      + Current Agent State
      + Compact Summary
      + Recent Conversation
      -> ModelRequest
```

It never calls a model, never runs a tool and never decides routing.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable, Sequence

from harness.agent.dto import Message, ModelRequest
from harness.agent.state import AgentState
from harness.context.prompts import build_system_prompt
from harness.inference.config import ModelConfig

MemoryProvider = Callable[[str, int], Awaitable[Sequence[str]]]


class ContextBuilder:
    """Assembles a :class:`ModelRequest` from state + registries."""

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        workspace_root: str,
        tool_catalog: str = "",
        skill_catalog: str = "",
        subagent_catalog: str = "",
        tool_schemas: Iterable[dict[str, Any]] = (),
        extra_rules: str = "",
        base_prompt: str | None = None,
        memory_provider: MemoryProvider | None = None,
        memory_top_k: int = 5,
        max_recent_messages: int = 40,
        max_iterations: int | None = None,
        delta_callback: Callable[[str], None] | None = None,
        budget: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        self.model_config = model_config
        self.workspace_root = workspace_root
        self.tool_catalog = tool_catalog
        self.skill_catalog = skill_catalog
        self.subagent_catalog = subagent_catalog
        self.tool_schemas = list(tool_schemas)
        self.extra_rules = extra_rules
        self.base_prompt = base_prompt
        self.memory_provider = memory_provider
        self.memory_top_k = memory_top_k
        self.max_recent_messages = max_recent_messages
        self.max_iterations = max_iterations
        #: Set by the runtime for the duration of a turn; the gateway streams
        #: assistant text through it so the UI can render progress.
        self.delta_callback: Callable[[str], None] | None = delta_callback
        #: Returns ``(used_tokens, usable_budget)`` for context-usage events.
        self.budget = budget
        self._memory_cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ updating
    def update_tools(self, catalog: str, schemas: Iterable[dict[str, Any]]) -> None:
        self.tool_catalog = catalog
        self.tool_schemas = list(schemas)

    def update_skills(self, catalog: str) -> None:
        self.skill_catalog = catalog

    def _kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.base_prompt is not None:
            kwargs["base_prompt"] = self.base_prompt
        return kwargs

    # ------------------------------------------------------------- memory lookup
    async def _memories(self, state: AgentState) -> list[str]:
        if self.memory_provider is None:
            return []
        query = self._memory_query(state)
        if not query:
            return []
        cache_key = query[:400]
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        try:
            found = list(await self.memory_provider(query, self.memory_top_k))
        except Exception:  # memory must never break the loop
            found = []
        self._memory_cache[cache_key] = found
        return found

    @staticmethod
    def _memory_query(state: AgentState) -> str:
        messages = list(state.get("messages") or [])
        for message in reversed(messages):
            if message.role == "user":
                return message.content.strip()
        return (state.get("user_input") or "").strip()

    # -------------------------------------------------------------------- build
    def _system_prompt(self, state: AgentState, memories: Iterable[str]) -> str:
        loaded = state.get("loaded_skills") or []
        details = state.get("skill_details") or {}
        state_lines = [
            f"- iteration: {state.get('iteration', 0)}",
            *(
                [
                    f"- tool/model loop budget: {self.max_iterations} iterations maximum; "
                    "each iteration may contain multiple tool calls, so plan the work and "
                    "stop with a concise partial result before the budget is exhausted."
                ]
                if self.max_iterations is not None
                else []
            ),
            f"- loaded skills: {', '.join(loaded) if loaded else '(none)'}",
        ]
        if state.get("compact_count"):
            state_lines.append(f"- context compactions so far: {state['compact_count']}")
        return build_system_prompt(
            workspace_root=self.workspace_root,
            tool_catalog=self.tool_catalog,
            skill_catalog=self.skill_catalog,
            subagent_catalog=self.subagent_catalog,
            loaded_skills=[(name, details.get(name, "")) for name in loaded],
            memories=memories,
            state_lines=state_lines,
            extra_rules=self.extra_rules,
            **self._kwargs(),
        )

    @staticmethod
    def _tail(messages: list[Message], limit: int) -> list[Message]:
        if limit <= 0 or len(messages) <= limit:
            return list(messages)
        cut = len(messages) - limit
        while cut > 0 and messages[cut].role == "tool":
            cut -= 1
        return list(messages[cut:])

    def _metadata(self, state: AgentState) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "iteration": state.get("iteration", 0),
            "thread_id": state.get("thread_id"),
            "loaded_skills": list(state.get("loaded_skills") or []),
        }
        if self.delta_callback is not None:
            metadata["on_delta"] = self.delta_callback
        return metadata

    async def build(self, state: AgentState) -> ModelRequest:
        memories = await self._memories(state)
        system = Message(role="system", content=self._system_prompt(state, memories))

        messages: list[Message] = [system]
        summary = state.get("compact_summary")
        if summary:
            messages.append(
                Message(
                    role="user",
                    content=(
                        "[Context compacted - summary of earlier work in this task]\n"
                        f"{summary}"
                    ),
                )
            )
        messages.extend(self._tail(list(state.get("messages") or []), self.max_recent_messages))

        return ModelRequest(
            messages=messages,
            tools=list(self.tool_schemas) or None,
            model=self.model_config.model,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
            metadata=self._metadata(state),
        )


__all__ = ["ContextBuilder", "MemoryProvider"]
