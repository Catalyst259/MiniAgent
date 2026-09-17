"""Summary: memory formation after a task ends.

Distinct from compact: this runs when the *task* is finished, decides what is
worth keeping in long-term memory, and writes it to the memory store.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from harness.agent.dto import MemoryRecord, Message, ModelRequest
from harness.agent.state import AgentState
from harness.context.prompts import SUMMARY_PROMPT
from harness.inference.gateway import ModelGateway
from harness.memory.service import MemoryService, transcript_of

log = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class SummaryService:
    """Extracts durable memory records from a finished task."""

    def __init__(
        self,
        gateway: ModelGateway,
        memory: MemoryService,
        *,
        max_facts: int = 5,
        max_transcript_chars: int = 60_000,
    ) -> None:
        self.gateway = gateway
        self.memory = memory
        self.max_facts = max(1, max_facts)
        self.max_transcript_chars = max_transcript_chars

    async def summarize_task(
        self,
        state: AgentState,
        *,
        repo: str | None = None,
        extra: str | None = None,
    ) -> list[MemoryRecord]:
        if not self.memory.enabled:
            return []
        transcript = transcript_of(state.get("messages") or [], limit=80)
        if extra:
            transcript += f"\n\nADDITIONAL CONTEXT:\n{extra}"
        if len(transcript) > self.max_transcript_chars:
            transcript = transcript[: self.max_transcript_chars]

        prompt = SUMMARY_PROMPT.format(max_facts=self.max_facts, transcript=transcript)
        try:
            response = await self.gateway.chat(
                ModelRequest(
                    messages=[
                        Message(role="system", content="You extract durable project memory. Reply with JSON only."),
                        Message(role="user", content=prompt),
                    ],
                    temperature=0.0,
                    metadata={"kind": "summary"},
                )
            )
        except Exception as exc:
            log.warning("summary model call failed: %s", exc)
            return []

        records = self._parse(response.text or "", state, repo)
        if records:
            written = await self.memory.remember(records)
            log.debug("memory formation wrote %d record(s)", written)
        return records
    # ------------------------------------------------------------------ parsing
    def _parse(self, text: str, state: AgentState, repo: str | None) -> list[MemoryRecord]:
        payload = self._extract_json(text)
        if not payload:
            return []
        raw_items = payload.get("memories") if isinstance(payload, dict) else payload
        if not isinstance(raw_items, list):
            return []
        records: list[MemoryRecord] = []
        for item in raw_items[: self.max_facts]:
            if isinstance(item, str):
                content, memory_type, metadata = item, "task_summary", {}
            elif isinstance(item, dict):
                content = str(item.get("content") or "").strip()
                memory_type = str(item.get("memory_type") or "task_summary")
                metadata = item.get("metadata") or {}
            else:
                continue
            if not content:
                continue
            records.append(
                MemoryRecord(
                    content=content,
                    memory_type=memory_type,
                    source="task_summary",
                    task_id=state.get("task_id"),
                    repo=repo,
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )
        return records

    @staticmethod
    def _extract_json(text: str) -> Any:
        if not text.strip():
            return None
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*\n?", "", candidate)
            candidate = re.sub(r"\n?```$", "", candidate).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            match = _JSON_BLOCK.search(candidate)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None


__all__ = ["SummaryService"]
