"""Prompt templates.

Kept in one place so the agent's identity, rules and section layout are easy to
review and change.  The context builder is the only consumer.
"""

from __future__ import annotations

from typing import Iterable

BASE_SYSTEM_PROMPT = """You are a terminal coding agent. You work inside a single workspace \
directory and you act by calling tools - you never pretend a tool ran.

Core rules:
- Ground every claim in something you actually read or ran. If you did not check it, say so.
- Inspect before you edit: locate the real code with `glob`/`grep`/`read_file` before patching.
- Prefer `apply_patch` for edits; use `write_file` only for new files or full rewrites.
- Make the smallest change that fixes the problem, and keep unrelated code untouched.
- After editing, verify: run the narrowest relevant test or command, then report the result.
- Load a skill with `load_skill` when its topic matches the task, and follow its procedure.
- Delegate with `delegate` to the `Planner` (planning, read-only) or `Explorer` (code finding, \
read-only) when the task is large or the repository is unfamiliar.
- For broad repository analysis or onboarding, load the matching exploration skill first and
    delegate repository mapping to the Explorer when available. For an explicit multi-step plan,
    delegate to the Planner. Do not skip a matching skill or subagent merely because direct
    filesystem tools are available.
- Never invent tool names. Use only the tools listed below.
- Finish with a concise report: what changed, where (file paths), and how you verified it.
"""

SECTIONS = {
    "environment": "Environment",
    "memory": "Relevant long-term memory",
    "tools": "Available tools",
    "skills": "Available skills",
    "loaded_skills": "Loaded skill details",
    "state": "Current state",
}


def build_system_prompt(
    *,
    workspace_root: str,
    tool_catalog: str = "",
    skill_catalog: str = "",
    subagent_catalog: str = "",
    loaded_skills: Iterable[tuple[str, str]] = (),
    memories: Iterable[str] = (),
    state_lines: Iterable[str] = (),
    extra_rules: str = "",
    base_prompt: str = BASE_SYSTEM_PROMPT,
) -> str:
    """Assemble the stable, cacheable part of the context."""

    blocks: list[str] = [base_prompt.strip()]

    env_lines = [f"- workspace root: {workspace_root}"]
    env_lines.extend(state_lines)
    blocks.append(_section(SECTIONS["environment"], "\n".join(env_lines)))

    memory_lines = [f"- {line}" for line in memories if str(line).strip()]
    if memory_lines:
        blocks.append(_section(SECTIONS["memory"], "\n".join(memory_lines)))

    if tool_catalog:
        blocks.append(_section(SECTIONS["tools"], tool_catalog.strip()))

    if skill_catalog:
        blocks.append(_section(SECTIONS["skills"], skill_catalog.strip()))

    if subagent_catalog:
        blocks.append(_section("Available subagents", subagent_catalog.strip()))

    details = [(name, body) for name, body in loaded_skills if body.strip()]
    if details:
        rendered = "\n\n".join(f"### {name}\n{body.strip()}" for name, body in details)
        blocks.append(_section(SECTIONS["loaded_skills"], rendered))

    if extra_rules.strip():
        blocks.append(_section("Additional instructions", extra_rules.strip()))

    return "\n\n".join(blocks) + "\n"


def _section(title: str, body: str) -> str:
    return f"## {title}\n{body}"


COMPACT_PROMPT = """You are compacting the conversation history of a coding agent so it can \
continue the same task in a smaller context window.

Rewrite the transcript below into a dense, factual summary. Keep, in this order:
1. the user's goal and every explicit constraint or preference,
2. every file path touched, with what changed and why,
3. exact commands run and their outcomes (test results, errors),
4. decisions made and the reasoning that is still relevant,
5. what is still open: the next concrete step and any blocker.

Rules:
- No preamble, no apologies, no "the user asked".
- Preserve exact identifiers, paths, commands and error text.
- Drop chit-chat, duplicated file contents and anything already resolved.
- Target {target_tokens} tokens or fewer.
"""

COMPACT_USER_TEMPLATE = """Transcript to compact:

{transcript}

---
Write the summary now."""


def render_transcript(entries: Iterable[str], *, max_chars: int = 120_000) -> str:
    text = "\n\n".join(entry for entry in entries if entry)
    if len(text) > max_chars:
        head = int(max_chars * 0.6)
        text = text[:head] + "\n\n... [older transcript elided] ...\n\n" + text[-(max_chars - head) :]
    return text


SUMMARY_PROMPT = """You extract durable, reusable facts from a finished coding task.

Return at most {max_facts} short memory entries as JSON:
{{"memories": [{{"content": "...", "memory_type": "repo_fact|solution|gotcha|preference|trajectory",
  "metadata": {{"files": ["..."], "tags": ["..."]}}}}]}}

Only include things worth remembering in *future* tasks: repository structure facts,
root causes and fixes, environment gotchas, user preferences, or a proven approach.
Skip anything task-specific and already finished. If nothing is worth keeping,
return {{"memories": []}}.

Task transcript:
{transcript}
"""


__all__ = [
    "BASE_SYSTEM_PROMPT",
    "SECTIONS",
    "build_system_prompt",
    "COMPACT_PROMPT",
    "COMPACT_USER_TEMPLATE",
    "SUMMARY_PROMPT",
    "render_transcript",
]
