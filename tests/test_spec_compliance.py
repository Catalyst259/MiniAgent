"""Acceptance tests: the concrete inventory demanded by `Detail.md`.

Every test here maps one-to-one onto a row of that document, so the deliverable can
be audited mechanically instead of by reading prose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.agent.dto import Message, ToolCall
from harness.agent.state import new_state
from harness.agent.termination import TerminationPolicy
from harness.cli.composer.slash_commands import build_default_registry
from harness.core import AgentHarness
from harness.inference.config import ModelConfig
from harness.inference.mock_gateway import MockGateway
from harness.infra.config import HarnessConfig
from harness.skills.registry import SkillRegistry
from harness.subagents.registry import SubAgentRegistry
from harness.tools.fs_tools import ALL_TOOL_NAMES, TOOL_FUNCS

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_TOOLS = {
    "list_dir",
    "glob",
    "grep",
    "read_file",
    "write_file",
    "apply_patch",
    "shell",
    "git_diff",
}
REQUIRED_SKILLS = {"repo_exploration", "debugging", "testing", "code_review"}
REQUIRED_SUBAGENTS = {"planner", "explorer"}
REQUIRED_COMMANDS = {
    "/help",
    "/status",
    "/model",
    "/tools",
    "/skills",
    "/agents",
    "/permissions",
    "/compact",
    "/clear",
    "/exit",
}


@pytest.fixture()
def harness(tmp_path) -> AgentHarness:
    config = HarnessConfig.load(ROOT / "config.yaml")
    config.models = {"main": ModelConfig(provider="mock", model="mock-main")}
    config.default_model = "main"
    config.runtime.workspace_root = str(tmp_path)
    config.memory.enabled = True
    config.memory.backend = "memory"
    config.embedding.backend = "hashing"
    config.embedding.dimensions = 64
    config.checkpoint.enabled = False
    instance = AgentHarness(config, workspace_root=tmp_path)
    instance.build()
    instance.set_gateway(MockGateway(config.resolve_model("main")))
    return instance


# ------------------------------------------------------------------------ Tools
def test_all_required_tools_are_registered(harness):
    assert set(ALL_TOOL_NAMES) == REQUIRED_TOOLS
    assert set(harness.tool_runtime.visible_tools()) == REQUIRED_TOOLS
    for name in REQUIRED_TOOLS:
        spec = harness.tool_registry.get(name)
        assert spec.description.strip()
        assert spec.input_schema.get("type") == "object"
        assert spec.input_schema.get("properties")


def test_native_tools_are_offered_to_the_model(harness):
    names = {schema["function"]["name"] for schema in harness.context_builder.tool_schemas}
    assert {"load_skill", "delegate"} <= names


def test_read_file_supports_line_ranges(harness, tmp_path):
    target = tmp_path / "many.py"
    target.write_text("".join(f"line {index}\n" for index in range(1, 51)), encoding="utf-8")
    out = harness.tool_registry.call("read_file", {"path": "many.py", "offset": 10, "limit": 3})
    assert "showing 10-12" in out
    assert "10\tline 10" in out and "12\tline 12" in out
    assert "38 more lines" in out


def test_apply_patch_supports_local_edits(harness, tmp_path):
    target = tmp_path / "app.py"
    target.write_text("def main():\n    return 1\n", encoding="utf-8")
    for patch in (
        "*** Begin Patch\n*** Update File: app.py\n@@\n-    return 1\n+    return 2\n*** End Patch",
        "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def main():\n-    return 1\n+    return 2\n",
        "*** Update File: app.py\n<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE",
    ):
        target.write_text("def main():\n    return 1\n", encoding="utf-8")
        out = harness.tool_registry.call("apply_patch", {"patch": patch})
        assert "1/1 files ok" in out, patch[:40]
        assert "return 2" in target.read_text()


def test_shell_and_git_diff_work(harness, tmp_path):
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    assert "exit_code: 0" in harness.tool_registry.call("shell", {"command": "echo hi"})
    diff = harness.tool_registry.call("git_diff", {})
    assert isinstance(diff, str) and diff


# ----------------------------------------------------------------------- Skills
def test_required_skills_exist_and_are_loadable(harness):
    registry = SkillRegistry([ROOT / "skills"])
    metadata = registry.discover()
    assert set(metadata) == REQUIRED_SKILLS
    for name in REQUIRED_SKILLS:
        assert metadata[name].description
        body = harness.skill_loader.load(name)
        assert f"BEGIN SKILL: {name}" in body
        assert len(body) > 400, name


def test_skill_metadata_only_is_injected(harness):
    system = harness.context_builder.skill_catalog
    for name in REQUIRED_SKILLS:
        assert name in system
    # bodies must not be preloaded
    assert "BEGIN SKILL" not in system


# -------------------------------------------------------------------- SubAgents
def test_required_subagents_exist(harness):
    registry = SubAgentRegistry([ROOT / "subagents"])
    specs = registry.discover()
    assert set(specs) == REQUIRED_SUBAGENTS


def test_planner_contract(harness):
    spec = harness.subagent_registry.get("planner")
    body = spec.body
    for heading in ("## Goal", "## Steps", "## Affected Areas", "## Risks", "## Verification Plan"):
        assert heading in body, heading
    assert body.index("## Goal") < body.index("## Steps") < body.index("## Affected Areas")
    assert "write_file" not in spec.tools and "apply_patch" not in spec.tools
    assert set(spec.tools) <= {"list_dir", "glob", "grep", "read_file", "git_diff"}


def test_explorer_contract_and_read_only_tools(harness):
    spec = harness.subagent_registry.get("explorer")
    for heading in ("## Relevant Files", "## Important Symbols", "## Call Relationships", "## Findings"):
        assert heading in spec.body, heading
    assert set(spec.tools) == {"list_dir", "glob", "grep", "read_file"}
    assert "shell" not in spec.tools
    assert "write_file" not in spec.tools and "apply_patch" not in spec.tools


async def test_explorer_harness_cannot_write(harness):
    """Delegation must produce a child whose tool ceiling is read-only."""

    captured = {}
    original = harness._harness_factory

    def factory(**kwargs):
        child = original(**kwargs)
        captured["allowed"] = None

        original_build = child.build

        def build():
            original_build()
            captured["allowed"] = set(child.tool_runtime.visible_tools())

        child.build = build
        return child

    harness._harness_factory = factory
    child = harness._subagent_factory(harness.subagent_registry.get("explorer"), "find stuff")
    assert captured["allowed"] == {"list_dir", "glob", "grep", "read_file", "git_diff"} or captured[
        "allowed"
    ] == {"list_dir", "glob", "grep", "read_file"}
    assert "write_file" not in (captured["allowed"] or set())
    assert "apply_patch" not in (captured["allowed"] or set())
    assert "shell" not in (captured["allowed"] or set())
    assert child.subagent_runtime.enabled is False


# ------------------------------------------------------------- Slash commands
def test_all_slash_commands_implemented():
    assert {command.display for command in build_default_registry({}).all()} == REQUIRED_COMMANDS


async def test_every_slash_command_dispatches(harness, tmp_path):
    """No slash command may raise; each must report something through the CLI."""

    from harness.cli.app import MiniAgentApp, Session
    from harness.cli.render import Renderer

    class Out:
        def __init__(self):
            self.seen: list[tuple[str, str]] = []

        def print(self, *args, **kwargs):
            self.seen.append(("print", " ".join(str(arg) for arg in args)))

        def info(self, text):
            self.seen.append(("info", text))

        def error(self, text):
            self.seen.append(("error", text))

        def banner(self):
            self.seen.append(("banner", ""))

    session = Session(harness.config)
    session.harness = harness
    console = Out()
    app = MiniAgentApp(session=session, renderer=Renderer(console=console, use_live_tail=False))
    await app.setup()

    for name in sorted(REQUIRED_COMMANDS - {"/exit"}):
        session.history = [Message(role="user", content="prior turn")]
        assert await app.handle_input(name) is False, name
    assert await app.handle_input("/exit") is True
    errors = [entry for entry in console.seen if entry[0] == "error"]
    assert not errors, errors


# ----------------------------------------------------------- Termination guard
def _state(messages):
    state = new_state("t", thread_id="x", task_id="y")
    state["messages"] = messages
    return state


def test_termination_guard_covers_every_reason():
    policy = TerminationPolicy(max_iterations=5, max_repeated_tool_calls=2)

    final = policy.check(_state([Message(role="assistant", content="all done")]))
    assert final.terminate and final.reason == "final_answer"

    working = policy.check(
        _state([Message(role="assistant", content="", tool_calls=[ToolCall(name="grep", arguments={"pattern": "a"})])])
    )
    assert not working.terminate

    looping = _state(
        [
            Message(role="assistant", content="", tool_calls=[ToolCall(name="read_file", arguments={"path": "a.py"})]),
            Message(role="assistant", content="", tool_calls=[ToolCall(name="read_file", arguments={"path": "a.py"})]),
        ]
    )
    decision = policy.check(looping)
    assert decision.terminate and decision.reason == "repeated_tool_call"

    iterations = _state(
        [
            Message(role="assistant", content="", tool_calls=[ToolCall(name="grep", arguments={"pattern": f"p{i}"})])
            for i in range(3)
        ]
    )
    iterations["iteration"] = 5
    decision = policy.check(iterations)
    assert decision.terminate and decision.reason == "max_iterations"

    explicit = _state([Message(role="assistant", content="x")])
    explicit["termination_status"] = "model_error"
    decision = policy.check(explicit)
    assert decision.terminate and decision.reason == "model_error"


# ------------------------------------------------------------- Context pipeline
async def test_context_has_system_tools_skills_and_state(harness):
    state = new_state("do the thing", thread_id="t", task_id="1")
    request = await harness.context_builder.build(state)
    system = request.messages[0].content
    assert "## Environment" in system and "## Available tools" in system
    assert "## Available skills" in system
    assert "read_file" in system and "apply_patch" in system
    assert request.tools
    tool_names = {schema["function"]["name"] for schema in request.tools}
    assert REQUIRED_TOOLS <= tool_names


async def test_compact_and_summary_are_separate_services(harness):
    from harness.context.compact import CompactService
    from harness.memory.summary import SummaryService

    assert isinstance(harness.context_manager.compact_service, CompactService)
    assert isinstance(harness.summary, SummaryService)
    assert harness.context_manager.compact_service is not harness.summary

    # compact only touches the conversation; summary only writes memory records
    state = new_state("task", thread_id="t", task_id="1")
    state["messages"] = [Message(role="user", content=f"m{i}") for i in range(12)]
    outcome = await harness.context_manager.compact(state)
    assert outcome.applied
    patch = outcome.state_patch(state)
    assert set(patch) == {"messages", "compact_summary", "compact_count"}
