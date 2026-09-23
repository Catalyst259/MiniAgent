"""Phase 3: persistent grants and skill permission ceilings.

Two rules are under test here:

* **persistence must live outside the workspace** - a store the agent can write
  to is a store the agent can widen;
* **a skill's tool list is a ceiling, never a grant** - loading a skill may
  narrow what the agent can do and can never widen it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.agent.dto import ToolCall
from harness.core import AgentHarness
from harness.inference.config import ModelConfig
from harness.inference.mock_gateway import MockGateway, ScriptedResponse
from harness.infra.config import HarnessConfig
from harness.permission import (
    DEFAULT_PERSISTENT_PATH,
    PersistentMemory,
    PermissionMemory,
    PermissionPolicy,
    SessionMemory,
    build_permission_stack,
    from_arguments,
)
from tests.approval_helpers import ScriptedProvider
from harness.permission.decision import Permission
from harness.permission.evaluator import PermissionEvaluator
from harness.permission.gate import PermissionGate
from harness.permission.rules import Rule
from harness.skills.registry import SkillRegistry
from harness.tools.paths import Workspace

ROOT = Path(__file__).resolve().parents[1]

SKILL = """---
name: deploy
description: Deploy the service.
keywords: [deploy, ship]
tools: [read_file, shell]
---

Steps:
1. run the tests
"""


# ------------------------------------------------------------------ skill data
def test_skill_front_matter_declares_tools(tmp_path):
    skill_dir = tmp_path / "deploy"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(SKILL, encoding="utf-8")

    registry = SkillRegistry([tmp_path])
    registry.discover()
    metadata = registry.get("deploy")
    assert metadata.tools == ["read_file", "shell"]
    assert metadata.keywords == ["deploy", "ship"]


def test_skill_without_a_tool_list_declares_no_ceiling(tmp_path):
    skill_dir = tmp_path / "plain"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: plain\ndescription: x\n---\nbody\n", encoding="utf-8"
    )
    registry = SkillRegistry([tmp_path])
    registry.discover()
    assert registry.get("plain").tools == []


def test_shipped_skills_parse(tmp_path):
    registry = SkillRegistry([ROOT / "skills"])
    discovered = registry.discover()
    assert discovered, "the repo ships skills"
    for name in registry.names():
        assert isinstance(registry.get(name).tools, list)


# ------------------------------------------------------------- skill ceiling
def make_gate(tmp_path, skills=(), name="", mode="auto"):
    policy = PermissionPolicy(mode=mode, workspace=Workspace(tmp_path))
    if skills:
        policy.apply_skill_ceiling(skills, name)
    memory = PermissionMemory()
    evaluator = PermissionEvaluator(policy, memory)
    return PermissionGate(evaluator, ScriptedProvider(["session"]))


def test_skill_ceiling_denies_tools_the_skill_did_not_declare(tmp_path):
    gate = make_gate(tmp_path, ["read_file"], "review")
    allowed = gate.evaluator.evaluate(from_arguments("read_file", {"path": "a.py"}))
    assert allowed.permission is Permission.ALLOW or allowed.permission is Permission.ASK

    denied = gate.evaluator.evaluate(from_arguments("shell", {"command": "pytest"}))
    assert denied.permission is Permission.DENY
    assert denied.source == "skill"
    assert "review" in denied.reason


def test_skill_ceiling_cannot_grant_what_the_policy_denies(tmp_path):
    """``Skill Permission <= Tool Permission`` in the direction that matters."""

    policy = PermissionPolicy(
        mode="auto",
        workspace=Workspace(tmp_path),
        rules=[Rule(permission=Permission.DENY, tool="shell", program="rm")],
    )
    policy.apply_skill_ceiling(["shell"], "reckless")
    gate = PermissionGate(PermissionEvaluator(policy, PermissionMemory()), ScriptedProvider())
    verdict = gate.evaluator.evaluate(from_arguments("shell", {"command": "rm -rf /"}))
    assert verdict.permission is Permission.DENY
    assert verdict.source != "skill", "the policy deny wins over the skill's grant"


def test_empty_skill_list_clears_the_ceiling(tmp_path):
    gate = make_gate(tmp_path, ["read_file"], "review")
    assert gate.evaluator.policy.active_skill_tools == ("read_file",)
    gate.evaluator.policy.apply_skill_ceiling([], "")
    assert gate.evaluator.policy.active_skill_tools == ()
    # and the previously denied tool is reachable again
    assert gate.evaluator.evaluate(
        from_arguments("read_file", {"path": "a.py"})
    ).permission is not Permission.DENY


def test_ceiling_is_reported_in_the_policy_summary(tmp_path):
    policy = PermissionPolicy(mode="auto", workspace=Workspace(tmp_path))
    policy.apply_skill_ceiling(["read_file", "grep"], "review")
    assert policy.skill_ceiling() == {"name": "review", "tools": ["read_file", "grep"]}


def test_ceiling_denies_without_a_prompt_even_when_the_mode_asks(tmp_path):
    gate = make_gate(tmp_path, ["read_file"], "review", mode="ask")
    verdict = gate.evaluator.evaluate(from_arguments("write_file", {"path": "a.py"}))
    assert verdict.permission is Permission.DENY, "a ceiling is not a question"
    assert verdict.source == "skill"


# ------------------------------------------------------- skill ceiling in loop
def make_harness(tmp_path, responses, *, permissions=True, skills_dir=None):
    config = HarnessConfig.load(ROOT / "config.yaml")
    config.models = {"main": ModelConfig(provider="mock", model="mock-main")}
    config.default_model = "main"
    config.runtime.workspace_root = str(tmp_path)
    config.memory.enabled = False
    config.checkpoint.enabled = False
    if skills_dir is not None:
        config.skills.paths = [str(skills_dir)]
    if not permissions:
        config.permissions.mode = "off"

    harness = AgentHarness(config, workspace_root=tmp_path, approval_provider=ScriptedProvider(["session"]))
    harness.build()
    harness.set_gateway(MockGateway(config.resolve_model("main"), responses=responses))
    return harness


async def test_loop_applies_the_loaded_skills_ceiling(tmp_path):
    """A skill loaded in this same turn already governs that turn's other calls."""

    skill_dir = tmp_path / "skills" / "readonly"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: readonly\ndescription: read only\ntools: [read_file]\n---\nbody\n",
        encoding="utf-8",
    )
    (tmp_path / "calc.py").write_text("x = 1\n", encoding="utf-8")

    harness = make_harness(
        tmp_path,
        [
            ScriptedResponse(
                tool_calls=[
                    ToolCall(name="load_skill", arguments={"name": "readonly"}, id="s1"),
                    ToolCall(
                        name="write_file",
                        arguments={"path": "calc.py", "content": "x = 999\n"},
                        id="w1",
                    ),
                ]
            ),
            ScriptedResponse.say("done"),
        ],
        skills_dir=tmp_path / "skills",
    )
    result = await harness.run("review this")

    write = [obs for obs in result["observations"] if obs.tool_name == "write_file"]
    assert write and not write[0].ok, "the write must be refused by the skill ceiling"
    assert "readonly" in write[0].error
    assert (tmp_path / "calc.py").read_text(encoding="utf-8") == "x = 1\n"

    # the skill itself loaded: the ceiling narrows tools, it does not block skills
    loaded = [obs for obs in result["observations"] if obs.tool_name == "load_skill"]
    assert loaded and loaded[0].ok


async def test_ceiling_lets_the_model_switch_skills(tmp_path):
    """A ceiling must never trap the agent: loading another skill stays possible."""

    skills = tmp_path / "skills"
    for name, tools in (("narrow", "[read_file]"), ("wide", "[read_file, shell]")):
        directory = skills / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: d\ntools: {tools}\n---\nbody\n",
            encoding="utf-8",
        )
    harness = make_harness(
        tmp_path,
        [
            ScriptedResponse.tool("load_skill", name="narrow"),
            ScriptedResponse.tool("load_skill", name="wide"),
            ScriptedResponse.say("done"),
        ],
        skills_dir=skills,
    )
    result = await harness.run("switch skills")
    assert all(obs.ok for obs in result["observations"]), result["observations"]


async def test_loop_without_a_loaded_skill_keeps_the_normal_policy(tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\n", encoding="utf-8")
    harness = make_harness(
        tmp_path,
        [
            ScriptedResponse.tool(
                "write_file", path="calc.py", content="x = 2\n"
            ),
            ScriptedResponse.say("done"),
        ],
    )
    await harness.run("write it")
    assert (tmp_path / "calc.py").read_text(encoding="utf-8") == "x = 2\n"


def test_harness_resolves_the_newest_skill_ceiling(tmp_path):
    skills = tmp_path / "skills"
    for name, tools in (("first", "[read_file]"), ("second", "[grep]")):
        directory = skills / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: d\ntools: {tools}\n---\nbody\n",
            encoding="utf-8",
        )
    harness = make_harness(tmp_path, [ScriptedResponse.say("done")], skills_dir=skills)
    assert harness._skill_tool_ceiling(["first"]) == (["read_file"], "first")
    assert harness._skill_tool_ceiling(["first", "second"]) == (["grep"], "second")
    assert harness._skill_tool_ceiling([]) == ([], "")
    assert harness._skill_tool_ceiling(["nope"]) == ([], "")


# ------------------------------------------------------- persistence location
def test_persistent_store_inside_the_workspace_is_refused(tmp_path):
    """The agent can write in its workspace, so its rules must not live there."""

    store = PersistentMemory(tmp_path / "approvals.json", workspace_root=tmp_path)
    assert store.enabled is False
    assert "inside the workspace" in store.unsafe_reason
    assert store.save() is False, "a refused store must not be written at all"
    assert not (tmp_path / "approvals.json").exists()


def test_persistent_store_within_a_subdirectory_is_refused(tmp_path):
    store = PersistentMemory(tmp_path / ".miniagent" / "approvals.json", workspace_root=tmp_path)
    assert store.enabled is False


def test_persistent_store_outside_the_workspace_is_accepted(tmp_path):
    elsewhere = tmp_path / "home" / "approvals.json"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = PersistentMemory(elsewhere, workspace_root=workspace)
    assert store.enabled is True
    assert store.unsafe_reason == ""
    assert store.add(Rule(permission=Permission.ALLOW, tool="write_file")) is None
    assert elsewhere.exists()


def test_stack_refuses_an_in_workspace_persistent_path(tmp_path):
    """The config cannot be used to place the store where the agent can edit it."""

    config = HarnessConfig()
    config.permissions.mode = "ask"
    config.permissions.persistent = True
    config.permissions.persistent_path = str(tmp_path / "approvals.json")
    stack = build_permission_stack(
        config, workspace=Workspace(tmp_path), approver=ScriptedProvider(["persistent"])
    )
    assert stack.memory.persistent is not None
    assert stack.memory.persistent.enabled is False


# ------------------------------------------------------ persistence round trip
async def test_always_allow_survives_a_restart(tmp_path):
    store_path = tmp_path / "home" / "approvals.json"
    config = HarnessConfig()
    config.permissions.mode = "ask"
    config.permissions.persistent = True
    config.permissions.persistent_path = str(store_path)
    workspace = Workspace(tmp_path / "workspace")
    workspace.root.mkdir()

    first = build_permission_stack(
        config, workspace=workspace, approver=ScriptedProvider(["persistent"])
    )
    call = ToolCall(name="write_file", arguments={"path": "a.py", "content": "x"}, id="c1")
    results = await first.gate.check_batch([call])
    assert results[0].allowed
    assert store_path.exists()

    # a brand new stack, as if the process had restarted
    second = build_permission_stack(config, workspace=workspace, approver=ScriptedProvider())
    verdict = second.evaluator.evaluate(from_arguments("write_file", {"path": "b.py"}))
    assert verdict.permission is Permission.ALLOW
    assert verdict.source == "persistent"
    assert verdict.approval == "persistent"


def test_persistent_grants_cannot_answer_a_sandbox_denial(tmp_path):
    store_path = tmp_path / "home" / "approvals.json"
    workspace = Workspace(tmp_path / "workspace")
    workspace.root.mkdir()
    memory = PermissionMemory(persistent=PersistentMemory(store_path, workspace_root=workspace.root).load())
    memory.persistent.add(Rule(permission=Permission.ALLOW, tool="write_file"))
    policy = PermissionPolicy(
        mode="ask",
        workspace=workspace,
        sandbox_deny=[Rule(permission=Permission.DENY, target=".env")],
    )
    evaluator = PermissionEvaluator(policy, memory)
    verdict = evaluator.evaluate(from_arguments("write_file", {"path": ".env"}))
    assert verdict.permission is Permission.DENY
    assert verdict.source == "sandbox"


def test_clear_session_keeps_saved_grants(tmp_path):
    store_path = tmp_path / "home" / "approvals.json"
    workspace = Workspace(tmp_path / "workspace")
    workspace.root.mkdir()
    memory = PermissionMemory(persistent=PersistentMemory(store_path, workspace_root=workspace.root).load())
    memory.session.add(Rule(permission=Permission.ALLOW, tool="shell", program="npm"))
    memory.persistent.add(Rule(permission=Permission.ALLOW, tool="write_file"))
    memory.clear_session()
    assert memory.session.rules() == []
    assert [rule.tool for rule in memory.persistent.rules()] == ["write_file"]


def test_persistent_store_is_owner_only(tmp_path):
    store_path = tmp_path / "approvals.json"
    store = PersistentMemory(store_path)
    store.add(Rule(permission=Permission.ALLOW, tool="shell", program="npm"))
    assert oct(store_path.stat().st_mode)[-3:] == "600"


def test_persistent_store_survives_a_corrupt_file(tmp_path):
    store_path = tmp_path / "approvals.json"
    store_path.write_text("[not, a, dict]", encoding="utf-8")
    store = PersistentMemory(store_path).load()
    assert store.rules() == [], "a corrupt store degrades to 'no grants', never a crash"


def test_forget_removes_one_grant(tmp_path):
    store_path = tmp_path / "approvals.json"
    store = PersistentMemory(store_path)
    npm = Rule(permission=Permission.ALLOW, tool="shell", program="npm")
    pip = Rule(permission=Permission.ALLOW, tool="shell", program="pip")
    store.add(npm)
    store.add(pip)
    assert store.forget(npm) is True
    assert [rule.program for rule in PersistentMemory(store_path).load().rules()] == ["pip"]


def test_default_store_path_is_the_documented_one():
    assert DEFAULT_PERSISTENT_PATH == "~/.config/miniagent/approvals.json"


def test_shipped_config_does_not_persist_by_default():
    """Enable it deliberately: 'always allow' writes a file in the user's home."""

    config = HarnessConfig.load(ROOT / "config.yaml")
    assert config.permissions.persistent is False
    assert config.permissions.persistent_path is None
