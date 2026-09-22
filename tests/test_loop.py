"""End-to-end tests of the agent loop, termination guard, skills and delegation."""

from __future__ import annotations

import json

import pytest

from harness.agent.dto import Message, Observation, ToolCall
from harness.agent.state import new_state
from harness.agent.termination import TerminationPolicy
from harness.core import AgentHarness
from harness.inference.config import ModelConfig
from harness.inference.mock_gateway import MockGateway, ScriptedResponse
from harness.infra.checkpoint import memory_checkpointer
from harness.infra.config import HarnessConfig

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture()
def workspace(tmp_path):
    (tmp_path / "calc.py").write_text(
        "def add(a, b):\n    return a - b\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    return tmp_path


def make_config(workspace, **overrides) -> HarnessConfig:
    config = HarnessConfig(
        models={"main": ModelConfig(provider="mock", model="mock-main", temperature=0.0)},
        default_model="main",
    )
    config.base_dir = str(ROOT)
    config.runtime.workspace_root = str(workspace)
    config.memory.enabled = False
    config.context.max_input_tokens = 60_000
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_harness(
    workspace,
    responses=None,
    *,
    on_event=None,
    interaction_provider=None,
    **overrides,
):
    config = make_config(workspace, **overrides)
    gateway = MockGateway(config.resolve_model("main"), responses=responses)
    harness = AgentHarness(
        config,
        workspace_root=workspace,
        on_event=on_event,
        interaction_provider=interaction_provider,
    )
    harness.build()
    harness.set_gateway(gateway)
    gateway.skills = {
        name: harness.skill_registry.get(name).keywords for name in harness.skill_registry.names()
    }
    return harness, gateway


# ------------------------------------------------------------------ termination
def _state_with(messages):
    state = new_state("t", thread_id="x", task_id="y")
    state["messages"] = messages
    return state


def test_final_answer_terminates():
    state = _state_with([Message(role="user", content="hi"), Message(role="assistant", content="done")])
    decision = TerminationPolicy().check(state)
    assert decision.terminate and decision.reason == "final_answer"
    assert decision.final_answer == "done"


def test_text_with_tool_calls_does_not_terminate():
    assistant = Message(
        role="assistant",
        content="let me look",
        tool_calls=[ToolCall(name="read_file", arguments={"path": "a.py"})],
    )
    decision = TerminationPolicy().check(_state_with([assistant]))
    assert not decision.terminate


def test_max_iterations_terminates():
    state = _state_with(
        [
            Message(
                role="assistant",
                content="working",
                tool_calls=[ToolCall(name="grep", arguments={"pattern": f"x{i}"})],
            )
            for i in range(3)
        ]
    )
    state["iteration"] = 40
    decision = TerminationPolicy(max_iterations=40).check(state)
    assert decision.terminate and decision.reason == "max_iterations"


def test_repeated_tool_call_terminates():
    calls = [ToolCall(name="read_file", arguments={"path": "same.py"}) for _ in range(3)]
    messages = [Message(role="assistant", content="again", tool_calls=[call]) for call in calls]
    decision = TerminationPolicy(max_repeated_tool_calls=3).check(_state_with(messages))
    assert decision.terminate and decision.reason == "repeated_tool_call"


def test_fatal_tool_error_terminates():
    state = _state_with(
        [
            Message(
                role="assistant",
                content="x",
                tool_calls=[ToolCall(name="read_file", arguments={"path": "a.py"})],
            )
        ]
    )
    state["observations"] = [
        Observation(
            tool_call_id="1",
            tool_name="shell",
            ok=False,
            content="",
            error="FATAL: disk on fire",
        )
    ]
    decision = TerminationPolicy().check(state)
    assert decision.terminate and decision.reason == "fatal_tool_error"


# ------------------------------------------------------------------------ loop
async def test_loop_reads_patches_and_verifies(workspace):
    responses = [
        ScriptedResponse.tool("load_skill", name="debugging"),
        ScriptedResponse.tool("read_file", path="calc.py", limit=50),
        ScriptedResponse.tool(
            "apply_patch",
            patch=(
                "*** Begin Patch\n*** Update File: calc.py\n@@\n-    return a - b\n"
                "+    return a + b\n*** End Patch"
            ),
        ),
        ScriptedResponse.tool("shell", command="python -m pytest -q test_calc.py"),
        ScriptedResponse.say("Fixed the subtraction bug in calc.py and the test passes."),
    ]
    events: list[str] = []
    harness, gateway = make_harness(
        workspace, responses, on_event=lambda event: events.append(event.type)
    )
    harness.context_manager.builder.memory_provider = None

    result = await harness.run("Fix the bug in calc.py")

    assert result["termination_status"] == "final_answer"
    assert "calc.py" in (result["final_answer"] or "")
    assert (workspace / "calc.py").read_text().splitlines()[1] == "    return a + b"
    assert sorted(set(events)) == sorted(
        {
            "context",
            "token_guard",
            "iteration",
            "tool_call",
            "tool_start",
            "tool_result",
            "skill_load",
            "assistant_text",
            "assistant_message",
            # every call passes the permission gate, which reports its verdict even
            # when the layer is switched off (mode=off in this harness)
            "permission_decision",
            "terminate",
        }
    )
    # the skill body was injected into the state and therefore into the context
    assert "debugging" in result["loaded_skills"]
    last_request = gateway.calls[-1]
    assert any("Debugging" in message.content for message in last_request.messages)
    # every assistant tool call has a matching tool observation
    observations = result["observations"]
    assert len(observations) == 4 and all(observation.ok for observation in observations)


async def test_loop_terminates_without_tool_calls(workspace):
    harness, _ = make_harness(workspace, [ScriptedResponse.say("nothing to do")])
    result = await harness.run("Say something")
    assert result["final_answer"] == "nothing to do"
    assert result["iteration"] == 1


async def test_loop_exposes_and_executes_user_input_tool(workspace):
    class ScriptedInteraction:
        available = True

        def __init__(self) -> None:
            self.questions = []

        async def choose(self, question, options, *, title="", detail=""):
            self.questions.append((title, question, list(options)))
            return options[1]

    interaction = ScriptedInteraction()
    events = []
    responses = [
        ScriptedResponse.tool(
            "request_user_input",
            question="Which strategy?",
            options=[
                {"value": "fast", "label": "Fast"},
                {"value": "safe", "label": "Safe"},
            ],
        ),
        ScriptedResponse.say("Using safe."),
    ]
    harness, gateway = make_harness(
        workspace,
        responses,
        on_event=events.append,
        interaction_provider=interaction,
    )
    result = await harness.run("Choose a strategy")

    offered = {schema["function"]["name"] for schema in gateway.calls[0].tools}
    assert "request_user_input" in offered
    assert interaction.questions[0][1] == "Which strategy?"
    observation = next(
        item for item in result["observations"] if item.tool_name == "request_user_input"
    )
    assert observation.ok and "safe" in observation.content
    assert {event.type for event in events} >= {
        "interaction_request",
        "interaction_resolved",
    }


async def test_loop_reports_tool_errors_to_the_model(workspace):
    responses = [
        ScriptedResponse.tool("read_file", path="does_not_exist.py"),
        ScriptedResponse.say("I could not read that file."),
    ]
    harness, gateway = make_harness(workspace, responses)
    result = await harness.run("read a missing file")
    observation = result["observations"][0]
    assert not observation.ok
    assert "does not exist" in (observation.error or "")
    tool_messages = [m for m in gateway.calls[-1].messages if m.role == "tool"]
    assert tool_messages and tool_messages[0].content.startswith("ERROR:")


async def test_loop_rejects_invalid_arguments(workspace):
    responses = [
        ScriptedResponse.tool("read_file", nonsense=1),
        ScriptedResponse.say("retrying"),
    ]
    harness, _ = make_harness(workspace, responses)
    result = await harness.run("read something")
    observation = result["observations"][0]
    assert not observation.ok
    assert "missing required argument" in (observation.error or "")


async def test_mock_heuristic_runs_offline(workspace):
    harness, gateway = make_harness(workspace)
    result = await harness.run("list the files in this repo")
    assert result["termination_status"] == "final_answer"
    assert result["iteration"] >= 2
    tool_names = [
        call.name for message in result["messages"] if message.role == "assistant" for call in message.tool_calls
    ]
    assert "list_dir" in tool_names
    assert "[mock model" in (result["final_answer"] or "")


# -------------------------------------------------------------------- checkpoint
async def test_checkpoint_resumes_state(workspace):
    harness, _ = make_harness(workspace, [ScriptedResponse.say("done")])
    checkpointer = memory_checkpointer()
    harness.orchestrator.checkpointer = checkpointer
    harness.orchestrator.build()
    harness.orchestrator.thread_id = "resume-me"
    state = harness.initial_state("long task", thread_id="resume-me")
    await harness.orchestrator.ainvoke(state)

    snapshot = await harness.orchestrator.aget_state()
    assert snapshot["final_answer"] == "done"
    assert snapshot["thread_id"] == "resume-me"


# ------------------------------------------------------------------------ skills
async def test_unknown_skill_is_reported(workspace):
    responses = [
        ScriptedResponse.tool("load_skill", name="does_not_exist"),
        ScriptedResponse.say("no such skill"),
    ]
    harness, _ = make_harness(workspace, responses)
    result = await harness.run("load a skill")
    assert not result["observations"][0].ok
    assert "unknown skill" in (result["observations"][0].error or "")


# -------------------------------------------------------------------- subagents
async def test_delegation_isolates_context_and_tools(workspace):
    responses = [
        ScriptedResponse.tool(
            "delegate", agent="explorer", task="find the add function", context="calc.py is relevant"
        ),
        ScriptedResponse.say("Explorer found calc.py."),
    ]
    harness, gateway = make_harness(workspace, responses)
    child_gateways: list[MockGateway] = []

    original_factory = harness._harness_factory

    def factory(**kwargs):
        child = original_factory(**kwargs)
        original_build = child.build

        def build_then_capture():
            original_build()
            child_gateways.append(child.gateway)
            return child

        child.build = build_then_capture
        return child

    harness._harness_factory = factory

    result = await harness.run("Ask the explorer where add() lives")

    assert result["delegate_history"]
    entry = result["delegate_history"][0]
    assert entry["agent"] == "explorer" and entry["ok"]

    # 1. the subagent got its own context, not the parent's message list
    child_gateway = child_gateways[0]
    child_system = child_gateway.calls[0].messages[0].content
    assert "Explorer subagent" in child_system
    assert "maximum tool/model loop is 20 iterations" in child_system
    parent_system = gateway.calls[-1].messages[0].content
    assert "Explorer subagent" not in parent_system

    # 2. tool isolation: the explorer only ever sees read-only tools
    child_tools = {schema["function"]["name"] for schema in child_gateway.calls[0].tools}
    assert child_tools == {"list_dir", "glob", "grep", "read_file", "load_skill"}

    # 3. the parent sees a compact result, not the child's transcript
    tool_messages = [m for m in gateway.calls[-1].messages if m.role == "tool"]
    assert any("Subagent `explorer` result" in message.content for message in tool_messages)


async def test_unknown_subagent_is_reported(workspace):
    responses = [
        ScriptedResponse.tool("delegate", agent="nobody", task="do something"),
        ScriptedResponse.say("that subagent does not exist"),
    ]
    harness, _ = make_harness(workspace, responses)
    result = await harness.run("delegate to nobody")
    assert not result["observations"][0].ok
    assert "unknown subagent" in (result["observations"][0].error or "")


async def test_subagent_internal_events_do_not_leak_to_parent(workspace):
    from harness.agent.events import Event

    events: list[Event] = []
    responses = [
        ScriptedResponse.tool("delegate", agent="explorer", task="map the repository"),
        ScriptedResponse.say("summarized"),
    ]
    harness, _ = make_harness(workspace, responses, on_event=events.append)
    result = await harness.run("map the repository")

    assert result["termination_status"] == "final_answer"
    kinds = [event.type for event in events]
    assert "delegate_start" in kinds and "delegate_end" in kinds
    start = kinds.index("delegate_start")
    end = kinds.index("delegate_end", start)
    assert not any(kind in {"iteration", "tool_call", "tool_result"} for kind in kinds[start + 1 : end])


async def test_subagent_does_not_construct_parent_memory_store(workspace):
    harness, _ = make_harness(workspace, [ScriptedResponse.say("done")])
    harness.build()
    child = harness._subagent_factory(harness.subagent_registry.get("explorer"), "inspect files")

    assert child.use_memory is False
    assert child.memory is None


# -------------------------------------------------------------------- compaction
async def test_compaction_triggers_and_keeps_task_alive(workspace):
    responses = [
        ScriptedResponse.tool("list_dir", path="."),
        ScriptedResponse.tool("read_file", path="calc.py"),
        ScriptedResponse.tool("grep", pattern="return"),
        ScriptedResponse.tool("glob", pattern="**/*.py"),
        ScriptedResponse.tool("list_dir", path="."),
        ScriptedResponse.tool("read_file", path="calc.py"),
        ScriptedResponse.tool("list_dir", path="."),
        ScriptedResponse.say("compacted summary text"),
        ScriptedResponse.say("finished after compacting"),
    ]

    harness, _ = make_harness(workspace, responses)
    harness.config.context.max_input_tokens = 2600
    harness.config.context.reserve_output_tokens = 100
    harness.config.context.compact_trigger_ratio = 0.5
    harness.context_manager.policy.max_input_tokens = 2600
    harness.context_manager.policy.reserve_output_tokens = 100
    harness.context_manager.policy.compact_trigger_ratio = 0.5

    result = await harness.run("Investigate this repository thoroughly")

    assert result["compact_count"] >= 1
    assert result["compact_summary"]
    assert result["termination_status"] == "final_answer"
    # the compacted summary is part of the next model input
    assert any(
        "[Context compacted" in message.content for message in harness.gateway.calls[-1].messages
    )


# ----------------------------------------------------------------------- memory
async def test_memory_formation_and_recall(workspace):
    config = make_config(workspace)
    config.memory.enabled = True
    config.memory.backend = "memory"
    config.embedding.backend = "hashing"
    config.embedding.dimensions = 64
    gateway = MockGateway(
        config.resolve_model("main"),
        responses=[
            ScriptedResponse.say(
                '{"memories": [{"content": "calc.py holds the add function used by the tests",'
                ' "memory_type": "repo_fact"}]}'
            ),
        ],
    )
    harness = AgentHarness(config, workspace_root=workspace)
    harness.build()
    harness.gateway = gateway
    harness.orchestrator.gateway = gateway
    harness.summary.gateway = gateway

    state = new_state("fix add", thread_id="t", task_id="1")
    written = await harness.finish(state)
    assert written == 1
    assert await harness.memory.store.count() == 1

    lines = await harness.memory_lines("where is the add function defined?")
    assert any("calc.py holds the add function" in line for line in lines)


async def test_memory_status_and_disabled(workspace):
    harness, _ = make_harness(workspace)
    info = await harness.status()
    assert info["memory"]["enabled"] is False
    assert info["tools"] == 8
    assert info["subagents"] == ["explorer", "planner"]
    assert info["skills"] == 4


# ---------------------------------------------------------------- config & tools
def test_status_after_build(workspace):
    harness, _ = make_harness(workspace)
    assert harness.tool_registry.names() == sorted(
        [
            "list_dir",
            "glob",
            "grep",
            "read_file",
            "write_file",
            "apply_patch",
            "shell",
            "git_diff",
        ]
    )
    info = harness.subagent_registry.get("explorer")
    assert info.tools == ["list_dir", "glob", "grep", "read_file"]


async def test_context_builder_injects_catalogs(workspace):
    harness, _ = make_harness(workspace)
    state = new_state("hello", thread_id="t", task_id="1")
    request = await harness.context_builder.build(state)
    system = request.messages[0].content
    assert "workspace root" in system
    assert "read_file" in system and "apply_patch" in system
    assert "debugging" in system
    assert request.tools and any(
        schema["function"]["name"] == "delegate" for schema in request.tools
    )


# --------------------------------------------------- conversation validity
def _tool_message_violations(messages) -> list[str]:
    """Every assistant ``tool_call_id`` must be answered by a tool message."""

    problems: list[str] = []
    for index, message in enumerate(messages):
        if message.role != "assistant" or not message.tool_calls:
            continue
        answered: list[str] = []
        for later in messages[index + 1 :]:
            if later.role == "tool":
                answered.append(later.tool_call_id)
            elif later.role == "assistant":
                break
        missing = [call.id for call in message.tool_calls if call.id not in answered]
        if missing:
            problems.append(
                f"assistant#{index} {[c.name for c in message.tool_calls]} missing={missing}"
            )
    return problems


async def test_one_turn_mixing_skill_delegate_and_tools_answers_every_call(workspace):
    """Regression: a mixed turn used to drop every call but one category.

    Providers reject such a history with "an assistant message with 'tool_calls'
    must be followed by tool messages responding to each 'tool_call_id'", which
    kills the session.
    """

    responses = [
        ScriptedResponse(
            tool_calls=[
                ToolCall(name="load_skill", arguments={"name": "debugging"}, id="call_skill"),
                ToolCall(name="list_dir", arguments={"path": "."}, id="call_list"),
            ]
        ),
        ScriptedResponse(
            tool_calls=[
                ToolCall(name="delegate", arguments={"agent": "explorer", "task": "find it"}, id="call_del")
            ]
        ),
        ScriptedResponse(
            tool_calls=[ToolCall(name="read_file", arguments={"path": "calc.py"}, id="call_read")]
        ),
        ScriptedResponse.say("all done"),
    ]
    harness, _ = make_harness(workspace, responses)
    result = await harness.run("mixed turn")

    assert result["termination_status"] == "final_answer"
    assert _tool_message_violations(result["messages"]) == []
    answers = {message.tool_call_id for message in result["messages"] if message.role == "tool"}
    assert answers == {"call_skill", "call_list", "call_del", "call_read"}
    assert "debugging" in result["loaded_skills"]
    assert result["delegate_history"] and result["delegate_history"][0]["agent"] == "explorer"


async def test_tool_calls_in_one_message_all_run(workspace):
    responses = [
        ScriptedResponse(
            tool_calls=[
                ToolCall(name="list_dir", arguments={"path": "."}, id="c1"),
                ToolCall(name="glob", arguments={"pattern": "**/*.py"}, id="c2"),
                ToolCall(name="grep", arguments={"pattern": "def"}, id="c3"),
            ]
        ),
        ScriptedResponse.say("three tools done"),
    ]
    harness, _ = make_harness(workspace, responses)
    result = await harness.run("use three tools at once")
    assert _tool_message_violations(result["messages"]) == []
    assert {observation.tool_name for observation in result["observations"]} == {
        "list_dir",
        "glob",
        "grep",
    }


async def test_delegate_summary_reaches_the_tool_message(workspace):
    responses = [
        ScriptedResponse(
            tool_calls=[
                ToolCall(name="delegate", arguments={"agent": "explorer", "task": "find add"}, id="d1")
            ]
        ),
        ScriptedResponse.say("done"),
    ]
    events: list = []
    harness, _ = make_harness(workspace, responses, on_event=events.append)
    result = await harness.run("delegate once")

    tool_messages = [message for message in result["messages"] if message.role == "tool"]
    assert tool_messages and "Subagent `explorer` result" in tool_messages[0].content

    # the UI gets the task and the returned summary, not just a status line
    started = [event for event in events if event.type == "delegate_start"]
    finished = [event for event in events if event.type == "delegate_end"]
    assert started and started[0].data.get("task") == "find add"
    assert finished and finished[0].data.get("summary")
