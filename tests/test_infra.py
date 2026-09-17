"""Tests for config, logging, checkpointing, skills, subagents and inference DTOs."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from harness.agent.dto import (
    Message,
    ModelRequest,
    ModelResponse,
    Observation,
    TokenUsage,
    ToolCall,
)
from harness.agent.events import Event
from harness.agent.state import ReplaceMessages, append_messages, new_state
from harness.context.builder import ContextBuilder
from harness.context.compact import CompactService
from harness.context.prompts import build_system_prompt
from harness.context.token_budget import TokenBudgetPolicy, split_for_compaction
from harness.inference.config import ModelConfig
from harness.inference.mock_gateway import MockGateway, ScriptedResponse
from harness.inference.openai_compatible import normalize_response
from harness.inference.tokenizer import HeuristicTokenCounter, build_token_counter
from harness.infra.checkpoint import list_threads, memory_checkpointer, sqlite_checkpointer
from harness.infra.config import HarnessConfig
from harness.skills.loader import SkillLoader
from harness.skills.registry import SkillRegistry, parse_front_matter
from harness.subagents.registry import SubAgentRegistry

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------- DTO layer
def test_message_to_openai_roundtrip():
    message = Message(role="assistant", content="", tool_calls=[ToolCall(name="read_file", arguments={"path": "a.py"})])
    payload = message.to_openai()
    assert payload["role"] == "assistant"
    assert payload["content"] is None
    call = payload["tool_calls"][0]
    assert call["function"]["name"] == "read_file"
    assert '"path": "a.py"' in call["function"]["arguments"]


def test_tool_call_normalizes_broken_json():
    call = ToolCall.from_openai({"id": "1", "function": {"name": "grep", "arguments": "{not json"}})
    assert call.name == "grep" and call.arguments == {}
    assert call.raw_arguments == "{not json"


def test_tool_call_normalizes_dict_arguments():
    call = ToolCall.from_openai({"name": "grep", "arguments": {"pattern": "x"}})
    assert call.arguments == {"pattern": "x"}


def test_observation_to_message_marks_errors():
    observation = Observation(tool_call_id="1", tool_name="shell", ok=False, content="", error="boom")
    message = observation.to_message()
    assert message.role == "tool" and message.content.startswith("ERROR: boom")
    assert message.tool_call_id == "1"


def test_token_usage_adds():
    total = TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3) + TokenUsage(
        prompt_tokens=10, completion_tokens=20, total_tokens=30
    )
    assert (total.prompt_tokens, total.completion_tokens, total.total_tokens) == (11, 22, 33)


def test_normalize_response_from_dict_and_object():
    payload = {
        "id": "x",
        "model": "m",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "reasoning_content": "think",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "shell", "arguments": '{"command": "ls"}'}}
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    response = normalize_response(payload)
    assert response.text is None and response.reasoning == "think"
    assert response.tool_calls[0].arguments == {"command": "ls"}
    assert response.usage.total_tokens == 12 and response.wants_tools


def test_normalize_response_without_choices_raises():
    from harness.agent.errors import ModelError

    with pytest.raises(ModelError):
        normalize_response({"choices": []})


def test_config_loads_dotenv_for_model_api_key(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        "  main:\n"
        "    provider: openai_compatible\n"
        "    api_key_env: TEST_DOTENV_API_KEY\n"
        "    model: test-model\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("TEST_DOTENV_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TEST_DOTENV_API_KEY", raising=False)

    config = HarnessConfig.load(config_path)

    assert config.resolve_model().api_key == "from-dotenv"


# ------------------------------------------------------------------- tokenizer
def test_heuristic_counter_is_monotonic():
    counter = HeuristicTokenCounter()
    assert counter.count_text("") == 0
    assert counter.count_text("hello world") >= 2
    assert counter.count_text("你好世界，这是一个测试") > 3
    request = ModelRequest(
        messages=[Message(role="system", content="x" * 400), Message(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "a", "description": "b"}}],
    )
    assert counter.count_request(request) > counter.count_text("hi")


def test_build_token_counter_falls_back_to_heuristic():
    counter = build_token_counter("auto", model="some-unknown-model")
    assert counter.count_text("hello") > 0


# ---------------------------------------------------------------- token budget
def test_budget_policy_thresholds():
    policy = TokenBudgetPolicy(max_input_tokens=1000, reserve_output_tokens=200, compact_trigger_ratio=0.5)
    assert policy.usable_tokens == 800
    assert policy.compact_threshold == 400
    assert not policy.exceeded(700)
    assert policy.exceeded(900)
    assert policy.should_compact(500, compactable_messages=3)
    assert not policy.should_compact(500, compactable_messages=0)
    assert "input tokens" in policy.describe(500)


def test_split_for_compaction_respects_tool_exchanges():
    messages = [
        Message(role="user", content="1"),
        Message(role="assistant", content="", tool_calls=[ToolCall(name="read_file")]),
        Message(role="tool", content="result", tool_call_id="1"),
        Message(role="assistant", content="done"),
        Message(role="user", content="2"),
        Message(role="assistant", content="answer"),
    ]
    head, tail = split_for_compaction(messages, keep_recent=2)
    assert tail[0].role == "user"  # never starts the tail with a tool result
    assert len(head) + len(tail) == len(messages)


def test_split_keeps_everything_when_history_is_short():
    messages = [Message(role="user", content="1"), Message(role="assistant", content="2")]
    head, tail = split_for_compaction(messages, keep_recent=8)
    assert head == [] and tail == messages


def test_replace_messages_reducer_drops_compacted_history():
    old = [Message(role="user", content="old")]
    retained = [Message(role="user", content="recent")]
    assert append_messages(old, ReplaceMessages(retained)) == retained


# --------------------------------------------------------------------- prompts
def test_system_prompt_sections():
    prompt = build_system_prompt(
        workspace_root="/tmp/ws",
        tool_catalog="- read_file: reads",
        skill_catalog="Available skills",
        subagent_catalog="- explorer: maps the repository",
        loaded_skills=[("debugging", "step 1")],
        memories=["remember this"],
        state_lines=["- iteration: 3"],
    )
    for expected in ("## Environment", "## Relevant long-term memory", "## Available tools", "## Available skills", "## Available subagents", "## Loaded skill details", "/tmp/ws", "step 1", "explorer"):
        assert expected in prompt
    assert "broad repository analysis" in prompt


def test_context_builder_exposes_iteration_budget():
    from harness.context.builder import ContextBuilder
    from harness.inference.config import ModelConfig

    builder = ContextBuilder(
        model_config=ModelConfig(provider="mock", model="mock"),
        workspace_root="/tmp/ws",
        max_iterations=40,
    )
    prompt = builder._system_prompt({"iteration": 7, "loaded_skills": []}, [])
    assert "40 iterations maximum" in prompt
    assert "each iteration may contain multiple tool calls" in prompt


# ---------------------------------------------------------------------- skills
def test_front_matter_parsing():
    front, body = parse_front_matter("---\nname: x\ndescription: y\nkeywords: [a, b]\n---\n\nBody text\n")
    assert front["name"] == "x" and front["keywords"] == ["a", "b"]
    assert body.strip() == "Body text"
    assert parse_front_matter("no front matter") == ({}, "no front matter")


def test_skill_registry_discovers_four_skills():
    registry = SkillRegistry([ROOT / "skills"])
    metadata = registry.discover()
    assert set(metadata) == {"repo_exploration", "debugging", "testing", "code_review"}
    assert metadata["debugging"].keywords
    catalog = registry.catalog()
    assert "debugging" in catalog and "load_skill" in catalog


def test_skill_loader_progressive_disclosure():
    registry = SkillRegistry([ROOT / "skills"])
    registry.discover()
    loader = SkillLoader(registry)
    body = loader.load("debugging")
    assert "BEGIN SKILL: debugging" in body
    assert "Reproduce" in body
    assert "already loaded" not in body
    assert "already loaded" in loader.load("debugging", ["debugging"])


def test_skill_loader_rejects_unknown():
    from harness.agent.errors import SkillNotFound

    registry = SkillRegistry([ROOT / "skills"])
    registry.discover()
    with pytest.raises(SkillNotFound):
        SkillLoader(registry).load("nope")


def test_missing_skill_path_is_ignored(tmp_path):
    registry = SkillRegistry([tmp_path / "absent"])
    assert registry.discover() == {}
    assert registry.catalog() == ""


# ------------------------------------------------------------------- subagents
def test_subagent_registry_reads_definitions():
    registry = SubAgentRegistry([ROOT / "subagents"])
    specs = registry.discover()
    assert set(specs) == {"planner", "explorer"}
    assert specs["explorer"].tools == ["list_dir", "glob", "grep", "read_file"]
    assert "shell" not in specs["explorer"].tools
    schema = registry.tool_schema()["function"]
    assert schema["name"] == "delegate"
    assert set(schema["parameters"]["properties"]["agent"]["enum"]) == {"planner", "explorer"}
    assert "planner" in registry.catalog()


def test_subagent_registry_case_insensitive_and_unknown():
    from harness.agent.errors import SubAgentNotFound

    registry = SubAgentRegistry([ROOT / "subagents"])
    registry.discover()
    assert registry.get("Explorer").name == "explorer"
    with pytest.raises(SubAgentNotFound):
        registry.get("nobody")


# ---------------------------------------------------------------------- config
def test_config_defaults_when_no_file():
    config = HarnessConfig.load(None)
    assert config.default_model in config.models
    assert config.context.max_input_tokens > 0
    assert "apply_patch" in config.tools.enabled


def test_config_from_yaml_and_env(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
models:
  main:
    provider: openai_compatible
    base_url: http://localhost:8000/v1
    api_key_env: TEST_KEY
    model: Qwen-Test
default_model: main
context:
  max_input_tokens: 1234
runtime:
  max_iterations: 7
tools:
  enabled: [read_file, grep, shell]
memory:
  enabled: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_KEY", "secret")
    monkeypatch.setenv("MINIAGENT_CONTEXT__MAX_INPUT_TOKENS", "4321")
    monkeypatch.setenv("MINIAGENT_RUNTIME__MAX_ITERATIONS", "9")

    config = HarnessConfig.load(config_file)
    assert config.context.max_input_tokens == 4321
    assert config.runtime.max_iterations == 9
    assert config.tools.enabled == ["read_file", "grep", "shell"]
    assert config.memory.enabled is False
    model = config.resolve_model("main")
    assert model.model == "Qwen-Test" and model.resolved_api_key == "secret"
    assert config.workspace_root == tmp_path.resolve()
    assert config.checkpoint_path.parent.parent == tmp_path.resolve()


def test_resolve_path_and_unknown_model(tmp_path):
    from harness.agent.errors import ConfigError

    config = HarnessConfig.load(None)
    config.base_dir = str(tmp_path)
    assert config.resolve_path("skills") == tmp_path / "skills"
    with pytest.raises(ConfigError):
        config.resolve_model("ghost")


def test_model_config_requires_key(monkeypatch):
    from harness.agent.errors import ConfigError

    monkeypatch.delenv("MISSING_KEY", raising=False)
    config = ModelConfig(model="x", api_key_env="MISSING_KEY")
    with pytest.raises(ConfigError):
        _ = config.resolved_api_key
    assert ModelConfig(model="x", api_key="EMPTY").resolved_api_key == "EMPTY"


def test_config_describe():
    described = HarnessConfig.load(None).describe()
    assert "workspace" in described and "model" in described


# ------------------------------------------------------------------- checkpoint
async def test_memory_checkpointer_smoke():
    saver = memory_checkpointer()
    assert saver is not None


def test_sqlite_checkpointer_roundtrip(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    with sqlite_checkpointer(str(path)) as saver:
        assert saver is not None
    assert path.exists()
    assert list_threads(str(path)) == []


async def test_sqlite_checkpoint_stores_state(tmp_path):
    """Runtime state really is persisted through the LangGraph SQLite saver."""

    from harness.core import AgentHarness
    from harness.infra.checkpoint import AsyncCheckpointStore

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "f.txt").write_text("hi\n")
    config = HarnessConfig(models={"main": ModelConfig(provider="mock", model="m")}, default_model="main")
    config.base_dir = str(ROOT)
    config.runtime.workspace_root = str(workspace)
    config.memory.enabled = False
    config.checkpoint.path = str(tmp_path / "cp.sqlite")

    async with AsyncCheckpointStore(str(config.checkpoint_path)) as checkpointer:
        harness = AgentHarness(config, workspace_root=workspace)
        harness.build()
        harness.set_gateway(MockGateway(config.resolve_model("main"), responses=[ScriptedResponse.say("persisted")]))
        harness.orchestrator.checkpointer = checkpointer
        harness.orchestrator.build()
        harness.orchestrator.thread_id = "thread-1"
        await harness.run("hello", thread_id="thread-1")
        snapshot = await harness.orchestrator.aget_state()
        assert snapshot["final_answer"] == "persisted"

    assert "thread-1" in list_threads(str(config.checkpoint_path))


# --------------------------------------------------------------- context manager
async def test_context_manager_reports_status(tmp_path):
    from harness.context.manager import ContextManager

    builder = ContextBuilder(
        model_config=ModelConfig(provider="mock", model="m"),
        workspace_root=str(tmp_path),
        tool_catalog="- read_file: reads",
        skill_catalog="skills",
        tool_schemas=[{"type": "function", "function": {"name": "read_file"}}],
    )
    manager = ContextManager(
        builder,
        HeuristicTokenCounter(),
        TokenBudgetPolicy(max_input_tokens=1000, reserve_output_tokens=100),
    )
    state = new_state("do a thing", thread_id="t", task_id="1")
    status = await manager.status(state)
    assert status["budget"] == 900
    assert status["messages"] == 1
    assert status["counter"] == "heuristic"
    prepared = await manager.prepare(state)
    assert prepared.tokens > 0 and not prepared.should_compact
    assert manager.force_compact_available(state) is False

    outcome = await manager.compact(state)
    assert outcome.applied is False and outcome.summary == ""


async def test_compact_service_folds_with_model(tmp_path):
    gateway = MockGateway(ModelConfig(provider="mock", model="m"))
    service = CompactService(gateway, keep_recent_messages=2)
    state = new_state("task", thread_id="t", task_id="1")
    state["messages"] = [Message(role="user", content=f"msg {i}") for i in range(10)]
    outcome = await service.compact(state)
    assert outcome.applied and outcome.folded == 8
    assert len(outcome.tail) == 2
    assert "mock compact summary" in outcome.summary
    patch = outcome.state_patch(state)
    assert patch["compact_count"] == 1 and len(patch["messages"]) == 2


async def test_compact_service_keeps_previous_summary_on_empty(tmp_path):
    class EmptyGateway(MockGateway):
        async def chat(self, request):
            self.calls.append(request)
            return ModelResponse(text="   ")

    gateway = EmptyGateway(ModelConfig(provider="mock", model="m"))
    service = CompactService(gateway, keep_recent_messages=1)
    state = new_state("task", thread_id="t", task_id="1")
    state["messages"] = [Message(role="user", content=f"m{i}") for i in range(6)]
    state["compact_summary"] = "previous"
    outcome = await service.compact(state)
    assert not outcome.applied and outcome.summary == "previous"


# ----------------------------------------------------------------------- events
def test_event_serializes():
    event = Event(type="tool_call", message="read_file", data={"id": "1"})
    assert event.model_dump()["type"] == "tool_call"
    assert str(event).startswith("[tool_call]")


# ------------------------------------------------------------------ model config
def test_trust_env_defaults_and_override():
    assert ModelConfig(model="x").trust_env is True
    assert ModelConfig(model="x", trust_env=False).trust_env is False


async def test_gateway_builds_a_proxy_free_client_when_trust_env_is_false(monkeypatch):
    """`trust_env: false` must bypass ambient proxy variables entirely.

    A shell whose NO_PROXY contains a bracketed IPv6 entry (``[::1]``) makes httpx
    raise ``InvalidURL`` the moment it parses the environment, so the adapter has
    to hand the SDK its own client instead of letting it read the environment.
    """

    from harness.inference.openai_compatible import OpenAICompatibleGateway

    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("NO_PROXY", "localhost,[::1]")

    gateway = OpenAICompatibleGateway(
        ModelConfig(provider="openai_compatible", model="m", api_key="EMPTY", trust_env=False)
    )
    inner = gateway.client._client
    assert inner.trust_env is False
    # this is what used to explode when the SDK read the environment itself
    assert inner._mounts == {}
