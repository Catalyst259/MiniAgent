"""MCP protocol tests: server, stdio client adapter and registry/runtime wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from harness.agent.dto import ToolCall
from harness.tools.fs_tools import TOOL_FUNCS, ToolContext
from harness.tools.local_backend import LocalToolBackend
from harness.tools.mcp_client import MCPClientAdapter, build_stdio_backend
from harness.tools.mcp_server import build_server
from harness.tools.paths import Workspace
from harness.tools.registry import ToolRegistry
from harness.tools.runtime import ToolRuntime, validate_arguments

@pytest.fixture(scope="module")
def workspace() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mcp_backend(workspace):
    backend = build_stdio_backend(workspace)
    yield backend
    backend.close()


# ------------------------------------------------------------------ server side
def test_server_exposes_every_tool(workspace):
    server = build_server(workspace)
    names = set(server._tool_manager._tools)
    assert names == set(TOOL_FUNCS)


def test_mcp_schema_matches_local_schema(workspace):
    server = build_server(workspace)
    for name, spec in TOOL_FUNCS.items():
        tool = server._tool_manager.get_tool(name)
        schema = tool.parameters
        properties = set((schema.get("properties") or {}))
        assert set(spec.parameters["properties"]) == properties, name
        assert set(spec.parameters.get("required") or []) <= set(schema.get("required") or []), name


# ------------------------------------------------------------------ client side
def test_mcp_backend_lists_tools_with_schemas(mcp_backend):
    specs = {spec.name: spec for spec in mcp_backend.list_specs()}
    assert set(specs) == set(TOOL_FUNCS)
    assert specs["apply_patch"].input_schema["required"] == ["patch"]
    assert "read_file" in specs["apply_patch"].description or "patch" in specs["apply_patch"].description


def test_mcp_backend_calls_tools(mcp_backend):
    out = mcp_backend.call("read_file", {"path": "pyproject.toml", "limit": 2})
    assert "pyproject.toml" in out and "1\t" in out

    listing = mcp_backend.call("list_dir", {})
    assert "harness/" in listing

    matches = mcp_backend.call("grep", {"pattern": "miniagent", "glob": "*.toml"})
    assert "pyproject.toml" in matches


def test_mcp_backend_reports_missing_arguments(mcp_backend):
    # The MCP layer validates required arguments before the tool body runs.
    out = mcp_backend.call("read_file", {})
    assert out.startswith("ERROR:") and "path" in out


def test_mcp_backend_tolerates_extra_arguments(mcp_backend):
    # Unknown keys are dropped by MCP input validation, so the call still works.
    out = mcp_backend.call("read_file", {"path": "pyproject.toml", "limit": 1, "bogus": 1})
    assert "pyproject.toml" in out and not out.startswith("ERROR")


def test_mcp_backend_unknown_tool(mcp_backend):
    out = mcp_backend.call("no_such_tool", {})
    assert out.startswith("ERROR:") and "no_such_tool" in out


# --------------------------------------------------------------- registry/runtime
def test_registry_merges_backends(workspace, mcp_backend):
    registry = ToolRegistry()
    registry.register_backend(LocalToolBackend(ToolContext(workspace=Workspace(workspace))))
    registry.register_backend(mcp_backend)
    assert registry.names() == sorted(TOOL_FUNCS)
    schemas = registry.openai_schemas(only=["read_file", "grep"])
    assert [schema["function"]["name"] for schema in schemas] == ["read_file", "grep"]


async def test_runtime_through_mcp_backend(workspace, mcp_backend):
    registry = ToolRegistry()
    registry.register_backend(mcp_backend)
    runtime = ToolRuntime(registry, allowed=["read_file", "shell"], timeout_seconds=60)
    assert runtime.visible_tools() == ["read_file", "shell"]

    observation = await runtime.run(ToolCall(name="read_file", arguments={"path": "README.md", "limit": 3}))
    assert observation.ok and "README" in observation.content

    denied = await runtime.run(ToolCall(name="write_file", arguments={"path": "x", "content": "y"}))
    assert not denied.ok and "not available" in (denied.error or "")


async def test_agent_loop_can_use_mcp_transport(workspace, tmp_path):
    """Full loop with tools served by the MCP stdio server instead of in-process."""

    from harness.agent.state import new_state
    from harness.core import AgentHarness
    from harness.inference.config import ModelConfig
    from harness.inference.mock_gateway import MockGateway, ScriptedResponse
    from harness.infra.config import HarnessConfig

    (tmp_path / "hello.py").write_text("print('hello')\n", encoding="utf-8")
    config = HarnessConfig(
        models={"main": ModelConfig(provider="mock", model="mock-main")}, default_model="main"
    )
    config.base_dir = str(workspace)
    config.runtime.workspace_root = str(tmp_path)
    config.memory.enabled = False
    config.tools.transport = "mcp-stdio"
    harness = AgentHarness(config, workspace_root=tmp_path)
    harness.build()
    harness.set_gateway(
        MockGateway(
            config.resolve_model("main"),
            responses=[
                ScriptedResponse.tool("read_file", path="hello.py"),
                ScriptedResponse.say("read it over MCP"),
            ],
        )
    )
    try:
        result = await harness.run("read hello.py", state=new_state("t", thread_id="mcp", task_id="1"))
    finally:
        await harness.close()
    assert result["termination_status"] == "final_answer"
    assert result["observations"] and result["observations"][0].ok
    assert "hello" in result["observations"][0].content


# ------------------------------------------------------------------- validation
def test_validate_arguments_type_and_required():
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    call = ToolCall(name="read_file", arguments={})
    assert "missing required" in validate_arguments(call, schema)
    assert "unexpected argument" in validate_arguments(
        ToolCall(name="read_file", arguments={"path": "a", "b": 1}), schema
    )
    assert "must be a integer" in validate_arguments(
        ToolCall(name="read_file", arguments={"path": "a", "limit": "5"}), schema
    )
    assert validate_arguments(ToolCall(name="read_file", arguments={"path": "a", "limit": 5}), schema) is None
    assert validate_arguments(ToolCall(name="x", arguments={"path": "a", "limit": True}), schema)
