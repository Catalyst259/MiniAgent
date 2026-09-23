"""Subprocess tests: the real CLI binary, the real MCP server, the real SQLite file."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
RUNNER = str(PYTHON if PYTHON.exists() else Path(sys.executable))


def run_cli(
    args: list[str] | tuple[str, ...] = (),
    *,
    stdin: str = "",
    timeout: int = 120,
    workspace: Path | None = None,
    transport: str = "local",
    cwd: Path = ROOT,
    checkpoint: Path | None = None,
    session: bool = False,
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["MINIAGENT_MODELS__MAIN__PROVIDER"] = "mock"
    env["MINIAGENT_MODELS__MAIN__MODEL"] = "mock-main"
    env["MINIAGENT_MEMORY__ENABLED"] = "false"
    env["MINIAGENT_CHECKPOINT__ENABLED"] = "true" if checkpoint else "false"
    if workspace:
        env["MINIAGENT_RUNTIME__WORKSPACE_ROOT"] = str(workspace)
    if checkpoint:
        env["MINIAGENT_CHECKPOINT__PATH"] = str(checkpoint)
    env["MINIAGENT_TOOLS__TRANSPORT"] = transport
    command = [RUNNER, "-m", "harness.main", *args]
    if session:
        # Exercise sessions programmatically; the CLI itself requires a terminal.
        command = [RUNNER, "-c", """
import asyncio
import sys
from harness.cli.app import MiniAgentApp, Session, load_config

async def run():
    app = MiniAgentApp(session=Session(load_config()))
    await app.setup()
    app.renderer.banner()
    async with app.session:
        for line in sys.stdin:
            if await app.handle_input(line.strip()):
                break

asyncio.run(run())
"""]
    return subprocess.run(
        command,
        input=stdin,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def demo_workspace() -> Path:
    directory = Path(tempfile.mkdtemp(prefix="harness-demo-"))
    (directory / "calc.py").write_text(
        "def add(a, b):\n    return a - b\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (directory / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8"
    )
    return directory


# ----------------------------------------------------------------- CLI binary
@pytest.mark.parametrize("stdin", ["", "你好\n/exit\n"])
def test_cli_rejects_piped_input(stdin):
    result = run_cli(stdin=stdin)
    assert result.returncode == 1
    assert result.stderr.strip() == "Error: stdin is not a terminal"
    assert not result.stdout


@pytest.mark.parametrize("args", [["--mock"], ["--help"], ["list files"]])
def test_cli_rejects_arguments(args):
    result = run_cli(args)
    assert result.returncode == 2
    assert "without arguments" in result.stderr


@pytest.mark.parametrize("transport", ["local", "mcp-stdio"])
def test_session_task_uses_tools(demo_workspace, transport):
    result = run_cli(session=True, workspace=demo_workspace, transport=transport, stdin="list files\n/exit\n")
    assert result.returncode == 0, result.stderr
    assert "list_dir" in result.stdout
    assert "calc.py" in result.stdout


def test_session_slash_commands(demo_workspace):
    stdin = "/status\n/skills\n/agents\n/tools\n/model\n/compact\n/clear\n/help\n/exit\n"
    result = run_cli(session=True, workspace=demo_workspace, stdin=stdin)
    assert result.returncode == 0
    out = result.stdout
    assert "MiniAgent" in out
    assert "workspace" in out and "thread" in out
    assert "debugging" in out and "repo_exploration" in out
    assert "planner" in out and "explorer" in out
    assert "apply_patch" in out
    assert "mock-main" in out
    assert "Nothing to compact" in out
    assert "conversation cleared" in out
    assert "/compact" in out


def test_session_runs_a_task_and_keeps_history(demo_workspace):
    stdin = "read calc.py\n/status\n/exit\n"
    result = run_cli(session=True, workspace=demo_workspace, stdin=stdin)
    assert result.returncode == 0
    assert "read_file" in result.stdout
    # /status after one turn must report a non-zero turn counter and a message history
    match = re.search(r"turn\s+(\d+)", result.stdout)
    assert match is not None and int(match.group(1)) >= 1
    history = re.search(r"messages\s+(\d+)", result.stdout)
    assert history is not None and int(history.group(1)) >= 2


# ------------------------------------------------------------------ MCP server
def test_mcp_server_is_importable_and_lists_tools():
    code = (
        "from pathlib import Path;"
        "from harness.tools.mcp_server import build_server;"
        "s = build_server(Path('.'));"
        "print(sorted(s._tool_manager._tools))"
    )
    result = subprocess.run(
        [RUNNER, "-c", code], capture_output=True, text=True, cwd=str(ROOT), timeout=60
    )
    assert result.returncode == 0
    assert "apply_patch" in result.stdout and "read_file" in result.stdout


# --------------------------------------------------------------- sqlite threads
def test_sqlite_checkpoint_file_is_created_and_listed(tmp_path, demo_workspace):
    checkpoint = tmp_path / "state" / "checkpoints.sqlite"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
models:
  main:
    provider: mock
    model: mock-main
default_model: main
runtime:
  workspace_root: {demo_workspace}
memory:
  enabled: false
  backend: memory
checkpoint:
  path: {checkpoint}
skills:
  paths: [{ROOT / 'skills'}]
subagents:
  paths: [{ROOT / 'subagents'}]
""",
        encoding="utf-8",
    )
    result = run_cli(session=True, cwd=tmp_path, checkpoint=checkpoint, stdin="list files\n/exit\n")
    assert result.returncode == 0, result.stderr
    assert checkpoint.exists()

    from harness.infra.checkpoint import list_threads

    assert list_threads(str(checkpoint)), "the run should have written at least one thread"
