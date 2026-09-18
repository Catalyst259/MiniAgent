"""Regression tests for the findings of the adversarial audit.

Every test here reproduces an attack that *worked* against an earlier revision of
the permission layer. They are grouped by the audit's finding ids so a future
change that re-opens one of them fails loudly and points at the original report.

The recurring root cause behind all five: the layer reasoned about the argument a
rule matched on, while the tool's real side effect lived somewhere else - a
patch's ``move_to``, a program's own ``--output`` flag, ``grep``'s ``glob``
filter, ``glob``'s ``pattern``. Each test below asserts that the *side effect* is
what gets inspected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.agent.dto import ToolCall
from harness.inference.config import ModelConfig
from harness.infra.config import HarnessConfig
from harness.permission import (
    PermissionPolicy,
    ScriptedProvider,
    build_permission_stack,
    from_arguments,
)
from harness.permission.decision import Permission
from harness.permission.evaluator import PermissionEvaluator
from harness.permission.memory import PermissionMemory
from harness.permission.rules import Rule, matches_path
from harness.tools.fs_tools import ToolContext
from harness.tools.fs_tools import glob as glob_tool
from harness.tools.paths import Workspace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def shipped(tmp_path):
    """The shipped policy, on a real workspace."""

    workspace = Workspace(tmp_path)
    config = HarnessConfig.load(ROOT / "config.yaml")
    stack = build_permission_stack(
        config, workspace=workspace, approver=ScriptedProvider(["session"])
    )
    return stack, workspace


# ------------------------------------------------------------------------- E1
async def test_e1_mode_off_does_not_fail_open(tmp_path):
    """``mode: off`` disables the rules, never the ceiling.

    The gate used to return ALLOW for every call in ``mode: off`` *before*
    consulting ``ceiling_check()``, so ``denied_tools``, the read-only ceiling and
    the workspace boundary were all lifted - and ``mode`` is what a config that
    omits the ``permissions:`` section gets.
    """

    config = HarnessConfig()
    config.permissions.mode = "off"
    config.permissions.denied_tools = ["write_file"]
    stack = build_permission_stack(config, workspace=Workspace(tmp_path))
    assert stack.enabled is False

    results = await stack.gate.check_batch(
        [ToolCall(name="write_file", arguments={"path": "x.py"}, id="c1")]
    )
    assert results[0].denied
    assert results[0].verdict.source == "sandbox"


async def test_e1_subagent_style_ceiling_holds_when_permissions_are_off(tmp_path):
    config = HarnessConfig()
    config.permissions.mode = "off"
    stack = build_permission_stack(config, workspace=Workspace(tmp_path))
    stack.policy.read_only = True
    stack.evaluator.policy.read_only = True

    results = await stack.gate.check_batch(
        [
            ToolCall(name="write_file", arguments={"path": "x.py"}, id="w"),
            ToolCall(name="read_file", arguments={"path": "x.py"}, id="r"),
        ]
    )
    assert results[0].denied
    assert results[1].allowed


async def test_e1_matches_the_evaluator_verdict(tmp_path):
    """The gate and the evaluator must never disagree about the same action."""

    for mode in ("off", "ask", "auto"):
        config = HarnessConfig()
        config.permissions.mode = mode
        config.permissions.denied_tools = ["shell"]
        stack = build_permission_stack(config, workspace=Workspace(tmp_path))
        action = from_arguments("shell", {"command": "ls"})
        gate_result = await stack.gate.check_batch(
            [ToolCall(name="shell", arguments={"command": "ls"}, id="c1")]
        )
        assert gate_result[0].verdict.permission is stack.evaluator.evaluate(action).permission, mode


# ------------------------------------------------------------------------- E2
def test_e2_patch_move_destination_is_a_target():
    payload = (
        "*** Begin Patch\n"
        "*** Update File: app.py\n"
        "*** Move to: secrets/stolen.pem\n"
        "+key\n"
        "*** End Patch"
    )
    action = from_arguments("apply_patch", {"patch": payload})
    assert "secrets/stolen.pem" in action.targets()
    assert "app.py" in action.targets()


async def test_e2_patch_cannot_move_into_a_protected_path(shipped):
    stack, _workspace = shipped
    payload = (
        "*** Begin Patch\n"
        "*** Update File: app.py\n"
        "*** Move to: secrets/stolen.pem\n"
        "+key\n"
        "*** End Patch"
    )
    verdict = stack.evaluator.evaluate(from_arguments("apply_patch", {"patch": payload}))
    assert verdict.permission is Permission.DENY
    assert verdict.source == "sandbox"


async def test_e2_the_patch_engine_really_honours_move_to(tmp_path):
    """Guards the assumption behind the test above."""

    from harness.tools.fs_tools import TOOL_FUNCS

    root = tmp_path
    (root / "app.py").write_text("one\ntwo\n", encoding="utf-8")
    ctx = ToolContext(workspace=Workspace(root))
    TOOL_FUNCS["apply_patch"].fn(
        ctx,
        patch=(
            "*** Begin Patch\n*** Update File: app.py\n"
            "*** Move to: moved.py\n@@\n one\n-two\n+three\n*** End Patch"
        ),
    )
    assert (root / "moved.py").exists(), "apply_patch must implement `*** Move to:`"


# ------------------------------------------------------------------------- E6
# Not in the original report, but found while reproducing its findings: the
# enforcement lived only in the graph's permission *node*, so any code that called
# `ToolRuntime.run` directly executed a call the gate had denied.


async def test_e6_tool_runtime_enforces_the_gate_on_its_own(tmp_path):
    """`ToolRuntime.run` is the execution boundary, so it decides too."""

    root = tmp_path
    cfg = HarnessConfig(
        workspace_root=str(root),
        permissions={"mode": "auto", "default": "deny", "denied_tools": ["write_file"]},
    )
    from harness.core import AgentHarness

    harness = AgentHarness(cfg, workspace_root=root)
    harness.build()

    observation = await harness.tool_runtime.run(
        ToolCall(name="write_file", arguments={"path": "pwned.txt", "content": "x"}, id="c1")
    )
    assert not observation.ok
    assert "PERMISSION DENIED" in observation.error
    assert not (root / "pwned.txt").exists()
    await harness.close()


async def test_e6_a_call_the_gate_already_decided_is_not_asked_again(tmp_path):
    """The act node passes its decisions to the runtime, so nothing is asked twice.

    Regression for N1: this used to rely on a ``ContextVar`` written by the
    permission node, which LangGraph never propagates to a sibling node, so every
    call was decided and prompted twice.
    """

    from harness.core import AgentHarness
    from harness.permission import ScriptedProvider

    root = tmp_path
    cfg = HarnessConfig(
        workspace_root=str(root),
        permissions={"mode": "ask", "ask": ["write_file"]},
    )
    approver = ScriptedProvider(["once", "once"])
    harness = AgentHarness(cfg, workspace_root=root, approval_provider=approver)
    harness.build()

    call = ToolCall(name="write_file", arguments={"path": "a.txt", "content": "x"}, id="c1")
    results = await harness.permission.gate.check_batch([call])
    assert results[0].allowed
    assert len(approver.requests) == 1

    observation = await harness.tool_runtime.run(call, decided={call.id})
    assert observation.ok, observation.error
    assert (root / "a.txt").exists()
    assert len(approver.requests) == 1, "the runtime must not ask the same question twice"
    await harness.close()


async def test_e6_a_call_with_no_decision_is_still_gated(tmp_path):
    """The execution boundary decides a call that arrives without a verdict."""

    from harness.core import AgentHarness
    from harness.permission import ScriptedProvider

    cfg = HarnessConfig(
        workspace_root=str(tmp_path),
        permissions={"mode": "ask", "ask": ["write_file"]},
    )
    approver = ScriptedProvider(["reject"])
    harness = AgentHarness(cfg, workspace_root=tmp_path, approval_provider=approver)
    harness.build()

    observation = await harness.tool_runtime.run(
        ToolCall(name="write_file", arguments={"path": "b.txt", "content": "x"}, id="never-decided")
    )
    assert not observation.ok
    assert len(approver.requests) == 1, "the runtime must decide it itself"
    assert not (tmp_path / "b.txt").exists()
    await harness.close()


async def test_e6_unanswerable_question_fails_closed_without_blocking(tmp_path):
    """A batch whose caller knows no UI is live must refuse, not hang."""

    from harness.permission import ScriptedProvider

    root = tmp_path
    cfg = HarnessConfig(
        workspace_root=str(root),
        permissions={"mode": "ask", "ask": ["write_file"]},
    )
    stack = build_permission_stack(
        cfg, workspace=Workspace(root), approver=ScriptedProvider(["session"])
    )
    call = ToolCall(name="write_file", arguments={"path": "a.txt", "content": "x"}, id="c1")
    results = await stack.gate.check_batch([call], allow_when_unavailable=False)
    assert results[0].denied
    assert "no approval prompt can be shown" in results[0].verdict.reason


async def test_e6_decisions_do_not_leak_across_turns(tmp_path):
    """A reused call id must be decided again, not inherited as "already decided".

    Models generate ids like ``call_0`` every turn.  The decision set is scoped to
    one execution path (a ``run_many`` batch, one ``!command``), never to a turn or
    a session, so a later call with the same id is looked at afresh.
    """

    from harness.core import AgentHarness
    from harness.permission import ScriptedProvider
    from harness.tools.runtime import calls_already_decided, is_decided

    root = tmp_path
    cfg = HarnessConfig(
        workspace_root=str(root),
        permissions={"mode": "ask", "ask": ["write_file"]},
    )
    approver = ScriptedProvider(["once", "once"])
    harness = AgentHarness(cfg, workspace_root=root, approval_provider=approver)
    harness.build()

    call = ToolCall(name="write_file", arguments={"path": "a.txt", "content": "x"}, id="call_0")

    # inside the execution path the call is treated as already decided
    with calls_already_decided([call.id]):
        assert is_decided("call_0")
        await harness.tool_runtime.run(call)
    assert len(approver.requests) == 0, "a decided call must not be asked about"

    # outside it, the decision does not linger
    assert not is_decided("call_0")
    second = await harness.tool_runtime.run(call)
    assert len(approver.requests) == 1, "a call with no live decision must be gated"
    assert second.ok, second.error

    # and a rejecting approver still refuses a reused id
    rejecting = ScriptedProvider(["reject"])
    other = AgentHarness(cfg, workspace_root=root, approval_provider=rejecting)
    other.build()
    denied = await other.tool_runtime.run(
        ToolCall(name="write_file", arguments={"path": "b.txt", "content": "y"}, id="call_0")
    )
    assert not denied.ok
    assert not (root / "b.txt").exists()
    await harness.close()
    await other.close()


# ------------------------------------------------------------------------- N1
async def test_n1_each_call_is_prompted_exactly_once(tmp_path):
    """Regression: every gated call used to be decided (and asked about) twice.

    The permission node marked its batch in a ``ContextVar`` and the ``act`` node
    re-read it - but LangGraph runs each node in its own task, so sibling nodes
    never see each other's ``ContextVar`` writes.  The decision now travels in the
    turn's scratch dict and is handed to ``run_many``.
    """

    from harness.core import AgentHarness
    from harness.inference.mock_gateway import MockGateway, ScriptedResponse
    from harness.permission import ScriptedProvider

    cfg = HarnessConfig.load(ROOT / "config.yaml")
    cfg.models = {"main": ModelConfig(provider="mock", model="mock-main")}
    cfg.default_model = "main"
    cfg.runtime.workspace_root = str(tmp_path)
    cfg.memory.enabled = False
    cfg.checkpoint.enabled = False
    cfg.permissions.mode = "ask"
    cfg.permissions.rules = [{"tool": "write_file", "permission": "ask"}]
    cfg.permissions.allow = []
    cfg.permissions.ask = []
    cfg.permissions.deny = []

    approver = ScriptedProvider(["once", "once", "once", "once"])
    harness = AgentHarness(cfg, workspace_root=tmp_path, approval_provider=approver)
    harness.build()
    harness.set_gateway(
        MockGateway(
            cfg.resolve_model("main"),
            responses=[
                ScriptedResponse(
                    tool_calls=[
                        ToolCall(name="write_file", arguments={"path": "a.txt", "content": "A"}, id="c1"),
                        ToolCall(name="write_file", arguments={"path": "b.txt", "content": "B"}, id="c2"),
                    ]
                ),
                ScriptedResponse.say("done"),
            ],
        )
    )
    await harness.run("write two files")

    assert len(approver.requests) == 2, (
        f"each of the two calls must be asked about once, got {len(approver.requests)}: "
        f"{[action.describe() for action, _ in approver.requests]}"
    )
    assert [action.describe() for action, _ in approver.requests] == [
        "write_file(a.txt)",
        "write_file(b.txt)",
    ]
    assert (tmp_path / "a.txt").exists() and (tmp_path / "b.txt").exists()
    await harness.close()


async def test_n1_a_session_grant_answers_the_rest_of_the_batch(tmp_path):
    from harness.core import AgentHarness
    from harness.inference.mock_gateway import MockGateway, ScriptedResponse
    from harness.permission import ScriptedProvider

    cfg = HarnessConfig.load(ROOT / "config.yaml")
    cfg.models = {"main": ModelConfig(provider="mock", model="mock-main")}
    cfg.default_model = "main"
    cfg.runtime.workspace_root = str(tmp_path)
    cfg.memory.enabled = False
    cfg.checkpoint.enabled = False
    cfg.permissions.mode = "ask"
    cfg.permissions.rules = [{"tool": "write_file", "permission": "ask"}]
    cfg.permissions.allow = []
    cfg.permissions.ask = []
    cfg.permissions.deny = []

    approver = ScriptedProvider(["session", "session"])
    harness = AgentHarness(cfg, workspace_root=tmp_path, approval_provider=approver)
    harness.build()
    harness.set_gateway(
        MockGateway(
            cfg.resolve_model("main"),
            responses=[
                ScriptedResponse(
                    tool_calls=[
                        ToolCall(name="write_file", arguments={"path": "a.txt", "content": "A"}, id="c1"),
                        ToolCall(name="write_file", arguments={"path": "b.txt", "content": "B"}, id="c2"),
                    ]
                ),
                ScriptedResponse.say("done"),
            ],
        )
    )
    await harness.run("write two files")
    assert len(approver.requests) == 1, "one session answer covers the whole batch"
    assert (tmp_path / "a.txt").exists() and (tmp_path / "b.txt").exists()
    await harness.close()


# ------------------------------------------------------------------------- N2
@pytest.mark.parametrize(
    "command",
    [
        "git diff --no-index a.txt /etc/hostname",
        "git diff --no-index a.txt ../outside.txt",
        "git diff --no-index=/dev/null a.txt /etc/passwd",
        "git diff --no-index -- a.txt /etc/passwd",
    ],
)
async def test_n2_shell_read_operands_cannot_reach_outside(shipped, command):
    """``git diff --no-index <a> <b>`` prints any file the user can read.

    Regression: shell targets came only from *write* options, so a read-only
    allow rule for ``git diff`` was a licence to read the whole filesystem.
    """

    stack, _workspace = shipped
    verdict = stack.evaluator.evaluate(from_arguments("shell", {"command": command}))
    assert verdict.permission is Permission.DENY, command
    assert verdict.source == "sandbox"


@pytest.mark.parametrize(
    "command",
    [
        "git diff",
        "git diff a.txt",
        "git diff --stat",
        "git diff --cached",
        "git diff HEAD~1",
        "git log --oneline -3",
        "git log -p",
        "git log --follow -- a.txt",
        "git status --porcelain",
        "ls -la",
    ],
)
async def test_n2_ordinary_read_only_commands_are_unaffected(shipped, command):
    """The read-operand guard must not cost the rules their usefulness."""

    stack, _workspace = shipped
    verdict = stack.evaluator.evaluate(from_arguments("shell", {"command": command}))
    assert verdict.permission is Permission.ALLOW, command


def test_n2_read_operand_extraction():
    from harness.permission import shell as shell_mod

    tokens = shell_mod.parse("git diff --no-index a.txt /etc/passwd").tokens
    assert "/etc/passwd" in shell_mod.reading_targets(tokens)
    # a bare word operand is not a path
    assert shell_mod.reading_targets(shell_mod.parse("git diff --no-index a b").tokens) == []
    # no operand option, no read targets
    assert shell_mod.reading_targets(shell_mod.parse("git diff a.txt").tokens) == []


# ------------------------------------------------------------------------- E3
@pytest.mark.parametrize(
    "command",
    [
        "git log -p --output=/tmp/exfil.txt",
        "git log --output /tmp/exfil.txt",
        "git diff --output=/tmp/exfil.txt",
        "git log -o/tmp/exfil.txt",
    ],
)
async def test_e3_shell_write_options_cannot_escape_the_workspace(shipped, command):
    """A read-only command can still be told to write outside the workspace.

    ``git log -p --output=<path>`` satisifed the shipped ``prefix: git log`` allow
    rule and wrote workspace content to an arbitrary absolute path.
    """

    stack, _workspace = shipped
    verdict = stack.evaluator.evaluate(from_arguments("shell", {"command": command}))
    assert verdict.permission is Permission.DENY, command
    assert verdict.source == "sandbox"


async def test_e3_relative_write_escape_is_refused(shipped):
    stack, _workspace = shipped
    verdict = stack.evaluator.evaluate(
        from_arguments("shell", {"command": "git log --output=../../exfil.txt"})
    )
    assert verdict.permission is Permission.DENY


async def test_e3_the_allowed_forms_still_work(shipped):
    """The guard must not break the read-only commands the rules allow."""

    stack, _workspace = shipped
    for command in ("git log --oneline -5", "git status", "git diff", "ls -la"):
        verdict = stack.evaluator.evaluate(from_arguments("shell", {"command": command}))
        assert verdict.permission is Permission.ALLOW, command


async def test_e3_write_option_is_never_a_standing_grant(shipped):
    """A command carrying an output path can never be memorised as a rule."""

    stack, _workspace = shipped
    gate = stack.gate
    grant = gate.grant_rule(from_arguments("shell", {"command": "git log --output=/tmp/x"}))
    assert grant is None


# ------------------------------------------------------------------------- E4
def test_e4_glob_never_leaves_the_workspace(tmp_path):
    """An absolute pattern used to walk the real filesystem root."""

    workspace = Workspace(tmp_path)
    (tmp_path / "inside.txt").write_text("x\n", encoding="utf-8")
    ctx = ToolContext(workspace=workspace)

    for pattern in ("/etc/passwd", "/etc/*.conf", "/home/*/.ssh/*"):
        out = glob_tool(ctx, pattern=pattern)
        assert "no files match" in out, pattern

    # the pattern is interpreted inside the workspace instead
    assert "inside.txt" in glob_tool(ctx, pattern="/**/*.txt")


def test_e4_glob_rejects_parent_traversal(tmp_path):
    from harness.agent.errors import ToolPermissionError

    ctx = ToolContext(workspace=Workspace(tmp_path))
    with pytest.raises(ToolPermissionError):
        glob_tool(ctx, pattern="../*.txt")


async def test_e4_glob_absolute_pattern_is_denied_by_the_layer(shipped):
    stack, _workspace = shipped
    verdict = stack.evaluator.evaluate(from_arguments("glob", {"pattern": "/etc/*.conf"}))
    assert verdict.permission is Permission.DENY


# ------------------------------------------------------------------------- E5
async def test_e5_grep_glob_filter_is_inspected(shipped):
    """``grep(glob="*.pem")`` reads private keys; the filter is a target."""

    stack, _workspace = shipped
    verdict = stack.evaluator.evaluate(
        from_arguments("grep", {"pattern": "KEY", "glob": "*.pem", "path": "."})
    )
    assert verdict.permission is Permission.DENY
    assert verdict.source == "sandbox"


async def test_e5_grep_over_a_protected_directory_is_denied(shipped):
    stack, _workspace = shipped
    verdict = stack.evaluator.evaluate(
        from_arguments("grep", {"pattern": "KEY", "path": ".ssh"})
    )
    assert verdict.permission is Permission.DENY


async def test_e5_ordinary_grep_is_unaffected(shipped):
    stack, _workspace = shipped
    verdict = stack.evaluator.evaluate(
        from_arguments("grep", {"pattern": "def main", "glob": "*.py", "path": "src"})
    )
    assert verdict.permission is Permission.ALLOW


# ------------------------------------------------------------------------- B1
@pytest.mark.parametrize(
    "path",
    [
        "server.pem",
        "certs/server.pem",
        "private.key",
        "id_rsa",
        "keys/id_rsa",
        ".ssh/config",
        "home/.ssh/config",
        ".aws/credentials",
        ".env",
        "sub/.env",
    ],
)
def test_b1_protected_paths_cover_the_workspace_root(shipped, path):
    """``fnmatch`` alone never gives ``**/`` its "zero directories" meaning."""

    stack, _workspace = shipped
    verdict = stack.evaluator.evaluate(from_arguments("read_file", {"path": path}))
    assert verdict.permission is Permission.DENY, path


@pytest.mark.parametrize(
    ("candidate", "pattern", "expected"),
    [
        ("server.pem", "**/*.pem", True),
        ("certs/server.pem", "**/*.pem", True),
        ("id_rsa", "**/id_rsa*", True),
        (".ssh/config", "**/.ssh/**", True),
        ("a/.ssh/config", "**/.ssh/**", True),
        (".aws/credentials", "**/.aws/**", True),
        ("app.py", "**/*.py", True),
        ("src/app.py", "**/*.py", True),
        ("src/app.py", "**/*.pem", False),
        (".env", ".env", True),
        ("sub/.env", "**/.env", True),
        ("src/app.py", "src/*.py", True),
        ("src/deep/app.py", "src/*.py", False),
        # path spellings that name the same file must match the same rules
        ("./.env", ".env", True),
        ("./.env", "**/.env", True),
        ("./.ssh/config", "**/.ssh/**", True),
        ("src/./app.py", "src/*.py", True),
        ("./certs/server.pem", "**/*.pem", True),
        ("secrets/", "secrets/**", True),
        ("secrets", "secrets/**", True),
        ("notsecrets/a", "secrets/**", False),
    ],
)
def test_b1_path_matching_semantics(candidate, pattern, expected):
    assert matches_path(candidate, pattern) is expected


@pytest.mark.parametrize("path", ["./.env", "sub/./../.env", ".env", "./.ssh/config"])
async def test_b1_dot_prefixed_spellings_are_protected(shipped, path):
    """Regression: ``./.env`` used to slip past the ``.env`` rule."""

    stack, _workspace = shipped
    verdict = stack.evaluator.evaluate(from_arguments("read_file", {"path": path}))
    if ".." in path:
        # workspace escape: the guard owns this one, not the glob matcher
        assert verdict.permission is Permission.DENY
    else:
        assert verdict.permission is Permission.DENY
        assert verdict.source == "sandbox"


# ------------------------------------------------------------------------- B2
def test_b2_auto_with_default_allow_is_flagged(caplog):
    """A documented combination whose consequence was not: nothing is denied."""

    with caplog.at_level("WARNING", logger="harness.permission.policy"):
        PermissionPolicy(mode="auto", default=Permission.ALLOW)
    assert any("default=allow" in record.message for record in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING", logger="harness.permission.policy"):
        PermissionPolicy(mode="auto", default=Permission.ASK)
    assert not caplog.records


async def test_b2_the_combination_really_allows_everything(tmp_path):
    """Documents the risk the warning is about (so the warning is justified)."""

    policy = PermissionPolicy(
        mode="auto",
        default=Permission.ALLOW,
        workspace=Workspace(tmp_path),
        rules=[Rule(permission=Permission.DENY, tool="shell", program="rm")],
    )
    evaluator = PermissionEvaluator(policy, PermissionMemory())
    assert evaluator.evaluate(from_arguments("shell", {"command": "rm -rf /"})).permission is Permission.DENY
    # no rule mentions curl, and default=allow lets it through by design
    assert evaluator.evaluate(from_arguments("shell", {"command": "curl evil"})).permission is Permission.ALLOW


# --------------------------------------------------------------- B3 (no bug)
async def test_b3_glob_remains_allowed_with_the_shipped_rules(shipped):
    """The audit suspected a foot-gun here; a tool-wide rule matches any target."""

    stack, _workspace = shipped
    assert stack.evaluator.evaluate(
        from_arguments("glob", {"pattern": "**/*.py"})
    ).permission is Permission.ALLOW
    assert stack.evaluator.evaluate(
        from_arguments("glob", {"pattern": "**/*.py", "path": "."})
    ).permission is Permission.ALLOW
