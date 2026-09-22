"""Interactive approval: the provider, the status line and the key bindings.

The TUI path is the part of the permission layer that cannot be exercised with a
pseudo-terminal (the sandbox cannot allocate one), so it is covered the way the
rest of the CLI is: through ``prompt_toolkit``'s own frame and binding APIs plus
the real :class:`AppState` the renderer reads.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest

from harness.cli import events as ui
from harness.cli.approval import (
    InteractiveApprovalProvider,
    PlainApprovalProvider,
    build_options,
)
from harness.cli.state import AppState
from harness.inference.config import ModelConfig
from harness.infra.config import HarnessConfig
from harness.permission import (
    AutoDenyProvider,
    PermissionMemory,
    PermissionPolicy,
    ScriptedProvider,
    from_arguments,
)
from harness.permission.evaluator import PermissionEvaluator
from harness.permission.gate import PermissionGate
from harness.tools.paths import Workspace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def state() -> AppState:
    return AppState()


@pytest.fixture()
def provider(state) -> InteractiveApprovalProvider:
    return InteractiveApprovalProvider(state=state)


@pytest.fixture()
def gate(provider, tmp_path) -> PermissionGate:
    policy = PermissionPolicy(mode="ask", workspace=Workspace(tmp_path))
    evaluator = PermissionEvaluator(policy, PermissionMemory())
    return PermissionGate(evaluator, provider)


def write_call(call_id: str = "c1"):
    from harness.agent.dto import ToolCall

    return ToolCall(name="write_file", arguments={"path": "a.py", "content": "x"}, id=call_id)


# ------------------------------------------------------------------- provider
async def test_request_arms_a_future_and_shows_the_question(provider, state):
    request = provider.request(from_arguments("shell", {"command": "npm install x"}), "no rule")
    assert provider.waiting
    assert state.interaction is not None
    assert not request.future.done(), "the question must wait for a real answer"

    provider.offer({"can_session": True, "can_persist": False})
    assert [option.value for option in state.interaction.options] == ["once", "session", "reject"]


async def test_answer_resolves_the_awaiting_gate(provider, gate, state):
    task = asyncio.create_task(gate.check_batch([write_call()]))
    await asyncio.sleep(0)
    assert provider.waiting, "the gate must be blocked on the question"

    assert provider.resolve("once") is True
    results = await task
    assert results[0].allowed
    assert results[0].verdict.approval == "once"
    assert not provider.waiting
    assert state.interaction is None


async def test_reject_produces_a_denial(provider, gate):
    task = asyncio.create_task(gate.check_batch([write_call()]))
    await asyncio.sleep(0)
    assert provider.resolve("reject") is True
    results = await task
    assert results[0].denied
    assert "user rejected" in results[0].verdict.reason


async def test_session_choice_is_remembered_so_the_next_call_is_silent(provider, gate):
    first = asyncio.create_task(gate.check_batch([write_call("c1")]))
    await asyncio.sleep(0)
    provider.resolve("session")
    await first
    assert gate.evaluator.memory.session.rules()

    # second call: no question, answered from memory
    results = await gate.check_batch([write_call("c2")])
    assert results[0].allowed
    assert results[0].verdict.source == "session"
    assert not provider.waiting


async def test_unavailable_scope_is_refused_instead_of_silently_narrowed(provider):
    provider.request(from_arguments("write_file", {"path": "a.py"}), "no rule")
    provider.offer({"can_session": False, "can_persist": False})
    assert provider.resolve("persistent") is False, "must not answer with a scope it never offered"
    assert provider.waiting
    assert provider.resolve("once") is True


async def test_only_one_question_is_live_at_a_time(provider, gate):
    calls = [write_call("c1"), write_call("c2")]
    task = asyncio.create_task(gate.check_batch(calls))
    await asyncio.sleep(0)
    assert provider.pending is not None
    provider.resolve("once")
    await asyncio.sleep(0)
    assert provider.pending is not None, "the second call still needs an answer"
    provider.resolve("once")
    results = await task
    assert [result.allowed for result in results] == [True, True]


async def test_move_changes_the_selected_option(provider, state):
    provider.request(from_arguments("write_file", {"path": "a.py"}), "no rule")
    provider.offer({"can_session": True, "can_persist": True})
    assert state.interaction.current.value == "once"
    provider.move(1)
    assert state.interaction.current.value == "session"
    provider.move(-1)
    assert state.interaction.current.value == "once"
    # wrapping
    provider.move(-1)
    assert state.interaction.current.value == "reject"


async def test_accept_selection_uses_the_highlighted_option(provider):
    provider.request(from_arguments("write_file", {"path": "a.py"}), "no rule")
    provider.offer({"can_session": True, "can_persist": True})
    provider.move(1)
    assert provider.accept_selection() is True
    assert provider.history == [("write_file(a.py)", "session")]


# --------------------------------------------------------------------- options
def test_options_hide_scopes_the_action_cannot_keep():
    assert [option[0] for option in build_options({"can_session": False, "can_persist": False})] == [
        "once",
        "reject",
    ]
    assert [option[0] for option in build_options({"can_session": True, "can_persist": True})] == [
        "once",
        "session",
        "persistent",
        "reject",
    ]
    # persistent implies a writable store; session does not
    assert [option[0] for option in build_options({"can_session": True, "can_persist": False})] == [
        "once",
        "session",
        "reject",
    ]


def make_bindings(provider):
    from harness.cli.composer.composer import build_key_bindings

    class _Composer:
        popup = None

        def sync_from_buffer(self, *args, **kwargs):  # pragma: no cover - unused
            pass

        def sync_popups(self):  # pragma: no cover - unused
            pass

    return build_key_bindings(_Composer(), interaction=provider.interaction)


def test_every_option_has_a_distinct_key():
    options = build_options({"can_session": True, "can_persist": True})
    keys = [option[2] for option in options]
    assert keys == ["1", "2", "3", "4"]


async def test_activity_string_reports_waiting_for_approval(state):
    seen: list[str] = []
    provider = InteractiveApprovalProvider(state=state, activity=seen.append)
    provider.request(from_arguments("write_file", {"path": "a.py"}), "no rule")
    assert seen[-1] == "waiting for your approval"
    provider.resolve("once")
    assert seen[-1] == ""


# ---------------------------------------------------------------- key bindings
#: prompt_toolkit reports terminal keys by their control-code names.
_KEY_ALIASES = {"enter": "c-m", "tab": "c-i", "escape": "escape"}


def _pick(bindings, key: str):
    """The enabled single-key binding for ``key``.

    prompt_toolkit searches its binding registry in reverse registration order and
    calls the first one whose filter passes, so that is exactly what this does.
    Only exact one-key bindings are considered: multi-key sequences such as
    ``escape, enter`` (newline) must not be mistaken for the binding under test.
    """

    from prompt_toolkit.keys import Keys

    names = {key, _KEY_ALIASES.get(key, key)}
    wanted = {member for member in Keys if member.value in names}
    for binding in reversed(bindings.bindings):
        if len(binding.keys) != 1:
            continue
        item = binding.keys[0]
        if getattr(item, "value", item) in names or item in wanted:
            if binding.filter():
                return binding
    return None


def _all_for(bindings, key: str) -> list:
    """Every one-key binding for ``key``, in registration order."""

    from prompt_toolkit.keys import Keys

    names = {key, _KEY_ALIASES.get(key, key)}
    wanted = {member for member in Keys if member.value in names}
    found = []
    for binding in bindings.bindings:
        if len(binding.keys) != 1:
            continue
        item = binding.keys[0]
        if getattr(item, "value", item) in names or item in wanted:
            found.append(binding)
    return found


class _FakeApp:
    """prompt_toolkit calls ``event.app.invalidate()`` after every handler."""

    def __init__(self) -> None:
        self.invalidated = 0
        self.exited = False

    def invalidate(self) -> None:
        self.invalidated += 1

    def exit(self, result=None) -> None:  # noqa: ANN001 - prompt_toolkit signature
        self.exited = True


class FakeBuffer:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.cursor_position = len(text)


class FakeEvent:
    def __init__(self, text: str = "") -> None:
        self.current_buffer = FakeBuffer(text)
        self.app = _FakeApp()


async def test_number_keys_answer_only_while_a_question_is_open(provider):
    bindings = make_bindings(provider)

    # no question: the digit must reach the composer
    by_key = {}
    for binding in bindings.bindings:
        for key in binding.keys:
            by_key.setdefault(key, []).append(binding)
    ones = by_key.get("1", [])
    assert not any(binding.filter() for binding in ones), "digits must type normally when idle"

    provider.request(from_arguments("write_file", {"path": "a.py"}), "no rule")
    provider.offer({"can_session": True, "can_persist": True})
    assert any(binding.filter() for binding in by_key["1"])

    _pick(bindings, "1").call(FakeEvent())
    assert not provider.waiting
    assert provider.history[-1][1] == "once"


async def test_number_keys_select_the_matching_scope(provider):
    for key, expected in [("1", "once"), ("2", "session"), ("3", "persistent"), ("4", "reject")]:
        provider.request(from_arguments("write_file", {"path": "a.py"}), "no rule")
        provider.offer({"can_session": True, "can_persist": True})
        bindings = make_bindings(provider)
        _pick(bindings, key).call(FakeEvent())
        assert provider.history[-1][1] == expected, key


async def test_left_right_move_and_enter_confirms(provider):
    provider.request(from_arguments("write_file", {"path": "a.py"}), "no rule")
    provider.offer({"can_session": True, "can_persist": True})
    # bindings are built after the question exists: their filter reads live state
    bindings = make_bindings(provider)
    _pick(bindings, "right").call(FakeEvent())
    assert provider.state.interaction.current.value == "session"
    _pick(bindings, "left").call(FakeEvent())
    assert provider.state.interaction.current.value == "once"

    picked = _pick(bindings, "enter")
    assert picked.handler.__name__ == "_interaction_accept", "Enter must confirm, not submit"
    picked.call(FakeEvent())
    assert not provider.waiting
    assert provider.history[-1][1] == "once"


async def test_ctrl_r_rejects(provider):
    provider.request(from_arguments("write_file", {"path": "a.py"}), "no rule")
    bindings = make_bindings(provider)
    binding = _pick(bindings, "c-r")
    assert binding is not None
    binding.call(FakeEvent())
    assert provider.history[-1][1] == "reject"


async def test_enter_still_submits_when_no_question_is_open(provider):
    submitted: list[str] = []
    from harness.cli.composer.composer import build_key_bindings

    class _Composer:
        popup = None

        def sync_from_buffer(self, *args, **kwargs):
            pass

        def sync_popups(self):
            pass

    bindings = build_key_bindings(
        _Composer(), interaction=provider.interaction, on_submit=submitted.append
    )
    binding = _pick(bindings, "enter")
    assert binding is not None
    binding.call(FakeEvent("hello"))
    assert submitted == ["hello"]


async def test_pipe_input_answers_a_real_application(provider):
    """End-to-end: a real prompt_toolkit Application, driven by real keystrokes.

    The sandbox cannot allocate a pseudo-terminal, so keys are fed through
    ``create_pipe_input`` and the UI writes to a dummy output - the same layer
    the interactive prompt uses.  The application never exits on its own (it is
    a long-lived prompt), so the test cancels it after the answer lands.
    """

    from prompt_toolkit.application import Application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.output import DummyOutput

    bindings = make_bindings(provider)
    provider.request(from_arguments("shell", {"command": "npm install x"}), "no rule")
    provider.offer({"can_session": True, "can_persist": True})

    with create_pipe_input() as pipe:
        app = Application(
            layout=Layout(Window(FormattedTextControl(lambda: "approve?"))),
            key_bindings=bindings,
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text("3")  # "Always allow"
        task = asyncio.create_task(app.run_async())
        for _ in range(50):
            await asyncio.sleep(0.01)
            if not provider.waiting:
                break
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown
            pass

    assert not provider.waiting, "the key press must have answered the question"
    assert provider.history[-1][1] == "persistent"


async def test_cancel_releases_a_pending_question(provider, state):
    """Ctrl+C must not leave the gate waiting on a future nobody will resolve."""

    provider.request(from_arguments("write_file", {"path": "a.py"}), "no rule")
    future = provider.pending.future
    assert provider.cancel() is True
    assert future.done()
    assert (await future).scope == "reject"
    assert not provider.waiting
    assert state.interaction is None
    assert provider.history[-1][1] == "cancelled"
    # idempotent: a second cancel is a no-op, not a crash
    assert provider.cancel() is False


async def test_app_cancel_clears_the_question():
    from harness.cli.app import MiniAgentApp, Session
    from harness.cli.render.renderer import Renderer

    config = HarnessConfig()
    app = MiniAgentApp(session=Session(config), renderer=Renderer())
    app.interaction._answerable = lambda: True
    app.approval.request(from_arguments("write_file", {"path": "a.py"}), "no rule")
    assert app.approval.waiting
    app._on_cancel()
    assert not app.approval.waiting
    assert app.state.interaction is None


# -------------------------------------------------------------- plain provider
class FakeStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    def readline(self) -> str:
        return self._lines.pop(0) if self._lines else ""


class FakeOut:
    def __init__(self) -> None:
        self.text = ""

    def write(self, text: str) -> None:
        self.text += text

    def flush(self) -> None:
        pass


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("1", "once"),
        ("2", "session"),
        ("3", "persistent"),
        ("4", "reject"),
        ("y", "once"),
        ("n", "reject"),
        ("always", "persistent"),
        ("", "reject"),
    ],
)
async def test_plain_provider_maps_answers(answer, expected):
    out = FakeOut()
    provider = PlainApprovalProvider(stream=FakeStream([answer + "\n"]), out=out)
    request = provider.request(from_arguments("shell", {"command": "pytest"}), "no rule")
    approval = await request
    assert approval.scope == expected
    assert "Agent wants to run" in out.text
    assert "Reject" in out.text


async def test_plain_provider_fails_closed_on_eof():
    provider = PlainApprovalProvider(stream=FakeStream([]), out=FakeOut())
    request = provider.request(from_arguments("shell", {"command": "pytest"}), "no rule")
    approval = await request
    assert approval.scope == "reject"
    assert "no input" in approval.note


async def test_plain_provider_rejects_an_unrecognised_answer():
    out = FakeOut()
    provider = PlainApprovalProvider(stream=FakeStream(["banana\n"]), out=out)
    request = provider.request(from_arguments("shell", {"command": "pytest"}), "no rule")
    approval = await request
    assert approval.scope == "reject"
    assert "cannot answer" in out.text


async def test_plain_provider_hides_scopes_the_action_cannot_keep():
    """The stdin menu follows the same rule as the TUI menu."""

    out = FakeOut()
    provider = PlainApprovalProvider(stream=FakeStream(["3\n"]), out=out)
    # a chained command can never be matched by a standing rule
    provider.offer({"can_session": False, "can_persist": False})
    request = provider.request(from_arguments("shell", {"command": "npm i; rm -rf ~"}), "no rule")
    approval = await request
    assert approval.scope == "reject", "an unavailable answer must not be granted"
    assert "Always allow" not in out.text
    assert "cannot answer" in out.text


# --------------------------------------------------------- !command gating
def _entries(app) -> str:
    """Everything the app rendered, whichever console it was built with."""

    console = app.renderer.console
    if hasattr(console, "plain"):
        return console.plain()
    return "\n".join(str(entry) for entry in getattr(console, "entries", []))


def make_shell_app(tmp_path, approver, *, mode="ask", rules=None):
    """An app whose session runs with an explicit permission posture."""

    from harness.cli.app import MiniAgentApp, Session
    from harness.cli.render.renderer import Renderer
    from tests.helpers import RecordingConsole

    config = HarnessConfig.load(ROOT / "config.yaml")
    config.models = {"main": ModelConfig(provider="mock", model="mock-main")}
    config.default_model = "main"
    config.runtime.workspace_root = str(tmp_path)
    config.memory.enabled = False
    config.checkpoint.enabled = False
    config.permissions.mode = mode
    if rules is not None:
        config.permissions.rules = rules
        config.permissions.allow = []
        config.permissions.ask = []
        config.permissions.deny = []
    return MiniAgentApp(
        session=Session(config, approval_provider=approver),
        renderer=Renderer(console=RecordingConsole(), use_live_tail=False),
    )


async def test_bang_command_prompts_and_runs_when_allowed(tmp_path):
    """`!command` goes through the same gate as the model's tools."""

    out = io.StringIO()
    approver = PlainApprovalProvider(stream=io.StringIO("2\n"), out=out)
    app = make_shell_app(
        tmp_path, approver, rules=[{"tool": "shell", "permission": "ask"}]
    )
    await app.setup()
    async with app.session:
        await app.handle_input("!echo gated-ok")

    text = _entries(app)
    assert "gated-ok" in text, "the approved command must have run"
    assert [choice for _action, choice in approver.history] == ["session"]
    # the prompt reached the user through the provider
    assert "Agent wants to run" in out.getvalue()


async def test_bang_command_is_refused_without_running(tmp_path):
    approver = PlainApprovalProvider(stream=io.StringIO("4\n"), out=io.StringIO())
    app = make_shell_app(
        tmp_path, approver, rules=[{"tool": "shell", "permission": "ask"}]
    )
    await app.setup()
    async with app.session:
        await app.handle_input("!echo should-not-run")

    text = _entries(app)
    # the command line is echoed in the trace, but it never produced output
    assert "exit_code" not in text, "a refused command must not run"
    assert "permission denied" in text.lower()
    assert [choice for _action, choice in approver.history] == ["reject"]


async def test_bang_command_denied_by_rule_without_prompting(tmp_path):
    """A configured deny needs no question, and must not block the loop."""

    approver = PlainApprovalProvider(stream=io.StringIO(""), out=io.StringIO())
    app = make_shell_app(
        tmp_path, approver, rules=[{"risk": "high", "permission": "deny"}]
    )
    await app.setup()
    async with app.session:
        await app.handle_input("!echo nope")

    text = _entries(app)
    assert "exit_code" not in text, "a denied command must not run"
    assert "permission denied" in text.lower()
    assert approver.history == [], "a deny rule must not ask"


async def test_bang_command_without_a_live_ui_fails_closed(tmp_path):
    """A question that cannot be shown must be refused, never hang the turn.

    Regression: the `!command` path used to consult the interactive provider even
    when no application loop was running, so the turn waited forever on a future
    no keystroke could resolve.
    """

    provider = InteractiveApprovalProvider(state=AppState())
    provider.interaction._answerable = lambda: False
    app = make_shell_app(
        tmp_path, provider, rules=[{"tool": "shell", "permission": "ask"}]
    )
    await app.setup()
    async with app.session:
        await app.handle_input("!echo silent-hang")

    text = _entries(app)
    assert "exit_code" not in text, "the command must not run"
    assert "no approval prompt can be shown" in text
    assert not provider.waiting, "no question may be left open"


async def test_bang_command_allowed_by_rule_runs_without_prompting(tmp_path):
    approver = PlainApprovalProvider(stream=io.StringIO(""), out=io.StringIO())
    app = make_shell_app(
        tmp_path, approver, rules=[{"tool": "shell", "prefix": "echo", "permission": "allow"}]
    )
    await app.setup()
    async with app.session:
        await app.handle_input("!echo allowed-direct")

    text = _entries(app)
    assert "allowed-direct" in text
    assert approver.history == []


# ------------------------------------------------------------------ app wiring
async def test_app_wires_the_provider_into_the_harness(tmp_path):
    """The session's harness must be built with the TUI approver, not auto-deny."""

    from harness.cli.app import MiniAgentApp, Session
    from harness.cli.render.renderer import Renderer

    config = HarnessConfig.load(ROOT / "config.yaml")
    config.models = {"main": ModelConfig(provider="mock", model="mock-main")}
    config.default_model = "main"
    config.runtime.workspace_root = str(tmp_path)
    config.memory.enabled = False
    config.checkpoint.enabled = False

    app = MiniAgentApp(session=Session(config), renderer=Renderer())
    async with app.session:
        assert isinstance(app.session.harness.approval_provider, InteractiveApprovalProvider)
        assert app.session.harness.permission.gate.approver is app.approval
        assert app.session.harness.permission.evaluator.can_prompt is True


async def test_app_keeps_a_supplied_non_interactive_provider(tmp_path):
    """A batch/plain run must not be handed a key-driven approver.

    Regression: installing the TUI provider unconditionally made a one-shot run
    wait forever for a key press that could never come.
    """

    import io

    from harness.cli.app import MiniAgentApp, Session
    from harness.cli.render.renderer import Renderer

    config = HarnessConfig.load(ROOT / "config.yaml")
    config.models = {"main": ModelConfig(provider="mock", model="mock-main")}
    config.default_model = "main"
    config.runtime.workspace_root = str(tmp_path)
    config.memory.enabled = False
    config.checkpoint.enabled = False

    plain = PlainApprovalProvider(stream=io.StringIO("1\n"), out=io.StringIO())
    app = MiniAgentApp(session=Session(config, approval_provider=plain), renderer=Renderer())
    assert app.approval is plain
    assert app.session.approval_provider is plain
    async with app.session:
        assert app.session.harness.permission.gate.approver is plain
        assert app.session.harness.permission.evaluator.can_prompt is True


def test_app_tolerates_a_session_without_a_harness():
    """``__post_init__`` must not explode before the session is entered."""

    from harness.cli.app import MiniAgentApp, Session
    from harness.cli.render.renderer import Renderer

    config = HarnessConfig()
    app = MiniAgentApp(session=Session(config), renderer=Renderer())
    assert app.approval is not None


async def test_permission_decision_event_becomes_a_transcript_cell():
    from harness.cli.app import MiniAgentApp, Session
    from harness.cli.render.renderer import Renderer

    config = HarnessConfig()
    app = MiniAgentApp(session=Session(config), renderer=Renderer())
    app.emit(
        ui.PermissionDecided(
            call_id="c1", tool="write_file", permission="deny", reason="rule deny", source="rule"
        )
    )
    assert app.state.history_cells
    assert "permission denied" in app.state.history_cells[-1].message

    app.emit(
        ui.PermissionDecided(
            call_id="c2",
            tool="write_file",
            permission="allow",
            reason="allowed for this session",
            source="approval",
            approval="session",
        )
    )
    assert "allowed for this session" in app.state.history_cells[-1].message


async def test_tool_denied_event_fails_the_running_cell():
    from harness.cli.app import MiniAgentApp, Session
    from harness.cli.render.renderer import Renderer

    config = HarnessConfig()
    app = MiniAgentApp(session=Session(config), renderer=Renderer())
    app.emit(ui.ToolStarted(call_id="c1", tool="write_file", arguments={"path": "a.py"}))
    app.emit(ui.ToolFailed(call_id="c1", tool="write_file", error="PERMISSION DENIED"))
    cell = app.state.history_cells[-1]
    assert cell.status.value == "failed"


def test_bridge_translates_permission_events():
    from harness.agent.events import Event
    from harness.cli.events_bridge import translate

    denied = translate(
        Event(
            type="permission_decision",
            message="deny",
            data={"id": "c1", "tool": "write_file", "permission": "deny", "reason": "rule"},
        )
    )
    assert isinstance(denied, ui.PermissionDecided)
    assert denied.permission == "deny"

    # the question itself is rendered by the provider, not as a cell
    assert translate(Event(type="permission_ask", message="x", data={})) is None
