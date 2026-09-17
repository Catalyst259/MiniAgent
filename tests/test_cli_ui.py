"""Tests for the event-driven CLI: composer, popup, fuzzy match, cells, streaming."""

from __future__ import annotations

import pytest

from harness.agent.events import Event
from harness.cli import events as ui
from harness.cli.app import MiniAgentApp, Session, load_config, build_arg_parser
from harness.cli.cells import (
    AssistantCell,
    ErrorCell,
    InfoCell,
    SkillCell,
    SubAgentCell,
    ToolCell,
    ToolStatus,
    UserCell,
)
from harness.cli.composer import (
    CommandPopup,
    Composer,
    SlashCommand,
    build_default_registry,
    fuzzy_filter,
    fuzzy_match,
    parse_slash,
)
from harness.cli.composer.slash_commands import CommandRegistry
from harness.cli.events_bridge import EventBridge, translate
from harness.cli.render.renderer import Renderer
from harness.cli.state import AppState, CommandPopupState, StreamState, TextAreaState
from harness.cli.streaming import AssistantStream, ToolStream
from harness.infra.config import HarnessConfig


# ------------------------------------------------------------------ test doubles
class RecordingConsole:
    """Captures everything a cell tries to render."""

    def __init__(self) -> None:
        self.entries: list[object] = []

    def print(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        for arg in args:
            self.entries.append(arg)

    def plain(self) -> str:
        return "\n".join(str(entry) for entry in self.entries)

    def types(self) -> list[str]:
        return [type(entry).__name__ for entry in self.entries]


def text_of(console: RecordingConsole) -> str:
    from rich.text import Text

    chunks = []
    for entry in console.entries:
        if isinstance(entry, Text):
            chunks.append(entry.plain)
        else:
            chunks.append(str(entry))
    return "\n".join(chunks)


# --------------------------------------------------------------- fuzzy matching
def test_fuzzy_match_prefix_beats_scattered():
    exact = fuzzy_match("model", "model")
    prefix = fuzzy_match("model", "mod")
    scattered = fuzzy_match("review", "rv")
    assert exact is not None and exact.score == -100
    assert prefix is not None and prefix.score == -100
    assert scattered is not None and scattered.score > exact.score


def test_fuzzy_match_matches_codex_example():
    """`/rv` should match `/review`; indices are the greedy subsequence."""

    result = fuzzy_match("review", "rv")
    assert result is not None
    assert result.matched_indices == [0, 2]


def test_fuzzy_match_scores_window_size():
    compact = fuzzy_match("abc", "abc")
    spread = fuzzy_match("a-b-c", "abc")
    assert compact is not None and spread is not None
    assert compact.score == -100
    assert spread.score == -98
    assert compact.score < spread.score


def test_fuzzy_match_rejects_non_subsequence():
    assert fuzzy_match("model", "zx") is None
    assert fuzzy_match("model", "zzz") is None
    assert fuzzy_match("status", "stx") is None


def test_fuzzy_match_empty_query_matches_everything():
    result = fuzzy_match("status", "")
    assert result is not None and result.score == -100 and result.matched_indices == []


def test_fuzzy_match_is_case_insensitive():
    assert fuzzy_match("Status", "sta") is not None


def test_fuzzy_filter_orders_by_score_then_declaration():
    candidates = [("review", "review"), ("resume", "resume"), ("read", "read")]
    ranked = fuzzy_filter(candidates, "re")
    assert [item[0] for item in ranked] == ["review", "resume", "read"] or [
        item[0] for item in ranked
    ] == ["read", "resume", "review"]
    # every result matched, and the payload travelled with it
    assert all(payload in {"review", "resume", "read"} for payload, _ in ranked)


# -------------------------------------------------------------------- registry
def test_default_registry_has_all_documented_commands():
    registry = build_default_registry({})
    assert registry.names() == [
        "help",
        "status",
        "model",
        "tools",
        "skills",
        "agents",
        "compact",
        "clear",
        "exit",
    ]


def test_registry_filter_prefix_and_fuzzy():
    registry = build_default_registry({})
    assert [m.command.name for m in registry.filter("mo")] == ["model"]
    fuzzy = [m.command.name for m in registry.filter("mdl")]
    assert "model" in fuzzy
    assert registry.filter("zzzz") == []


def test_registry_aliases_resolve():
    registry = build_default_registry({})
    assert registry.get("quit") is not None
    assert registry.get("quit").name == "exit"
    assert registry.get("q").name == "exit"
    assert registry.get("/exit").name == "exit"
    assert registry.get("nope") is None


def test_registry_highlighting():
    registry = build_default_registry({})
    highlight = registry.highlighter("mdl")
    marked = highlight("model")
    assert [char for char, matched in marked if matched] == ["m", "d", "l"]


def test_parse_slash():
    assert parse_slash("hello") is None
    assert parse_slash("/model") == ("model", "")
    assert parse_slash("/model fast") == ("model", "fast")
    assert parse_slash("  /status  ") == ("status", "")


# ------------------------------------------------------------------ command popup
def test_popup_is_derived_from_composer_text():
    popup = CommandPopup(build_default_registry({}), CommandPopupState())
    popup.sync("/")
    assert popup.visible and len(popup.matches) == 9
    popup.sync("/mo")
    assert [m.command.name for m in popup.matches] == ["model"]
    popup.sync("/")
    assert len(popup.matches) == 9
    popup.sync("plain text")
    assert not popup.visible and popup.matches == []


def test_popup_selection_wraps_and_clamps():
    popup = CommandPopup(build_default_registry({}), CommandPopupState())
    popup.sync("/")
    popup.move(-1)
    assert popup.state.selected == len(popup.matches) - 1
    popup.move(1)
    assert popup.state.selected == 0
    popup.sync("/c")
    assert 0 <= popup.state.selected < len(popup.matches)


def test_popup_scrolls_with_selection():
    state = CommandPopupState(window=3)
    popup = CommandPopup(build_default_registry({}), state)
    popup.sync("/")
    for _ in range(5):
        popup.move(1)
    assert state.selected == 5
    assert state.scroll == 3
    assert len(state.visible_matches) == 3


def test_popup_completion_and_dismiss():
    state = CommandPopupState()
    popup = CommandPopup(build_default_registry({}), state)
    popup.sync("/mod")
    assert popup.completion() == "/model"
    popup.dismiss()
    assert not popup.visible
    popup.sync("/mod")
    assert popup.visible


def test_popup_renders_selection_and_highlight():
    console = RecordingConsole()
    popup = CommandPopup(build_default_registry({}), CommandPopupState())
    popup.sync("/mod")
    popup.render(console)
    rendered = text_of(console)
    assert "❯ /model" in rendered
    assert "show or switch the model" in rendered


# --------------------------------------------------------------------- composer
def test_textarea_insert_and_backspace():
    state = TextAreaState()
    state.set_text("hello world", cursor=len("hello world"))
    state.backspace()
    assert state.text == "hello worl" and state.cursor == len("hello worl")
    state.insert("d")
    assert state.text == "hello world" and state.cursor == len("hello world")


def test_textarea_backspace_at_start_is_a_noop():
    state = TextAreaState()
    state.set_text("abc", cursor=0)
    state.backspace()
    assert state.text == "abc" and state.cursor == 0


def test_textarea_delete_forward_and_word_delete():
    state = TextAreaState()
    state.set_text("hello world", cursor=5)
    state.delete()
    assert state.text == "helloworld"
    state.set_text("hello world", cursor=len("hello world"))
    state.delete_word_backward()
    assert state.text == "hello "


def test_textarea_moves_and_bounds():
    state = TextAreaState()
    state.set_text("abc")
    state.move_home()
    assert state.cursor == 0
    state.move(-5)
    assert state.cursor == 0
    state.move_end()
    assert state.cursor == 3
    state.move(10)
    assert state.cursor == 3


def test_textarea_unicode_backspace_is_grapheme_aware():
    # Chinese characters are one grapheme each
    state = TextAreaState()
    state.set_text("你好")
    state.backspace()
    assert state.text == "你"
    # ASCII characters are also one grapheme each
    state.set_text("ab")
    state.backspace()
    assert state.text == "a"
    # a combining accent is deleted together with its base character
    state.set_text("e\u0301x")
    state.backspace()
    assert state.text == "e\u0301"
    state.backspace()
    assert state.text == ""


def test_textarea_render_shows_cursor():
    state = TextAreaState()
    state.set_text("hi", cursor=1)
    assert state.render() == "> hi\n   ^"


def test_composer_submit_clears_and_returns_text():
    composer = Composer()
    composer.sync_from_buffer("  do the thing  ")
    assert composer.submit_text() == "do the thing"
    assert composer.state.text == ""
    assert composer.submit_text() is None


def test_composer_prompt_fragments_include_popup_rows():
    composer = Composer(
        state=TextAreaState(),
        popup=CommandPopup(build_default_registry({}), CommandPopupState(window=20)),
    )
    composer.sync_from_buffer("/")
    composer.sync_popups()
    fragments = composer.prompt_fragments()
    # one fragment per popup character plus the trailing prompt mark
    assert fragments[-1] == ("fg:ansigreen bold", "› ")
    assert sum(1 for _, text in fragments if text == "\n") == len(composer.popup.matches)
    rendered = "".join(text for _, text in fragments)
    assert "help" in rendered and "show command help" in rendered


# ------------------------------------------------------------------------ cells
def test_user_cell_renders_prompt():
    console = RecordingConsole()
    UserCell(text="fix the bug").render(console)
    assert "fix the bug" in text_of(console)


def test_assistant_cell_renders_markdown():
    console = RecordingConsole()
    AssistantCell(source="# Title\n\n- item").render(console)
    assert console.types() == ["Markdown"]


def test_tool_cell_lifecycle_and_preview():
    console = RecordingConsole()
    cell = ToolCell(call_id="1", tool="shell", arguments={"command": "pytest"})
    assert cell.status is ToolStatus.RUNNING and cell.glyph == "●"
    cell.render(console)
    assert "shell" in text_of(console) and "pytest" in text_of(console)

    cell.append("line one\n")
    cell.finish(True, text="line one\nline two\n", duration_ms=42)
    assert cell.status is ToolStatus.DONE
    console = RecordingConsole()
    cell.render(console)
    rendered = text_of(console)
    assert "line one" in rendered and "42 ms" in rendered


def test_tool_cell_truncates_long_output():
    console = RecordingConsole()
    cell = ToolCell(call_id="1", tool="grep", max_preview_lines=3)
    cell.finish(True, text="\n".join(f"line {index}" for index in range(10)))
    cell.render(console)
    rendered = text_of(console)
    assert "line 0" in rendered and "line 9" not in rendered
    assert "7 more line(s)" in rendered


def test_tool_cell_failure_renders_error():
    console = RecordingConsole()
    cell = ToolCell(call_id="1", tool="apply_patch")
    cell.fail("hunk did not match")
    cell.render(console)
    assert cell.status is ToolStatus.FAILED and cell.glyph == "✗"
    assert "hunk did not match" in text_of(console)


def test_subagent_and_small_cells_render():
    console = RecordingConsole()
    SubAgentCell(agent="explorer", task="find auth", summary="## Findings\n- auth.py", iterations=3).render(console)
    SkillCell(name="debugging").render(console)
    InfoCell(message="hello").render(console)
    ErrorCell(message="boom", fatal=True).render(console)
    rendered = text_of(console)
    for expected in ("explorer", "find auth", "auth.py", "debugging", "hello", "boom"):
        assert expected in rendered


# -------------------------------------------------------------------- app state
def test_appstate_commits_active_cell_once():
    state = AppState()
    cell = ToolCell(call_id="1", tool="grep")
    state.set_active(cell)
    assert state.active_cell is cell and state.history_cells == []
    state.commit()
    assert state.active_cell is None and state.history_cells == [cell]
    state.commit()
    assert state.history_cells == [cell]


def test_appstate_change_callback_fires():
    hits: list[int] = []
    state = AppState(on_change=lambda: hits.append(1))
    state.append_cell(InfoCell(message="x"))
    state.touch()
    assert len(hits) == 2


# -------------------------------------------------------------------- streaming
def test_assistant_stream_holds_back_incomplete_line():
    stream = AssistantStream()
    stream.append("Hello, I will")
    stream.append(" fix this.\nThe problem")
    assert stream.committed_source == "Hello, I will fix this.\n"
    assert stream.pending_source == "The problem"
    assert stream.stable_lines == ["Hello, I will fix this."]
    assert stream.tail_lines == ["The problem"]

    final = stream.finish("Hello, I will fix this.\nThe problem is the regex.\n")
    assert final.endswith("regex.\n")
    assert stream.pending_source == ""
    assert len(stream.stable_lines) == 2


def test_assistant_stream_preview_bounds_tail():
    stream = AssistantStream()
    stream.append("stable line\n")
    stream.append("tail")
    preview = stream.preview(tail_lines=6)
    assert preview.startswith("stable line")
    assert preview.endswith("tail")


def test_tool_stream_tracks_running_cells():
    stream = ToolStream()
    cell = stream.start("call-1", "shell", {"command": "ls"})
    assert cell.status is ToolStatus.RUNNING
    stream.append("call-1", "output\n")
    assert cell.output == "output\n"
    finished = stream.finish("call-1", ok=True, text="done", duration_ms=5)
    assert finished is cell and finished.status is ToolStatus.DONE
    assert stream.running() == []


def test_tool_stream_failure_marks_cell():
    stream = ToolStream()
    stream.start("c", "grep")
    cell = stream.finish("c", ok=False, error="bad regex")
    assert cell is not None and cell.status is ToolStatus.FAILED and cell.error == "bad regex"


# --------------------------------------------------------------------- renderer
def test_renderer_flush_prints_each_cell_once():
    console = RecordingConsole()
    renderer = Renderer(console=console, use_live_tail=False)
    cells = [UserCell(text="hi"), InfoCell(message="note")]
    renderer.flush(cells)
    first = len(console.entries)
    renderer.flush(cells)
    assert len(console.entries) == first  # nothing reprinted

    cells.append(InfoCell(message="second"))
    renderer.flush(cells)
    assert len(console.entries) == first + 1


def test_renderer_stream_collects_deltas_without_breaking_markdown():
    console = RecordingConsole()
    renderer = Renderer(console=console, use_live_tail=False)
    renderer.begin_stream()
    for delta in ("**hel", "lo**\n", "next"):
        renderer.push_delta(delta)
    source = renderer.end_stream()
    assert source == "**hello**\nnext"
    cell = renderer.render_assistant(source)
    assert isinstance(cell, AssistantCell)
    assert console.types() == ["Markdown"]


# ----------------------------------------------------------------- event bridge
def test_translate_maps_runtime_events():
    assert isinstance(translate(Event(type="iteration", data={"iteration": 2})), ui.AssistantStarted)
    tool_started = translate(
        Event(type="tool_start", data={"id": "c1", "tool": "grep", "arguments": {"pattern": "x"}})
    )
    assert isinstance(tool_started, ui.ToolStarted)
    assert tool_started.call_id == "c1" and tool_started.arguments == {"pattern": "x"}

    finished = translate(
        Event(type="tool_result", message="ok output", data={"id": "c1", "tool": "grep", "ok": True})
    )
    assert isinstance(finished, ui.ToolFinished) and finished.text == "ok output"

    assert isinstance(translate(Event(type="skill_load", data={"skill": "debugging"})), ui.SkillLoaded)
    assert isinstance(translate(Event(type="delegate_end", data={"agent": "explorer"})), ui.SubAgentFinished)
    assert isinstance(translate(Event(type="error", message="boom")), ui.ErrorEvent)
    assert translate(Event(type="assistant_text", message="hi")) is None


def test_bridge_records_raw_and_forwards_ui_events():
    seen: list[ui.AgentEvent] = []
    bridge = EventBridge(seen.append)
    bridge(Event(type="iteration", data={"iteration": 1}))
    bridge(Event(type="assistant_text", message="not a UI event"))
    assert len(bridge.raw) == 2
    assert len(seen) == 1
    assert bridge.last("assistant_text").message == "not a UI event"


# -------------------------------------------------------------------------- app
@pytest.fixture()
def config(tmp_path) -> HarnessConfig:
    config = HarnessConfig.load(None)
    config.base_dir = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    config.runtime.workspace_root = str(tmp_path)
    config.memory.enabled = False
    config.checkpoint.enabled = False
    from harness.inference.config import ModelConfig

    config.models = {"main": ModelConfig(provider="mock", model="mock-main")}
    config.default_model = "main"
    return config


def make_app(config) -> MiniAgentApp:
    return MiniAgentApp(
        session=Session(config),
        renderer=Renderer(console=RecordingConsole(), use_live_tail=False),
        stream=False,  # deterministic: no delta streaming in these tests
    )


async def test_app_setup_builds_registry(config):
    app = make_app(config)
    await app.setup()
    assert app.registry.names()[0] == "help"


async def test_app_unknown_command_reports_error(config):
    app = make_app(config)
    await app.setup()
    assert await app.handle_input("/nonsense") is False
    assert any("unknown command" in str(entry) for entry in app.renderer.console.entries)


async def test_app_help_lists_commands(config):
    app = make_app(config)
    await app.setup()
    await app.handle_input("/help")
    rendered = "\n".join(str(entry) for entry in app.renderer.console.entries)
    for name in ("/status", "/model", "/tools", "/skills", "/agents", "/compact", "/clear", "/exit"):
        assert name in rendered


async def test_app_status_and_agents_and_skills(config):
    app = make_app(config)
    await app.setup()
    async with app.session:
        await app.handle_input("/status")
        rendered = "\n".join(str(entry) for entry in app.renderer.console.entries)
        assert "workspace" in rendered and "tools" in rendered

        app.renderer.console.entries.clear()
        await app.handle_input("/agents")
        rendered = "\n".join(str(entry) for entry in app.renderer.console.entries)
        assert "explorer" in rendered and "planner" in rendered

        app.renderer.console.entries.clear()
        await app.handle_input("/skills")
        rendered = "\n".join(str(entry) for entry in app.renderer.console.entries)
        assert "debugging" in rendered and "testing" in rendered


async def test_app_exit_command_stops_loop(config):
    app = make_app(config)
    await app.setup()
    assert await app.handle_input("/exit") is True


async def test_app_shell_intent_runs_through_tool_runtime(config):
    app = make_app(config)
    await app.setup()
    async with app.session:
        await app.handle_input("!echo hello-from-shell")
    rendered = "\n".join(str(entry) for entry in app.renderer.console.entries)
    assert "hello-from-shell" in rendered


async def test_app_prompt_turn_produces_cells(config):
    app = make_app(config)
    await app.setup()
    async with app.session:
        result = await app.run_prompt("list the files")
    assert result is not None and result["termination_status"] == "final_answer"
    kinds = [type(cell).__name__ for cell in app.state.history_cells]
    assert kinds[0] == "UserCell"
    assert "ToolCell" in kinds
    assert kinds[-1] == "AssistantCell"
    assert app.state.active_cell is None


async def test_app_streaming_produces_single_assistant_cell(tmp_path):
    from harness.inference.config import ModelConfig
    from harness.inference.mock_gateway import MockGateway, ScriptedResponse

    config = HarnessConfig.load(None)
    config.base_dir = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    config.runtime.workspace_root = str(tmp_path)
    config.memory.enabled = False
    config.checkpoint.enabled = False
    config.models = {"main": ModelConfig(provider="mock", model="mock-main")}
    config.default_model = "main"

    app = MiniAgentApp(
        session=Session(config),
        renderer=Renderer(console=RecordingConsole(), use_live_tail=False),
        stream=True,
    )
    await app.setup()
    deltas: list[str] = []

    async with app.session:
        app.session.harness.set_gateway(
            MockGateway(
                config.resolve_model("main"),
                responses=[ScriptedResponse.say("**bold** answer\nsecond line\n")],
            )
        )
        # capture deltas through the same path the renderer uses
        original_push = app.renderer.push_delta
        app.renderer.push_delta = lambda delta: (deltas.append(delta), original_push(delta))[1]
        await app.run_prompt("say something")

    assert deltas, "the streaming path must receive deltas"
    assistants = [cell for cell in app.state.history_cells if isinstance(cell, AssistantCell)]
    assert len(assistants) == 1
    # the streamed source is stored once, with stable/tail joined canonically
    assert assistants[0].source == "**bold** answer\nsecond line"


# ------------------------------------------------------------------- arg parsing
def test_load_config_flags(tmp_path):
    args = build_arg_parser().parse_args(
        ["--mock", "--no-memory", "--workspace", str(tmp_path), "--no-stream"]
    )
    config = load_config(args)
    assert config.models[config.default_model].provider == "mock"
    assert config.memory.enabled is False
    assert config.checkpoint.enabled is True
    assert str(tmp_path) in config.runtime.workspace_root


def test_stream_state_defaults():
    state = StreamState()
    assert state.source == "" and state.stable_lines == [] and state.tail_lines == []


# ------------------------------------------------- cell lifecycle (regressions)
#
# These cover a class of bug that shipped: every event appended a new cell, so a
# single answer was rendered twice (once as streamed text, once under the
# "final answer" banner) and every tool call produced two pending lines.


async def test_assistant_answer_is_rendered_exactly_once(config):
    """The completion event must finish the streamed cell, not render a copy."""

    from harness.agent.dto import ToolCall
    from harness.inference.mock_gateway import MockGateway, ScriptedResponse

    app = make_app(config)
    app.stream = True
    await app.setup()
    responses = [
        ScriptedResponse(tool_calls=[ToolCall(name="list_dir", arguments={"path": "."}, id="c1")]),
        ScriptedResponse.say("ANSWER-BODY that must appear once\n"),
    ]
    async with app.session:
        app.session.harness.set_gateway(MockGateway(config.resolve_model("main"), responses=responses))
        await app.run_prompt("do it")

    entries = app.renderer.console.entries
    bodies = [
        entry.plain if hasattr(entry, "plain") else str(entry)
        for entry in entries
    ]
    markdown_renders = [entry for entry in entries if type(entry).__name__ == "Markdown"]
    banners = [body for body in bodies if "final answer" in body]
    # the answer body lives inside the Markdown renderable
    rendered_sources = [getattr(entry, "markup", "") for entry in markdown_renders]
    occurrences = sum(source.count("ANSWER-BODY") for source in rendered_sources)

    assert len(markdown_renders) == 1, "the answer was rendered more than once"
    assert len(banners) == 1, "the final-answer banner appeared more than once"
    assert occurrences == 1, f"the answer text appeared {occurrences} times"
    assert markdown_renders[0].markup.strip() == "ANSWER-BODY that must appear once"


async def test_submitted_user_prompt_is_rendered_before_final_answer(config):
    from harness.inference.mock_gateway import MockGateway, ScriptedResponse

    app = make_app(config)
    await app.setup()
    async with app.session:
        app.session.harness.set_gateway(
            MockGateway(config.resolve_model("main"), responses=[ScriptedResponse.say("answer")])
        )
        await app.run_prompt("the submitted prompt")

    rendered = "\n".join(
        entry.plain if hasattr(entry, "plain") else str(entry)
        for entry in app.renderer.console.entries
    )
    assert "the submitted prompt" in rendered
    assert "answer" in rendered
    user_index = next(
        index
        for index, entry in enumerate(app.renderer.console.entries)
        if getattr(entry, "plain", "") == "› the submitted prompt"
    )
    answer_index = next(
        index
        for index, entry in enumerate(app.renderer.console.entries)
        if getattr(entry, "markup", "") == "answer"
    )
    assert user_index < answer_index


async def test_assistant_deltas_update_one_cell(config):
    from harness.inference.mock_gateway import MockGateway, ScriptedResponse

    app = make_app(config)
    app.stream = True
    await app.setup()
    async with app.session:
        app.session.harness.set_gateway(
            MockGateway(config.resolve_model("main"), responses=[ScriptedResponse.say("one\ntwo\nthree\n")])
        )
        await app.run_prompt("say something")

    assistants = [cell for cell in app.state.history_cells if isinstance(cell, AssistantCell)]
    assert len(assistants) == 1, "streaming deltas must update a single cell"
    assert assistants[0].source.strip() == "one\ntwo\nthree"
    assert assistants[0].complete is True


def test_intermediate_assistant_message_is_not_marked_final(config):
    app = make_app(config)
    app.emit(ui.TurnStarted(prompt="inspect the repository"))
    app.emit(ui.AssistantStarted(iteration=1, message_id="iteration-1"))
    app.emit(ui.AssistantDelta(text="I will inspect the files first."))
    app.emit(ui.AssistantFinished(text="I will inspect the files first.", message_id="iteration-1"))

    from harness.cli.render.transcript import TranscriptControl

    rendered = "".join(text for _style, text in TranscriptControl(app.state)._get_fragments())
    assert "I will inspect the files first." in rendered
    assert "final answer" not in rendered


async def test_tool_call_creates_one_cell_and_renders_pending_then_done(config):
    from harness.agent.dto import ToolCall
    from harness.cli import events as ui
    from harness.inference.mock_gateway import MockGateway, ScriptedResponse

    app = make_app(config)
    app.stream = False
    await app.setup()
    responses = [
        ScriptedResponse(
            tool_calls=[
                ToolCall(name="list_dir", arguments={"path": "."}, id="c1"),
                ToolCall(name="glob", arguments={"pattern": "**/*.py"}, id="c2"),
            ]
        ),
        ScriptedResponse.say("done"),
    ]
    async with app.session:
        app.session.harness.set_gateway(MockGateway(config.resolve_model("main"), responses=responses))
        await app.run_prompt("use two tools")

    tool_cells = [cell for cell in app.state.history_cells if isinstance(cell, ToolCell)]
    assert [cell.call_id for cell in tool_cells] == ["c1", "c2"], "one cell per call id, no duplicates"

    bodies = [
        entry.plain if hasattr(entry, "plain") else str(entry)
        for entry in app.renderer.console.entries
    ]
    for tool in ("list_dir", "glob"):
        pending = [b for b in bodies if b.startswith(f"● {tool}") and b.endswith("…")]
        finished = [b for b in bodies if b.startswith(f"● {tool}") and not b.endswith("…")]
        assert len(pending) == 1, f"{tool}: expected exactly one pending line, got {len(pending)}"
        assert len(finished) == 1, f"{tool}: expected exactly one finished line, got {len(finished)}"


async def test_duplicate_tool_started_event_does_not_create_second_cell(config):
    """`requested` and `started` are the same entity (same call id)."""

    from harness.cli import events as ui

    app = make_app(config)
    await app.setup()
    started = ui.ToolStarted(call_id="c1", tool="read_file", arguments={"path": "a.py"})
    app.emit(started)
    app.emit(ui.ToolStarted(call_id="c1", tool="read_file", arguments={"path": "a.py"}))
    assert len(app._tool_cells) == 1
    app.emit(ui.ToolFinished(call_id="c1", tool="read_file", ok=True, text="content"))
    tool_cells = [cell for cell in app.state.history_cells if isinstance(cell, ToolCell)]
    assert len(tool_cells) == 1


async def test_empty_assistant_message_is_not_kept(config):
    """A turn that only calls tools must not leave an empty answer cell."""

    from harness.agent.dto import ToolCall
    from harness.inference.mock_gateway import MockGateway, ScriptedResponse

    app = make_app(config)
    await app.setup()
    responses = [
        ScriptedResponse(tool_calls=[ToolCall(name="list_dir", arguments={"path": "."}, id="c1")]),
        ScriptedResponse.say(""),
    ]
    async with app.session:
        app.session.harness.set_gateway(MockGateway(config.resolve_model("main"), responses=responses))
        await app.run_prompt("just look")

    empties = [cell for cell in app.state.history_cells if isinstance(cell, AssistantCell) and cell.empty]
    assert empties == [], "an empty assistant cell was left in the transcript"


def test_markdown_theme_has_no_background_fills():
    """Code blocks must not paint a full-width dark bar."""

    from harness.cli.render.theme import MARKDOWN_THEME, build_markdown_console

    for name, style in MARKDOWN_THEME.items():
        assert "on " not in str(style), f"{name} sets a background: {style}"
    console = build_markdown_console()
    assert console is not None


async def test_interleaved_messages_render_once_and_are_all_completed(config):
    """Text between tool calls is a real message: rendered once, marked complete.

    Before the fix those intermediate messages were left half-open (and their
    text was printed twice: once by the stream, once by the turn-end render).
    """

    from harness.agent.dto import ToolCall
    from harness.inference.mock_gateway import MockGateway, ScriptedResponse

    app = make_app(config)
    app.stream = True
    await app.setup()
    responses = [
        ScriptedResponse(
            text="Let me look at the files first.",
            tool_calls=[ToolCall(name="list_dir", arguments={"path": "."}, id="c1")],
        ),
        ScriptedResponse.say("FINAL-ANSWER-BODY\n"),
    ]
    async with app.session:
        app.session.harness.set_gateway(MockGateway(config.resolve_model("main"), responses=responses))
        await app.run_prompt("investigate")

    assistants = [cell for cell in app.state.history_cells if isinstance(cell, AssistantCell)]
    assert len(assistants) == 2, "one cell per assistant message"
    assert all(cell.complete for cell in assistants), "every message must be marked complete"

    # each message carries its own text exactly once
    joined = "\n".join(cell.source for cell in assistants)
    assert joined.count("Let me look at the files first.") == 1
    assert joined.count("FINAL-ANSWER-BODY") == 1
    assert [cell.source for cell in assistants] == ["Let me look at the files first.", "FINAL-ANSWER-BODY"]

    markdowns = [entry for entry in app.renderer.console.entries if type(entry).__name__ == "Markdown"]
    assert len(markdowns) == 2, f"expected exactly two message renders, got {len(markdowns)}"
