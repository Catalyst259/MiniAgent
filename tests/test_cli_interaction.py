"""The model/user interaction seam, adapters and real terminal panel."""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.input import create_pipe_input

from harness.cli.interaction import InteractiveInteractionProvider
from harness.cli.state import AppState
from harness.interaction import Choice, InteractionCancelled, choices_from_payload
from tests.helpers import SizedDummyOutput, render_application_rows


def test_choice_payload_is_validated_and_normalized():
    choices = choices_from_payload(
        [
            {"value": "fast", "label": "Fast", "description": "Less checking"},
            {"value": "safe", "label": "Safe", "description": "More checking"},
        ]
    )
    assert choices == [
        Choice("fast", "Fast", "Less checking"),
        Choice("safe", "Safe", "More checking"),
    ]
    with pytest.raises(ValueError, match="between 2 and 4"):
        choices_from_payload([{"label": "Only"}])


async def test_interactive_provider_resolves_the_selected_choice():
    state = AppState()
    provider = InteractiveInteractionProvider(state=state)
    future = provider.request(
        "How should I continue?",
        [Choice("fast", "Fast"), Choice("safe", "Safe")],
        title="Strategy",
    )

    assert provider.waiting and state.interaction is provider.pending
    provider.move(1)
    assert provider.accept_selection()
    assert await future == Choice("safe", "Safe")
    assert not provider.waiting and state.interaction is None


async def test_cancelled_interaction_releases_the_waiter():
    provider = InteractiveInteractionProvider(state=AppState())
    future = provider.request("Continue?", [Choice("yes", "Yes"), Choice("no", "No")])
    assert provider.cancel()
    with pytest.raises(InteractionCancelled):
        await future


async def test_model_choice_panel_is_visible_and_drives_a_real_turn(app):
    """Tool call -> modal frame -> user answer -> tool observation -> final answer."""

    from harness.inference.mock_gateway import MockGateway, ScriptedResponse

    await app.setup()
    responses = [
        ScriptedResponse.tool(
            "request_user_input",
            title="Strategy",
            question="How should I continue?",
            options=[
                {"value": "fast", "label": "Fast", "description": "Fewer checks"},
                {"value": "safe", "label": "Safe", "description": "Run all checks"},
            ],
        ),
        ScriptedResponse.say("Using the safe strategy."),
    ]

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=SizedDummyOutput()):
            application = app.create_application()
            from harness.cli.output import ConsoleOutput, TranscriptOutput

            app.terminal.application = application
            app.presenter.output = TranscriptOutput(app.terminal.invalidate)
            async with app.session:
                gateway = MockGateway(app.session.harness.model_config, responses=responses)
                app.session.harness.set_gateway(gateway)
                turn = asyncio.create_task(app.run_prompt("Choose a strategy"))
                try:
                    for _ in range(100):
                        if app.interaction.waiting or turn.done():
                            break
                        await asyncio.sleep(0.01)
                    if turn.done():
                        await turn
                    assert app.interaction.waiting, "the model's choice never reached the UI"

                    visible = "\n".join(render_application_rows(app, columns=80))
                    assert "How should I continue?" in visible
                    assert "1) Fast" in visible and "2) Safe" in visible

                    assert app.interaction.resolve_index(1)
                    result = await asyncio.wait_for(turn, timeout=5)
                finally:
                    app.interaction.cancel()
                    if not turn.done():
                        turn.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await turn
        app.terminal.application = None
        app.presenter.output = ConsoleOutput(app.renderer)

    assert result["final_answer"] == "Using the safe strategy."
    interaction_observations = [
        item for item in result["observations"] if item.tool_name == "request_user_input"
    ]
    assert interaction_observations[0].ok
    assert "safe" in interaction_observations[0].content


@pytest.fixture()
def app(tmp_path):
    from pathlib import Path

    from harness.cli.app import MiniAgentApp, Session
    from harness.cli.render import Renderer
    from harness.inference.config import ModelConfig
    from harness.infra.config import HarnessConfig
    from tests.helpers import RecordingConsole

    root = Path(__file__).resolve().parents[1]
    config = HarnessConfig.load(root / "config.yaml")
    config.models = {"main": ModelConfig(provider="mock", model="mock-main")}
    config.default_model = "main"
    config.runtime.workspace_root = str(tmp_path)
    config.memory.enabled = False
    config.checkpoint.enabled = False
    return MiniAgentApp(
        session=Session(config),
        renderer=Renderer(console=RecordingConsole(), use_live_tail=False),
    )
