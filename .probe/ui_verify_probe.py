#!/usr/bin/env python3
"""End-to-end UI verification: running tools, chain of thought, command output."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, "/home/catalyst259/Personal-Files/Agent/.probe")
os.chdir("/home/catalyst259/Personal-Files/Agent")
os.environ.pop("DEEPSEEK_API_KEY", None)
os.environ.pop("EMBEDDING_API_KEY", None)
os.environ["TERM"] = "xterm-256color"

from ansi_emulator import Screen  # noqa: E402
from headless_app_driver import build_app, make_capture_output  # noqa: E402
from prompt_toolkit.application.current import create_app_session  # noqa: E402
from prompt_toolkit.input.defaults import create_pipe_input  # noqa: E402
from prompt_toolkit.patch_stdout import patch_stdout  # noqa: E402

results: list[str] = []


def say(*parts) -> None:
    results.append(" ".join(str(p) for p in parts))


def show(label: str, screen: Screen) -> None:
    rows = [line.rstrip() for line in screen.visible()]
    say(f"--- {label} ---")
    for line in rows:
        if line.strip():
            say("   | " + line[:96])


async def main() -> int:
    from harness.cli import events as ui

    app = build_app(30, 100)
    await app.setup()
    log: list[str] = []
    output, _ = make_capture_output(30, 100, log, [0])
    screen = Screen(30, 100)
    cpr_r, cpr_w = os.pipe()
    os.set_blocking(cpr_w, False)

    def record(data: str) -> None:
        log.append(data)
        screen.feed(data.encode("utf-8", "replace"))
        if "\x1b[6n" in data:
            try:
                os.write(cpr_w, f"\x1b[{screen.row + 1};{screen.col + 1}R".encode())
            except OSError:
                pass

    output.write_raw = record
    output.write = lambda data: record(data.replace("\x1b", "?"))

    async with app.session:
        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input, output=output):
                application = app.create_application()
                app._ui_app = application
                app.renderer.banner()
                with patch_stdout(raw=True):
                    run_task = asyncio.ensure_future(application.run_async())
                    await asyncio.sleep(0.8)

                    # 1. a running tool with arguments (never committed to history)
                    app.emit(
                        ui.ToolStarted(
                            call_id="probe-1",
                            tool="shell",
                            arguments={"command": "pytest -q tests/test_cli_render.py"},
                        )
                    )
                    app.emit(ui.ToolOutput(call_id="probe-1", text="collecting ...\nrun 1\n"))
                    await asyncio.sleep(0.5)
                    show("running tool", screen)
                    say("   可见参数:", "pytest -q" in "\n".join(screen.visible()))
                    say("   运行中 body 行:", sum(1 for l in screen.visible() if l.startswith("  │")))

                    # 2. chain of thought + final answer
                    app.emit(ui.AssistantStarted(iteration=1, message_id="m1"))
                    app.emit(ui.AssistantDelta(text="the answer is 42"))
                    app.emit(
                        ui.AssistantFinished(
                            text="the answer is 42",
                            reasoning="step 1: read the file\nstep 2: count the lines",
                            message_id="m1",
                        )
                    )
                    await asyncio.sleep(0.5)
                    show("chain of thought", screen)
                    joined = "\n".join(screen.visible())
                    say("   显示 thinking:", "thinking" in joined)
                    say("   显示 step 1:", "step 1: read the file" in joined)

                    # 3. slash command output must be inside the transcript
                    await app.handle_input("/help")
                    await asyncio.sleep(0.5)
                    show("after /help", screen)
                    joined = "\n".join(screen.visible())
                    say("   /help 输出在 transcript:", "run a shell command" in joined)
                    say("   history_cells 数:", len(app.state.history_cells))

                    # 4. scroll back: PageUp twice then Ctrl+End
                    pipe_input.send_text("\x1b[5~")
                    await asyncio.sleep(0.5)
                    show("after PageUp", screen)
                    pipe_input.send_text("\x1b[1;5F")
                    await asyncio.sleep(0.5)
                    show("after Ctrl+End", screen)
                    say("   at_bottom:", app._transcript_pane.at_bottom)

                    app._ui_app.exit()
                    try:
                        await asyncio.wait_for(run_task, timeout=5)
                    except Exception:
                        run_task.cancel()
    print("\n".join(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
