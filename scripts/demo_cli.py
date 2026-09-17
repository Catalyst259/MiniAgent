#!/usr/bin/env python3
"""Non-interactive demo of the MiniAgent CLI: banner, slash commands, a task, !shell.

Runs the same ``MiniAgentApp`` used by the interactive TUI, but feeds it a script
instead of keystrokes so it works without a terminal.

    .venv/bin/python scripts/demo_cli.py --workspace /tmp/demo
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.cli.app import MiniAgentApp, Session, load_config  # noqa: E402
from harness.cli.render import Renderer  # noqa: E402

SCRIPT = [
    "/help",
    "/tools",
    "!echo shell-intent-works",
    "list the files in this workspace",
    "/status",
]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--real", action="store_true", help="use the configured model instead of --mock")
    parser.add_argument("--show-events", action="store_true")
    args = parser.parse_args()

    config = load_config(
        argparse.Namespace(
            config=None,
            workspace=args.workspace,
            mock=not args.real,
            model=None,
            no_memory=True,
            no_checkpoint=True,
            mcp_stdio=False,
        )
    )
    app = MiniAgentApp(
        session=Session(config),
        renderer=Renderer(),
        stream=True,
        show_events=args.show_events,
    )
    await app.setup()
    app.renderer.banner()
    async with app.session:
        for line in SCRIPT:
            print(f"\n› {line}")
            if await app.handle_input(line):
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
