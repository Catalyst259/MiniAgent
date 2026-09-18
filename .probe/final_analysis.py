#!/usr/bin/env python3
"""Final analysis for the MiniAgent TUI byte-stream captures.

Computes, per capture:
  * mouse / alt-screen / cursor-visibility sequence counts
  * bytes and bytes-per-second in the post-turn idle window
  * how many times the final answer text appears
  * duplicate analysis of the reconstructed scrollback
  * the exact byte pattern of one idle repaint cycle
"""

from __future__ import annotations

import json
import re
import sys

PROBE = "/home/catalyst259/Personal-Files/Agent/.probe"

MOUSE = [
    "\x1b[?1000h", "\x1b[?1002h", "\x1b[?1003h", "\x1b[?1006h", "\x1b[?1015h",
    "\x1b[?1000l", "\x1b[?1002l", "\x1b[?1003l", "\x1b[?1006l", "\x1b[?1015l",
    "\x1b[?1049h", "\x1b[?1049l", "\x1b[?25l", "\x1b[?25h", "\x1b[?2004h", "\x1b[?2004l",
    "\x1b[?1004h", "\x1b[?7h", "\x1b[?7l",
]


def load(name: str) -> tuple[bytes, dict, dict]:
    raw = open(f"{PROBE}/raw_{name}.bin", "rb").read()
    meta = json.load(open(f"{PROBE}/meta_{name}.json"))
    screen = json.load(open(f"{PROBE}/screen_{name}.json"))
    return raw, meta, screen


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def analyse(name: str, needle: str) -> None:
    raw, meta, screen = load(name)
    text = raw.decode("utf-8", "replace")
    section(f"{name}: {len(raw)} bytes, size {meta['cols']}x{meta['rows']}")

    print("-- terminal mode sequences --")
    for seq in MOUSE:
        print(f"   {seq!r:14s} count={text.count(seq)}")

    marks = meta["marks"]
    print("-- phase byte counts --")
    for index in range(1, len(marks)):
        prev, cur = marks[index - 1], marks[index]
        span = cur["at"] - prev["at"]
        rate = (cur["offset"] - prev["offset"]) / span if span else 0
        print(
            f"   {prev['label'][:34]:36s} -> {cur['label'][:30]:32s}"
            f" {cur['offset'] - prev['offset']:6d} B in {span:6.3f}s = {rate:8.1f} B/s"
        )

    turn_end = next((m for m in marks if str(m["label"]).startswith("turn_end")), None)
    if turn_end:
        tail = marks[-1]
        print(
            f"-- idle after turn: {tail['offset'] - turn_end['offset']} B over "
            f"{tail['at'] - turn_end['at']:.3f}s = "
            f"{(tail['offset'] - turn_end['offset']) / (tail['at'] - turn_end['at']):.1f} B/s"
        )

    print("-- writes --", meta.get("writes"))
    print("-- final scrollback lines:", meta.get("final_scrollback_len"))

    print("-- answer-text occurrences in the byte stream --")
    for probe in (
        "[mock model - no API key configured]",
        "This run used the offline deterministic model",
        "final answer",
        "reasoning",
        "(thinking)",
        "reasoning_content",
    ):
        print(f"   {probe!r:50s} {text.count(probe)}")

    scrollback = screen["final"]["scrollback"]
    if scrollback:
        from collections import Counter

        counts = Counter(line for line in scrollback if line.strip())
        dupes = {line: n for line, n in counts.items() if n > 1}
        dup_lines = sum(n - 1 for n in dupes.values())
        print(
            f"-- scrollback: {len(scrollback)} lines, "
            f"{len(dupes)} distinct lines duplicated, {dup_lines} redundant copies"
        )
        for line, n in list(dupes.items())[:5]:
            print(f"   x{n} {line!r}")
    else:
        print("-- scrollback empty: nothing was ever pushed off the top --")

    visible = screen["final"]["visible"]
    print("-- final visible screen (non-blank rows) --")
    for index, line in enumerate(visible):
        if line.strip():
            print(f"   {index:3d}|{line}")

    # one idle repaint cycle
    burst = re.findall(r"(?:\x1b\[\?12l\x1b\[\?25h\x1b\[\?25l\x1b\[\?7l\x1b\[\?7h\x1b\[0m){2,}", text)
    if burst:
        print(f"-- idle repaint cycle sample (x{len(burst)}): {burst[0]!r}")

    # newline / scroll pressure
    print(f"-- LF bytes: {raw.count(b'\\n')}, CR bytes: {raw.count(b'\\r')}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["hl2_turn", "hl3_keys", "hl4_width60", "hl1_startup"]
    for name in names:
        try:
            analyse(name, "")
        except FileNotFoundError:
            print("missing capture:", name)
