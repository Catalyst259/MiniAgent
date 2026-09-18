#!/usr/bin/env python3
"""Minimal ANSI/CSI screen emulator + byte-stream analyzers for MiniAgent TUI probes.

Library + CLI.  The emulator handles the subset prompt_toolkit actually emits:

  \\r  \\n  \\b  \\t  \\x07
  ESC [ <n> A/B/C/D          cursor up/down/right/left
  ESC [ <n> G                cursor to column
  ESC [ <n> ; <m> H / f      cursor position
  ESC [ <n> K                erase in line (0 = to end, 1 = to start, 2 = whole)
  ESC [ <n> J                erase in display (2 = whole screen)
  ESC [ ... m                SGR: skipped (styles not tracked)
  ESC [ ? Pm h/l             DEC private mode: recorded, not applied to the grid
  ESC 7 / ESC 8              save/restore cursor
  ESC ] ... BEL / ST         OSC: ignored
  anything else ESC-lit      ignored (escape aborted)

LF on the bottom row scrolls the top row into `scrollback` (this is what a real
terminal emulator does, and the whole point of task 4).
"""

from __future__ import annotations

import re
import unicodedata

MAX_SCROLLBACK = 200000


def _char_width(ch: str) -> int:
    if unicodedata.combining(ch):
        return 0
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 2
    return 1


class Screen:
    def __init__(self, rows: int = 24, cols: int = 100) -> None:
        self.rows = rows
        self.cols = cols
        self.grid: list[list[str]] = [[" "] * cols for _ in range(rows)]
        self.row = 0
        self.col = 0
        self.saved = (0, 0)
        self.scrollback: list[str] = []
        self.modes: dict[int, bool] = {}
        self.mode_events: list[tuple[int, bool]] = []
        self.other_events: list[str] = []
        self.unknown: list[str] = []
        self.total_cells = 0

    # ------------------------------------------------------------------ helpers
    def _line(self, index: int) -> str:
        return "".join(self.grid[index]).rstrip()

    def _erase_line(self, mode: int) -> None:
        if mode == 0:
            span = range(self.col, self.cols)
        elif mode == 1:
            span = range(0, min(self.col + 1, self.cols))
        else:
            span = range(0, self.cols)
        for column in span:
            self.grid[self.row][column] = " "

    def _erase_display(self, mode: int) -> None:
        if mode == 2 or mode == 3:
            self.grid = [[" "] * self.cols for _ in range(self.rows)]
        elif mode == 0:
            self._erase_line(0)
            for line in range(self.row + 1, self.rows):
                self.grid[line] = [" "] * self.cols
        elif mode == 1:
            self._erase_line(1)
            for line in range(0, self.row):
                self.grid[line] = [" "] * self.cols

    def _lf(self) -> None:
        if self.row >= self.rows - 1:
            self.scrollback.append(self._line(0))
            if len(self.scrollback) > MAX_SCROLLBACK:
                del self.scrollback[:1000]
            self.grid.pop(0)
            self.grid.append([" "] * self.cols)
        else:
            self.row += 1

    def _put(self, ch: str) -> None:
        width = _char_width(ch)
        if width == 0:
            return
        if self.col >= self.cols:
            self.col = 0
            self._lf()
        if self.col + width <= self.cols:
            self.grid[self.row][self.col] = ch
            if width == 2 and self.col + 1 < self.cols:
                self.grid[self.row][self.col + 1] = ""
        self.col += width
        self.total_cells += 1

    # -------------------------------------------------------------------- feed
    CSI_RE = re.compile(rb"\x1b\[([0-9;?<>=]*)([@-~])")

    def feed(self, data: bytes) -> None:
        i = 0
        n = len(data)
        while i < n:
            byte = data[i]
            if byte == 0x1B:
                if i + 1 >= n:
                    break
                nxt = data[i + 1]
                if nxt == 0x5B:  # CSI
                    match = self.CSI_RE.match(data, i)
                    if match:
                        self._csi(match.group(1).decode("latin-1"), chr(match.group(2)[0]))
                        i = match.end()
                        continue
                    i += 2
                    continue
                if nxt == 0x5D:  # OSC ... BEL or ST
                    j = i + 2
                    while j < n and data[j] not in (0x07, 0x1B):
                        j += 1
                    i = j + 1 if j < n else n
                    continue
                if nxt in (0x37, 0x38):  # ESC 7 / ESC 8
                    if nxt == 0x37:
                        self.saved = (self.row, self.col)
                    else:
                        self.row, self.col = self.saved
                    i += 2
                    continue
                if nxt in (0x28, 0x29, 0x2A, 0x2B, 0x23):  # charset selection etc.
                    i += 3
                    continue
                if nxt == 0x3D or nxt == 0x3E:  # DECKPAM / DECKPNM
                    i += 2
                    continue
                self.unknown.append(repr(data[i : i + 4]))
                i += 2
                continue
            if byte == 0x0D:
                self.col = 0
            elif byte == 0x0A:
                self._lf()
            elif byte == 0x08:
                self.col = max(0, self.col - 1)
            elif byte == 0x09:
                self.col = min(self.cols - 1, (self.col // 8 + 1) * 8)
            elif byte == 0x07:
                pass
            elif byte < 0x20:
                pass
            else:
                # decode one UTF-8 character
                length = 1
                if byte >= 0xF0:
                    length = 4
                elif byte >= 0xE0:
                    length = 3
                elif byte >= 0xC0:
                    length = 2
                chunk = data[i : i + length]
                try:
                    self._put(chunk.decode("utf-8"))
                except UnicodeDecodeError:
                    try:
                        self._put(bytes([byte]).decode("latin-1"))
                    except Exception:
                        pass
                i += length
                continue
            i += 1

    def _csi(self, params: str, final: str) -> None:
        private = params.startswith("?")
        body = params[1:] if private else params
        values = [int(part) if part.isdigit() else 0 for part in body.split(";")] if body else []

        def arg(index: int, default: int = 1) -> int:
            if index < len(values) and values[index] != 0:
                return values[index]
            return default

        if private:
            if final in ("h", "l"):
                for value in values:
                    state = final == "h"
                    self.modes[value] = state
                    self.mode_events.append((value, state))
                return
            self.other_events.append(f"CSI ?{params}{final}")
            return

        if final == "A":
            self.row = max(0, self.row - arg(0))
        elif final == "B":
            self.row = min(self.rows - 1, self.row + arg(0))
        elif final == "C":
            self.col = min(self.cols - 1, self.col + arg(0))
        elif final == "D":
            self.col = max(0, self.col - arg(0))
        elif final == "E":
            self.row = min(self.rows - 1, self.row + arg(0))
            self.col = 0
        elif final == "F":
            self.row = max(0, self.row - arg(0))
            self.col = 0
        elif final == "G" or final == "`":
            self.col = max(0, min(self.cols - 1, arg(0) - 1))
        elif final == "d":
            self.row = max(0, min(self.rows - 1, arg(0) - 1))
        elif final in ("H", "f"):
            row = arg(0) - 1
            col = (values[1] - 1) if len(values) > 1 and values[1] else 0
            self.row = max(0, min(self.rows - 1, row))
            self.col = max(0, min(self.cols - 1, col))
        elif final == "K":
            self._erase_line(values[0] if values else 0)
        elif final == "J":
            self._erase_display(values[0] if values else 0)
        elif final == "m":
            pass  # SGR ignored
        elif final == "r":
            self.other_events.append("CSI r (scroll region)")
        elif final == "L":
            for _ in range(arg(0)):
                self.grid.insert(self.row, [" "] * self.cols)
                self.grid.pop()
        elif final == "M":
            for _ in range(arg(0)):
                self.grid.pop(self.row)
                self.grid.append([" "] * self.cols)
        elif final == "P":
            count = arg(0)
            line = self.grid[self.row]
            del line[self.col : self.col + count]
            line.extend([" "] * count)
        elif final == "@":
            count = arg(0)
            line = self.grid[self.row]
            for _ in range(count):
                line.insert(self.col, " ")
            del line[self.cols :]
        elif final in ("h", "l"):
            self.other_events.append(f"CSI {params}{final}")
        elif final == "n" or final == "c" or final == "t":
            self.other_events.append(f"CSI {params}{final} (query)")
        else:
            self.unknown.append(f"CSI {params}{final}")

    # ------------------------------------------------------------------- views
    def visible(self) -> list[str]:
        return [self._line(index) for index in range(self.rows)]

    def visible_text(self) -> str:
        return "\n".join(self.visible())


def analyze(path: str, rows: int = 24, cols: int = 100) -> dict:
    data = open(path, "rb").read()
    screen = Screen(rows, cols)
    screen.feed(data)
    return {
        "path": path,
        "bytes": len(data),
        "screen": screen,
        "visible": screen.visible(),
        "scrollback": screen.scrollback,
        "modes": screen.modes,
        "mode_events": screen.mode_events,
        "other_events": screen.other_events[:20],
        "unknown": screen.unknown[:20],
    }


def duplicate_scrollback(lines: list[str]) -> tuple[int, int, dict[str, int]]:
    """Return (total, non_empty, counts) for scrollback lines."""

    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    non_empty = [line for line in lines if line.strip()]
    return len(lines), len(non_empty), counts


if __name__ == "__main__":
    import sys

    result = analyze(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 24,
                     int(sys.argv[3]) if len(sys.argv) > 3 else 100)
    print("=== VISIBLE SCREEN ===")
    for index, line in enumerate(result["visible"]):
        print(f"{index:3d}|{line}")
    print(f"=== SCROLLBACK ({len(result['scrollback'])} lines) ===")
    for index, line in enumerate(result["scrollback"][:60]):
        print(f"{index:4d}|{line}")
    print("modes:", result["modes"])
    print("unknown:", result["unknown"])
