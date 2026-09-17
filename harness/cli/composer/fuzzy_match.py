"""Codex-style subsequence fuzzy matcher.

The scoring is the one described in ``CLI_Design.md`` section 35: a
case-insensitive subsequence match whose score is the size of the window the
match spans, with a bonus for matching from the start of the haystack (smaller
score wins).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FuzzyResult:
    matched_indices: list[int]
    score: int


def fuzzy_match(haystack: str, needle: str) -> FuzzyResult | None:
    """Return match positions and score, or ``None`` when there is no match."""

    if needle == "":
        return FuzzyResult([], -100)
    lowered_haystack = haystack.lower()
    lowered_needle = needle.lower()

    positions: list[int] = []
    cursor = 0
    for char in lowered_needle:
        found = lowered_haystack.find(char, cursor)
        if found == -1:
            return None
        positions.append(found)
        cursor = found + 1

    window = positions[-1] - positions[0] + 1 - len(lowered_needle)
    score = max(window, 0)
    if positions[0] == 0:
        score -= 100
    return FuzzyResult(positions, score)


def fuzzy_filter(
    candidates: list[tuple[str, object]],
    query: str,
) -> list[tuple[object, FuzzyResult]]:
    """Filter ``(text, payload)`` pairs, ordered by ``(score, original order)``."""

    matches: list[tuple[int, object, FuzzyResult]] = []
    for order, (text, payload) in enumerate(candidates):
        result = fuzzy_match(text, query)
        if result is not None:
            matches.append((order, payload, result))
    matches.sort(key=lambda item: (item[2].score, item[0]))
    return [(payload, result) for _order, payload, result in matches]


__all__ = ["FuzzyResult", "fuzzy_match", "fuzzy_filter"]
