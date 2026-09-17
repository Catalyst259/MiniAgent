---
name: debugging
description: Use when a test, build, or runtime path fails and you must find the real cause before changing code.
keywords: [debugging, errors, traces, reproduce, hypothesis, diagnosis, bisect, rootcause]
---

# Debugging

Fix the cause, not the symptom. A change you cannot explain is a coincidence, and
coincidences come back. Work the loop below in order; do not skip to step 5.

## Procedure

1. **Reproduce first.** Get a command that fails on demand, and run it yourself.
   If you cannot reproduce it, you cannot verify a fix - say so and gather more
   detail (exact input, version, environment, flags) before touching code.
2. **Read the FULL error.** The whole traceback or log block, top to bottom, not
   the last line. Note every frame in *your* code, the exception type, and the
   exact message. Most bugs are already named here.
3. **Locate the smallest relevant region.** Use `grep` for the failing symbol or
   message, then `read_file` with a line range around it - roughly 30 lines of
   context. Read the whole enclosing function at minimum.
4. **Form one concrete falsifiable hypothesis before editing.** It must predict
   an observation: "if X is None here, then the log line at 42 should print None."
   Write it down. If you cannot state what would disprove it, you have a hunch,
   not a hypothesis.
5. **Probe the hypothesis cheaply.** Add a temporary print, run the narrowest
   test, or inspect the intermediate value. Prefer observing reality over
   reasoning about it - but time-box this to one or two probes.
6. **Make the smallest reasonable change.** One logical edit that addresses the
   cause. No drive-by refactors, no reformatting, no unrelated fixes.
7. **Re-run the narrowest test** that exercises the failure. Confirm it passes
   for the predicted reason, not because the test was skipped or weakened.
8. **Re-run the reproduction** from step 1 with the original input, then the
   neighbouring tests, then the wider suite.

## Triage table

| Symptom | Likely cause | First probe |
| --- | --- | --- |
| `TypeError: 'NoneType'` deep in a call | earlier function returned `None` on an unhandled branch | `grep` the function's `return` statements; check the caller's input |
| Test passed before, fails now, code unchanged | environment, fixture ordering, or a shared mutable default | run the test alone (`pytest path::test -x -q`); check module-level state |
| `ImportError` / `ModuleNotFoundError` | wrong package root, missing dependency, circular import, shadowed name | run from the repo root; print `sys.path` and the module `__file__` |
| Hangs, no output, no error | blocking I/O, deadlock, or an unbounded retry loop | re-run with a timeout and a stack dump; check loops and client timeouts |
| Works locally, fails in CI | version drift, missing env var, path or locale assumptions | diff the pinned versions; `grep` for hardcoded absolute paths |
| Wrong value, no exception | off-by-one, wrong operator, unit mismatch, stale cache | assert the intermediate value at the boundary; check index ranges |
| Intermittent failure | unseeded randomness, time dependence, race, dict ordering | run the test in a loop; seed RNG; freeze the clock |
| Failure only for some inputs | unhandled edge case (empty, unicode, huge, negative) | parameterize over the boundary inputs explicitly |

## Anti-patterns

- **Shotgun edits**: changing several things at once, so you cannot tell what fixed
  it - and you have introduced untested changes.
- **Log-and-pray**: scattering prints without a hypothesis, then reading the noise.
- **Fixing the symptom**: catching and swallowing the exception, clamping the bad
  value, or adding a `None` guard where the real bug is an earlier wrong branch.
- **Editing tests to match buggy behaviour** instead of fixing the code.
- **Assuming the failing line is the broken line** - the cause is usually upstream.
- **Rewriting the module** when the diff should be three lines.
- **Declaring victory without re-running** the original reproduction.

## Done means

The original reproduction passes, a regression test covers it, and you can
explain in one sentence what was wrong and why the change fixes it.
