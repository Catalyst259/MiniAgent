---
name: testing
description: Use when running, narrowing, adding, or trusting tests for a change you are about to make.
keywords: [testing, pytest, failures, regression, fixtures, suites, assertions, coverage]
---

# Testing

Tests are the only evidence that your change works. Run them; do not assume them.
Never edit a test into passing - if a test is wrong, say so explicitly and explain
why, and let the caller decide.

## 1. Discover the runner and the exact command

Look before you run:

- Manifest and config: `pyproject.toml` (`[tool.pytest.ini_options]`, `testpaths`),
  `package.json` scripts, `Makefile`, `tox.ini`, CI workflow files.
- Test layout: `tests/`, `test/`, `*_test.go`, `*.spec.ts`, `__tests__/`.
- Fixtures and conftest: `conftest.py`, `tests/fixtures/`, factory helpers.

Record the literal command you will use, for example
`.venv/bin/python -m pytest` or `pytest tests/test_api.py`. Use the interpreter
that has the dependencies installed, not whichever `python` is first on `PATH`.

## 2. Run the narrowest test first

Start as small as possible; a wide run hides signal in noise and wastes time:

```
pytest tests/test_module.py::test_case -x -q      # single test, stop on first failure
pytest tests/test_module.py -x -q                 # single module
pytest -q -k "keyword"                            # selection by name
```

Then widen: module -> package -> full suite. `-x` for triage, full run for the
final verdict.

## 3. Read failures completely

Read the whole failure block: the test name, the assertion line, the actual vs expected values, and the captured stdout/stderr (`-vv`, `--show-capture=all` when needed). A failure message you skimmed is a failure you will misdiagnose.

## 4. Classify the failure before fixing it

The class tells you where to look:

- **Assertion failure** - the code ran and produced a wrong value. The bug is in
  the code under test (or the expectation is genuinely outdated).
- **Collection error** - pytest could not even import/discover the module. Syntax
  error, bad decorator, duplicate test name, unreadable path.
- **Import error** - `ModuleNotFoundError` / `ImportError`. Wrong root, missing
  dependency, circular import, or a package not installed in this interpreter.
- **Fixture error** - setup raised or a fixture is missing. Look at the fixture
  body and its scope, not at the test body.
- **Error (not assertion)** - an exception escaped. Treat it as a bug until proven
  otherwise; do not wrap it in `pytest.raises` unless raising is the contract.
- **Skipped / xfail** - not evidence. Check why it skipped before counting it green.

## 5. Never weaken tests

Forbidden: deleting a failing test, loosening an assertion, adding `skip`/`xfail`
to a real failure, widening a tolerance to hide a regression, or mocking the very
thing under test. If a test truly encodes obsolete behaviour, change it
deliberately and report the change with its justification.

## 6. Add a regression test for every bug fix

A fix without a test is a fix that will silently regress. The regression test must:

1. Fail on the old code (verify by reasoning, or by reverting the fix briefly).
2. Pass on the new code.
3. Assert the specific behaviour that was broken - the exact input that broke, and
   the boundary next to it.

Place it next to the existing tests for that module and follow their style and
fixtures instead of inventing new ones.

## 7. When there is no test infrastructure

Do not pretend. Options, in order:

1. Check for an existing runner you missed (scripts, CI, `Makefile` targets).
2. Write a minimal executable check: a small script that calls the code with a
   known input and asserts the output, run via `shell`.
3. For CLI-level work, exercise the real command and inspect its exit code and
   output.
4. Report clearly that verification was manual, and state exactly what you ran.

## 8. Final full-suite run

Before claiming completion, run the whole suite, not just the tests you touched.
Report the exact command and the result (pass/fail counts). If pre-existing
failures remain, name them and confirm they are unrelated to your change. Never
report success from a partial run without labelling it as partial.
