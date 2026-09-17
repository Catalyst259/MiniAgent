---
name: repo_exploration
description: Use when entering an unfamiliar repository or module and you need a working map before editing anything.
keywords: [exploration, repository, codebase, structure, onboarding, architecture, navigation, discovery]
---

# Repository Exploration

Build a small, correct map of the codebase before you touch it. Exploration is
finished when you can name the entry point, the module that owns the behaviour
you will change, and the command that exercises it - not when you have read
everything.

## 1. Orient on manifests and entry points

Read the files that declare intent before reading implementation:

- `pyproject.toml`, `requirements.txt`, `package.json`, `go.mod`, `Cargo.toml`,
  `Makefile` - language, dependencies, scripts, test runner.
- `README.md`, `docs/`, `CONTRIBUTING.md` - stated architecture and conventions.
- CI config (`.github/workflows/`, `.gitlab-ci.yml`) - the commands that must pass.

Then locate entry points: `main.py`, `__main__.py`, `cli.py`, `manage.py`,
`cmd/*/main.go`, `src/index.ts`. Use `glob` with a pattern rather than guessing
paths one by one.

## 2. Map the directory shape

Call `list_dir` on the top two levels only. Write one line per package: what it
owns. Do not descend into vendored code, build output, or `.venv/`.

Rule of thumb: a directory with one or two files is a leaf; a directory imported
by many siblings is a hub. Bugs and changes concentrate in hubs.

## 3. Read the 3-8 files that define behaviour

Do not read files alphabetically. Select by evidence:

1. Start at the entry point and follow imports one hop at a time toward your task.
2. `grep` for an exact token from the task - an error message, flag name, route,
   or config key. That match is your anchor file.
3. `grep` for definitions of the symbol you must change: `def name`,
   `class Name`, `func name`, `function name`.
4. Read the anchor file plus its immediate callers and callees.

Bound the cost:

- Prefer `grep` with a path filter over reading whole directories.
- Use line ranges when a file is long; never pull 2000 lines to see one function.
- Read the module's tests - they encode the real contract faster than prose.

## 4. Record the map

Before editing, write down (in notes or in a message back to the caller):

- Entry point and how control reaches your area.
- Modules you will touch, as paths.
- Data flow in one line: input -> transform -> output.
- Commands that build, run, and test.
- Open unknowns you could not resolve.

Keep it short. A map longer than the code it describes is not a map.

## 5. Decide when to delegate to `Explorer`

Delegate when the search is broad but shallow and isolation helps:

- "Where is X handled, and what calls it?" spanning many packages.
- Three or more independent questions about the same repository.
- You are near the context budget and need conclusions, not raw code.

Do not delegate when you already know the file, when the answer is one `grep`
away, or when you need the code in your own context in order to edit it.

## 6. Stop conditions

Stop exploring and start acting when all of these hold:

- You can name the file and function you will change.
- You can state how to reproduce the current behaviour.
- You know the exact test command.
- Remaining unknowns are details you will learn while editing.

Red flags: reading at random to "get familiar"; continuing to explore after you
already know the fix; opening more than roughly ten files without recording what
you learned; assuming directory names describe behaviour; trusting README claims
that contradict the code.
