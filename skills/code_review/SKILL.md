---
name: code_review
description: Use after implementing a change to self-review the diff before declaring the task complete.
keywords: [review, diff, correctness, minimality, security, edgecases, cleanup, findings]
---

# Code Review (self-review)

Review your own change as if a hostile reviewer wrote it. Do this after the tests
pass and before you say the task is done. The diff is the unit of review - not
your memory of what you intended to write.

## Procedure

1. Re-read the diff with `git_diff`. Read every hunk in full; do not skim.
2. Walk the checklist below, one dimension at a time. Mixing dimensions makes you
   miss things.
3. Fix what you find, then re-run the narrowest tests plus the full suite.
4. Report findings in the format at the end, even when the list is empty.

## Checklist

**Correctness**
- Does the change do what the task asked - all of it, and only it?
- Are the boundary conditions right: empty input, zero, one element, negative,
  maximum, unicode, duplicate keys, `None`?
- Are return values, error paths, and defaults consistent with the callers?
- Any inverted boolean, off-by-one, wrong variable, or copy-paste of a similar block?
- Does it still work when the obvious "happy path" assumption fails?

**Minimality and churn**
- Is every hunk necessary for this task?
- Any unrelated refactor, rename, reformat, reordering of imports, or whitespace
  change? Revert it - it hides the real change and inflates review cost.
- Any dead code, leftover debug print, commented-out block, or TODO you added?
- Does the new code duplicate something that already exists in the repo?

**Errors and resources**
- Are failures handled, or silently swallowed by a bare `except` / empty catch?
- Are exceptions raised with enough context to debug later?
- Are files, sockets, connections, and locks closed on every path, including the
  error path? Prefer context managers and `try/finally`.
- Any unbounded loop, retry, recursion, or allocation?

**Consistency**
- Do names describe what the thing is, matching the vocabulary already used in
  that file and its neighbours?
- Does the new code look like the code around it - same error style, same logging,
  same typing conventions?
- If you introduced a new pattern, is it justified, or should you follow the
  existing one?

**Security**
- Injection: is untrusted input concatenated into a shell command, SQL string,
  template, or `eval`? Parameterize, quote, or avoid.
- Path traversal: is a user-supplied path joined and used without normalizing and
  confining it to an allowed root?
- Secrets: no hardcoded keys, tokens, passwords, or credentials - in code, tests,
  fixtures, config, or log output.
- Deserialization, SSRF, permissive CORS, or disabled TLS verification introduced
  for convenience.
- Does an error message or log leak internal paths or sensitive values?

**Tests**
- Does a test cover the new behaviour and, for a bug fix, fail without the fix?
- Are the tests asserting real behaviour rather than mocking it away?
- Did any existing test get weakened, skipped, or deleted?

## Output format

Report findings as a short list, highest severity first:

```
Findings
- [high]   path/to/file.py:88 - user-supplied name is interpolated into the SQL
           string; parameterize the query.
- [medium] path/to/file.py:41 - error is swallowed; log and re-raise.
- [low]    path/to/file.py:12 - leftover debug print.
- [nit]    path/to/file.py:30 - local name `d` shadows the module-level `d`.
```

Severities: **high** (wrong results, data loss, security, crash), **medium**
(incorrect edge behaviour, resource leak, misleading error), **low** (clarity,
dead code), **nit** (style).

Rules:

- Every finding cites a file and line.
- **Fix all high-severity findings before declaring the task complete.** Do not
  ship a known high-severity issue and describe it as "follow-up work".
- If you accept a medium or low finding without fixing it, say so explicitly and
  why.
- If the diff is clean, say `Findings: none` rather than inventing filler.
