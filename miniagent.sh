#!/usr/bin/env bash
# Launch the interactive MiniAgent session: ./miniagent.sh
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "no virtualenv found; create one with: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi
exec "$PYTHON" -m harness.main "$@"
