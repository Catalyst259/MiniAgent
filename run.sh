#!/usr/bin/env bash
# Compatibility wrapper: the launcher is now ./miniagent.sh
exec "$(dirname "$0")/miniagent.sh" "$@"
