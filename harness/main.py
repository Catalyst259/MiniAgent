"""Convenience entry point: ``python -m harness.main`` (same as ``python -m harness``)."""

from __future__ import annotations

import sys

from harness.cli.app import main

if __name__ == "__main__":
    sys.exit(main())
