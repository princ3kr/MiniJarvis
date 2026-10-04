"""`python -m jarvis` -- the CLI. One line, because it is the entry point.

Everything lives in `jarvis.cli`. This module exists so the package has a `__main__`
target, and so the docstring that documents the console has somewhere to live.
"""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

def main(argv: list[str] | None = None) -> int:
    """Kept so `from jarvis.__main__ import main` (the console script entry) works."""
    from .cli import main as _main
    return _main(argv)
