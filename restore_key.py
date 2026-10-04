"""Restore the inline API key into .env from the newest available copy.

The key was once hard-coded in the notebook. It now lives in `.env`, which is
git-ignored, so this exists to move it between places without ever printing it.

Idempotent, and never writes the key to stdout, a log, or an exception message --
that is the only property that matters for a file whose whole job is to handle a
secret. If the copy cannot be found it says so and exits non-zero rather than
leaving a half-written .env.

    python restore_key.py              # restore from the newest copy found
    python restore_key.py --check      # report only, change nothing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
ENV = PROJECT / ".env"
BACKUP_DIRS = [PROJECT, Path("/tmp/opencode/orig")]

# Same shape as jarvis.redaction: enough to recognise the key without handling it.
KEY_LINES = re.compile(
    r"^\s*(?:LLM_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|OPENROUTER_API_KEY|GROQ_API_KEY)\s*=\s*(.+?)\s*$",
    re.M,
)


def _find_in_notebook() -> list[tuple[float, str]]:
    """(mtime, key) for every notebook that carries a key. Reads bytes only."""
    found: list[tuple[float, str]] = []
    for nb in PROJECT.glob("*.ipynb"):
        try:
            raw = nb.read_text(errors="replace")
        except OSError:
            continue
        for m in KEY_LINES.finditer(raw):
            val = m.group(1).strip().strip('"').strip("'")
            if len(val) > 20 and not val.startswith("<"):
                found.append((nb.stat().st_mtime, val))
    return found


def _find_in_backup() -> list[tuple[float, str]]:
    found: list[tuple[float, str]] = []
    for d in BACKUP_DIRS:
        if not d.is_dir():
            continue
        for f in d.glob("env.*"):
            try:
                raw = f.read_text(errors="replace")
            except OSError:
                continue
            for m in KEY_LINES.finditer(raw):
                val = m.group(1).strip().strip('"').strip("'")
                if len(val) > 20:
                    found.append((f.stat().st_mtime, val))
    return found


def current_key() -> str | None:
    if not ENV.is_file():
        return None
    for m in KEY_LINES.finditer(ENV.read_text(errors="replace")):
        return m.group(1).strip().strip('"').strip("'")
    return None


def write_key(key: str) -> None:
    """Rewrite .env preserving every other line. Written 0600."""
    existing = ENV.read_text().splitlines() if ENV.is_file() else []
    out: list[str] = []
    replaced = False
    for line in existing:
        if KEY_LINES.match(line) and not replaced:
            name = line.split("=", 1)[0].strip()
            out.append(f"{name} = {key}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"LLM_API_KEY = {key}")
    # Write via a temp file then rename, so an interrupted run cannot leave .env
    # truncated -- the failure mode that would lose the key it was restoring.
    tmp = ENV.with_suffix(".env.tmp")
    tmp.write_text("\n".join(out) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(ENV)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only")
    args = ap.parse_args()

    have = current_key()
    if args.check:
        print(f"api key available: {'YES' if have else 'NONE SET'}")
        return 0 if have else 1

    candidates = _find_in_notebook() + _find_in_backup()
    if not candidates:
        print("no key found in any notebook or backup; .env unchanged", file=sys.stderr)
        return 1
    _, key = max(candidates, key=lambda kv: kv[0])
    write_key(key)
    # Report only whether it worked. Never the value, never a prefix of it.
    print("api key restored into .env (0600)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
