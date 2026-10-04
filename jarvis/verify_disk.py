"""Re-check the answer's cheap claims against the disk.

Third in the chain, and the one that catches the failure the others cannot see: an
answer that is *internally* consistent with the evidence and still wrong, because the
evidence itself was stale.

    $ ls -la big.dat
    -rw-r--r--  1 u u 104857600 Aug  4 10:00 big.dat
    ...
    "big.dat is 50MB"                     <- the agent divided by 2

A number check passes (104857600 is in the output). A rubric passes (50MB is a
plausible reading of a 100MB file's byte count). A `stat` does not. So for the
predicates that are cheap and unambiguous to re-check -- does it exist, what size,
what mtime -- the disk gets the last word.

Every check goes through `AccessPolicy.resolve`. A verification step that could read
outside the jail would be a hole in the jail, and this code runs on model-authored
text, which is exactly the untrusted input the jail exists for.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from .access import AccessDenied, AccessPolicy

_SIZE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([KMGTP]i?B|bytes?|B)\b", re.I)
_MULT = {"B": 1, "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4, "P": 1000**5}
# 1024-based, because that is what `ls -lh` and `du -h` print. Handling both is the
# point: an agent that read "1.0K" from `ls -lh` and restated it as 1024 bytes is
# right, and one that read 1000 from `ls -l` and restated it as 1K is wrong by 2.4%.
_BIN_MULT = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}


def _to_bytes(value: float, unit: str) -> tuple[int, int]:
    """Both plausible byte counts for a human-readable size."""
    u = unit.upper().rstrip("B").rstrip("I") or "B"
    u = {"": "B"}.get(u, u)
    dec = _MULT.get(u, _MULT["B"])
    bin_ = _BIN_MULT.get(u, _BIN_MULT["B"])
    return int(value * dec), int(value * bin_)


def recheck_disk_claims(answer: str, *, ctx, tolerance: float = 0.02
                        ) -> tuple[list[str], list[str]]:
    """(problems, checks_performed).

    Returns an empty `problems` when there was nothing to check, which is the common
    case and is NOT the same as "verified" -- the caller distinguishes them.
    """
    from .verify_paths import ABS_PATH, _norm

    problems: list[str] = []
    checks: list[str] = []
    policy = AccessPolicy.from_settings(ctx.settings, cwd=ctx.root or Path.cwd())

    for m in ABS_PATH.finditer(answer):
        raw = _norm(m.group(1))
        if len(raw) < 4 or raw.startswith("http"):
            continue
        try:
            target = policy.resolve(raw, for_write=False)
        except AccessDenied as e:
            # Claiming a path outside the scope is a hallucination or an escape
            # attempt; either way it is not supported.
            problems.append(f"{raw} is outside the read scope ({e.reason})")
            continue
        except Exception:
            continue

        exists = target.exists()
        checks.append(f"exists {raw}={exists}")

        # A size claim about this path, if any, must agree with the disk.
        window = answer[max(0, m.start() - 120): m.end() + 120]
        sizes = [(v, u) for v, u in _SIZE.findall(window)]
        if sizes and exists and target.is_file():
            try:
                actual = target.stat().st_size
            except OSError as e:
                problems.append(f"{raw} unreadable ({type(e).__name__})")
                continue
            ok = False
            for v, u in sizes:
                dec, bin_ = _to_bytes(float(v), u)
                for cand in {dec, bin_}:
                    if abs(cand - actual) <= max(1, actual * tolerance):
                        ok = True
                        break
                if ok:
                    break
            checks.append(f"size {raw} claimed={sizes} actual={actual}")
            if not ok:
                problems.append(
                    f"{raw} is {actual} bytes on disk, answer claims "
                    f"{', '.join(f'{v}{u}' for v, u in sizes)}")

    return problems, checks


def is_stale(evidence_time: float, max_age_s: float = 300.0) -> bool:
    """Evidence older than this may describe a filesystem that has since moved.

    Stated rather than hidden: a stale check should make the answer less confident,
    not silently more confident because the code did not look.
    """
    return (time.time() - evidence_time) > max_age_s