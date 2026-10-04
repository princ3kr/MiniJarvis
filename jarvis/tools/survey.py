from __future__ import annotations

import time
from pathlib import Path

from ..access import AccessPolicy, AccessDenied
from ..config import CATEGORIES, Settings
from ..context import TaskContext
from .observation import Observation


def _find_raw_survey(pattern: str | None, base: Path, policy: AccessPolicy, settings: Settings) -> list[str]:
    import subprocess
    from ..redaction import scrubbed_env
    cmd = ["find", str(base)]
    # prune skip dirs
    cmd += ["("]
    skip_list = sorted(policy.skip_dirs)
    for i, d in enumerate(skip_list):
        if i:
            cmd.append("-o")
        cmd += ["-name", d]
    cmd += [")", "-prune", "-o", "-type", "f", "-print"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=settings.limits.timeout_s, check=False,
                             env=scrubbed_env(workspace=base), shell=False)
    except subprocess.TimeoutExpired:
        return []
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    return lines


def survey_files(*, ctx: TaskContext, root: Path | str | None = None,
                 categories: list[str] | str | None = None, extensions: list[str] | None = None) -> Observation:
    t0 = time.perf_counter()
    policy = AccessPolicy.from_settings(ctx.settings, cwd=ctx.root or Path.cwd())
    try:
        base = policy.resolve(root or ctx.root or ctx.settings.read_roots[0])
    except AccessDenied as e:
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="survey", status="blocked", reason=str(e), duration_ms=dur)

    cats_list = ([categories] if isinstance(categories, str) else list(categories or CATEGORIES.keys()))
    wanted: set[str] = set()
    for c in cats_list:
        wanted |= CATEGORIES.get(c, set())
    if extensions:
        wanted |= {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

    by_ext: dict[str, int] = {}
    by_cat: dict[str, int] = {c: 0 for c in cats_list}
    by_top: dict[str, int] = {}
    sample: list[str] = []
    seen = 0

    for raw in _find_raw_survey(None, base, policy, ctx.settings):
        try:
            p = policy.resolve(raw)
        except AccessDenied:
            continue
        if not policy.walkable(p.parent):
            continue  # respect skip+deny dirs
        seen += 1
        ext = p.suffix.lower()
        by_ext[ext or "(none)"] = by_ext.get(ext or "(none)", 0) + 1
        try:
            rel_parts = p.relative_to(base).parts
        except Exception:
            rel_parts = p.parts[-2:] if len(p.parts) >= 2 else p.parts
        top = rel_parts[0] if rel_parts else "(root)"
        by_top[top] = by_top.get(top, 0) + 1
        if ext in wanted:
            for c in cats_list:
                if ext in CATEGORIES.get(c, set()):
                    by_cat[c] += 1
            if len(sample) < 200:
                sample.append(str(p))

    dur = (time.perf_counter() - t0) * 1000
    return Observation(
        tool="survey",
        status="ok",
        items=[],
        text="",
        total_lines=seen,
        n_lines=seen,
        meta={
            "root": str(base),
            "files_seen": seen,
            "by_extension": dict(sorted(by_ext.items(), key=lambda kv: -kv[1])[:25]),
            "by_category": by_cat,
            "by_toplevel": dict(sorted(by_top.items(), key=lambda kv: -kv[1])[:15]),
            "matched_count": sum(by_cat.values()),
            "sample": sample[:12],
            "excluded_dirs": sorted(policy.skip_dirs),
            "note": "Counts EXCLUDE the pruned directories. Ask about a specific folder to count within it.",
        },
        duration_ms=dur,
    )
