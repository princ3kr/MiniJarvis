from __future__ import annotations

import difflib
import fnmatch
import re
import time
from pathlib import Path
from typing import Any

from ..access import AccessPolicy, AccessDenied
from ..config import Settings
from ..context import TaskContext
from .base import run_tool
from .observation import Observation


def _stem(name: str) -> str:
    return Path(name).stem.lower()


def _normalized_tokens(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _pattern_variants(pattern: str) -> list[str]:
    """Generate conservative filename variants for punctuation/case mismatches.

    This catches common user transcription errors like `ask-my-repo-mainV2,zip`
    versus the real file `Ask-My-Repo-mainV2.zip` without broadening to unrelated
    names.
    """
    variants: list[str] = []
    seen: set[str] = set()
    for candidate in (
        pattern,
        pattern.lower(),
        pattern.replace(",", "."),
        pattern.replace(".", ","),
        pattern.replace("_", "-"),
        pattern.replace("-", "_"),
        pattern.replace("-", ""),
        pattern.replace("_", ""),
    ):
        safe = candidate.strip()
        if not safe or safe in seen:
            continue
        seen.add(safe)
        variants.append(safe)
    base = _normalized_tokens(pattern)
    if base:
        variants.append(base)
    return variants


def _prune_args(policy: AccessPolicy) -> list[str]:
    """Build find prune args for skip dirs. Group alternatives with parens."""
    args = ["("]
    skip_list = sorted(policy.skip_dirs)
    for i, d in enumerate(skip_list):
        if i:
            args.append("-o")
        args += ["-name", d]
    args += [")", "-prune", "-o"]
    return args


def _match_kind(pattern: str, path: Path, settings: Settings) -> tuple[int | None, float, str | None]:
    name = path.name
    low, pl = name.lower(), pattern.lower()
    ps, ns = _stem(pl), _stem(low)
    limits = settings.limits
    fml = limits.fuzzy_min_len
    fco = limits.fuzzy_cutoff
    if low == pl:
        return 0, 1.0, "exact"
    if _normalized_tokens(low) == _normalized_tokens(pl):
        return 0, 0.99, "normalized"
    if fnmatch.fnmatch(low, pl):
        return 1, 0.95, "glob"
    same_ext = (Path(name).suffix.lower() == Path(pattern).suffix.lower()
                or not Path(pattern).suffix)
    if pl and (pl in low or (ps and ps in ns)):
        return (2, 0.80, "substring") if same_ext else (2, 0.70, "stem_other_ext")
    if min(len(ns), len(ps)) < fml:
        return None, 0.0, None
    ratio = difflib.SequenceMatcher(None, ns, ps).ratio() if (ns and ps) else 0.0
    if ratio >= fco:
        return 3, round(ratio, 3), "fuzzy"
    return None, round(ratio, 3), None


def _find_raw(pattern: str | None, base: Path, policy: AccessPolicy, settings: Settings,
              limit: int | None = None, kind: str = "f", ctx: TaskContext | None = None) -> list[str]:
    cmd = ["find", str(base), *_prune_args(policy), "-type", kind]
    if pattern:
        # -iname MUST come before -print
        cmd += ["-iname", pattern]
    cmd.append("-print")
    # One runner for every tool: scrubbed env, rlimits, process-group kill and a
    # byte cap. A timeout is a status, not an exception that escapes the tool.
    out = run_tool(cmd, cwd=base, timeout_s=settings.limits.timeout_s)
    if out.status == "timeout" and ctx:
        try:
            ctx.log("find", "search", status="timeout")
        except Exception:
            pass
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    return lines[:limit] if limit else lines


def _rank(pattern: str, raw: str, want_dirs: bool, policy: AccessPolicy, settings: Settings) -> dict | None:
    try:
        rp = policy.resolve(raw)
    except AccessDenied:
        return None
    is_dir = rp.is_dir()
    if is_dir != want_dirs:
        return None
    rank, score, kind = _match_kind(pattern, rp, settings)
    if kind is None:
        return None
    return {"path": str(rp), "rank": rank, "kind": kind, "score": score, "is_dir": is_dir}


def find_files(pattern: str, *, ctx: TaskContext, root: Path | str | None = None,
               max_results: int | None = None, fuzzy: bool = True, include_dirs: bool = True,
               min_size: int | None = None, max_size: int | None = None,
               newer_than_days: int | None = None, type_: str | None = None) -> Observation:
    t0 = time.perf_counter()
    policy = AccessPolicy.from_settings(ctx.settings, cwd=ctx.root or Path.cwd())
    try:
        base = policy.resolve(root or ctx.root or ctx.settings.read_roots[0])
    except AccessDenied as e:
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="find", status="blocked", reason=str(e), duration_ms=dur)

    limits = ctx.settings.limits
    max_res = max_results if max_results is not None else limits.find_max_results

    pattern_wild = pattern if any(ch in pattern for ch in "*?[") else f"*{pattern}*"

    best: dict[str, dict] = {}

    def absorb(raws, want_dirs):
        for raw in raws:
            hit = _rank(pattern, raw, want_dirs=want_dirs, policy=policy, settings=ctx.settings)
            if hit and hit["path"] not in best:
                best[hit["path"]] = hit

    absorb(_find_raw(pattern_wild, base, policy, ctx.settings, kind="f"), False)
    if include_dirs:
        absorb(_find_raw(pattern_wild, base, policy, ctx.settings, kind="d"), True)

    if not best:
        for variant in _pattern_variants(pattern):
            variant_wild = variant if any(ch in variant for ch in "*?[") else f"*{variant}*"
            absorb(_find_raw(variant_wild, base, policy, ctx.settings, kind="f"), False)
            if include_dirs:
                absorb(_find_raw(variant_wild, base, policy, ctx.settings, kind="d"), True)
            if len(best) >= max_res:
                break

    if fuzzy and len(best) < max_res:
        ext = Path(pattern).suffix
        pool = f"*{ext}" if ext else None
        absorb(_find_raw(pool, base, policy, ctx.settings, limit=limits.fuzzy_pool_cap + 1, kind="f"), False)
        if include_dirs:
            absorb(_find_raw(None, base, policy, ctx.settings, limit=limits.fuzzy_pool_cap + 1, kind="d"), True)

    out_items = sorted(best.values(), key=lambda d: (d["rank"], -d["score"], d["path"]))
    out_items = out_items[:max_res]

    dur = (time.perf_counter() - t0) * 1000
    status = "ok" if out_items else "not_found"
    return Observation(
        tool="find",
        status=status,
        items=out_items,
        text="",
        total_lines=len(out_items),
        n_lines=len(out_items),
        start_line=1,
        truncated=len(out_items) >= max_res,
        duration_ms=dur,
    )
