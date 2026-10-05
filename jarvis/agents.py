"""The `searchFiles` agent: the only agent implemented in Phase 0.

Shape of the run
----------------
    intent  -> route -> recall -> read -> describe

`extract_intent` and `detect_query_type` live in `jarvis/router.py` (they decide the
route, and the router is the natural owner of them). Everything that TOUCHES the
filesystem lives here.

Two routes, and they are not interchangeable:

  * `aggregate` -- "how many images?" must walk the tree and COUNT. Answering it by
    name-matching folders called `images` reports a confident zero on a machine full of
    images. So aggregate queries never call find/grep at all.
  * `locate` -- "where is X?" recalls by name first (cheap, precise), then by folder
    name, then falls back to content, and only then offers near misses.

Return shape
------------
`run_agent` returns a plain dict with a fixed set of keys. That is deliberate: the
prompt bundle, the CLI, and the notebook's live-run cell all read these keys by name,
so the shape is a public contract of this module. The aggregate route additionally
returns a `survey` key, which the bundle formats as a scoped count.

The ranking itself is NOT here. It lives in `jarvis/tools/find.py` and is called, not
reimplemented -- the notebook had two copies of the ranking rules that drifted, and the
copy that drifted is how `find -iname` ended up AFTER `-print` in one of them.
"""
from __future__ import annotations

import difflib
import os
import re
import time
from pathlib import Path
from typing import Any

from .access import AccessDenied, AccessPolicy
from .context import TaskContext
from .planner import AgentResult
from .router import (AGENT_NAME_LITERAL, detect_query_type, extract_intent,
                     is_simple_count_query)
from .tools import TOOL_MANIFEST
from .tools import find as _find_tool
from .tools.observation import Observation

AGENT_NAME = AGENT_NAME_LITERAL


# --------------------------------------------------------------------------- tool seam
def call_tool(name: str, *, ctx: TaskContext, **kwargs) -> Observation:
    """The one place the agent touches a tool. Fails CLOSED on an unknown name.

    Same allowlist shape as the notebook's `call_tool`, but the manifest is now the
    registry of record, so a new tool is reachable the moment it is registered and an
    unregistered name cannot be reached at all.

    A tool that raises propagates. That is intentional and it is the agent boundary
    that catches: `execute_plan` wraps the whole agent call, so a tool fault becomes a
    failed attempt with a reason, not an exception that kills the run.
    """
    spec = TOOL_MANIFEST.get(name)
    if spec is None:
        raise ValueError(f"tool {name!r} not allowed; allowed = {list(TOOL_MANIFEST)}")
    ctx.charge_tool(name)
    return spec.handler(ctx=ctx, **kwargs)


# --------------------------------------------------------------------------- near misses
def near_misses(pattern: str, *, ctx: TaskContext, root: Path | str | None = None,
                limit: int | None = None) -> list[dict[str, Any]]:
    """Names that ALMOST match, ranked, for "did you mean?" framing.

    A near miss is a BAND, not a threshold: >= `fuzzy_cutoff` is a real match and is
    excluded (an exact hit must never be reported as a near miss), < `near_miss_min` is
    noise and is excluded. The adaptive floor of 0.55 for a large pool exists because
    `difflib` scores short strings on incidental overlap, so on a big tree the
    near_miss_min band fills up with garbage.

    Reuses `tools.find._find_raw` -- a private helper, deliberately. It is the only way
    to get the unranked `*<ext>` pool this scoring needs, and duplicating the prune args
    is exactly how the two copies of `find` drifted apart in the first place.
    """
    t0 = time.perf_counter()
    lim = ctx.settings.limits
    limit = lim.near_miss_max if limit is None else limit
    policy = AccessPolicy.from_settings(ctx.settings, cwd=ctx.root or Path.cwd())
    try:
        base = policy.resolve(root or ctx.root or ctx.settings.read_roots[0])
    except AccessDenied:
        return []
    ext = Path(pattern).suffix
    pool = _find_tool._find_raw(f"*{ext}" if ext else None, base, policy, ctx.settings,
                                limit=lim.fuzzy_pool_cap, kind="f", ctx=ctx)
    ps = Path(pattern).stem.lower()
    scored: list[dict[str, Any]] = []
    for raw in pool:
        try:
            rp = policy.resolve(raw)
        except AccessDenied:
            continue
        ns = rp.stem.lower()
        if not (ns and ps) or ns == ps:
            continue
        ratio = difflib.SequenceMatcher(None, ns, ps).ratio()
        if ratio >= lim.fuzzy_cutoff or ratio < lim.near_miss_min:
            continue
        scored.append({"path": rp, "score": round(ratio, 3)})

    min_ratio = max(lim.near_miss_min, 0.55) if len(pool) > 20 else lim.near_miss_min
    scored = [s for s in scored if s["score"] >= min_ratio]
    scored.sort(key=lambda d: -d["score"])
    out = scored[:limit]
    ctx.log("tool", "near_miss", detail=f"{pattern} -> {len(out)}",
            status="ok" if out else "none", t0=t0, tool="near_miss")
    return out


# --------------------------------------------------------------------------- folders
def _describe_dir(d: Path | str, *, ctx: TaskContext,
                  max_children: int = 40) -> dict[str, Any]:
    """Summarise a FOLDER so it can be read as evidence.

    A folder is findable (`find -type d`) but is not a file, so `read` refuses it. Without
    this, "show me the rust project" returned a folder hit with `read_status: not_found`
    and the answer said the folder could not be read -- technically true, useless.

    Both halves of the summary -- the child listing and the recursive histogram --
    prune through `policy.walkable`. That was a KNOWN BUG (removal scheduled, Phase 3)
    and it is fixed here rather than carried, because the fix is small and the bug had
    two faces:

      * the child listing filtered `skip_dirs` while the histogram did not, so
        `__pycache__` was hidden from the listing yet its files were still counted in
        "N file(s) in total". Two halves of one summary disagreeing about what exists
        is worse than either being wrong alone.
      * the name check was not the jail. `.ssh` is not in `skip_dirs`, so the listing
        announced `.ssh/` as a child folder. No secret leaked -- this function does not
        read contents -- but a summary that enumerates a private-key directory by name
        is one edit away from doing so.

    `os.walk` with an in-place prune of `dirnames`, rather than `rglob` filtered
    afterwards: pruning the descent means an excluded tree is never entered and never
    stat-ed, and "a file exists here" is itself a disclosure.
    """
    t0 = time.perf_counter()
    policy = AccessPolicy.from_settings(ctx.settings, cwd=ctx.root or Path.cwd())
    try:
        d = policy.resolve(d)
        # Listing uses `nameable` (deny-list only) and the histogram uses
        # `walkable` (deny-list + skip_dirs). The split is deliberate: the listing
        # should say `node_modules/` exists, since hiding it makes the answer wrong;
        # the histogram should not count 30,000 vendored files, since that is noise.
        # `.ssh` is denied by both, so it appears in neither.
        kids = [c for c in sorted(d.iterdir()) if policy.nameable(c)]
        kids_dirs = [c.name + "/" for c in kids if c.is_dir()]
        kids_files = [c.name for c in kids if c.is_file()]
    except (OSError, PermissionError) as e:
        return {"path": str(d), "kind": "folder", "is_dir": True, "score": 0.0,
                "read_status": "error", "truncated": False,
                "excerpt": f"(could not list: {e})"}

    # histogram over the whole subtree; counting children AND rglob double-counts
    counts: dict[str, int] = {}
    sub_files = sub_dirs = 0
    try:
        # os.walk, not rglob: mutating `dirs` in place is the documented way to prune
        # a descent, so an excluded tree is never entered -- and never stat-ed, which
        # matters because "a file exists here" is itself a disclosure.
        for dirpath, dirnames, filenames in os.walk(d):
            keep = [x for x in dirnames if policy.walkable(Path(dirpath) / x)]
            sub_dirs += len(keep)
            dirnames[:] = keep          # prune, do not merely filter
            for fn in filenames:
                p = Path(dirpath) / fn
                if policy.try_resolve(p, for_write=False) is None:
                    continue
                sub_files += 1
                ext = p.suffix.lower() or "(no ext)"
                counts[ext] = counts.get(ext, 0) + 1
    except OSError:
        pass
    top_ext = sorted(counts.items(), key=lambda kv: -kv[1])[:6]

    lines = [f"FOLDER: {d}",
             f"  child folders ({len(kids_dirs)}): "
             + (", ".join(kids_dirs[:max_children]) or "(none)"),
             "  types in tree: "
             + (", ".join(f"{e} x{n}" for e, n in top_ext) or "(none)")]
    if kids_files:
        lines.append("  files here: " + ", ".join(kids_files[:20]))
    if sub_dirs or sub_files:
        lines.append(f"  recursively: {sub_dirs} subfolder(s), "
                     f"{sub_files} file(s) in total")

    ctx.log("tool", "read", detail=f"{d.name}/ ({len(kids_dirs)}d {len(kids_files)}f)",
            status="ok", t0=t0)
    return {"path": str(d), "kind": "folder", "is_dir": True, "score": 0.0,
            "read_status": "ok", "truncated": len(kids_dirs) > max_children,
            "excerpt": "\n".join(lines)}


def _aggregate_root(query: str, root: Path | str | None, *, ctx: TaskContext) -> Path:
    """Use a uniquely named immediate child as the scope for aggregate queries."""
    policy = AccessPolicy.from_settings(ctx.settings, cwd=ctx.root or Path.cwd())
    base = policy.resolve(root or ctx.root or ctx.settings.read_roots[0])
    low = query.lower()
    if re.search(r"\b(system|machine|computer|everywhere|whole\s+system)\b", low):
        return base

    normalized_query = re.sub(r"[^a-z0-9]+", "", low)
    candidates: list[Path] = []
    try:
        for child in base.iterdir():
            if not child.is_dir() or not policy.nameable(child):
                continue
            normalized_name = re.sub(r"[^a-z0-9]+", "", child.name.lower())
            if normalized_name and normalized_name in normalized_query:
                candidates.append(policy.resolve(child))
    except OSError:
        return base
    return candidates[0] if len(candidates) == 1 else base


# --------------------------------------------------------------------------- the agent
def run_agent(user_query: str, *, ctx: TaskContext, top_files: int = 3,
              root: Path | str | None = None,
              exhaustive: bool | None = None) -> dict[str, Any]:
    """Answer one locate-or-count question from the filesystem.

    Returns the fixed-shape dict documented in the module docstring. Never raises for a
    tool problem -- a blocked root degrades to `not_found` evidence-free rather than an
    exception, so the caller always gets something it can format.
    """
    t_start = time.perf_counter()
    intent = extract_intent(user_query)
    qtype = detect_query_type(user_query)
    intent["query_type"] = qtype
    if exhaustive is None:
        exhaustive = bool(intent["filenames"])

    lim = ctx.settings.limits
    _rd = qtype["shape"] + (f" cats={qtype['categories']}" if qtype["categories"] else "")
    ctx.log("route", "route", detail=_rd, status="ok", shape=qtype["shape"],
            categories=qtype["categories"], cue=qtype["cue_hit"], tool="router")

    # ---- aggregate: "how many images are there?"
    if qtype["aggregate"]:
        cats = qtype["categories"]
        survey_root = _aggregate_root(user_query, root, ctx=ctx)
        sv = call_tool("survey", ctx=ctx, root=survey_root, categories=cats or None,
                       include_skipped_dirs=True,
                       count_only=is_simple_count_query(user_query))
        rec = ctx.trace.event("agent", "run_agent", detail=user_query[:60], status="ok",
                              t0=t_start, shape="aggregate", cats=cats, tool="agent")
        ok = sv.status == "ok" and bool(sv.meta.get("complete", True))
        if not ok:
            summary = f"Survey incomplete: {sv.reason or sv.status}."
        elif cats:
            summary = f"Counted {sv.meta['matched_count']} matching files under {survey_root}."
        else:
            summary = f"Counted {sv.meta['files_seen']} files under {survey_root}."
        return {"status": "ok" if ok else "error", "query": user_query, "intent": intent,
                "route": "aggregate", "near_misses": [],
                "searches": [{"by": "survey", "categories": cats,
                              "root": str(survey_root)}],
                "survey": sv.meta, "evidence": [], "similar_files": [],
                "summary": summary,
                "confidence": 1.0 if ok else 0.0, "elapsed_ms": rec.duration_ms}

    ranked: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    searches: list[dict[str, Any]] = []

    # 1. name recall: ALL filename candidates, union, no early break
    for fn in intent["filenames"]:
        found = call_tool("find", ctx=ctx, pattern=fn, root=root,
                          max_results=lim.find_max_results).items
        searches.append({"by": "find", "query": fn, "n": len(found),
                         "folders": sum(1 for f in found if f["is_dir"])})
        for h in found:
            if h["path"] not in seen_paths:
                seen_paths.add(h["path"])
                ranked.append(h)

    # 1b. bare keywords have no extension, so FILENAME_RE misses them:
    #     "show me the rust project" needs a folder lookup.
    #     fuzzy=False on purpose -- a folder hint must be a real name match, else
    #     instruction prose ("Search the codebase") matches src/ everywhere.
    for kw in intent["keywords"]:
        if len(kw) < 3 or any(kw.lower() == f.lower() for f in intent["filenames"]):
            continue
        hits = [f for f in call_tool("find", ctx=ctx, pattern=kw, root=root, fuzzy=False,
                                     max_results=lim.dir_hint_max).items if f["is_dir"]]
        if hits:
            searches.append({"by": "find_dir", "query": kw, "n": len(hits)})
        for h in hits:
            if h["path"] not in seen_paths:
                seen_paths.add(h["path"])
                ranked.append(h)

    # 2. content recall: only when no name matched, or when forced exhaustive
    if not ranked:
        for kw in (intent["keywords"] if exhaustive else intent["keywords"][:1]):
            found = call_tool("grep", ctx=ctx, query=kw, root=root,
                              max_results=lim.max_results).items
            searches.append({"by": "grep", "query": kw, "n": len(found)})
            for h in found:
                if h["path"] not in seen_paths:
                    seen_paths.add(h["path"])
                    ranked.append({"path": h["path"], "rank": 9, "kind": "content",
                                   "score": 0.0, "is_dir": False,
                                   "line_no": h["line_no"], "text": h["text"]})

    if not ranked:
        # offer near misses before giving up
        nm: list[dict[str, Any]] = []
        for fn in (intent["filenames"] or intent["keywords"][:2]):
            nm.extend(near_misses(fn, ctx=ctx, root=root))
        uniq, s = [], set()
        for c in nm:
            if c["path"] not in s:
                s.add(c["path"])
                uniq.append(c)
        uniq = uniq[:lim.near_miss_max]
        rec = ctx.trace.event("agent", "run_agent", detail=user_query[:60],
                              status="near_miss" if uniq else "not_found", t0=t_start,
                              near=len(uniq), tool="agent")
        return {"status": "near_miss" if uniq else "not_found",
                "query": user_query, "intent": intent, "searches": searches,
                "evidence": [], "similar_files": [], "near_misses": uniq,
                "summary": (f"No exact match, {len(uniq)} similar name(s) exist."
                            if uniq else "No file matched."),
                "confidence": 0.3 if uniq else 0.0, "elapsed_ms": rec.duration_ms}

    # 3. read top N files, centred on the match; folders get summarised
    evidence: list[dict[str, Any]] = []
    for h in ranked:
        if len([e for e in evidence if e["kind"] != "folder"]) >= top_files:
            break
        if h.get("is_dir"):
            evidence.append(_describe_dir(h["path"], ctx=ctx))
            continue
        line = h.get("line_no")
        r = call_tool("read", ctx=ctx, path=h["path"],
                      start_line=(line - lim.read_context if line else 1))
        evidence.append({"path": str(h["path"]), "kind": h["kind"], "score": h["score"],
                         "is_dir": False, "match_line": line,
                         "match_text": h.get("text"), "read_status": r.status,
                         "truncated": r.truncated, "excerpt": (r.text or "")[:6000]})

    similar = [{"path": str(h["path"]), "kind": h["kind"], "score": h["score"],
                "is_dir": bool(h.get("is_dir"))} for h in ranked]
    by_grep = any(s["by"] == "grep" for s in searches)
    rec = ctx.trace.event("agent", "run_agent", detail=user_query[:60], status="ok",
                          t0=t_start, files=len(evidence), similar=len(similar),
                          tool="agent")
    return {"status": "ok", "query": user_query, "intent": intent, "route": "locate",
            "searches": searches, "evidence": evidence, "similar_files": similar,
            "near_misses": [],
            "summary": f"Found {len(similar)} similar file(s); read {len(evidence)}.",
            "confidence": 0.8 if not by_grep else 0.6, "elapsed_ms": rec.duration_ms}


# --------------------------------------------------------------------------- adapter
def run_search_files(instruction: str, context: dict[int, Any], *,
                     ctx: TaskContext, top_files: int = 3) -> AgentResult:
    """Adapt the searchFiles agent to the AgentResult contract.

    `context` carries verified results from dependency steps. It is deliberately
    NOT concatenated into the query: the search agent does its own intent parsing and
    prepending prose would only pollute it. That was a real finding -- prepending a
    dependency's summary made every downstream search return `not_found`.
    """
    t0 = time.perf_counter()
    res = run_agent(instruction, ctx=ctx, top_files=top_files)
    ctx.log("agent", "searchFiles_adapter",
            detail=f"{res['status']} in {round((time.perf_counter() - t0) * 1000)}ms",
            status=res["status"], tool="agent")
    if res.get("route") == "aggregate":
        sv = res["survey"]
        return AgentResult(
            status="ok",
            summary=res["summary"],
            evidence=[{"path": s, "kind": "sample"} for s in sv["sample"][:8]],
            confidence=res["confidence"],
            meta={"route": "aggregate", "counts": sv["by_category"],
                  "files_seen": sv["files_seen"], "excluded": sv["excluded_dirs"]})

    if res["status"] == "near_miss":
        return AgentResult(status="near_miss",
                           summary=res["summary"],
                           evidence=[{"path": str(c["path"]), "score": c["score"]}
                                     for c in res["near_misses"]],
                           confidence=res["confidence"],
                           meta={"route": "locate", "near_misses": res["near_misses"]})

    if res["status"] == "not_found":
        return AgentResult(status="not_found", summary=res["summary"],
                           evidence=[], confidence=0.0, meta={"route": "locate"})

    ev = [{"path": e["path"], "kind": e["kind"],
           "read_status": e["read_status"],
           "line": e.get("match_line"),
           "excerpt": e["excerpt"][:800]} for e in res["evidence"]]
    return AgentResult(status="ok", summary=res["summary"], evidence=ev,
                       confidence=res["confidence"],
                       meta={"route": "locate",
                             "similar": res["similar_files"][:10]})


AGENT_RUNNERS: dict[str, Any] = {"searchFiles": run_search_files}
"""Notebook's `AGENT_RUNNERS` was a module global mutated by the test suite to stub an
agent. It is still a registry, but the tests mutate the test's own reference and restore
it; the orchestrator treats an agent name absent from `IMPLEMENTED_AGENTS` as a hard
failure, so a registry entry is never enough to make an agent runnable."""
