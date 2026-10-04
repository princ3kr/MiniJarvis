"""The prompt bundle: turn an agent result into the exact text the model will read.

This module is the answer-quality layer, and its wording is not decoration. Three
encodings here fix real, observed failures:

  * `"50 images"` and `"1143 images"` are BOTH TRUE for different scopes. The aggregate
    bundle must therefore emit the count AND the exclusion list AND a `COUNTING CAVEAT`
    that says so explicitly, or the model picks whichever number it likes and states it
    as fact. The scope is part of the answer, not a footnote.
  * A near miss must never be presented as the requested file. The bundle has a
    dedicated `NO EXACT MATCH` heading for that, and the near-miss branch refuses to
    print an excerpt at all.
  * A file that could not be read is printed as `(not read: <status>)` rather than being
    silently omitted, so the model says "I could not read it" instead of inventing its
    contents.

`SYSTEM_PROMPT` encodes the same three rules at the model level, so a bundle that is
shown to a human and a bundle that is shown to a model teach the same lesson.

Output format is a contract. The bundle is plain text with fixed section headings
because that is what the live-run cell and the CLI both print, and because a stable
layout is what makes a bad answer diagnosable after the fact.
"""
from __future__ import annotations

from typing import Any

from .router import AGENT_NAME_LITERAL as AGENT_NAME

NOT_FOUND_REPLY = ("I couldn't find anything resembling that. I searched by filename and "
                   "by file contents, and got no matches — so rather than guess, here's "
                   "what I looked for.")

SYSTEM_PROMPT = """You are a precise assistant answering questions about files on a user's computer.

Rules:
- Use ONLY the evidence provided. Never invent file names, counts, line numbers, or code.
- Cite sources inline as path/to/file.ext:12.

For AGGREGATE queries (counts, totals, breakdowns):
- Give the exact number from the COUNT BY CATEGORY section. Do not estimate.
- ALWAYS state that the count excludes the pruned directories, and name a few.
  "50 images" and "1143 images" are both true for different scopes; say which you mean.
- Do not count the sample lines to get a number.

For NO EXACT MATCH:
- Say clearly that no exact match exists.
- Then offer the near misses as possibilities and ask which was intended.
- Never present a near miss as if it were the requested file.

For LOCATE queries:
- If the evidence does not answer it, say so plainly and name what is missing.
- If a file could not be read, say so rather than guessing its contents.
- When several similar files were found, list them and explain the difference.

Be concise and concrete. No preamble, no restating the question."""


def build_prompt_bundle(result: dict[str, Any]) -> str:
    """Render an agent result as the user half of the final LLM call.

    Three branches, and they must never be merged: an aggregate result has no excerpts
    to show, a near miss has no hit to show, and a not-found has nothing at all. Merging
    them is how "no exact match" started being answered with an excerpt of the closest
    file.

    KNOWN BUG (removal scheduled, Phase 3): the locate branch indexes
    `e["match_line"]` / `e["match_text"]` with `[]` rather than `.get()`, so any evidence
    row lacking those keys raises `KeyError` inside prompt construction instead of
    degrading. Every producer in `agents.py` sets them today, which is exactly why this
    has never fired. Phase 3 makes the evidence rows a typed dataclass so the shape is
    checked once, at construction, instead of being assumed here.
    """
    if result.get("route") == "aggregate":
        sv = result["survey"]
        cats = result["searches"][0]["categories"]
        out = [f"AGENT: {AGENT_NAME} (read-only)  [AGGREGATE QUERY]",
               f"QUERY: {result['query']}", f"FILES SURVEYED: {sv['files_seen']}",
               f"ROOT: {sv['root']}", ""]
        if cats:
            out.append(f"COUNT BY CATEGORY ({', '.join(cats)}):")
            for c, n in sorted(sv["by_category"].items(), key=lambda kv: -kv[1]):
                out.append(f"    {c:<12} {n}")
            out += [f"    {'TOTAL':<12} {sv['matched_count']}", "",
                    "EXTENSION BREAKDOWN (top 15):"]
            out += [f"    {e:<12} {n}" for e, n in list(sv["by_extension"].items())[:15]]
            if sv["sample"]:
                out += ["", "SAMPLE MATCHES (proof the count is real):"]
                out += [f"    {s}" for s in sv["sample"][:6]]
        else:
            out.append("No category word recognised. Top-level distribution:")
            out += [f"    {k:<22} {v}" for k, v in list(sv["by_toplevel"].items())[:12]]
        out += ["", "=" * 60, "COUNTING CAVEAT (must be stated in the answer):",
                f"  These counts EXCLUDE: {', '.join(sv['excluded_dirs'])}",
                "  This is the count of files the agent can see, NOT the whole disk.",
                f"CONFIDENCE: {result['confidence']}   ({result['elapsed_ms']}ms)",
                "INSTRUCTION: Answer with the exact number. State the exclusions. "
                "Do NOT re-count from the sample."]
        return "\n".join(out)

    if result["status"] == "near_miss":
        out = [f"AGENT: {AGENT_NAME} (read-only)  [NO EXACT MATCH]",
               f"QUERY: {result['query']}",
               f"SEARCHES ATTEMPTED: {result['searches']}", "",
               f"NEAR MISSES ({len(result['near_misses'])}):"]
        out += [f"    score {c['score']:.2f}  {c['path']}"
                for c in result["near_misses"]]
        out += ["=" * 60,
                "INSTRUCTION: No exact match exists. Say so plainly, then offer the "
                "closest matches and ask which was meant. Do NOT present a near miss "
                "as if it were the requested file."]
        return "\n".join(out)

    if result["status"] == "not_found":
        return (f"AGENT: {AGENT_NAME}\nQUERY: {result['query']}\nRESULT: not_found\n"
                f"SEARCHES ATTEMPTED: {result['searches']}\n"
                f"INSTRUCTION: Tell the user plainly that nothing matched. "
                f"Do NOT guess a filename or invent content.")

    sim = result.get("similar_files") or []
    out = [f"AGENT: {AGENT_NAME} (read-only)", f"QUERY: {result['query']}",
           f"SEARCHES: {result['searches']}"]
    if len(sim) > len(result["evidence"]):
        out.append(f"ALL {len(sim)} SIMILAR FILES FOUND (only the first "
                   f"{len(result['evidence'])} were read):")
        out += [f"    [{s['kind']:<9} {s['score']}] {s['path']}" for s in sim]

    out += ["", f"EVIDENCE ({len(result['evidence'])} item(s) read):", ""]
    for i, e in enumerate(result["evidence"], 1):
        label = "folder" if e.get("is_dir") else e["kind"]
        out.append(f"--- [{i}] {e['path']}   (matched by {label})")
        if e.get("is_dir"):
            out += ["    (folder listing, not file contents)", e["excerpt"], ""]
            continue
        if e["match_line"]:
            out.append(f"    line {e['match_line']}: {e['match_text']}")
        if e["read_status"] != "ok":
            out.append(f"    (not read: {e['read_status']})")
            continue
        if e["truncated"]:
            out.append("    (excerpt, truncated)")
        out += [e["excerpt"], ""]

    out += ["=" * 60, f"CONFIDENCE: {result['confidence']}   ({result['elapsed_ms']}ms)",
            "INSTRUCTION: Answer the query in natural language using ONLY the evidence "
            "above. Cite files as path:line. If the evidence does not answer it, say so "
            "plainly rather than guessing."]
    return "\n".join(out)


def build_llm_messages(result: dict[str, Any]) -> list[dict[str, str]]:
    """The two messages every synthesis call makes: our system prompt, then the bundle."""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt_bundle(result)}]


__all__ = ["AGENT_NAME", "SYSTEM_PROMPT", "NOT_FOUND_REPLY",
           "build_prompt_bundle", "build_llm_messages"]
