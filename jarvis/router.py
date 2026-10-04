"""The router: ONE LLM call decides direct | loop | plan.

This replaces a regex cue list, which the design explicitly called out for removal.
The first version of this file was `PLAN_NEEDED_CUES` matched with `in` against the
lowercased query. It had two failures, and the second is why the gate was a trap:

* substrings misfire -- `"both"` matches any query containing those four letters
* **no cue could ever select the loop**, so `strategy="auto"` had two reachable paths
  out of three. A diagnostic question got a one-shot file search and a confident
  wrong answer, and every query looked identical in the trace.

Why a model and not a smarter function
--------------------------------------
Because the decision is not lexical. It is about *what kind of work settles the
question*, and no word list captures that:

    "which file is >= 100MB"        needs `find -size`, i.e. a command
    "how many images are there"     one tool call
    "compare a.py and b.py"         two searches and a join

The first and third differ by one word ("which file is >= 100MB" vs "which files are
images"), and a cue list that separated them would be a coincidence. A model reads the
question and picks the tool that can answer it, which is the actual skill.

Why ONE call
------------
One, before any work. Not a call per stage, and not a call per candidate. The router
is a cost, and a router that costs more than the work it routes is not worth having.
If the choice is wrong the ladder recovers -- `escalate` moves direct -> loop -> plan
when a stage returns `not_found` -- so the router only has to be right most of the
time, not always. That is why it can be a single cheap call with no retry.

`plan` is for MULTI-PART tasks only. Anything a bounded loop can converge on is
`loop`, because a loop that escalates itself is strictly better than a plan that
guesses the decomposition up front.

Fallback
--------
If the model is unreachable -- no key, no endpoint, 401 -- `route()` returns `loop`.
Not `direct`: `loop` is the only rung whose failures are self-correcting, and a
degraded router should pick the path that verifies its own work. The regex cue list
is gone entirely rather than kept as a fallback, because a fallback nobody trusts is
a fallback that will be the primary path the first time the API is slow.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from typing import Literal

from . import llm
from .context import BudgetExhausted, TaskContext
from .llm import LLMConfigError, LLMUnavailable

Strategy = Literal["direct", "loop", "plan"]
STRATEGIES: tuple[str, ...] = ("direct", "loop", "plan")

# `plan` -> `loop` -> `direct` is the escalation ORDER for RETRIES. A plan that cannot
# be decomposed into steps is better served by a loop, and a loop that cannot converge
# is better served by a plan that thinks first. `direct` is last because it is the
# rung most likely to produce a confident wrong answer.
ESCALATION: dict[str, str] = {"direct": "loop", "loop": "plan", "plan": "loop"}

JUDGE_PROMPT = """\
You route a filesystem question to the cheapest strategy that can answer it.

You are NOT answering the question. You are choosing who does.

STRATEGIES

direct
  One cheap tool call: find, grep, read, survey, index_query. Nothing else.
  Use it ONLY when one of those calls plainly settles the question.
  "how many python files are under src/", "where is X defined",
  "does a file called notes.txt exist".

loop
  Run shell commands in a loop: pick a command, see its output, decide the next one,
  until the answer is supported by what the commands actually printed.
  Use it for anything where the answer IS command output, or where no single tool
  call is enough:
    - SIZE or TIME: "which file is >= 100MB", "how much disk space", "how old",
      "what is the size of X", "what changed recently"
    - COUNTS WITH CONDITIONS: "how many files are >= 100MB", "how many images are
      larger than 2MB", "how many lines of python are there"
    - "the system" / "my machine" / "everywhere": a scope wider than one directory,
      which needs find/du/df rather than a single directory listing
    - anything needing output you cannot see without running something
    - diagnostics: "why does X fail", "what is the error"

plan
  Decompose a genuinely MULTI-PART task into sub-goals, run them, combine the
  results. Use it only when the question has two or more INDEPENDENT parts that each
  need their own investigation:
    "compare a.py and b.py and tell me which is more readable"
    "find my resume, summarize it, and also check if there is a cover letter"
    "how much disk is Downloads and separately how much is Documents"
  If ONE loop could converge on it, choose loop. A loop that escalates itself beats
  a plan that guesses the decomposition.

TIE-BREAK, in order:
  1. If the answer depends on something you can only get by RUNNING something, loop.
  2. If one tool call settles it, direct.
  3. If it has two or more independent parts, plan.
  4. Otherwise loop. A loop that escalates is safer than a one-shot guess.

Reply with ONE JSON object and nothing else:
{"strategy": "direct|loop|plan", "reason": "under 15 words"}"""


@dataclass(frozen=True)
class Route:
    strategy: Strategy
    reason: str
    source: str = "llm"          # "llm" | "fallback"
    cost: float = 0.0

    @property
    def escalate_to(self) -> str | None:
        """The next rung on failure, or None to give up. `plan` retries as `loop`
        rather than stopping, because an undecomposable plan is a loop's problem."""
        return ESCALATION.get(self.strategy)


def route(query: str, *, ctx: TaskContext | None = None) -> Route:
    """One model call. Never raises, and always returns a usable strategy."""
    q = query.strip()
    try:
        out = llm.call(ctx, [{"role": "system", "content": JUDGE_PROMPT},
                             {"role": "user", "content": q}],
                       purpose="route", temperature=0.0, max_tokens=120)
        cost = float(out.get("usage", {}).get("total_tokens") or 0)
        data = llm.extract_json(out.get("text", ""), expect="object")
        strategy = str(data.get("strategy") or "").strip().lower()
        reason = str(data.get("reason") or "").strip()[:120]
    except llm.LLMUnavailable as e:
        # Degraded, not broken. `loop` because it is the only rung that verifies its
        # own work: escalating on `not_found` means a wrong router still converges.
        if ctx is not None:
            ctx.log("route", "judge", detail=f"llm unavailable: {e}", status="fallback",
                    strategy="loop", reason="llm unavailable", tool="router")
        return Route("loop", f"llm unavailable ({type(e).__name__}); defaulting to loop",
                     source="fallback")
    except BudgetExhausted as e:
        if ctx is not None:
            ctx.log("route", "judge", detail=f"budget: {e}", status="budget",
                    strategy="loop", reason="budget exhausted", tool="router")
        return Route("loop", "budget exhausted before routing", source="fallback")
    except LLMConfigError as e:
        # A CONFIG error is not a transient one. Degrading to a loop would paper over
        # "no API key" with an agent that appears to work and cannot answer, which is
        # the worst possible failure mode for diagnosing it. Say so loudly and still
        # return something runnable.
        if ctx is not None:
            ctx.log("route", "judge", detail=str(e)[:120], status="config_error",
                    strategy="loop", reason="llm misconfigured", tool="router")
        return Route("loop", f"LLM CONFIG ERROR: {e}", source="fallback")
    except Exception as e:
        if ctx is not None:
            ctx.log("route", "judge", detail=f"{type(e).__name__}: {str(e)[:60]}",
                    status="error", strategy="loop", reason="router error",
                    tool="router")
        return Route("loop", f"router error ({type(e).__name__}); defaulting to loop",
                     source="fallback")

    if strategy not in STRATEGIES:
        # A model that names a strategy that does not exist has not answered the
        # question. Treat it as unusable rather than coercing it: coercing would let
        # a typo silently pick a rung.
        if ctx is not None:
            ctx.log("route", "judge", detail=f"bad strategy {strategy!r}",
                    status="error", strategy="loop", reason="unrecognised strategy",
                    tool="router")
        return Route("loop", f"unrecognised strategy {strategy!r}; defaulting to loop",
                     source="fallback", cost=cost)

    if ctx is not None:
        ctx.log("route", "judge", detail=f"{strategy} ({reason})", status="ok",
                strategy=strategy, reason=reason, source="llm", tool="router")
    return Route(strategy, reason or "(no reason given)", source="llm", cost=cost)


# Imported late: context imports this module, so a top-level import would be circular.
from .context import BudgetExhausted  # noqa: E402

# ---------------------------------------------------------------------------
# FAST PATH -- exact lookups. Not a router.
#
# Everything below decides what a `direct` call should LOOK FOR, not which strategy
# to use. The strategy is `route()` above, which is a model call. Keeping these two
# concerns in one file but two clearly separated halves is deliberate: the plan
# removes the regex router, not the cheap exact-match fast path, and conflating them
# is how the cue list came to be load-bearing in the first place.
#
# Nothing here may influence strategy selection. `detect_query_type` in particular
# used to gate `needs_plan` and that is exactly the coupling being removed.
# ---------------------------------------------------------------------------
AGENT_NAME_LITERAL = "searchFiles"


# --------------------------------------------------------------------------- categories
# Natural-language word -> survey category. `None` means "recognised, but carries no
# category": "file"/"files" are stripped by `detect_query_type` so a query about "files"
# does not accidentally narrow the survey to the `code` category.
CATEGORY_WORDS: dict[str, str | None] = {
    "python": "code", "py": "code", "javascript": "code", "js": "code",
    "typescript": "code", "ts": "code", "java": "code", "rust": "code",
    "golang": "code", "cpp": "code", "csharp": "code", "ruby": "code",
    "image": "image", "images": "image", "picture": "image", "pictures": "image",
    "photo": "image", "photos": "image", "screenshot": "image", "icon": "image",
    "icons": "image", "video": "video", "videos": "video", "audio": "audio",
    "music": "audio", "song": "audio", "songs": "audio",
    "document": "document", "documents": "document", "doc": "document",
    "docs": "document", "pdf": "document", "pdfs": "document", "note": "document",
    "notes": "document", "text": "document",
    "spreadsheet": "spreadsheet", "spreadsheets": "spreadsheet", "csv": "spreadsheet",
    "notebook": "notebook", "notebooks": "notebook",
    "archive": "archive", "archives": "archive", "zip": "archive", "zips": "archive",
    "code": "code", "codes": "code", "script": "code", "scripts": "code",
    "source": "code", "file": None, "files": None,
    "data": "data", "dataset": "data", "datasets": "data",
    "binary": "binary", "binaries": "binary", "executable": "binary",
}

AGGREGATE_CUES = (
    "how many", "how much", "count", "number of", "total", "tally",
    "what type", "what types", "what kind", "what kinds", "what format",
    "which type", "which types", "what extension", "what extensions",
    "breakdown", "summary", "summarize", "summarise", "overview",
    "biggest", "largest", "smallest", "oldest", "newest",
    "list all", "show all", "statistics", "stats",
)


def detect_query_type(query: str) -> dict[str, Any]:
    """AGGREGATE vs LOCATE, plus the categories the query implies.

    An aggregate query must go to `survey`. A locate query must go to find/grep.
    Getting this wrong was a real failure: "how many images are there?" used to be
    answered by pattern-matching folders literally named `images`, which reports a
    confident zero on a machine with thousands of images.
    """
    q = query.lower()
    words = set(re.findall(r"[a-z]+", q))
    is_agg = any(cue in q for cue in AGGREGATE_CUES)
    cats = sorted({CATEGORY_WORDS[w] for w in words
                   if w in CATEGORY_WORDS and CATEGORY_WORDS[w]})
    return {"shape": "aggregate" if is_agg else "locate", "aggregate": is_agg,
            "categories": cats,
            "cue_hit": next((c for c in AGGREGATE_CUES if c in q), None)}


# --------------------------------------------------------------------------- intent
# Must cover archives and documents: when `zip` was missing,
# "what is inside Ask-My-Repo-mainV2.zip?" captured no filename at all.
FILENAME_EXTS = (
    "py|ipynb|md|txt|json|jsonl|ya?ml|toml|cfg|ini|env|lock|csv|tsv|log|sql|db|sqlite"
    "|js|jsx|mjs|cjs|ts|tsx|vue|svelte|html|htm|css|scss|sh|bash|zsh|fish|ps1|bat"
    "|c|h|cc|cpp|hpp|cxx|cs|java|kt|kts|go|rs|rb|php|pl|lua|r|jl|swift|dart|ex|exs"
    "|scala|clj|hs|ml|mli|erl|zig|nim|vim|el|asm|s"
    "|zip|tar|gz|tgz|bz2|xz|7z|rar|jar|whl"
    "|pdf|docx?|xlsx?|pptx?|epub|rtf|odt|ods"
    "|png|jpe?g|gif|webp|bmp|tiff?|svg|ico|mp4|mov|avi|mkv|webm|mp3|wav|flac"
)
FILENAME_RE = re.compile(r"[\w\-./~]*[\w\-]+\.(?:" + FILENAME_EXTS + r")\b", re.I)
BARE_FILENAME_RE = re.compile(
    r"\b(?:Dockerfile|Makefile|README|LICENSE|requirements|Procfile)\b", re.I)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "in", "on", "at", "to", "of", "and", "or",
    "for", "with", "why", "how", "what", "where", "which", "does", "do", "did", "can",
    "you", "i", "me", "my", "it", "this", "that", "there", "here", "be", "been",
    "not", "but", "if", "then", "so", "about", "from", "by", "please", "help",
    "show", "find", "tell", "explain", "look", "check", "file", "files", "code",
    "use", "using", "get",
    # instruction verbs/nouns. Without these, a planner instruction like
    # "Search the codebase for the definition of X" leaks `search` and `def` as
    # keywords, and a folder-hint lookup then matches src/ and ide/.
    "search", "searching", "locate", "identify", "definition", "defined",
    "defines", "codebase", "repository", "repo", "project", "folder",
    "directory", "list", "give", "return", "find", "report", "summarise",
    "count", "total", "number", "e.g", "eg", "example", "examples", "etc",
    "return", "contains", "containing", "across", "within",
}


def extract_intent(query: str) -> dict[str, Any]:
    """Pull filenames and keywords out of a natural-language query.

    The filename pass runs FIRST and its words are then removed from the keyword
    list. Without that ordering "Ask-My-Repo-mainV2.zip" contributes both a real
    filename and the junk keyword `Ask`, and `Ask` then fuzzy-matches unrelated
    folders.
    """
    filenames, seen = [], set()
    for pat in (FILENAME_RE, BARE_FILENAME_RE):
        for m in pat.finditer(query):
            fn = m.group(0).strip("./")
            if fn.lower() not in seen:
                seen.add(fn.lower())
                filenames.append(fn)
    keywords = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_.]{2,}", query)
                if w.lower() not in STOPWORDS and not w.endswith(".")]
    # drop fragments of captured filenames, or "Ask" from "Ask-My-Repo.zip"
    # triggers a junk folder search
    fn_words = {w.lower() for f in filenames for w in re.split(r"[.\-_/]", f) if w}
    keywords = [w for w in keywords if w.lower() not in fn_words]
    return {"filenames": filenames, "keywords": list(dict.fromkeys(keywords))[:5],
            "raw": query}


__all__ = ["route", "Route", "STRATEGIES", "ESCALATION", "JUDGE_PROMPT",
           "AGENT_NAME_LITERAL", "CATEGORY_WORDS", "AGGREGATE_CUES",
           "FILENAME_EXTS", "FILENAME_RE", "BARE_FILENAME_RE", "STOPWORDS",
           "detect_query_type", "extract_intent"]
