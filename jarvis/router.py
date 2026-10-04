"""The router: cheap lexical classification of a query into a SHAPE and an INTENT.

Two decisions live here, and they are deliberately different in kind:

  * `detect_query_type` -- AGGREGATE ("how many images?") vs LOCATE ("where is X?").
    An aggregate question must COUNT the tree, not pattern-match a folder name.
    That distinction was a real fix: "how many images" used to match folders
    literally named `images`.
  * `needs_plan` -- whether a task is multi-step. This is the broken gate. See its
    comment; it is kept verbatim so Phase 1 can remove it against a test.

There is also the intent extractor, which turns a natural-language query into
`filenames` + `keywords`. That lives here rather than in `agents.py` because the
router is the only consumer that needs it before an agent is chosen, and because
the STOPWORDS list is a routing concern: it exists to stop a PLANNER's prose
("Search the codebase for the definition of X") from leaking `search`/`codebase`/
`definition` into keywords, which then fuzzy-match every `src/` on disk.
"""
from __future__ import annotations

import re
from typing import Any

from .context import TaskContext

# The one spelling of the agent's name. The prompt bundle stamps it on every
# bundle so a trace can be traced back to which agent produced the evidence.
AGENT_NAME_LITERAL = "searchFiles"


# --------------------------------------------------------------------------- plan gate
# KNOWN BUG (removal scheduled, Phase 1): PLAN_NEEDED_CUES is matched as SUBSTRINGS,
# not as phrases. `"both"` matches "both files and folders", which is fine, but it also
# matches a single-file query that merely contains those letters; more importantly there
# is no cue that can ever select the ReAct path, so the ReAct loop is unreachable from
# `run_task(strategy="auto")`. Phase 1 replaces this whole gate with a DSPy classifier
# plus an escalation ladder (direct -> discovery -> plan -> react). Kept for Phase 0
# fidelity: several notebook regressions encode the CURRENT decision boundary.
PLAN_NEEDED_CUES = ("then", "and also", "after that", "compare", "compare them",
                    "both", "across", "summarize", "summarise", "then find",
                    "and count", "and report", "step by step")


def needs_plan(query: str, *, ctx: TaskContext | None = None) -> tuple[bool, str]:
    """Cheap gate. Logged so the threshold can be tuned from data later.

    KNOWN BUG (removal scheduled, Phase 1): substring cues misfire -- "both" matches
    any query containing that substring, and this gate never selects the ReAct path at
    all. Replaced by a DSPy classifier plus an escalation ladder. Kept for Phase 0
    fidelity.
    """
    q = query.lower()
    cue = next((c for c in PLAN_NEEDED_CUES if c in q), None)
    multi = len(re.findall(r"[a-z_]+\.(?:py|md|ipynb|json|yaml|txt)", q)) >= 2
    decision = bool(cue) or multi
    reason = f"cue={cue!r}" if cue else ("multiple files named" if multi else "single-step")
    if ctx is not None:
        ctx.log("route", "gate", detail=f"needs_plan={decision} ({reason})", status="ok",
                decision=decision, reason=reason, tool="router")
    return decision, reason


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


__all__ = ["AGENT_NAME_LITERAL", "PLAN_NEEDED_CUES", "CATEGORY_WORDS", "AGGREGATE_CUES",
           "FILENAME_EXTS", "FILENAME_RE", "BARE_FILENAME_RE", "STOPWORDS",
           "needs_plan", "detect_query_type", "extract_intent"]
