"""Every path an answer names must appear in the evidence.

This is the first of the two code-first checks, and it exists because asking a model
"does this path appear in the evidence?" does not work. It says yes often enough to be
useless, and a hallucinated path is the single worst failure this system can produce:
a confident, plausible, non-existent filename is indistinguishable from a correct one
to the user reading it.

So it is checked with string matching, and the matching is deliberately strict. A
loose check here is worse than no check, because it manufactures false confidence --
the whole reason the verification chain exists.

Three things it will NOT do, each with a reason:

* **Not resolve through the jail.** The question is "did the agent SEE this", not "does
  this exist". Resolving would make a real-but-unseen path look verified, which is the
  opposite of the check.
* **Not accept a basename when the answer gives an absolute path.** `id_rsa` appears in
  the evidence and `/home/u/.ssh/id_rsa` does not, and answering "yes, verified"
  because a substring matched is how a fabricated absolute path gets through.
* **Not guess.** No fuzzy matching, no normalisation beyond the obvious. A check that
  matches approximately cannot support a claim of certainty.
"""
from __future__ import annotations

import re
from pathlib import Path

# An absolute path, or a path with at least one separator. Deliberately excludes bare
# words: "main.py" alone is too weak a signal to fail an answer over, so a bare
# filename is checked but reported differently (see `unverified_paths`).
ABS_PATH = re.compile(r"(?<![\w])((?:~|\.{0,2})?/[\w.\-+@/]{2,})")
REL_PATH = re.compile(r"(?<![\w./])([\w.\-]+\.(?:py|md|txt|json|ya?ml|toml|cfg|ini|"
                      r"zip|tar|gz|sh|js|ts|tsx|jsx|java|rs|go|c|cpp|h|css|html|xml|"
                      r"csv|tsv|log|db|sqlite|pdf|png|jpe?g|gif|mp[34]|so|dll|"
                      r"whl|safetensors|ipynb))\b")

# Trailing punctuation that belongs to the sentence, not the path.
_TRIM = " \t\n\r\"'`.,;:!?)]}>|"


def _norm(p: str) -> str:
    return p.strip(_TRIM)


def evidence_paths(evidence: str) -> set[str]:
    """Every path-like token in the evidence, in both absolute and basename form."""
    out: set[str] = set()
    for m in ABS_PATH.finditer(evidence):
        p = _norm(m.group(1))
        if p:
            out.add(p)
            out.add(Path(p).name)
    for m in REL_PATH.finditer(evidence):
        p = _norm(m.group(1))
        if p:
            out.add(p)
            out.add(Path(p).name)
    return out


def claimed_paths(answer: str) -> list[str]:
    """Paths the ANSWER asserts. Absolute paths are the ones that must be exact."""
    out: list[str] = []
    for m in ABS_PATH.finditer(answer):
        p = _norm(m.group(1))
        # Skip anything that is really a URL or a number-with-slashes.
        if p and not p.startswith("http") and len(p) > 3:
            out.append(p)
    return out


def unverified_paths(answer: str, evidence: str) -> list[str]:
    """Paths in `answer` with no counterpart in `evidence`.

    An ABSOLUTE path must appear verbatim. A bare filename matches on its basename,
    because "there are three main.py files" and "main.py has a bug" are both ordinary
    things to say and failing them over would make the check cry wolf.
    """
    seen = evidence_paths(evidence)
    ghosts: list[str] = []
    for p in claimed_paths(answer):
        if p in seen:
            continue
        if Path(p).name in seen:
            continue                      # basename-only mention, not a claim of location
        ghosts.append(p)
    return ghosts