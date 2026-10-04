"""The strategy ladder: direct -> plan -> react, chosen per query and escalated on failure.

This replaces the notebook's `needs_plan`, which was a substring match returning a
boolean. Two things were wrong with it, and the second is the one that mattered:

  * substring cues misfire -- `"both"` matches any query containing those letters
  * **there was no cue that could ever select ReAct**, so `strategy="auto"` could
    never reach the ReAct loop. It was unreachable code that looked wired up.

Why a ladder and not a classifier
---------------------------------
A three-way classifier has to be right about every query on the first try, and a
wrong guess is expensive: pick `direct` for something needing six steps and the user
gets a confident one-shot answer that is simply incomplete.

So routing is a *starting guess* and the ladder escalates when a stage cannot settle
the question. That inverts the burden: the router only has to be roughly right, and
the failure of a stage is itself the signal to try the next one. `run_task` escalates
direct -> plan -> react, and react is a terminal state that either answers or reports
`inconclusive`.

The react signals are the interesting part, and they are not about wording -- they are
about whether the question is answerable by READING. `find`, `grep`, `read`, `survey`
settle "where is X" and "how many Y". They cannot settle:

  * anything whose answer is the output of a command (disk usage, git status, file
    sizes, what a test prints, what an error says)
  * anything diagnostic ("why does X fail", "what is the error", "is X broken")
  * anything that wants a *change* ("fix it", "make it pass", "add a flag")

Those are react. The cue list is deliberately about the *asker wanting command
output*, not about incidental words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .context import TaskContext

Strategy = Literal["direct", "plan", "react"]

# A query naming two or more concrete artefacts usually wants them compared, which
# is two searches plus a join -- a plan, not one call.
_MULTI_ARTEFACT = re.compile(
    r"[a-z0-9_\-]+\.(?:py|md|ipynb|json|ya?ml|txt|cfg|toml|rs|go|js|ts|java|c|cpp)\b",
    re.I,
)
_SEQUENCE_CUES = (
    "and then", "after that", "step by step", "first.*then", "and also",
    "compare", "versus", " vs ", "difference between", "across", "summarize",
    "summarise", "then report", "and count", "count.*and",
)

# Questions whose answer IS command output, or which need something to happen.
_REACT_CUES = (
    r"\bhow (?:much|many) (?:space|disk|bytes|memory)\b",
    r"\bdisk usage\b", r"\bdu\b",
    r"\bgit (?:status|log|diff|branch|remote)\b",
    r"\bwhat(?:'s| is) (?:the )?(?:error|output|result|status|version|exit code)\b",
    r"\bwhy does\b", r"\bwhy (?:is|are|did|does)\b",
    r"\bwhat(?:'s| is) wrong\b", r"\bfail(?:s|ing|ed)?\b", r"\bbroken\b",
    r"\brun the tests?\b", r"\bdoes it (?:pass|compile|build|work)\b",
    r"\bcheck if .{0,40}\b(?:work|pass|compile|build|run)\b",
    r"\bfix (?:it|this|that|the)\b", r"\bmake it (?:pass|work)\b",
    r"\b(?:traceback|stack ?trace|segfault|panic)\b",
    r"\bpermissions?\b", r"\bowner\b", r"\bprocess(?:es)? (?:is|are) running\b",
)
_REACT_RE = re.compile("|".join(_REACT_CUES), re.I)

# Reading tools settle a LOCATE or AGGREGATE question on their own.
_READING_ANSWERS = (
    r"\bwhere (?:is|are)\b", r"\bwhich (?:file|folder|directory|path)\b",
    r"\bfind\b", r"\bhow many (?:files?|images?|folders?|directories|documents?)\b",
    r"\blist\b", r"\bshow me\b", r"\bwhat(?:'s| is) in\b", r"\bsearch\b",
    r"\bdoes .{0,40}\b(?:exist|contain|have)\b", r"\bany file\b",
)


@dataclass(frozen=True)
class Route:
    strategy: Strategy
    reason: str
    signals: tuple[str, ...] = ()

    @property
    def escalate_to(self) -> Strategy | None:
        """The next rung, or None at the top. This is what makes the ladder work."""
        return {"direct": "plan", "plan": "react", "react": None}[self.strategy]


def route(query: str, *, ctx: TaskContext | None = None) -> Route:
    """Pick a starting strategy. Cheap, deterministic, and logged.

    Deterministic on purpose. A model call here would cost a round trip before any
    work happens, and it would be a *second* place to be wrong -- the classifier
    model and the answering model would disagree, and neither would be debuggable
    from the trace. This function's whole output is one line in the trace, so a bad
    route is visible immediately rather than inferred.
    """
    q = query.strip().lower()
    signals: list[str] = []

    # REACT first. "compare the size of a.py and b.py" also matches a plan cue, but
    # the react cue is the stronger claim: the answer requires running something.
    if _REACT_RE.search(q):
        signals.append("diagnostic-or-command-output")

    artefacts = set(m.lower() for m in _MULTI_ARTEFACT.findall(query))
    if len(artefacts) >= 2:
        signals.append(f"{len(artefacts)}-artefacts")
    seq = [c for c in _SEQUENCE_CUES if re.search(c, q)]
    if seq:
        signals.append(f"sequence:{seq[0]}")

    if "diagnostic-or-command-output" in signals:
        r = Route("react", "answer needs command output or a diagnostic", tuple(signals))
    elif len(artefacts) >= 2 or seq:
        r = Route("plan", "multiple goals to sequence", tuple(signals))
    elif re.search("|".join(_READING_ANSWERS), q):
        r = Route("direct", "reading tools settle this in one pass", tuple(signals))
    else:
        # Unknown shape: start cheap and escalate. A one-shot `direct` costs one
        # call, so guessing low is nearly free and the ladder recovers the loss.
        r = Route("direct", "no strong signal; start cheap and escalate", tuple(signals))

    if ctx is not None:
        ctx.log("route", "gate",
                detail=f"strategy={r.strategy} ({r.reason})"
                       + (f" signals={list(r.signals)}" if r.signals else ""),
                status="ok", decision=r.strategy, reason=r.reason,
                signals=list(r.signals), tool="router")
    return r


def needs_plan(query: str, *, ctx: TaskContext | None = None) -> tuple[bool, str]:
    """Back-compat shim over `route`. Returns True for anything above `direct`.

    Kept because the Phase 0 tests assert the gate's decision boundary, and because
    something a planner step asks ("do I need to plan?") genuinely only wants the
    boolean. Note the semantics CHANGED: this used to be a substring match over
    `PLAN_NEEDED_CUES`, which could not select react. Now it is `route() != direct`.
    """
    r = route(query, ctx=ctx)
    return r.strategy != "direct", f"route={r.strategy} ({r.reason})"