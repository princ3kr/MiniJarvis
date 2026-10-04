"""The planner: turn one goal into a small ordered DAG of agent steps.

Phase 0 scope
-------------
This is a mechanical port of the notebook's planner cell. Every behaviour below that
looks wrong is wrong ON PURPOSE and is tagged `# KNOWN BUG (removal scheduled)` with
the phase that removes it. The regression net is `tests/test_planner.py`, which asserts
the CURRENT behaviour, including the wrong parts. That ordering matters: a later phase
must be able to show "test X failed, and X failing is the fix".

DSPy is used here, and only here, because a typed Pydantic output genuinely pays: a
plan is a JSON object graph, and a malformed one should be a validation error rather
than prose to be salvaged. It is used ONLY where an LM is already configured -- see
`run_task`, which configures DSPy lazily and only when a plan is actually needed.

`StepStatus` and `FailureType` are NOT redefined here: they live in `jarvis/trace.py`
and are re-exported, so the trace and the plan cannot disagree about the vocabulary.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .context import TaskContext
from .trace import FailureType, Trace, Verdict

# Re-exported, not redefined. Verdict is used by the verifier via this module too.
__all_reuse__ = (FailureType, Verdict)


# --------------------------------------------------------------------------- types
class StepStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    inconclusive = "inconclusive"   # verifier said `unsure` -- NOT a failure
    skipped = "skipped"


class Step(BaseModel):
    id: int
    agent: str
    instruction: str
    depends_on: list[int] = Field(default_factory=list)
    expected_output: str
    status: StepStatus = StepStatus.pending
    result: dict[str, Any] | None = None

    @field_validator("depends_on", mode="before")
    @classmethod
    def _null_deps(cls, v):
        # the model emits null for "no dependencies" despite being told []
        return [] if v is None else v


class Plan(BaseModel):
    goal: str
    steps: list[Step]

    @model_validator(mode="after")
    def _check_dag(self):
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate step ids: {ids}")
        for s in self.steps:
            for d in s.depends_on:
                if d not in ids:
                    raise ValueError(f"step {s.id} depends on unknown step {d}")
            if s.id in s.depends_on:
                raise ValueError(f"step {s.id} depends on itself")
        return self

    def by_id(self, i: int) -> Step | None:
        return next((s for s in self.steps if s.id == i), None)


@dataclass
class AgentResult:
    """What every agent must return to be verifiable. Uniform by construction."""
    status: Literal["ok", "near_miss", "not_found"]
    summary: str
    evidence: list[Any] = field(default_factory=list)
    confidence: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- registry
# ONLY what exists. `webSearch` and `assistant` were here for the whole of the
# notebook's life with nothing behind them: a plan step naming one failed at RUN time
# instead of at PLANNING time, so the failure surfaced as "agent not implemented"
# halfway through a task rather than as a rejected plan.
#
# That is worse than useless, because the vocabulary is what a model copies when it
# writes a plan. Naming an agent it can see in the registry teaches it to emit steps
# that cannot run. The vocabulary is now exactly the set with a runner, and adding
# one is a deliberate act: write the runner, then add the name.
# `planner` is a ROLE the orchestrator plays (it makes the plan), not an agent a step
# can be dispatched to -- a step naming it has no runner and would fail at run time.
# It stays in the registry so the vocabulary is complete and honest about the
# distinction, and out of IMPLEMENTED_AGENTS because that set is exactly "has a
# runner".
AGENT_REGISTRY = {"searchFiles", "planner", "loop"}
IMPLEMENTED_AGENTS = {"searchFiles", "loop"}

# model-invented aliases -> canonical name
# Keys are the agent name with every non-letter stripped, because normalise_agent()
# does exactly that before lookup. So "file_search_agent" -> "filesearchagent".
#
# Aliases for REMOVED agents are gone too. Keeping "websearch" -> something would mean
# the model could still reach a name that has no runner, which is the exact failure
# this registry was changed to remove.
AGENT_ALIASES = {
    # searchFiles
    "filesearchagent": "searchFiles", "filesearch": "searchFiles",
    "search": "searchFiles", "searchfiles": "searchFiles",
    "searchfile": "searchFiles", "explore": "searchFiles", "finder": "searchFiles",
    "filesystem": "searchFiles", "locate": "searchFiles", "grep": "searchFiles",
    # planner / loop
    "plan": "planner", "planning": "planner",
    "loop": "loop", "react": "loop", "reactloop": "loop", "shell": "loop",
    "command": "loop", "runcommand": "loop", "exec": "loop",
}


def normalise_agent(name: str) -> str:
    """Map an alias to its canonical agent. Raises on genuinely unknown agents.

    Lookup key strips non-letters, so `file_search_agent`, `FileSearchAgent`, and
    `filesearch` all collapse to the same entry.

    Invented-but-plausible names are a real failure mode of a small LM asked for JSON:
    it emits `file_search_agent` instead of `searchFiles` and, without this alias table,
    the whole plan is rejected and the task silently degrades to the 2-step fallback.
    """
    key = re.sub(r"[^a-z]", "", str(name).lower())
    canon = AGENT_ALIASES.get(key)
    if canon:
        return canon
    for a in AGENT_REGISTRY:
        if a.lower() == str(name).lower():
            return a
    raise ValueError(
        f"unknown agent {name!r}; available: {sorted(AGENT_REGISTRY)}")


# --------------------------------------------------------------------------- dspy
# KNOWN BUG (removal scheduled, Phase 2): DSPy is configured lazily and GLOBALLY, once
# per process, via `dspy.configure`. Two tasks in one process therefore share one LM,
# and `configure_dspy()` cannot be pointed at a per-task model. The notebook's own
# symptom was that reconfiguring between tasks silently changed the model in flight.
# Phase 2 makes the LM an explicit per-task argument.
try:
    import dspy
    # `from dspy.signature import ...` does NOT work on dspy 3.x -- there is no
    # `dspy.signature` submodule. The names are re-exported at package top level.
    from dspy import InputField, OutputField, Signature
    DSPY_OK = True
except ImportError as _e:                                    # pragma: no cover
    DSPY_OK = False
    _DSPY_IMPORT_ERROR = _e


def configure_dspy(settings=None) -> bool:
    """Point DSPy at the same OpenAI-compatible endpoint as our own client.

    Delegates to `llm.configure_dspy`, which owns the provider choice. The notebook
    hardcoded `provider="openai"` here while its own docstring said `provider="groq"`
    was REQUIRED: litellm otherwise strips the `openai/` prefix off
    `openai/gpt-oss-120b` and routes to api.openai.com, which 404s. That single
    mismatch is why the planner silently fell back to `make_plan_fallback` for weeks.

    `num_retries=0` is equally load-bearing: DSPy's internal retry storm turned ONE
    rubric call into 536 seconds of silent backoff when the provider returned 429.
    Fail fast instead and let the orchestrator see the error and degrade visibly.
    """
    from . import llm as _llm
    from .config import load_settings
    s = settings or load_settings()
    if not DSPY_OK:
        print(f"  dspy unavailable ({_DSPY_IMPORT_ERROR}) -- using make_plan_fallback")
        from .context import venv_python
        v = venv_python() or "python"
        print(f"  install it with: {v} -m pip install dspy")
        return False
    if not _llm.configure_dspy(s):
        if not s.llm.api_key:
            print("  no API key -> set GROQ_API_KEY")
        return False
    print(f"  dspy configured: openai/{s.llm.dspy_model or s.llm.model} @ {s.llm.base_url}")
    return True


if DSPY_OK:
    class MakePlan(Signature):
        """Break a file-inspection task into 2-4 short ordered steps.

        Use only the `searchFiles` agent. One concern per step. Each step must be
        independently checkable, so give a concrete expected_output.
        Steps with no dependency must use an empty depends_on list.
        """
        task: str = InputField(desc="what the user wants to find out")
        findings_so_far: str = InputField(desc="verified results from earlier steps, or 'none'")
        steps_json: str = OutputField(
            desc=('JSON array, no prose. Each element: '
                  '{"id": 1, "agent": "searchFiles", "instruction": "...", '
                  '"depends_on": [], "expected_output": "..."}'))

    class RevisePlan(Signature):
        """Revise a stalled plan by changing strategy, not by rewording.

        You are given attempts that already failed. Do NOT repeat any of them:
        change the search strategy, use a different tool surface, widen or narrow
        the query, or split the step differently. Keep steps that already passed.
        Return only the REMAINING steps, renumbered from 1.
        """
        goal: str = InputField(desc="the original task")
        completed: str = InputField(desc="steps already verified, with their results")
        dead_ends: str = InputField(desc="attempts that FAILED - never repeat these")
        failure: str = InputField(desc="why the plan stalled")
        steps_json: str = OutputField(
            desc='JSON array of remaining steps only, no prose, same element shape')

    # ChainOfThought where reasoning genuinely helps the DECISION, plain Predict
    # where it would bias the judgement. See planner.py's module docstring for why
    # these are deliberately different.
else:                                            # pragma: no cover
    MakePlan = RevisePlan = None                  # type: ignore[assignment]


def _predictor(signature_cls):
    """Build the predictor for a signature.

    The notebook resolved this at import time into module globals `_make` / `_revise`.
    That was module-level mutable state AND it froze the CoT flag from the environment
    at import, so a test could not flip `settings.planner_cot`. The predictor itself is
    cheap and stateless; only the LM it uses is global, and that is tracked separately.
    """
    import dspy
    return dspy.ChainOfThought(signature_cls)


PLAN_INSTRUCTIONS = (
    "Rules: 2-4 steps. agent must be exactly 'searchFiles'. "
    "depends_on must be [] or a list of earlier step ids. "
    "One concern per step. expected_output must be concrete and checkable "
    "(e.g. 'list of file paths with line numbers'). "
    "Do not read file contents yourself - only the agents can."
)


def _extract_json(text: str) -> Any:
    """Pull the first JSON array out of prose-wrapped output.

    KNOWN BUG (removal scheduled, Phase 2): `t.find("[")` takes the FIRST `[`
    anywhere in the output, not the first array. A model that writes "Here is step [1]
    of 3:" before the JSON makes this parse the wrong bracket and raise, and the whole
    plan is lost. `llm.extract_json` has the same shape today, so this is a duplicated
    defect rather than a fixed one. Phase 2 unifies them and makes the "first balanced
    JSON value" rule explicit.
    """
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t.strip())
    start = t.find("[")
    if start == -1:
        raise ValueError(f"no JSON array in output: {text[:120]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError("unterminated JSON array")


# KNOWN BUG (removal scheduled, Phase 1): the hard cap of 7 lives here as a literal,
# not as `MAX_STEPS_PER_PLAN`, so the orchestrator's constant is decorative. It also
# contradicts `PLAN_INSTRUCTIONS` ("2-4 steps"), so a model that obeys neither gets a
# plan trimmed mid-DAG. Phase 1 makes the cap a setting and reconciles the two.
MAX_PLAN_STEPS = 7


def parse_plan_json(raw: str, goal: str) -> Plan:
    """Repair the three predictable defects, then validate.

    1. `depends_on: null`            -> []
    2. invented agent names         -> canonical name
    3. prose around the JSON        -> stripped

    All three are real: a model told "depends_on must be []" emits `null` roughly half
    the time, and a model told "agent must be exactly 'searchFiles'" emits
    `file_search_agent`. Rejecting either one throws away an otherwise-usable plan.
    """
    data = _extract_json(raw)
    cleaned = []
    for i, s in enumerate(data, 1):
        if not isinstance(s, dict):
            continue
        deps = s.get("depends_on") or []
        if isinstance(deps, (int, str)):
            deps = [deps]
        cleaned.append({
            "id": int(s.get("id", i)),
            "agent": normalise_agent(s.get("agent", "searchFiles")),
            "instruction": str(s.get("instruction", "")).strip(),
            "depends_on": [int(d) for d in deps],
            "expected_output": str(s.get("expected_output", "")).strip(),
        })
    cleaned = [s for s in cleaned if s["instruction"]]
    if not cleaned:
        raise ValueError("plan had no usable steps")
    cleaned = cleaned[:MAX_PLAN_STEPS]          # long plans go stale; cap hard
    return Plan(goal=goal, steps=cleaned)


def make_plan(task: str, *, ctx: TaskContext, findings: str = "none") -> Plan:
    """THE SEAM: one Predict call. Swap in CoT / demos / an optimiser here."""
    if not DSPY_OK:
        raise RuntimeError("dspy unavailable")
    out = _predictor(MakePlan)(task=task + "\n\n" + PLAN_INSTRUCTIONS,
                               findings_so_far=findings)
    ctx.log("planner", "planner_reasoning",
            detail=str(getattr(out, "reasoning", ""))[:120],
            status="ok", cot=ctx.settings.planner_cot, tool="planner")
    return parse_plan_json(out.steps_json, goal=task)


def replan(goal: str, completed: str, failure: str, *,
           ctx: TaskContext, max_replans: int = 2) -> Plan:
    """Revise only the remaining steps. `ctx.trace.dead_ends()` is the do-not-repeat list.

    `max_replans` is accepted and IGNORED, exactly as in the notebook. The real cap is
    `ctx.budget.max_replans`, enforced by `ctx.charge_replan()`, so a caller passing
    `max_replans=99` gets no more replans than the budget allows. KNOWN BUG (removal
    scheduled, Phase 1): an ignored parameter is a lie in the signature; Phase 1 either
    honours it or deletes it.
    """
    if not DSPY_OK:
        raise RuntimeError("dspy unavailable")
    _ = max_replans
    out = _predictor(RevisePlan)(
        goal=goal + "\n\n" + PLAN_INSTRUCTIONS,
        completed=completed,
        dead_ends="; ".join(ctx.trace.dead_ends()) or "none",
        failure=failure)
    return parse_plan_json(out.steps_json, goal=goal)


def make_plan_fallback(task: str) -> Plan:
    """Deterministic plan, used when dspy is unavailable or keeps failing.

    Not a stub: it encodes the shape a good plan takes for this agent, so the
    orchestrator and verifier stay testable without a network call. It is also the
    reason the offline path still produces a REAL dependency chain -- step 2 depends on
    step 1's result, which is what exercises `ready_steps` and `blocked_steps` in
    production rather than only in tests.
    """
    s1 = Step(id=1, agent="searchFiles", instruction=task,
              depends_on=[], expected_output="list of file paths with line numbers")
    s2 = Step(id=2, agent="searchFiles",
              instruction=f"file type breakdown relevant to: {task[:80]}",
              depends_on=[1], expected_output="counts per file extension")
    return Plan(goal=task, steps=[s1, s2])
