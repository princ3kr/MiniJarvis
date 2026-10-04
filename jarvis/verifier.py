"""The verifier: decide whether an agent's result actually satisfies a step.

Two layers, in this order:

  1. `deterministic_verify` -- structural checks that need no model. Cited paths must
     exist and be inside the jail. A step whose `expected_output` asks for paths must
     have some. This is cheap and it is the layer that catches a hallucinated path.
  2. `rubric_verify` -- an LLM judgement, only for results that cleared layer 1.

Three behaviours here were bugs once and are now load-bearing. They are preserved with
their original comments because each one fixes a DIFFERENT failure, and removing any of
them re-opens a failure that was observed:

  * `unsure` is NOT failure. It means unproven. The orchestrator reports it as
    `inconclusive` and the do-not-repeat list EXCLUDES it.
  * `use_llm=False` accepts a structural pass, labelled "rubric skipped". Otherwise NO
    step could ever complete offline and a plan could not finish without a network.
  * The verifier NEVER raises. Any internal fault degrades to `unsure`.

The verifier must not become clever. A verifier that raises kills the orchestrator; one
that fails an unproven result teaches the agent to avoid correct work; one that passes
everything is decoration.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .access import AccessDenied, AccessPolicy
from .context import TaskContext
from .planner import AgentResult, Step
from .planner import DSPY_OK  # noqa: F401  (re-exported for the tests' stubbing)
from .trace import FailureType, Verdict  # noqa: F401  (FailureType unused here, kept for one import)

__all__ = ["deterministic_verify", "rubric_verify", "verify_step", "VerifyResult"]


def deterministic_verify(step: Step, result: AgentResult | None, *,
                         ctx: TaskContext) -> tuple[Verdict, str]:
    """Structural checks. No LLM. Returns (Verdict, reason).

    The jail check is the point of this function. An agent that cites
    `/etc/passwd` has either escaped the jail or invented the path, and both are a hard
    fail -- there is no reading under which a verified step cites a file the user cannot
    see.
    """
    if result is None:
        return Verdict.fail, "agent returned nothing"
    if result.status == "not_found":
        return Verdict.fail, "no files matched"
    if result.status == "near_miss":
        return Verdict.unsure, "only near matches, no exact hit"
    if not result.evidence:
        return Verdict.unsure, "ok but zero evidence items"
    if result.confidence <= 0.0:
        return Verdict.unsure, "zero confidence"

    policy = AccessPolicy.from_settings(ctx.settings, cwd=ctx.root or Path.cwd())

    # every cited path must exist and be inside the jail
    bad = []
    for e in result.evidence:
        p = e.get("path") if isinstance(e, dict) else None
        if not p:
            continue
        try:
            rp = policy.resolve(p)
            if not rp.exists():
                bad.append(f"{rp.name} missing")
        except PermissionError:
            bad.append(f"{p} outside jail")
    if bad:
        return Verdict.fail, f"unverifiable paths: {'; '.join(bad[:3])}"

    # expected_output mentions file paths -> require at least one that exists
    if "path" in step.expected_output.lower() and not any(
            isinstance(e, dict) and e.get("path") for e in result.evidence):
        return Verdict.fail, "expected file paths, got none"

    return Verdict.unsure, "structural checks passed; needs rubric"


if DSPY_OK:
    from dspy import InputField, OutputField, Signature

    class VerifyResult(Signature):
        """Judge whether a result satisfies a step's requirement.

        The `citable locations found:` lines are authoritative -- they are paths and
        line numbers that really exist. Excerpts are partial windows and may show
        prose, imports, or surrounding text; do NOT fail a step merely because an
        excerpt does not show the exact construct. Judge whether the LOCATIONS could
        satisfy the requirement. Fail only when the locations clearly cannot.
        """
        requirement: str = InputField(desc="what the step expected")
        instruction: str = InputField(desc="what the agent was asked to do")
        result: str = InputField(desc="agent summary plus citable locations and excerpts")
        verdict: str = OutputField(desc="exactly one of: pass, fail, unsure")
        reason: str = OutputField(desc="one short sentence justifying the verdict")

else:                                            # pragma: no cover
    VerifyResult = None                          # type: ignore[assignment]


def _format_evidence(result: AgentResult, limit: int = 6) -> str:
    """Render evidence as citable path:line lines FIRST, excerpts after.

    Feeding raw JSON to the rubric was a real failure: a correct hit at
    `glinerTest.ipynb:15` was buried inside a markdown excerpt, the rubric saw prose
    instead of code, and it returned a confident `fail` on evidence that DID answer the
    step. The citations must lead, and excerpts must be labelled as partial.
    """
    cites, excerpts = [], []
    for e in result.evidence[:limit]:
        if not isinstance(e, dict):
            continue
        p = e.get("path", "?")
        line = e.get("line")
        cites.append(f"  {p}:{line}" if line else f"  {p}")
        ex = (e.get("excerpt") or "").strip().replace("\n", " ")[:220]
        if ex:
            excerpts.append(f"  [{p}] {ex}")
    out = ["citable locations found:"] + (cites or ["  (none)"])
    if excerpts:
        out += ["excerpts (PARTIAL - the location lines above are authoritative):"] \
               + excerpts
    return "\n".join(out)


def _verify_predictor(*, cot: bool):
    """CoT OFF by default for the judge: reasoning before a verdict gives the model
    room to construct a justification for whatever it lands on. Flip
    `settings.verifier_cot` if you want to see the difference."""
    import dspy
    return dspy.ChainOfThought(VerifyResult) if cot else dspy.Predict(VerifyResult)


def rubric_verify(step: Step, result: AgentResult, *, ctx: TaskContext,
                  use_llm: bool = True) -> tuple[Verdict, str]:
    """LLM judgement, only for results that passed the structural checks."""
    if not use_llm or not DSPY_OK:
        return Verdict.unsure, "no rubric available; deferring to orchestrator"

    try:
        out = _verify_predictor(cot=ctx.settings.verifier_cot)(
            requirement=step.expected_output, instruction=step.instruction,
            result=f"agent summary: {result.summary}\n"
                   f"{_format_evidence(result)}")
    except Exception as e:
        # The verifier must NEVER crash the orchestrator. A rate limit or transport
        # fault downgrades to `unsure`, which the orchestrator reports honestly
        # rather than treating as pass or fail.
        ctx.log("tool", "verify", detail=f"{type(e).__name__}: {str(e)[:70]}",
                status="error", tool="verifier")
        return Verdict.unsure, f"rubric unavailable ({type(e).__name__})"
    v = str(out.verdict).strip().lower()
    verdict = {"pass": Verdict.pass_, "fail": Verdict.fail,
               "unsure": Verdict.unsure}.get(v, Verdict.unsure)
    reasoning = str(getattr(out, "reasoning", "") or "")
    if reasoning:
        # keeping the rubric's reasoning is the cheapest way to audit WHY a step
        # was accepted -- these are the decisions that are hardest to review later
        ctx.log("verifier", "rubric_reasoning", detail=reasoning[:120], status="ok",
                cot=ctx.settings.verifier_cot, tool="verifier")
    return verdict, str(out.reason).strip()[:200]


def verify_step(step: Step, result: AgentResult | None, *, ctx: TaskContext,
                use_llm: bool = True) -> tuple[Verdict, str]:
    """Structural checks, then the rubric only if they pass. NEVER raises.

    A verifier that crashes the orchestrator is worse than useless, so any internal
    fault degrades to `unsure`, which the orchestrator reports honestly rather than
    silently counting as pass or fail.

    If `use_llm=False` there is no rubric, and returning `unsure` would mean NO step
    could ever be marked done -- the plan could not complete offline at all. So a
    result clearing every structural check is accepted as `pass`, explicitly labelled
    `rubric skipped` so the weaker guarantee is visible in the log, not implied.

    KNOWN BUG (removal scheduled, Phase 2): the hand-off to the rubric is keyed on the
    EXACT reason string `"structural checks passed; needs rubric"`. Edit that string and
    the rubric silently stops being called, with no error anywhere -- a green run that
    verifies nothing. Phase 2 returns a structured reason code instead of prose.
    """
    try:
        v, why = deterministic_verify(step, result, ctx=ctx)
        if v is Verdict.fail:
            return v, why
        if v is Verdict.unsure and why == "structural checks passed; needs rubric":
            if not use_llm:
                return Verdict.pass_, "structural checks passed (rubric skipped)"
            return rubric_verify(step, result, ctx=ctx, use_llm=True)
        return v, why
    except Exception as e:
        ctx.log("tool", "verify", detail=f"{type(e).__name__}: {str(e)[:70]}",
                status="error", tool="verifier")
        return Verdict.unsure, f"verifier error ({type(e).__name__})"
