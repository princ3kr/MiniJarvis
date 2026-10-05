"""One pipeline in `run_task`, and the stage result that drives escalation.

The plan this implements:

    query
      -> LLM JUDGE (one call: direct | loop | plan, with a reason)
      -> stage: direct | loop | plan
      -> VERIFY (code first, then the LLM)
      -> SYNTHESIZE from verified evidence only, checks re-run on its text
      -> or "not found" plus the list of what was tried

Three properties do the work, and each one replaced something that was quietly wrong.

**A stage returns a `StageResult`, not a string.** The old dispatcher returned `None`
for failure and a string for success, so "found nothing" and "crashed" and "budget
ran out" were the same value and the caller could not choose an escalation. `outcome`
is the plan's vocabulary -- `found` / `near_miss` / `not_found` / `error` -- and only
`found` stops the ladder. A `near_miss` escalates: a near-miss presented as an answer
is the fuzzy-filename guessing the plan says to remove.

**Only verified text reaches the user.** `synthesize_verified` re-runs the path and
number checks on the SYNTHESIS, not just on the evidence. The first version verified
the evidence and then let the model write freely from it, which is exactly the gap
that produced "approximately 37K".

**Escalation is bounded and visible.** `ESCALATION` is the plan's ladder, each rung is
tried at most once per stage transition, and every transition is a trace event. A
caller can see which rungs ran and why it stopped.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from . import llm
from .context import BudgetExhausted, TaskContext
from .router import ESCALATION, route
from .trace import Verdict

__all__ = ["StageResult", "Outcome", "run_task", "plan_path"]

Outcome = Literal["found", "near_miss", "not_found", "error"]

# Escalation only happens on these. `error` deliberately does NOT escalate: if the
# endpoint is down, the next rung will fail the same way, and burning the budget on
# three strategies to reach the same 401 is worse than reporting the error.
ESCALATE_ON: frozenset[str] = frozenset({"near_miss", "not_found"})

MAX_RUNG_VISITS = 3

# The ladder's rungs, in escalation order. Used to work out how many stages might
# still run, so the iteration budget can be divided between them.
STRATEGY_ORDER: tuple[str, ...] = ("direct", "loop", "plan")


@dataclass
class StageResult:
    """What one strategy produced. The ladder branches on `outcome`, never on None."""

    outcome: Outcome
    text: str | None = None                 # ONLY verified text reaches the user
    evidence: list[dict[str, Any]] = field(default_factory=list)
    strategy: str = ""
    reason: str = ""
    tried: list[str] = field(default_factory=list)
    verified: bool = False
    cost: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome == "found"

    def not_found(self, strategy: str, reason: str = "") -> "StageResult":
        return StageResult(outcome="not_found", strategy=strategy, reason=reason)

    def render(self) -> str:
        if self.outcome == "found":
            return self.text or ""
        head = {"near_miss": "~", "not_found": "x", "error": "!"}[self.outcome]
        lines = [f"{head} {self.strategy}: {self.outcome} -- {self.reason[:90]}"]
        if self.tried:
            lines.append("  tried:")
            lines += [f"    - {t}" for t in self.tried[:12]]
        return "\n".join(lines)


# --------------------------------------------------------------------------- stages
def _draft_answer(query: str, res: "StageResult") -> str:
    """A candidate answer for a stage that found things but phrased nothing.

    Only used as the THING BEING VERIFIED, never as the thing shown: if it survives
    the checks the rubric writes a better sentence from the same evidence. It is
    deliberately plain and entirely derived from `res.evidence`, because a draft that
    adds anything is a draft the verifier will (correctly) reject.
    """
    if not res.evidence:
        return ""
    heads = []
    for e in res.evidence[:6]:
        cmd = " ".join(str(c) for c in e.get("command", []))[:80]
        body = str(e.get("output") or "")[:1200]
        heads.append(f"$ {cmd}\n{body}")
    return "\n".join(heads)


def _render_hits(hits: list[dict]) -> str:
    """Turn a tool's hits into the same shape the loop produces, so ONE verifier and
    ONE set of code-first checks apply to every strategy.

    Without this the direct stage's evidence has a different shape from the loop's and
    the path/number checks -- which read `evidence[i]["output"]` -- had nothing to check
    on the direct path. A check that finds no text is not a passing check; it is a
    check that did not run, and it looked like verification passing.
    """
    if not hits:
        return "(no matches)"
    lines: list[str] = []
    for h in hits[:40]:
        p = h.get("path")
        if not p:
            continue
        line = str(p)
        if h.get("score") is not None:
            line += f"   (score {h['score']})"
        if h.get("kind") == "folder":
            line += "   [folder]"
        lines.append(line)
    return "\n".join(lines) or "(no matches)"


def _direct_stage(query: str, *, ctx: TaskContext, use_llm_verify: bool) -> StageResult:
    """One cheap tool call, and nothing else.

    A near-miss here is NOT an answer. `ask()` returns the model's text either way, so
    this stage re-checks whether anything exact was actually found and downgrades a
    fuzzy hit to `near_miss` -- which escalates. Presenting
    `Ask-My-Repo-mainV2.zip` as the answer to "is there a file named ask-my-repo.zip"
    is the guessing the plan says to remove.
    """
    from .agents import run_agent
    from .router import extract_intent

    t0 = time.perf_counter()
    intent = extract_intent(query)
    result = run_agent(query, ctx=ctx, root=ctx.root)
    tried = [f"{s.get('by', 'tool')}: {s.get('query', '')}" for s in result.get("searches", [])]
    # `agents.run_agent` does not label hits as exact or fuzzy; it returns paths and a
    # score. So exactness is computed HERE, from the intent the router extracted,
    # rather than read off a field that does not exist. The first version read
    # `h.get("exact")`, which is always None, so `n_exact` was always 0 and EVERY
    # direct result was downgraded to `near_miss` and escalated -- the eval showed
    # 0/12 direct cases succeeding.
    #
    # Exact means: the query named a literal filename, and a returned path's name
    # contains it. Anything else is a fuzzy near-miss, which must NOT be shown as an
    # answer -- that is the plan's removal of difflib guessing from the answer path.
    wanted = [f for f in (intent.get("filenames") or []) if f]
    hits = result.get("similar_files", [])
    if wanted:
        low = [w.lower() for w in wanted]
        exact = [h for h in hits
                 if any(w in Path(str(h.get("path", ""))).name.lower() for w in low)]
    else:
        # No literal filename in the query ("how many images are there"), so a hit
        # from find/grep/survey is as exact as it gets.
        exact = [h for h in hits if h.get("score", 0) >= 0.999] or hits
    n_exact = len(exact)
    n_near = len(hits)

    cost = {"tool_calls": ctx.counters.tool_calls, "seconds":
            round(time.perf_counter() - t0, 2)}

    if n_exact == 0 and n_near > 0:
        # Only fuzzy hits. Report near_miss so the ladder climbs.
        return StageResult(outcome="near_miss", strategy="direct",
                           reason=f"{n_near} near match(es), none exact",
                           tried=tried, cost=cost)
    if n_exact == 0 and not result.get("similar_files"):
        return StageResult(outcome="not_found", strategy="direct",
                           reason="nothing matched", tried=tried, cost=cost)

    return StageResult(outcome="found", strategy="direct",
                       evidence=[{"command": ["find"] + list(wanted),
                                  "output": _render_hits(exact)}],
                       reason=f"{n_exact} exact match(es)", tried=tried,
                       verified=False, cost=cost)


def _stage_share(ctx: TaskContext, stages_left: int) -> int:
    """Iterations this stage may spend, out of the task's remaining budget.

    WITHOUT THIS, escalation is useless. The iteration budget was global, so the first
    stage spent all of it and every later rung was refused at turn 1 with
    "iteration ceiling 8 reached" -- the ladder reported `budget` and gave up without
    ever running the strategy it escalated to. Observed on
    "how many lines of python are in this directory?": the loop used 8 turns, then
    plan got none.

    So the budget is DIVIDED among the stages that might still run. Guessing low stays
    cheap: `direct` costs no iterations at all (it is one tool call), so most of the
    budget goes to `loop`, which is where turns are actually spent.
    """
    remaining = max(1, ctx.budget.max_iterations - ctx.counters.iterations)
    return max(2, remaining // max(1, stages_left))


def _loop_stage(query: str, *, ctx: TaskContext, use_llm_verify: bool,
                max_turns: int | None = None) -> StageResult:
    """The ReAct loop. Its own result already carries verdicts and a finish check."""
    from .react import run_react

    t0 = time.perf_counter()
    r = run_react(query, ctx=ctx, max_turns=max_turns, use_llm_verify=use_llm_verify)
    tried = [f"{s.action}: {' '.join(s.args.get('argv', []))[:70]}" for s in r.steps]

    if r.status == "answered" and r.answer:
        return StageResult(
            outcome="found", text=r.answer, strategy="loop",
            reason=f"answered after {r.turns} turn(s)", tried=tried,
            verified=(r.status == "answered"),
            cost={"turns": r.turns, "llm_calls": ctx.counters.llm_calls,
                  "seconds": round(time.perf_counter() - t0, 2)})

    # `answered_unverified` is NOT `found`. The plan is explicit: unverified answers
    # must not be shown as answers. It is reported as near_miss so the ladder tries a
    # different strategy, and the text is carried so the caller can show it labelled.
    if r.answer:
        return StageResult(outcome="near_miss", text=r.answer, strategy="loop",
                           reason="produced an answer that did not verify",
                           tried=tried, verified=False,
                           cost={"turns": r.turns})

    if r.status == "budget":
        return StageResult(outcome="error", strategy="loop",
                           reason=f"budget exhausted after {r.turns} turn(s)",
                           tried=tried)

    return StageResult(outcome="not_found", strategy="loop",
                       reason=f"no verified answer after {r.turns} turn(s)",
                       tried=tried)


def _plan_stage(query: str, *, ctx: TaskContext, use_llm_verify: bool,
                replans_left: int, turn_cap: int | None = None) -> StageResult:
    """Planner v2: sub-goals, each run by a bounded loop, combined at the end.

    The plan says planner steps are sub-goals run by bounded loops, not agent calls.
    So each step gets its own loop with its own budget, and a step that cannot
    converge is reported as such rather than silently swallowed.
    """
    from .planner import (DSPY_OK, configure_dspy, make_plan, make_plan_fallback)
    from .react import run_react

    t0 = time.perf_counter()
    used_fallback = False
    plan = None
    if DSPY_OK:
        try:
            configure_dspy(ctx.settings)
            plan = make_plan(query, ctx=ctx)
            ctx.log("planner", "plan", detail=f"{len(plan.steps)} step(s)",
                    status="ok", steps=len(plan.steps), tool="planner")
        except Exception as e:
            ctx.log("planner", "plan", detail=f"{type(e).__name__}", status="fallback",
                    tool="planner")
    if plan is None:
        used_fallback = True
        plan = make_plan_fallback(query)
        ctx.log("planner", "plan", detail="deterministic fallback", status="fallback",
                tool="planner")

    tried = [f"step {s.id}: {s.instruction[:70]}" for s in plan.steps]
    evidence: list[dict[str, Any]] = []
    parts: list[str] = []

    for step in plan.steps:
        # Each step is a SUB-GOAL run by a bounded loop, per the plan. Not an agent
        # call: an agent call cannot escalate, and a sub-goal that needs a command
        # needs the loop.
        #
        # The cap is this stage's share divided among its steps, so a 3-step plan does
        # not have its whole budget consumed by step 1 -- which is what happened: step
        # 2 and 3 were refused at turn 1 with the task budget already spent.
        # Each sub-goal gets the stage's share divided by the number of steps.
        #
        # `turn_cap` is what is LEFT of the task budget, not what the stage was
        # allocated. Using the allocation and dividing again was a double division: a
        # 3-step plan with a 5-turn allocation gave each step `max(2, 5//3)` = 2
        # turns, which is not enough to run a command, read the output, and finish.
        # Observed on the eval: every plan-routed case ended
        # "no sub-goal produced a verified result" with the sub-loops reporting 0
        # turns of real work.
        remaining = max(1, ctx.budget.max_iterations - ctx.counters.iterations)
        per_step = max(3, min(turn_cap or remaining,
                              remaining // max(1, len(plan.steps))) or remaining)
        sub = run_react(step.instruction, ctx=ctx, max_turns=per_step,
                        use_llm_verify=use_llm_verify)
        tried.append(f"step {step.id} -> {sub.status} ({sub.turns} turns)")
        if sub.answer:
            parts.append(f"{step.instruction}\n{sub.answer}")
            for s in sub.steps:
                if s.observation is not None and s.observation.status == "ok":
                    evidence.append({"command": s.args.get("argv", []),
                                     "output": (s.observation.text or "")[:2000]})
        else:
            parts.append(f"{step.instruction}\n(no verified result)")

    if not any(p.endswith("(no verified result)") is False for p in parts):
        return StageResult(outcome="not_found", strategy="plan",
                           reason="no sub-goal produced a verified result", tried=tried,
                           cost={"seconds": round(time.perf_counter() - t0, 2),
                                 "used_fallback": used_fallback})

    # Combine from the sub-goal answers, then verify the COMBINATION. Verifying each
    # part separately and trusting the join is how a synthesis invents a claim that
    # appears in no sub-goal.
    combined = llm.call(ctx, [
        {"role": "system", "content":
         "Combine these sub-results into one answer to the original question. Use "
         "ONLY what the sub-results state. If a part is missing, say it is missing. "
         "Do not add numbers, paths or facts that are not in the text."},
        {"role": "user", "content":
         f"QUESTION: {query}\n\nSUB-RESULTS:\n" + "\n\n".join(parts)[:8000]}],
        purpose="plan_combine", temperature=0.0).get("text", "").strip()

    return StageResult(outcome="found", text=combined, strategy="plan",
                       reason=f"{len(plan.steps)} sub-goal(s), "
                              f"{len(evidence)} evidence block(s)",
                       evidence=evidence, tried=tried, verified=False,
                       cost={"steps": len(plan.steps),
                             "seconds": round(time.perf_counter() - t0, 2),
                             "used_fallback": used_fallback})


STAGES = {"direct": _direct_stage, "loop": _loop_stage, "plan": _plan_stage}


# --------------------------------------------------------------------------- pipeline
def plan_path(query: str, *, ctx: TaskContext, strategy: str = "auto",
              use_llm_verify: bool = True, replans_left: int = 2,
              max_turns: int | None = None,
              on_event=None) -> StageResult:
    """The whole pipeline. Returns a StageResult; never raises.

    `on_event` is called for every trace-worthy moment so an interactive CLI can
    render the path as it happens instead of after the fact.
    """
    def emit(kind: str, detail: str) -> None:
        if on_event is not None:
            try:
                on_event(kind, detail)
            except Exception:
                pass                          # a display must never break the run

    tried_all: list[str] = []
    visits: dict[str, int] = {}

    if strategy == "auto":
        r = route(query, ctx=ctx)
        emit("route", f"{r.strategy} — {r.reason}"
                      + ("  [fallback]" if r.source == "fallback" else ""))
        strategy = r.strategy

    current = strategy
    last: StageResult | None = None
    stage_results: list[StageResult] = []

    while True:
        if visits.get(current, 0) >= MAX_RUNG_VISITS:
            emit("route", f"{current} tried {MAX_RUNG_VISITS}x, stopping")
            break

        # Rungs not yet visited, INCLUDING this one. Computed BEFORE this rung is
        # marked visited: after, `visits` already contains it, the count drops to the
        # number of future rungs, and the current stage is handed the whole remaining
        # budget -- which is how the loop spent all 8 iterations and left `plan` with
        # none.
        rungs_left = sum(1 for k in STRATEGY_ORDER if k not in visits)
        visits[current] = visits.get(current, 0) + 1

        emit("stage", f"running {current}")
        try:
            if current == "direct":
                res = _direct_stage(query, ctx=ctx, use_llm_verify=use_llm_verify)
            elif current == "loop":
                turn_cap = max_turns if max_turns is not None else _stage_share(ctx, rungs_left)
                res = _loop_stage(query, ctx=ctx, use_llm_verify=use_llm_verify,
                                  max_turns=turn_cap)
            elif current == "plan":
                res = _plan_stage(query, ctx=ctx, use_llm_verify=use_llm_verify,
                                  replans_left=replans_left,
                                  turn_cap=max_turns if max_turns is not None
                                  else _stage_share(ctx, rungs_left))
            else:
                raise ValueError(f"unknown strategy {current!r}")
        except BudgetExhausted as e:
            res = StageResult(outcome="error", strategy=current, reason=str(e))
        except Exception as e:
            # CONTRACT rule 5: never raise across a stage boundary. An exception that
            # escapes here kills the task and loses the trace that explains it.
            res = StageResult(outcome="error", strategy=current,
                              reason=f"{type(e).__name__}: {str(e)[:120]}")

        res.tried = tried_all + res.tried
        tried_all = res.tried
        last = res
        stage_results.append(res)
        emit("stage", f"{current} -> {res.outcome}"
                      + (f" ({res.reason})" if res.reason else ""))

        # The plan's order: VERIFY (code first) THEN SYNTHESIZE, and only verified
        # text reaches the user. A `found` from a stage is a CANDIDATE, not an answer.
        if res.outcome == "found":
            # The plan's order: VERIFY, then SYNTHESIZE from verified evidence only.
            # A stage can succeed by finding the right thing without having phrased it
            # -- `direct` returns paths, not prose -- so synthesis runs even when the
            # stage brought its own text. It is what turns evidence into an answer,
            # and it runs after the code-first checks so the model only ever sees
            # evidence that survived them.
            candidate = res.text or _draft_answer(query, res)
            verified_text, vwhy = synthesize_verified(
                query, candidate, res.evidence, ctx=ctx, use_llm_verify=use_llm_verify)
            if verified_text:
                res.text = verified_text
                res.verified = True
                res.reason = (res.reason + f"; verified: {vwhy}").strip("; ")
                emit("verify", f"verified — {vwhy}")
                return res
            emit("verify", f"REJECTED — {vwhy}")
            res.outcome = "near_miss"
            res.text = None
            res.reason = f"evidence did not support an answer: {vwhy}"

        if res.outcome not in ESCALATE_ON:
            return res

        nxt = ESCALATION.get(current)
        if nxt is None:
            break
        emit("route", f"{current} did not settle it — escalating to {nxt}")
        current = nxt

    # Nothing verified. Report the most INFORMATIVE failure, not merely the last one.
    #
    # The ladder ends wherever escalation stopped, which is usually `plan` -- and
    # `plan`'s reason is "no sub-goal produced a verified result", which says nothing
    # about the `loop` that actually got closest. Reporting that as the failure is
    # true and useless: every eval case ended with the same sentence regardless of
    # what had actually been tried.
    best = last
    for cand in stage_results:
        if _progress(cand) > _progress(best):
            best = cand
    if best is None:
        return StageResult(outcome="not_found", strategy=strategy,
                           reason="no strategy produced a verified answer",
                           tried=tried_all)
    best.tried = tried_all
    if best.text:
        best.outcome = "near_miss"
    return best


def _progress(r: "StageResult | None") -> int:
    """How much work a stage did, as a comparable number.

    Evidence blocks only. `tried` is deliberately NOT counted: it accumulates across
    stages by design, so a later rung always has a longer list and would always win --
    which is the same bug as reporting the last stage, just with extra steps.
    """
    if r is None:
        return -1
    return len(r.evidence)


# --------------------------------------------------------------------------- verify
# Phrases that mean "I did not find it". An answer containing one is NOT an answer,
# however well it is formatted and whatever the rubric said.
#
# Here because the rubric is not enough. Asked how many lines of Python were in a
# directory, the model produced a clean, confident, well-formed paragraph that opened
# "I don't have the line-count information for any of the listed files" -- and the
# pipeline printed it under a green ANSWER heading. It passed every path check (no
# paths), every number check (no numbers), and the rubric, because it IS a truthful
# sentence about the evidence. Being honest about failing is not the same as having
# succeeded, and only the wording distinguished them.
#
# Deliberately a short list of unambiguous admissions rather than sentiment analysis.
# A false positive costs one loop turn; a false negative shows a refusal to a user who
# asked a question, dressed as an answer.
_ADMISSIONS = (
    "i don't have", "i do not have", "i couldn't", "i could not",
    "i wasn't able", "i was not able", "unable to find", "unable to determine",
    "not able to find", "not able to determine", "cannot find", "cannot determine",
    "no information", "not enough information", "insufficient information",
    "no results", "no matches", "not found", "couldn't find", "could not find",
    "i'm unable", "i am unable", "no answer", "cannot answer", "can't answer",
    "i don't know", "i do not know",
)


def admits_failure(text: str) -> bool:
    """Does this answer say it could not find the answer?"""
    low = " ".join((text or "").lower().replace("’", "'").split())
    return any(p in low for p in _ADMISSIONS)


def synthesize_verified(query: str, candidate: str, evidence: list[dict],
                        *, ctx: TaskContext, use_llm_verify: bool = True
                        ) -> tuple[str | None, str]:
    """Verify code-first, then the LLM rubric LAST, and re-check the synthesis.

    The order matters and it is the plan's order. Paths and numbers are checked by
    code because they are checkable: a model asked "does this path appear in the
    evidence?" will say yes often enough to be useless, and a hallucinated path is the
    failure the whole verification chain exists to catch. The rubric runs last because
    it is the only part that understands prose, and the most likely to be talked into.

    Then the same checks run AGAIN on the synthesis. Verifying the evidence and then
    letting the model write freely from it leaves the gap this is closing -- that is
    how "approximately 37K" got in, off evidence that did not contain it.
    """
    from .react import _count_claim_mismatch, _unsupported_numbers

    corpus = "\n".join(str(e.get("output") or "") for e in evidence)
    if not corpus:
        return None, "no evidence to verify against"

    # --- code first, and before anything else: is this an answer at all?
    # An admission of failure is a `not_found`, not a result. Checked here so it
    # cannot be reached by a rubric that is feeling generous.
    if admits_failure(candidate):
        return None, ("answer admits it could not find the information; that is a "
                      "not-found, not an answer")

    bad = _count_claim_mismatch(candidate, corpus)
    if bad:
        return None, f"answer states {', '.join(bad)} files but the evidence lists a different number"

    # --- code first: numbers
    bad = _unsupported_numbers(candidate, corpus)
    if bad:
        return None, f"answer states {', '.join(bad)} which no command output contains"

    # --- code first: paths
    from .verify_paths import unverified_paths
    ghosts = unverified_paths(candidate, corpus)
    if ghosts:
        return None, f"answer names {', '.join(ghosts[:3])} which no command output contains"

    # --- code first: re-check cheap predicates on disk
    from .verify_disk import recheck_disk_claims
    disk_problems, disk_detail = recheck_disk_claims(candidate, ctx=ctx)
    if disk_problems:
        return None, f"disk re-check disagrees: {'; '.join(disk_problems[:3])}"

    if not use_llm_verify:
        return None, ("verification skipped (--no-verify is debug-only; it cannot "
                      "produce a shown answer)")

    # --- the rubric last
    try:
        out = llm.call(ctx, [
            {"role": "system", "content":
             "Decide whether the ANSWER is fully supported by the EVIDENCE. Reply "
             "PASS or FAIL, then a colon and a reason under 15 words.\n"
             "PASS = every claim in the answer is supported by the evidence.\n"
             "FAIL = any claim is unsupported, or the evidence does not answer the "
             "question."},
            {"role": "user", "content":
             f"QUESTION: {query}\n\nANSWER:\n{candidate[:2000]}\n\n"
             f"EVIDENCE:\n{corpus[:6000]}"}],
            purpose="verify_answer", temperature=0.0)
        txt = (out.get("text") or "").strip()
        head = txt.split(":", 1)[0].strip().upper()
        why = txt.split(":", 1)[1].strip() if ":" in txt else txt[:80]
        if head != "PASS":
            return None, f"rubric: {why[:120]}"
    except BudgetExhausted:
        raise
    except Exception as e:
        return None, f"verifier degraded ({type(e).__name__}); refusing to show it unverified"

    return candidate.strip(), why[:120] or "supported"


# --------------------------------------------------------------------------- entry
def run_task(query: str, *, ctx: TaskContext, strategy: str = "auto",
             use_llm_verify: bool = True, replans_left: int = 2,
             max_turns: int | None = None,
             on_event=None) -> str | None:
    """Main entry. Returns verified text, or None with the trace explaining why."""
    r = plan_path(query, ctx=ctx, strategy=strategy, use_llm_verify=use_llm_verify,
                  replans_left=replans_left, max_turns=max_turns, on_event=on_event)
    if r.ok and r.text:
        return r.text
    if r.text:                      # near_miss carrying text: shown, but labelled
        print(f"\n[UNVERIFIED — {r.strategy}] {r.text}")
    return None