"""The orchestrator and the entry point: run steps, judge them, replan, answer.

What is here
------------
    ready_steps / blocked_steps  -- the DAG scheduler
    execute_plan                 -- run, verify, repeat-reject, replan
    run_task                     -- strategy dispatch
    ask / synthesize             -- the direct (no-planner) path

State lives on `TaskContext`. The notebook kept `PATH_LOG` in a module global, which
meant a second task in the same kernel inherited the first task's do-not-repeat list and
silently rejected steps it had never tried. Every attempt here goes through
`ctx.attempt(...)`, and every read of the dead-end list goes through
`ctx.trace.dead_ends()`.

Four statuses, and the distinction between the last two is the most important thing in
this module:

    done          the verifier returned `pass`
    inconclusive  the verifier returned `unsure` -- NOT proven wrong
    failed        the verifier returned `fail`, or the agent raised
    skipped       a dependency failed, so this step can never run

`inconclusive` exists because conflating it with `failed` is a lie. A rate-limited
rubric is not a broken step. It is reported separately, excluded from the do-not-repeat
list, and never presented to the answer model as a finding.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import llm
from .agents import AGENT_RUNNERS, run_agent
from .context import BudgetExhausted, TaskContext
from .planner import (AgentResult, DSPY_OK, IMPLEMENTED_AGENTS, Plan, Step, StepStatus,
                      configure_dspy, make_plan, make_plan_fallback, replan)
from .prompts import NOT_FOUND_REPLY, build_llm_messages, build_prompt_bundle
from .strategy import route
from .trace import FailureType, Verdict
from .verifier import verify_step

# KNOWN BUG (removal scheduled, Phase 1): the notebook's cap of 7 lives as a literal in
# `planner.parse_plan_json` and never consults this constant, so this module advertises a
# limit it does not enforce. Phase 1 makes one setting, read from both places.
MAX_REPLANS = 2
MAX_STEPS_PER_PLAN = 7


# --------------------------------------------------------------------------- scheduling
def ready_steps(plan: Plan) -> list[Step]:
    """Steps whose dependencies are all done."""
    done = {s.id for s in plan.steps if s.status is StepStatus.done}
    return [s for s in plan.steps
            if s.status is StepStatus.pending and all(d in done for d in s.depends_on)]


def blocked_steps(plan: Plan) -> list[Step]:
    """Pending steps that can never run because a dependency failed."""
    failed = {s.id for s in plan.steps if s.status is StepStatus.failed}
    return [s for s in plan.steps
            if s.status is StepStatus.pending
            and any(d in failed for d in s.depends_on)]


# --------------------------------------------------------------------------- attempts
def _log_attempt(plan_version: int, step: Step, result: AgentResult | None,
                 verdict: Verdict, reason: str, *, ctx: TaskContext,
                 duration_ms: float | None = None):
    """Record one judged attempt on the task's trace.

    Replaces the notebook's `PATH_LOG.append(...)` plus its separate `log()` call. Two
    events are emitted: `ctx.attempt(...)` records the judged attempt (and carries the
    verdict, failure type and evidence count), and one `plan_step` event carries the
    agent and duration that the attempt record itself does not model.
    """
    a = ctx.attempt(plan_version=plan_version, step_id=step.id, agent=step.agent,
                    instruction=step.instruction,
                    status=(result.status if result else "none"),
                    result_summary=(result.summary if result else ""),
                    verdict=verdict, failure_reason=reason,
                    failure_type=_classify(result, verdict, reason),
                    evidence_checked=(f"{len(result.evidence)} item(s)" if result else ""),
                    duration_ms=duration_ms)
    ctx.log("plan", "plan_step",
            detail=f"v{plan_version} step {step.id} [{step.agent}] "
                   f"{verdict.value}: {reason[:50]}",
            status=verdict.value, step_id=step.id, agent=step.agent,
            attempt_id=a.attempt_id, duration_ms=duration_ms, tool="orchestrator")
    return a


def _classify(result: AgentResult | None, verdict: Verdict, reason: str) -> FailureType:
    """Why did this attempt not pass? `unsure` is its own answer.

    `unsure` means unproven, not wrong. Classifying it as FailureType.wrong
    made the path log claim the agent returned something off-target when in
    fact the verifier simply could not reach a verdict.
    """
    if verdict is Verdict.pass_:
        return FailureType.none
    if verdict is Verdict.unsure:
        return FailureType.inconclusive
    if "outside jail" in reason or "missing" in reason:
        return FailureType.wrong
    if "timeout" in reason.lower():
        return FailureType.timeout
    if result is None:
        return FailureType.error
    if result.status == "not_found":
        return FailureType.empty
    if not result.evidence:
        return FailureType.empty
    return FailureType.wrong


def _verified_digest(plan: Plan) -> str:
    """Compact record of what has PASSED, so a replan can reuse it.

    Note the `s.status is StepStatus.done` filter: an inconclusive step is NOT in here.
    Handing a replanner an unverified result as if it were verified is how an unproven
    claim gets repeated with more confidence.
    """
    lines = []
    for s in plan.steps:
        if s.status is StepStatus.done and s.result:
            lines.append(f"step {s.id}: {s.result.get('summary', '')[:120]}")
    return "; ".join(lines) or "none"


# --------------------------------------------------------------------------- execution
def execute_plan(plan: Plan, *, ctx: TaskContext, plan_version: int = 1,
                 use_llm_verify: bool = True,
                 replans_left: int = MAX_REPLANS) -> dict[str, Any]:
    """Run the plan to completion, replanning on failure. Returns a run record."""
    seen_sigs = set(ctx.trace.signatures())

    while True:
        for s in blocked_steps(plan):
            s.status = StepStatus.skipped
            _log_attempt(plan_version, s, None, Verdict.fail,
                         "dependency failed", ctx=ctx, duration_ms=None)

        ready = ready_steps(plan)
        if not ready:
            break

        for step in ready:
            sig = (step.agent, re.sub(r"\s+", " ", step.instruction.strip().lower()))
            if sig in seen_sigs:
                step.status = StepStatus.failed
                _log_attempt(plan_version, step, None, Verdict.fail,
                             "matches a logged dead end", ctx=ctx, duration_ms=None)
                continue

            if step.agent not in IMPLEMENTED_AGENTS:
                step.status = StepStatus.failed
                _log_attempt(plan_version, step, None, Verdict.fail,
                             f"agent {step.agent} not implemented", ctx=ctx,
                             duration_ms=None)
                continue

            step.status = StepStatus.running
            # by_id() can legitimately return None if a replan ever leaves a
            # dangling dep, so resolve defensively rather than crashing the run
            deps = {d.id: (d.result or {})
                    for d in (plan.by_id(x) for x in step.depends_on)
                    if d is not None}
            t0 = time.perf_counter()
            try:
                result = AGENT_RUNNERS[step.agent](step.instruction, deps, ctx=ctx)
                err = None
            except Exception as e:
                result, err = None, f"{type(e).__name__}: {e}"
            dt = round((time.perf_counter() - t0) * 1000, 1)

            if err:
                step.status = StepStatus.failed
                _log_attempt(plan_version, step, None, Verdict.fail, err, ctx=ctx,
                             duration_ms=dt)
                seen_sigs.add(sig)
                continue

            verdict, reason = verify_step(step, result, ctx=ctx, use_llm=use_llm_verify)
            step.result = asdict(result)
            if verdict is Verdict.pass_:
                step.status = StepStatus.done
                _log_attempt(plan_version, step, result, verdict, reason, ctx=ctx,
                             duration_ms=dt)
            elif verdict is Verdict.unsure:
                # `unsure` is NOT a failure. Treating it as one made a rate-limited
                # rubric look like a broken step, which is a different claim and a
                # wrong one. It gets its own status and is reported separately.
                step.status = StepStatus.inconclusive
                _log_attempt(plan_version, step, result, verdict, reason, ctx=ctx,
                             duration_ms=dt)
            else:
                step.status = StepStatus.failed
                _log_attempt(plan_version, step, result, verdict, reason, ctx=ctx,
                             duration_ms=dt)
                seen_sigs.add(sig)

        if all(s.status is not StepStatus.pending for s in plan.steps):
            break
        if not ready_steps(plan):
            break                     # stalled: only blocked steps remain
        if not any(s.status is StepStatus.failed for s in plan.steps):
            continue                  # nothing failed -- just keep executing
        if replans_left <= 0:
            break

        failed = [s for s in plan.steps if s.status is StepStatus.failed]
        pinned = [s for s in plan.steps if s.status is StepStatus.done]
        failure = "; ".join(f"step {s.id}: {s.result.get('summary','')[:60] if s.result else ''}"
                            for s in failed) or "unknown"
        try:
            ctx.charge_replan()
            new_plan = replan(plan.goal, _verified_digest(plan), failure,
                              ctx=ctx, max_replans=replans_left)
        except Exception as e:
            # KNOWN BUG (removal scheduled, Phase 1): a revised plan that depends on an
            # ALREADY-PINNED step is rejected here by `Plan`'s DAG validator, because the
            # revised plan is validated standalone and does not contain the pinned steps.
            # `RevisePlan` explicitly asks for "only the REMAINING steps", so any
            # dependency-chained plan cannot be replanned at all: the call fails, the run
            # stops at the current version, and the reason is buried in an attempt record.
            # This is why the notebook's own "verified results stay pinned" test passed
            # vacuously -- its fake replan had a dangling dep and never ran.
            _log_attempt(plan_version, Step(id=-1, agent="planner",
                                           instruction="replan", expected_output=""),
                         None, Verdict.fail,
                         f"replan failed: {type(e).__name__}: {str(e)[:70]}", ctx=ctx,
                         duration_ms=None)
            break

        # Keep verified steps pinned at their ORIGINAL ids, and remap the new
        # remainder through an explicit old->new map. Shifting every new id by a
        # flat offset is wrong: the planner's new deps refer to the ORIGINAL ids,
        # and pinned steps keep theirs, so a blind offset left deps pointing at ids
        # that no longer exist.
        #
        # KNOWN BUG (removal scheduled, Phase 1): the map is built pinned-first, so a new
        # step whose id COLLIDES with a pinned id OVERWRITES the pinned entry. The
        # planner is told "return only the REMAINING steps, renumbered from 1", so
        # collision is the expected case, not the exotic one: with pinned id 1, the map
        # becomes {1: 2, ...} and any new step that says `depends_on: [1]` is wired to the
        # FIRST NEW STEP instead of the pinned one. It runs in the wrong order and reports
        # a plausible-looking result. Phase 1 builds the map in two passes (pinned
        # entries win) and asserts the two id spaces are disjoint first.
        offset = max((s.id for s in pinned), default=0)
        id_map = {s.id: s.id for s in pinned}
        for n, s in enumerate(new_plan.steps, 1):
            id_map[s.id] = offset + n
        for n, s in enumerate(new_plan.steps, 1):
            s.id = offset + n
            s.depends_on = sorted({id_map[d] for d in s.depends_on if d in id_map})
        plan.steps = pinned + new_plan.steps
        # a dep that could not be remapped would silently never run
        live_ids = {s.id for s in plan.steps}
        for s in plan.steps:
            if any(d not in live_ids for d in s.depends_on):
                s.depends_on = [d for d in s.depends_on if d in live_ids]
        # KNOWN BUG (removal scheduled, Phase 1): `plan.steps = pinned + new` DROPS the
        # failed and inconclusive steps. Their ids can then be reissued to a retry step,
        # so `run["failed"]` and the trace can name the same id for two different
        # instructions. The attempts survive in `ctx.trace`; the plan does not.
        plan_version += 1
        replans_left -= 1
        _log_attempt(plan_version,
                     Step(id=-1, agent="planner", instruction="replan",
                          expected_output="revised remaining steps"),
                     AgentResult(status="ok",
                                 summary=f"plan v{plan_version} with "
                                         f"{len(new_plan.steps)} new step(s)",
                                 confidence=1.0),
                     Verdict.pass_, "replanned", ctx=ctx, duration_ms=None)

    return {"plan": plan, "version": plan_version,
            "done": [s.id for s in plan.steps if s.status is StepStatus.done],
            "failed": [s.id for s in plan.steps if s.status is StepStatus.failed],
            "inconclusive": [s.id for s in plan.steps
                             if s.status is StepStatus.inconclusive],
            "skipped": [s.id for s in plan.steps if s.status is StepStatus.skipped],
            "attempts": len(ctx.trace.attempts)}


# --------------------------------------------------------------------------- direct path
def synthesize(result: dict[str, Any], *, ctx: TaskContext) -> dict[str, Any] | None:
    """One LLM call over the prompt bundle.

    Returns None on TRANSIENT failure; raises `LLMConfigError` on a CONFIG failure.

    That split is the whole point. A 429 or a 5xx is a fact about the world and the
    caller degrades to raw evidence. A rejected key or a 404 model is a wiring bug and
    must be reported loudly -- conflating the two once made a wrong model name report as
    "no key", which sent the user looking in the wrong place for an hour.
    """
    try:
        return llm.call(ctx, build_llm_messages(result), purpose="synthesize")
    except llm.LLMConfigError as e:
        ctx.log("llm", "synthesize", detail=str(e)[:90], status="config_error", tool="llm")
        raise
    except llm.LLMUnavailable as e:
        ctx.log("llm", "synthesize", detail=str(e)[:90], status="error", tool="llm")
        return None


def ask(user_query: str, *, ctx: TaskContext, top_files: int = 3,
        root: Path | str | None = None,
        exhaustive: bool | None = None) -> str | None:
    """The no-planner path: one agent run, one LLM call, honest offline degradation."""
    result = run_agent(user_query, ctx=ctx, top_files=top_files, root=root,
                       exhaustive=exhaustive)
    print("=" * 62)
    print(f"AGENT: {result['status']} | similar={len(result.get('similar_files', []))} "
          f"read={len(result['evidence'])} | {result['elapsed_ms']}ms")
    print("=" * 62)

    try:
        out = synthesize(result, ctx=ctx)
    except llm.LLMConfigError as e:
        print("\n" + "!" * 62)
        print("!!! LLM MISCONFIGURED -- wiring bug, not a network blip")
        print("!" * 62)
        print(f"  {e}")
        from .config import load_settings
        cfg = load_settings().llm
        print(f"  key      : {'SET' if cfg.api_key else 'NOT SET'}")
        print(f"  model    : {cfg.model}")
        print(f"  base_url : {cfg.base_url}")
        print("\n  fix one of:")
        print('    export GROQ_API_KEY="gsk_..."')
        print('    export LLM_MODEL="openai/gpt-oss-120b"')
        print("\n  raw evidence follows:")
        print(build_prompt_bundle(result))
        return None

    if out:
        print("\n--- ANSWER ---\n")
        print(out["text"])
        print(f"\n[{out['model']} | {out['usage'].get('total_tokens', '?')} tokens | "
              f"conf {result['confidence']}]")
        return out["text"]

    if result["status"] == "not_found":
        print("\n--- ANSWER (offline) ---\n")
        print(NOT_FOUND_REPLY)
        return NOT_FOUND_REPLY

    print("\n--- LLM TEMPORARILY UNAVAILABLE: raw evidence instead ---")
    print(build_prompt_bundle(result))
    print("\n>> re-run check_llm() to confirm the service is back.")
    return None


# --------------------------------------------------------------------------- entry point
def run_task(query: str, *, ctx: TaskContext, strategy: str = "auto",
             use_llm_verify: bool = True,
             replans_left: int = MAX_REPLANS) -> str | None:
    """Main entry. `strategy="auto"` walks the ladder and escalates on failure.

    strategy:
      auto   -- route, then escalate direct -> plan -> react as stages fail to settle
      direct -- one searchFiles call, no planner
      plan   -- planner + orchestrator + verifier
      react  -- the shell loop: think, act, verify, repeat

    The ladder, not a classifier, is what makes `auto` trustworthy. A three-way
    classifier has to be right about every query on the first try, and guessing low
    (direct) costs one call, so `route()` only has to be roughly right -- the
    escalation recovers the rest. Concretely: if `direct` returns nothing, try
    `plan`; if the plan cannot settle it either, try `react`. This is what was
    missing: the old gate could only ever choose direct-or-plan, so a diagnostic
    question silently got a one-shot file search and a confident wrong answer.
    """
    if strategy == "auto":
        r = route(query, ctx=ctx)
        strategy = r.strategy
        ctx.log("route", "strategy",
                detail=f"{strategy} ({r.reason})", status="ok",
                strategy=strategy, reason=r.reason, tool="router")
        print(f"  strategy: {strategy}")
        return _escalate(query, ctx=ctx, start=strategy,
                         use_llm_verify=use_llm_verify,
                         replans_left=replans_left)

    if strategy == "direct":
        answer = ask(query, ctx=ctx)
        if answer is not None:
            return answer
        # Explicitly asked for direct and direct found nothing. Do NOT quietly
        # escalate: the caller pinned the strategy, and silently spending more
        # budget than asked for is its own bug. Say it and stop.
        ctx.note("strategy='direct' found nothing; not escalating because the "
                 "strategy was pinned. Use strategy='auto' to climb the ladder.")
        return None

    if strategy == "react":
        return _react_answer(query, ctx=ctx, use_llm_verify=use_llm_verify)

    if strategy != "plan":
        raise ValueError(f"unknown strategy {strategy!r}")

    return _plan_answer(query, ctx=ctx, use_llm_verify=use_llm_verify,
                        replans_left=replans_left)


def _escalate(query: str, *, ctx: TaskContext, start: str, use_llm_verify: bool,
              replans_left: int) -> str | None:
    """Walk the ladder from `start`, moving up when a stage cannot settle the query.

    A stage "fails" by returning nothing, not by raising. The last rung returns its
    own honest `None` -- `inconclusive` is a real outcome and pretending otherwise
    would mean guessing.
    """
    order = ["direct", "plan", "react"]
    try:
        i = order.index(start)
    except ValueError:
        raise ValueError(f"unknown strategy {start!r}")

    for stage in order[i:]:
        ctx.log("route", "escalate", detail=f"trying {stage}", status="ok",
                stage=stage, tool="router")
        answer = _run_stage(stage, query, ctx=ctx, use_llm_verify=use_llm_verify,
                            replans_left=replans_left)
        if answer:
            return answer
        nxt = order[order.index(stage) + 1] if order.index(stage) + 1 < len(order) else None
        if nxt is None:
            ctx.note("react exhausted its turns without settling the question; "
                     "reporting inconclusive rather than guessing.")
            return None
        ctx.note(f"{stage} could not settle the question; escalating to {nxt}.")
        print(f"  {stage} found nothing -> escalating to {nxt}")

    return None


def _run_stage(stage: str, query: str, *, ctx: TaskContext, use_llm_verify: bool,
               replans_left: int) -> str | None:
    if stage == "direct":
        return ask(query, ctx=ctx)
    if stage == "plan":
        return _plan_answer(query, ctx=ctx, use_llm_verify=use_llm_verify,
                            replans_left=replans_left)
    if stage == "react":
        return _react_answer(query, ctx=ctx, use_llm_verify=use_llm_verify)
    raise ValueError(f"unknown stage {stage!r}")


def _react_answer(query: str, *, ctx: TaskContext, use_llm_verify: bool) -> str | None:
    from .react import run_react

    print("  react: think -> act -> verify")
    r = run_react(query, ctx=ctx, use_llm_verify=use_llm_verify)
    if r.steps:
        print(r.render()[-4000:])
    if r.answer:
        return r.answer
    if r.status != "answered":
        ctx.note(f"react ended {r.status} after {r.turns} turn(s)")
    return None


def _plan_answer(query: str, *, ctx: TaskContext, use_llm_verify: bool,
                 replans_left: int) -> str | None:
    t0 = time.perf_counter()

    # DSPy is configured lazily, and ONLY when a plan is actually needed. Without
    # this the fallback path calls rubric_verify against an unconfigured LM and
    # raises "No LM is loaded" instead of degrading.
    used_fallback = False
    plan = None
    if DSPY_OK:
        configure_dspy(ctx.settings)
        try:
            plan = make_plan(query, ctx=ctx)
            ctx.log("planner", "plan", detail=f"v1 {len(plan.steps)} steps", status="ok",
                    steps=len(plan.steps), tool="planner")
        except Exception as e:
            ctx.log("planner", "plan", detail=f"{type(e).__name__}: {str(e)[:60]}",
                    status="fallback", tool="planner")
            plan = None
    if plan is None:
        used_fallback = True
        plan = make_plan_fallback(query)
        print("  (using the deterministic fallback plan -- DSPy plan unavailable)")

    print(f"\nPLAN (goal: {plan.goal[:70]})")
    for s in plan.steps:
        print(f"  {s.id}. [{s.agent}] {s.instruction[:64]}")
        print(f"     expects: {s.expected_output[:64]}  deps={s.depends_on}")

    run = execute_plan(plan, ctx=ctx, use_llm_verify=use_llm_verify,
                       replans_left=replans_left)

    print(f"\nRUN: version={run['version']} done={run['done']} "
          f"failed={run['failed']} inconclusive={run['inconclusive']} "
          f"skipped={run['skipped']} "
          f"({round((time.perf_counter()-t0)*1000)}ms, {run['attempts']} attempts)")

    # If we fell back to a deterministic plan, the LM was never configured, so any
    # rubric call inside execute_plan would have raised. Re-run without it.
    if used_fallback and use_llm_verify:
        print("  (re-running with structural-only verification: no LM configured)")
        for s in plan.steps:
            s.status = StepStatus.pending
            s.result = None
        ctx.trace.attempts.clear()
        ctx.trace.events.clear()
        run = execute_plan(plan, ctx=ctx, use_llm_verify=False, replans_left=0)

    if run["done"]:
        # verified results only. Failed/inconclusive steps contribute context,
        # never claims.
        ctx_text = "\n\n".join(
            f"STEP {s.id} (verified): {s.result['summary']}\n"
            + "\n".join(f"  - {e.get('path')}" for e in s.result["evidence"][:6]
                        if isinstance(e, dict) and e.get("path"))
            for s in run["plan"].steps if s.status is StepStatus.done and s.result)
        gaps = []
        for s in run["plan"].steps:
            if s.status is StepStatus.failed:
                gaps.append(f"step {s.id} [{s.agent}] FAILED")
            elif s.status is StepStatus.inconclusive:
                gaps.append(f"step {s.id} [{s.agent}] UNVERIFIED (could not be checked)")
            elif s.status is StepStatus.skipped:
                gaps.append(f"step {s.id} [{s.agent}] SKIPPED (dependency failed)")
        if gaps:
            ctx_text += "\n\nUNRESOLVED: " + "; ".join(gaps)

        messages = [
            {"role": "system", "content":
             "You are composing the final answer to a multi-step file-search task. "
             "Use ONLY the VERIFIED results below. Cite files as path:line. "
             "If some steps failed, say plainly which parts are unanswered and why. "
             "Never present an unverified result as fact. Be concise."},
            {"role": "user", "content":
             f"TASK: {query}\n\n{ctx_text}\n\n"
             f"DEAD ENDS TRIED ({len(ctx.trace.dead_ends())}): "
             f"{'; '.join(ctx.trace.dead_ends()) or 'none'}"},
        ]
        try:
            out = llm.call(ctx, messages, purpose="final")
            print("\n--- FINAL ANSWER ---\n")
            print(out["text"])
            print(f"\n[{out['model']} | {out['usage'].get('total_tokens','?')} tokens]")
            return out["text"]
        except (llm.LLMUnavailable, llm.LLMConfigError) as e:
            ctx.log("llm", "final", detail=str(e)[:60], status="error", tool="llm")
            print(f"\nLLM unavailable: {e}\n\nVERIFIED RESULTS:\n{ctx_text}")
            return None
        except BudgetExhausted as e:
            # A spent budget is an outcome, not a crash: TaskContext.__exit__ swallows
            # it for the same reason. Here it means the verified results were collected
            # but there is no budget left to phrase them.
            ctx.note(f"budget exhausted before the final answer: {e}")
            print(f"\nBUDGET EXHAUSTED before the final answer: {e}\n\n"
                  f"VERIFIED RESULTS:\n{ctx_text}")
            return None
    else:
        print("\nNo step verified. Reporting honestly rather than guessing.")
        for a in ctx.trace.attempts[-6:]:
            print(f"  attempt {a.attempt_id}: {a.compact()}")
        return None


__all__ = ["MAX_REPLANS", "MAX_STEPS_PER_PLAN", "ready_steps", "blocked_steps",
           "execute_plan", "run_task", "ask", "synthesize",
           "_escalate", "_react_answer", "_plan_answer"]
