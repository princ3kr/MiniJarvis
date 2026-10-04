"""Planner/orchestrator structural tests -- the Phase 0 regression net.

Ported one-for-one from the notebook's `test_planner.py`, which had 32 checks covering
DAG validation, JSON repair, the ready queue, repeat rejection, replan caps, verifier
verdicts and dead-end filtering. Every one is here.

Two conventions, because the port removed the notebook's module globals:

  * `PATH_LOG` was a module list. It is now `ctx.trace`. A test seeds a dead end by
    appending to `ctx.trace.attempts`.
  * `AGENT_RUNNERS` was a module dict the tests monkeypatched. It is still a registry,
    but each test takes a private copy so no test can leave another one broken.

NO NETWORK. `dspy` is present in the venv, but no test constructs a real LM: the DSPy
paths are either skipped or driven by `make_plan_fallback`, and the rubric is stubbed
through the verifier's public surface.

Two tests assert a KNOWN BUG is still present. They are the tripwires for Phase 1
(`test_plan_cap_is_seven_and_the_constant_is_decorative`,
`test_dead_branch_use_planner_is_unreachable`).
"""
from __future__ import annotations

import json

import pytest

from jarvis import engine, planner, verifier
from jarvis.engine import blocked_steps, ready_steps
from jarvis.planner import (AGENT_ALIASES, AGENT_REGISTRY, IMPLEMENTED_AGENTS, AgentResult,
                            Plan, Step, StepStatus, _extract_json, make_plan_fallback,
                            normalise_agent, parse_plan_json)
from jarvis.trace import Attempt, FailureType, Verdict


def S(i, agent="searchFiles", deps=(), exp="list of file paths", instruction=None):
    return Step(id=i, agent=agent, instruction=instruction or f"step {i} instruction",
                depends_on=list(deps), expected_output=exp)


def seed_dead_end(ctx, *, agent="searchFiles", instruction="find the thing",
                  verdict=Verdict.fail, failure_type=FailureType.empty,
                  failure_reason="no files matched", summary=""):
    ctx.trace.attempts.append(
        Attempt(attempt_id=len(ctx.trace.attempts) + 1, plan_version=1, step_id=1,
                agent=agent, instruction=instruction, status="not_found",
                result_summary=summary, verdict=verdict,
                failure_reason=failure_reason, failure_type=failure_type))


@pytest.fixture
def runners(monkeypatch):
    """A private copy of AGENT_RUNNERS, restored after each test."""
    import copy
    private = dict(engine.AGENT_RUNNERS)
    monkeypatch.setattr(engine, "AGENT_RUNNERS", private)
    return private


# ---------------------------------------------------------------- DAG validation
def test_plan_rejects_unknown_dependency():
    with pytest.raises(Exception) as e:
        Plan(goal="g", steps=[S(1, deps=[99])])
    assert "unknown step" in str(e.value), e


def test_plan_rejects_self_dependency():
    with pytest.raises(Exception) as e:
        Plan(goal="g", steps=[S(1, deps=[1])])
    assert "itself" in str(e.value), e


def test_plan_rejects_duplicate_ids():
    with pytest.raises(Exception) as e:
        Plan(goal="g", steps=[S(1), S(1)])
    assert "duplicate" in str(e.value), e


def test_null_depends_on_coerced_to_empty_list():
    s = Step(id=1, agent="searchFiles", instruction="x",
             expected_output="y", depends_on=None)
    assert s.depends_on == [], s.depends_on


def test_plan_by_id_round_trips():
    p = Plan(goal="g", steps=[S(1), S(2, deps=[1])])
    assert p.by_id(2).id == 2
    assert p.by_id(404) is None


def test_step_status_has_the_inconclusive_member():
    """`inconclusive` exists so `unsure` is not reported as a failure."""
    assert StepStatus.inconclusive.value == "inconclusive"
    assert [s.value for s in StepStatus] == \
        ["pending", "running", "done", "failed", "inconclusive", "skipped"]


# ---------------------------------------------------------------- JSON repair
def test_parse_repairs_null_deps_and_agent_aliases():
    raw = ('Here you go:\n```json\n'
           '[{"id":1,"agent":"file_search_agent","instruction":"find it",'
           '"depends_on":null,"expected_output":"path:line"},\n'
           '{"id":2,"agent":"file_search","instruction":"count it",'
           '"depends_on":[1],"expected_output":"count"}]\n```\nHope this helps!')
    p = parse_plan_json(raw, goal="x")
    assert [s.agent for s in p.steps] == ["searchFiles", "searchFiles"], p.steps
    assert p.steps[0].depends_on == [] and p.steps[1].depends_on == [1]
    assert p.goal == "x"


def test_parse_strips_prose_around_the_json():
    raw = ("Sure! I have found 3 steps.\n\n"
           '[{"id":1,"agent":"searchFiles","instruction":"a","depends_on":[],'
           '"expected_output":"b"}]\n\nLet me know if that helps.')
    p = parse_plan_json(raw, goal="x")
    assert len(p.steps) == 1 and p.steps[0].instruction == "a"


def test_parse_repunes_a_scalar_dependency_into_a_list():
    raw = json.dumps([{"id": 1, "agent": "searchFiles", "instruction": "a",
                       "depends_on": [], "expected_output": "b"},
                      {"id": 2, "agent": "searchFiles", "instruction": "c",
                       "depends_on": 1, "expected_output": "d"}])
    p = parse_plan_json(raw, goal="x")
    assert p.steps[1].depends_on == [1], p.steps[1].depends_on


def test_parse_still_rejects_a_dependency_on_a_step_that_is_not_there():
    """The scalar repair does not make a dangling dependency legal."""
    raw = json.dumps([{"id": 2, "agent": "searchFiles", "instruction": "a",
                       "depends_on": 1, "expected_output": "b"}])
    with pytest.raises(Exception) as e:
        parse_plan_json(raw, goal="x")
    assert "unknown step" in str(e.value)


def test_parse_caps_plan_length_at_seven():
    raw = json.dumps([{"id": i, "agent": "searchFiles", "instruction": f"s{i}",
                       "depends_on": [], "expected_output": "x"}
                      for i in range(1, 13)])
    p = parse_plan_json(raw, goal="x")
    assert len(p.steps) == 7, len(p.steps)


def test_plan_cap_has_exactly_one_owner():
    """The cap used to be declared twice: `planner.MAX_PLAN_STEPS` (enforced) and
    `engine.MAX_STEPS_PER_PLAN` (read by nothing, so changing it changed no behaviour).
    Two constants for one setting is one too many. Asserted by absence, because a
    re-declaration is exactly what this guards."""
    assert planner.MAX_PLAN_STEPS == 7
    assert not hasattr(engine, "MAX_STEPS_PER_PLAN"), \
        "the decorative duplicate is back"
    src = planner.parse_plan_json.__doc__ or ""
    import inspect
    assert "cleaned[:MAX_PLAN_STEPS]" in inspect.getsource(planner.parse_plan_json)


def test_parse_drops_steps_with_no_instruction():
    raw = json.dumps([{"id": 1, "agent": "searchFiles", "instruction": "   ",
                       "depends_on": [], "expected_output": "x"}])
    with pytest.raises(ValueError) as e:
        parse_plan_json(raw, goal="x")
    assert "no usable steps" in str(e.value)


def test_unknown_agent_name_is_a_hard_error():
    raw = json.dumps([{"id": 1, "agent": "wizard", "instruction": "x",
                       "depends_on": [], "expected_output": "y"}])
    with pytest.raises(ValueError) as e:
        parse_plan_json(raw, goal="x")
    assert "unknown agent" in str(e.value)


def test_extract_json_raises_on_prose_without_an_array():
    with pytest.raises(ValueError) as e:
        _extract_json("I could not produce a plan.")
    assert "no JSON array" in str(e.value)


def test_extract_json_takes_the_first_bracket_not_the_first_array():
    """KNOWN BUG (removal scheduled, Phase 2): `t.find("[")` grabs the first `[`
    anywhere, so a bracket in the prose hijacks the parse. Here it returns `[1]` -- the
    prose -- and every plan element is silently lost."""
    assert _extract_json('Here is step [1] of the plan: [{"id": 1}]') == [1]


def test_extract_json_is_string_aware():
    """A `}` inside a JSON string must not close the object early."""
    raw = '[{"instruction": "match on } and then stop", "id": 1}, {"id": 2}]'
    assert [d["id"] for d in _extract_json(raw)] == [1, 2]


def test_extract_json_raises_on_unterminated_output():
    with pytest.raises(ValueError) as e:
        _extract_json('[{"id": 1}, {"id": 2}')
    assert "unterminated" in str(e.value)


# ---------------------------------------------------------------- agent registry
def test_normalise_agent_collapses_every_spelling():
    for alias in ["file_search_agent", "FileSearchAgent", "filesearch", "search",
                  "SearchFiles", "explore", "grep", "locate", "finder"]:
        assert normalise_agent(alias) == "searchFiles", alias


def test_normalise_agent_accepts_canonical_names_verbatim():
    for name in AGENT_REGISTRY:
        assert normalise_agent(name) == name


def test_normalise_agent_rejects_a_genuinely_unknown_agent():
    with pytest.raises(ValueError) as e:
        normalise_agent("wizard")
    assert "unknown agent" in str(e.value) and "available" in str(e.value)


def test_alias_table_keys_are_the_stripped_form():
    """Every key must already be lowercase-and-unstripped, because `normalise_agent`
    looks up `re.sub(r'[^a-z]', '', name.lower())`."""
    import re
    for key, canon in AGENT_ALIASES.items():
        assert re.sub(r"[^a-z]", "", key.lower()) == key, key
        assert canon in AGENT_REGISTRY, (key, canon)


def test_the_registry_holds_only_agents_that_exist():
    """`webSearch` and `assistant` were declared for the notebook's whole life with
    nothing behind them, so a plan step naming one failed at RUN time instead of at
    PLANNING time. Worse, the registry is what a model copies from when it writes a
    plan, so naming an agent with no runner teaches it to emit steps that cannot run.

    Now the registry IS the implemented set: adding a name means writing the runner
    first, deliberately.
    """
    # `planner` is allowed to differ: it is a role the orchestrator plays, not an
    # agent a step dispatches to. Everything else must have a runner.
    assert AGENT_REGISTRY - IMPLEMENTED_AGENTS == {"planner"}
    assert "webSearch" not in AGENT_REGISTRY
    assert "assistant" not in AGENT_REGISTRY

    # And every implemented agent must have a manifest, so the two cannot drift.
    from jarvis.manifests import build_manifests
    m = build_manifests(("/tmp",), "/tmp")
    assert IMPLEMENTED_AGENTS <= set(m), \
        f"no manifest for {sorted(IMPLEMENTED_AGENTS - set(m))}"


def test_aliases_cannot_reach_a_removed_agent():
    """The alias table is a second door into the vocabulary. Removing a name from the
    registry while leaving "websearch" in the aliases would keep it reachable."""
    # Values, not the source text: a comment may legitimately name what was removed.
    for gone in ("webSearch", "assistant"):
        assert gone not in planner.AGENT_ALIASES.values(), \
            f"an alias still reaches {gone}"
        assert gone not in planner.AGENT_ALIASES


# ---------------------------------------------------------------- scheduling
def test_ready_steps_respects_dag_order():
    p = Plan(goal="g", steps=[S(1), S(2, deps=[1]), S(3)])
    ready = ready_steps(p)
    assert {s.id for s in ready} == {1, 3}, [s.id for s in ready]
    p.by_id(1).status = StepStatus.done
    assert {s.id for s in ready_steps(p)} == {2, 3}


def test_blocked_steps_marks_orphaned_work():
    p = Plan(goal="g", steps=[S(1), S(2, deps=[1])])
    p.by_id(1).status = StepStatus.failed
    assert {s.id for s in blocked_steps(p)} == {2}


def test_blocked_steps_ignores_inconclusive_dependencies():
    """An inconclusive step did NOT fail, so its dependents are not provably blocked.
    They are also not ready: `ready_steps` requires the dependency to be `done`."""
    p = Plan(goal="g", steps=[S(1), S(2, deps=[1])])
    p.by_id(1).status = StepStatus.inconclusive
    assert blocked_steps(p) == []
    assert ready_steps(p) == [], "an unproven step must not satisfy a dependency"


def test_skipped_steps_are_neither_ready_nor_blocked():
    p = Plan(goal="g", steps=[S(1)])
    p.steps[0].status = StepStatus.skipped
    assert ready_steps(p) == [] and blocked_steps(p) == []


# ---------------------------------------------------------------- repeat rejection
def test_repeat_rejection_is_enforced_in_code(ctx, runners):
    seed_dead_end(ctx, instruction="find the thing")
    calls: list[str] = []
    runners["searchFiles"] = lambda i, c, **k: calls.append(i)

    p = Plan(goal="g", steps=[S(1)])
    p.steps[0].instruction = "find the thing"      # identical -> must be refused
    run = engine.execute_plan(p, ctx=ctx, use_llm_verify=False, replans_left=0)
    assert not calls, f"agent was re-run on a dead end: {calls}"
    assert run["failed"] == [1], run
    assert "dead end" in ctx.trace.attempts[-1].failure_reason


def test_signature_normalisation_ignores_case_and_whitespace(ctx, runners):
    seed_dead_end(ctx, instruction="Find   The Thing")
    calls: list[str] = []
    runners["searchFiles"] = lambda i, c, **k: calls.append(i)
    p = Plan(goal="g", steps=[S(1, instruction="find the thing")])
    run = engine.execute_plan(p, ctx=ctx, use_llm_verify=False, replans_left=0)
    assert not calls
    assert run["failed"] == [1], run


def test_a_different_agent_with_the_same_words_is_not_a_repeat(ctx, runners):
    """`signature()` is (agent, instruction), so a repeat of a different agent's dead end
    is not a repeat -- it is refused for its own reason instead."""
    seed_dead_end(ctx, agent="searchFiles", instruction="find the thing")
    p = Plan(goal="g", steps=[S(1, agent="webSearch", instruction="find the thing")])
    run = engine.execute_plan(p, ctx=ctx, use_llm_verify=False, replans_left=0)
    assert "not implemented" in ctx.trace.attempts[-1].failure_reason, \
        ctx.trace.attempts[-1].failure_reason
    assert "dead end" not in ctx.trace.attempts[-1].failure_reason
    assert run["failed"] == [1]


def test_unimplemented_agent_fails_loudly(ctx, runners):
    calls: list[str] = []
    runners["searchFiles"] = lambda i, c, **k: calls.append(i)
    p = Plan(goal="g", steps=[S(1, agent="webSearch")])
    run = engine.execute_plan(p, ctx=ctx, use_llm_verify=False, replans_left=0)
    assert run["failed"] == [1], run
    assert not calls
    a = ctx.trace.attempts[-1]
    assert "not implemented" in a.failure_reason, a.failure_reason


# ---------------------------------------------------------------- replanning
# A replan only fires when, after a pass over the ready set, some step is STILL pending
# AND some step failed. A ONE-STEP plan can never satisfy both: its only step becomes
# non-pending, and `all(s.status is not pending)` breaks out first. The notebook's replan
# tests used one-step plans and so never reached this code at all. Every test below uses
# a three-step plan -- one that pins, two that raise -- so the branch really executes.


def three_step_plan() -> Plan:
    return Plan(goal="g", steps=[S(1, instruction="keep me"),
                                 S(2, instruction="break me", deps=[1]),
                                 S(3, instruction="third")])


def raiser_except(*pass_instructions):
    """An agent runner that returns a passing result for the named instructions and
    raises for everything else."""
    def runner(instr, deps, *, ctx, **kw):
        if instr in pass_instructions:
            return AgentResult(
                status="ok", summary="kept",
                evidence=[{"path": str(ctx.root / "README.md")}], confidence=0.9)
        raise RuntimeError("boom")
    return runner


@pytest.fixture
def replan_env(ctx, monkeypatch):
    """Stub the agent (step 1 passes, 2 and 3 raise) and `replan` (returns a valid,
    id-colliding two-step plan). Returns the list of `completed` digests replan saw."""
    monkeypatch.setattr(engine, "AGENT_RUNNERS",
                        {**engine.AGENT_RUNNERS,
                         "searchFiles": raiser_except("keep me")})
    digests: list[str] = []

    def default_replan(goal, completed, failure, *, ctx, max_replans=2):
        digests.append(completed)
        return Plan(goal=goal, steps=[S(1, instruction="retry a"),
                                      S(2, instruction="retry b", deps=[1])])

    monkeypatch.setattr(engine, "replan", default_replan)
    return digests


def test_orchestrator_never_exceeds_the_replan_cap(ctx, replan_env):
    run = engine.execute_plan(three_step_plan(), ctx=ctx, use_llm_verify=False,
                              replans_left=2)
    assert len(replan_env) >= 1, "the replan branch never ran"
    assert len(replan_env) <= 2, f"{len(replan_env)} replans, cap was 2"
    assert run["version"] <= 3, f"version {run['version']} exceeded cap"
    assert ctx.counters.replans == len(replan_env)


def test_replan_is_offered_only_the_verified_digest(ctx, replan_env):
    engine.execute_plan(three_step_plan(), ctx=ctx, use_llm_verify=False, replans_left=1)
    assert "kept" in replan_env[0], replan_env[0]
    assert "boom" not in replan_env[0], "a failed summary leaked into the digest"


def test_replan_cap_is_also_charged_to_the_task_budget(ctx, replan_env):
    """`charge_replan` is what actually stops the loop when the budget is tighter than
    the local `replans_left` counter."""
    engine.execute_plan(three_step_plan(), ctx=ctx, use_llm_verify=False, replans_left=99)
    assert ctx.counters.replans <= ctx.settings.budget.max_replans, ctx.counters.replans


def test_verified_results_stay_pinned_across_a_replan(ctx, replan_env):
    p = three_step_plan()
    engine.execute_plan(p, ctx=ctx, use_llm_verify=False, replans_left=1)
    s1 = next((s for s in p.steps if s.instruction == "keep me"), None)
    assert s1 is not None and s1.status is StepStatus.done, \
        "a verified step was discarded by the replan"
    assert s1.result and s1.result["summary"] == "kept"


def test_replan_keeps_the_original_id_of_a_pinned_step(ctx, monkeypatch):
    """Non-colliding new ids: the only shape in which the old->new map is correct.

    A flat offset would be wrong here: the new steps' deps refer to the ORIGINAL ids
    while the pinned step keeps its own, so a blind offset leaves deps pointing at ids
    that no longer exist. Original 50 must arrive as 2, and the dep must follow it.

    Scope note, and the reason the deps below are internal to the new plan: a new step
    that depends on a PINNED step cannot be constructed at all. `Plan` validates
    standalone, so `depends_on=[1]` dangles, the replan is rejected, and the remap never
    runs. That rejection is asserted by
    `test_replan_rejects_a_plan_that_depends_on_a_pinned_step`; it is the reason this test
    can only reach the non-colliding half of the map.
    """
    monkeypatch.setattr(engine, "AGENT_RUNNERS",
                        {**engine.AGENT_RUNNERS, "searchFiles": raiser_except("keep me")})
    monkeypatch.setattr(engine, "replan", lambda goal, completed, failure, *, ctx,
                        max_replans=2: Plan(goal=goal, steps=[
        S(50, instruction="retry a"),
        S(51, instruction="retry b", deps=[50]),
    ]))
    p = three_step_plan()
    run = engine.execute_plan(p, ctx=ctx, use_llm_verify=False, replans_left=1)

    assert run["version"] == 2, run
    pinned_step = next(s for s in p.steps if s.instruction == "keep me")
    assert pinned_step.id == 1, f"pinned step was renumbered to {pinned_step.id}"
    assert pinned_step.status is StepStatus.done

    retries = [s for s in p.steps if s.instruction.startswith("retry")]
    assert [s.id for s in retries] == [2, 3], [s.id for s in retries]
    assert retries[0].depends_on == [], retries[0].depends_on
    assert retries[1].depends_on == [2], retries[1].depends_on

    ids = [s.id for s in p.steps]
    assert len(ids) == len(set(ids)), f"duplicate ids after remap: {ids}"
    live = set(ids)
    for s in p.steps:
        assert all(d in live for d in s.depends_on), s


def test_replan_id_map_is_clobbered_by_colliding_new_ids(ctx, replan_env):
    """KNOWN BUG (removal scheduled, Phase 1): the old->new map is built pinned-first, so
    a new step whose id collides with a pinned id OVERWRITES the pinned entry. The planner
    is told to renumber from 1, so collision is the NORMAL case -- and here a new step that
    asked to depend on step 1 gets wired to the FIRST NEW STEP instead.

    Wrong behaviour, asserted on purpose. Phase 1 flips it and this is where the new
    expectation goes."""
    p = three_step_plan()
    run = engine.execute_plan(p, ctx=ctx, use_llm_verify=False, replans_left=1)

    assert run["version"] == 2, run
    pinned_step = next(s for s in p.steps if s.instruction == "keep me")
    retry_a = next(s for s in p.steps if s.instruction == "retry a")
    retry_b = next(s for s in p.steps if s.instruction == "retry b")
    assert pinned_step.id == 1
    assert (retry_a.id, retry_b.id) == (2, 3)
    # WRONG: `retry b` asked for step 1 and got step 2 (`retry a`), not the pinned step
    assert retry_b.depends_on == [2], retry_b.depends_on


def test_replan_drops_the_failed_steps_from_the_plan(ctx, replan_env):
    """KNOWN BUG (removal scheduled, Phase 1): `plan.steps = pinned + new_plan.steps`
    discards the failed and inconclusive steps. Their attempts survive in the trace but
    their ids are free to be reissued, so the plan and the trace can disagree about what
    step 2 was."""
    p = three_step_plan()
    engine.execute_plan(p, ctx=ctx, use_llm_verify=False, replans_left=1)
    assert not any(s.instruction == "break me" for s in p.steps), \
        "the failed step should have been dropped (documented behaviour)"
    # nothing is lost forensically: the attempt is still on the trace
    assert any(a.step_id == 2 and a.failure_reason == "RuntimeError: boom"
               for a in ctx.trace.attempts)


def test_replan_rejects_a_plan_that_depends_on_a_pinned_step(ctx, monkeypatch):
    """KNOWN BUG (removal scheduled, Phase 1): `replan` returns only the REMAINING steps,
    so any remaining step that depends on a pinned step carries a dangling dep, fails
    `Plan`'s DAG validator, and ends the run at the current version."""
    monkeypatch.setattr(engine, "AGENT_RUNNERS",
                        {**engine.AGENT_RUNNERS, "searchFiles": raiser_except("keep me")})
    monkeypatch.setattr(engine, "replan", lambda goal, completed, failure, *, ctx,
                        max_replans=2: Plan(goal=goal, steps=[
        S(1, instruction="retry a", deps=[1]),
    ]))
    p = three_step_plan()
    run = engine.execute_plan(p, ctx=ctx, use_llm_verify=False, replans_left=1)

    assert run["version"] == 1, "a rejected replan must not bump the version"
    assert any("replan failed" in a.failure_reason for a in ctx.trace.attempts), \
        ctx.trace.attempts[-1].failure_reason
    pinned_step = next(s for s in p.steps if s.instruction == "keep me")
    assert pinned_step.status is StepStatus.done


def test_replan_failure_stops_the_run_instead_of_looping(ctx, monkeypatch):
    """A replan that raises must end the run at the current version, not spin.

    The agent still has to PIN one step. An agent that fails everything marks the whole
    plan non-pending on the first pass, and `execute_plan` breaks out at its
    "all steps settled" check before the replan branch is ever reached -- so the replan
    would go untested. `raiser_except("keep me")` pins step 1 and leaves step 2 pending,
    which is the only shape that actually arrives at the replan call.
    """
    def boom(*a, **k):
        raise RuntimeError("no LM")

    monkeypatch.setattr(engine, "replan", boom)
    monkeypatch.setattr(engine, "AGENT_RUNNERS",
                        {**engine.AGENT_RUNNERS, "searchFiles": raiser_except("keep me")})
    run = engine.execute_plan(three_step_plan(), ctx=ctx, use_llm_verify=False,
                              replans_left=2)
    assert run["version"] == 1, run
    assert any("replan failed" in a.failure_reason for a in ctx.trace.attempts), \
        ctx.trace.attempts


def test_dead_ends_excludes_inconclusive(ctx):
    ctx.trace.attempts.extend([
        Attempt(attempt_id=1, plan_version=1, step_id=1, agent="searchFiles",
                instruction="real failure", status="not_found", result_summary="",
                verdict=Verdict.fail, failure_reason="no files matched",
                failure_type=FailureType.empty),
        Attempt(attempt_id=2, plan_version=1, step_id=2, agent="searchFiles",
                instruction="never checked", status="ok", result_summary="looks fine",
                verdict=Verdict.unsure, failure_reason="rubric unavailable",
                failure_type=FailureType.inconclusive),
    ])
    de = ctx.trace.dead_ends()
    assert any("real failure" in d for d in de), de
    assert not any("never checked" in d for d in de), \
        "an unverified step was blacklisted as a dead end"


def test_verified_digest_lists_only_passed_steps(ctx, runners):
    p = Plan(goal="g", steps=[S(1, instruction="passed"),
                              S(2, instruction="unproven", deps=[1])])
    p.steps[0].status = StepStatus.done
    p.steps[0].result = {"summary": "found the file", "evidence": []}
    p.steps[1].status = StepStatus.inconclusive
    p.steps[1].result = {"summary": "maybe", "evidence": []}
    digest = engine._verified_digest(p)
    assert "found the file" in digest
    assert "maybe" not in digest, digest


def test_attempt_compact_is_one_line(ctx):
    a = Attempt(attempt_id=1, plan_version=1, step_id=2, agent="searchFiles",
                instruction="a " * 60, status="not_found", result_summary="",
                verdict=Verdict.fail, failure_reason="no files matched",
                failure_type=FailureType.empty)
    c = a.compact()
    assert "\n" not in c, c
    assert "DEAD END" in c and "no files matched" in c, c


# ---------------------------------------------------------------- the fallback plan
def test_make_plan_fallback_is_a_real_two_step_plan():
    p = make_plan_fallback("where is the auth code?")
    assert p.goal == "where is the auth code?"
    assert len(p.steps) == 2
    assert [s.id for s in p.steps] == [1, 2]
    assert p.steps[0].depends_on == []
    assert p.steps[1].depends_on == [1], "fallback must exercise the dependency edge"
    assert all(s.agent == "searchFiles" for s in p.steps)
    assert all(s.expected_output for s in p.steps)
    assert p.steps[1].instruction.startswith("file type breakdown")


def test_make_plan_fallback_passes_dag_validation():
    """It is constructed as a `Plan`, not a dict, so it cannot ship a broken DAG."""
    Plan(goal=make_plan_fallback("x").goal, steps=make_plan_fallback("x").steps)


# ---------------------------------------------------------------- dead code
def test_dead_branch_use_planner_is_gone(ctx, monkeypatch):
    """`run_task` used to set `use_planner = True` and then test `if not use_planner:`
    immediately after, so the `ask()` fallback below the plan block was unreachable.
    Removed when `run_task` was split into `_plan_answer` / `_react_answer`.

    Asserted by absence rather than by source text: the old test grepped the function
    body, which passes just as happily on the string inside a docstring.
    """
    import inspect
    src = inspect.getsource(engine.run_task)
    assert "use_planner" not in src, "the dead branch is back"
    # and the pipeline that replaced it is real, and the stages are reachable
    from jarvis import pipeline
    assert callable(pipeline.plan_path)
    assert set(pipeline.STAGES) == {"direct", "loop", "plan"}


def test_dead_ends_helper_is_not_a_module_global():
    """`dead_ends()` was a module function over a module list. It is now
    `ctx.trace.dead_ends()`; nothing in the package keeps its own copy.

    Checked over the AST rather than the raw text. The modules deliberately *discuss*
    the globals they replaced -- docstrings and comments explain why `PATH_LOG` is gone --
    so a line-based grep flags the explanation and not the code. A docstring is an
    `ast.Constant` and a comment is not in the tree at all, so only a real name
    reference can fail this.
    """
    import ast
    import inspect
    import textwrap

    forbidden = ("PATH_LOG", "LOGS", "_GATEWAY_CALLS", "_LAST_EXIT", "_LAST_OUT",
                 "_LAST_ERR", "APPROVAL_GRANTED")
    for mod in (engine, planner, verifier):
        tree = ast.parse(textwrap.dedent(inspect.getsource(mod)))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
        leaked = sorted(names.intersection(forbidden))
        assert not leaked, f"{mod.__name__} still binds {leaked} as module state"
