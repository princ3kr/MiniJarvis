"""Structural tests for the planner/orchestrator/verifier.

Runs the notebook in the venv (dspy available), then asserts the offline
behaviour: DAG validation, JSON repair, ready-queue, repeat rejection,
replan caps, verifier verdicts. No network needed except the notebook's own
LLM preflight cell.
"""
import json
import pathlib
import re
import sys

NB = pathlib.Path("/home/princ3kr/Desktop/learnings/glinerTest.ipynb")

nb = json.loads(NB.read_text())
codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
g = {"__name__": "nb"}
for i, c in enumerate(codes):
    if i == len(codes) - 1:
        break
    exec(compile(c, f"<cell{i}>", "exec"), g)

Step = g["Step"]
Plan = g["Plan"]
StepStatus = g["StepStatus"]
Verdict = g["Verdict"]
AgentResult = g["AgentResult"]

results = []


def check(name):
    def deco(fn):
        try:
            results.append((name, "ok", str(fn())[:74]))
        except Exception as e:
            results.append((name, "RAISED", f"{type(e).__name__}: {e}"))
        return fn
    return deco


def S(i, agent="searchFiles", deps=(), exp="list of file paths", instruction=None):
    return Step(id=i, agent=agent, instruction=instruction or f"step {i} instruction",
                depends_on=list(deps), expected_output=exp)


# ---------------------------------------------------------------- DAG
@check("plan rejects unknown dependency")
def _():
    try:
        Plan(goal="g", steps=[S(1, deps=[99])])
    except Exception as e:
        assert "unknown step" in str(e), e
        return "ValueError raised"
    raise AssertionError("accepted a dangling dependency")


@check("plan rejects self-dependency")
def _():
    try:
        Plan(goal="g", steps=[S(1, deps=[1])])
    except Exception as e:
        assert "itself" in str(e), e
        return "ValueError raised"
    raise AssertionError("accepted a self-dependency")


@check("plan rejects duplicate ids")
def _():
    try:
        Plan(goal="g", steps=[S(1), S(1)])
    except Exception as e:
        assert "duplicate" in str(e), e
        return "ValueError raised"
    raise AssertionError("accepted duplicate ids")


@check("null depends_on coerced to []")
def _():
    s = Step(id=1, agent="searchFiles", instruction="x",
             expected_output="y", depends_on=None)
    assert s.depends_on == [], s.depends_on
    return "None -> []"


# ---------------------------------------------------------------- JSON repair
@check("parse repairs null deps + agent aliases")
def _():
    raw = ('Here you go:\n```json\n'
           '[{"id":1,"agent":"file_search_agent","instruction":"find it",'
           '"depends_on":null,"expected_output":"path:line"},\n'
           '{"id":2,"agent":"file_search","instruction":"count it",'
           '"depends_on":[1],"expected_output":"count"}]\n```\nHope this helps!')
    p = g["parse_plan_json"](raw, goal="x")
    assert [s.agent for s in p.steps] == ["searchFiles", "searchFiles"], p.steps
    assert p.steps[0].depends_on == [] and p.steps[1].depends_on == [1]
    return "null->[], aliases->searchFiles, prose stripped"


@check("parse caps plan length at 7")
def _():
    raw = json.dumps([{"id": i, "agent": "searchFiles", "instruction": f"s{i}",
                       "depends_on": [], "expected_output": "x"}
                      for i in range(1, 13)])
    p = g["parse_plan_json"](raw, goal="x")
    assert len(p.steps) == 7, len(p.steps)
    return "12 steps -> 7"


@check("unknown agent name is a hard error")
def _():
    raw = json.dumps([{"id": 1, "agent": "wizard", "instruction": "x",
                       "depends_on": [], "expected_output": "y"}])
    try:
        g["parse_plan_json"](raw, goal="x")
    except ValueError as e:
        assert "unknown agent" in str(e), e
        return "rejected"
    raise AssertionError("accepted a hallucinated agent")


# ---------------------------------------------------------------- orchestrator
@check("ready_steps respects DAG order")
def _():
    p = Plan(goal="g", steps=[S(1), S(2, deps=[1]), S(3)])
    ready = g["ready_steps"](p)
    assert {s.id for s in ready} == {1, 3}, [s.id for s in ready]
    p.by_id(1).status = StepStatus.done
    assert {s.id for s in g["ready_steps"](p)} == {2, 3}
    return "{1,3} then {2,3}"


@check("blocked_steps marks orphaned work")
def _():
    p = Plan(goal="g", steps=[S(1), S(2, deps=[1])])
    p.by_id(1).status = StepStatus.failed
    assert {s.id for s in g["blocked_steps"](p)} == {2}
    return "dependent of failed -> blocked"


@check("repeat rejection is enforced in code")
def _():
    # seed a dead end, then have the orchestrator encounter the identical step
    g["PATH_LOG"].clear()
    from dataclasses import asdict
    g["PATH_LOG"].append(g["Attempt"](
        attempt_id=1, plan_version=1, step_id=1, agent="searchFiles",
        instruction="find the thing", status="not_found", result_summary="",
        verdict=Verdict.fail, failure_reason="no files matched",
        failure_type=g["FailureType"].empty))

    calls = []
    g["AGENT_RUNNERS"]["searchFiles"] = lambda i, c, **k: calls.append(i)

    p = Plan(goal="g", steps=[S(1)])
    p.steps[0].instruction = "find the thing"      # identical -> must be refused
    run = g["execute_plan"](p, use_llm_verify=False, replans_left=0)
    assert not calls, f"agent was re-run on a dead end: {calls}"
    assert run["failed"] == [1], run
    g["AGENT_RUNNERS"]["searchFiles"] = g["run_search_files"]
    return "agent never invoked on a logged dead end"


@check("unimplemented agent fails loudly")
def _():
    calls = []
    g["AGENT_RUNNERS"]["searchFiles"] = lambda i, c, **k: calls.append(i)
    p = Plan(goal="g", steps=[S(1, agent="webSearch")])
    run = g["execute_plan"](p, use_llm_verify=False, replans_left=0)
    assert run["failed"] == [1], run
    assert not calls
    a = g["PATH_LOG"][-1]
    assert "not implemented" in a.failure_reason, a.failure_reason
    g["AGENT_RUNNERS"]["searchFiles"] = g["run_search_files"]
    return "webSearch -> failed, not silently skipped"


@check("orchestrator never exceeds replan cap")
def _():
    g["PATH_LOG"].clear()
    n = {"i": 0}
    def always_fail(instr, ctx):
        n["i"] += 1
        raise RuntimeError("simulated agent failure")
    g["AGENT_RUNNERS"]["searchFiles"] = always_fail

    versions = []
    orig_replan = g["replan"]
    def counting_replan(goal, completed, failure, max_replans=2):
        versions.append(max_replans)
        return Plan(goal=goal, steps=[S(900 + len(versions),
                                       instruction=f"retry {len(versions)}")])
    g["replan"] = counting_replan

    p = Plan(goal="g", steps=[S(1)])
    run = g["execute_plan"](p, use_llm_verify=False, replans_left=2)

    assert run["version"] <= 3, f"version {run['version']} exceeded cap"
    assert len(versions) <= 2, f"{len(versions)} replans, cap was 2"
    g["replan"] = orig_replan
    g["AGENT_RUNNERS"]["searchFiles"] = g["run_search_files"]
    return f"{len(versions)} replan(s), version={run['version']}"


@check("verified results stay pinned across a replan")
def _():
    g["PATH_LOG"].clear()
    good = AgentResult(status="ok", summary="found it",
                       evidence=[{"path": "/home/princ3kr/Desktop/learnings/glinerTest.ipynb"}],
                       confidence=0.9)

    def runner(instr, ctx):
        if instr == "good step":
            return good
        raise RuntimeError("boom")

    g["AGENT_RUNNERS"]["searchFiles"] = runner
    orig = g["replan"]
    g["replan"] = lambda goal, completed, failure, max_replans=2: Plan(
        goal=goal, steps=[S(50, instruction="next step", deps=[1])])

    p = Plan(goal="g", steps=[S(1, instruction="good step"),
                              S(2, instruction="bad step", deps=[1])])
    g["execute_plan"](p, use_llm_verify=False, replans_left=1)

    s1 = next((s for s in p.steps if s.instruction == "good step"), None)
    assert s1 is not None and s1.status is StepStatus.done, \
        "a verified step was discarded by the replan"
    assert s1.result and s1.result["summary"] == "found it"
    g["replan"] = orig
    g["AGENT_RUNNERS"]["searchFiles"] = g["run_search_files"]
    return "passed step survived replan with its result"


# ---------------------------------------------------------------- verifier
@check("verifier: not_found is a hard fail")
def _():
    v, why = g["deterministic_verify"](S(1), AgentResult(
        status="not_found", summary="nothing", confidence=0.0))
    assert v is Verdict.fail, (v, why)
    return f"fail: {why}"


@check("verifier: near_miss is unsure, not fail")
def _():
    v, why = g["deterministic_verify"](S(1), AgentResult(
        status="near_miss", summary="close ones",
        evidence=[{"path": "/home/princ3kr/Desktop/learnings/glinerTest.ipynb"}],
        confidence=0.3))
    assert v is Verdict.unsure, (v, why)
    return "unsure"


@check("verifier: cited paths must exist inside the jail")
def _():
    v, why = g["deterministic_verify"](S(1), AgentResult(
        status="ok", summary="s", confidence=0.8,
        evidence=[{"path": "/home/princ3kr/Desktop/learnings/glinerTest.ipynb"},
                  {"path": "/etc/passwd"}]))
    assert v is Verdict.fail and "outside jail" in why, (v, why)
    return "rejected /etc/passwd"


@check("verifier: non-existent cited path fails")
def _():
    v, why = g["deterministic_verify"](S(1), AgentResult(
        status="ok", summary="s", confidence=0.8,
        evidence=[{"path": "/home/princ3kr/definitely_not_here_98765.py"}]))
    assert v is Verdict.fail and "missing" in why, (v, why)
    return "rejected missing file"


@check("verifier: no evidence -> unsure, needs rubric")
def _():
    v, why = g["deterministic_verify"](S(1), AgentResult(
        status="ok", summary="s", confidence=0.9, evidence=[]))
    assert v is Verdict.unsure, (v, why)
    return "deferred to rubric"


# ---------------------------------------------------------------- agent contract
@check("searchFiles adapter returns AgentResult")
def _():
    r = g["run_search_files"]("show me glinerTest.ipynb", {})
    assert isinstance(r, AgentResult), type(r)
    assert r.status in ("ok", "near_miss", "not_found"), r.status
    assert r.summary and isinstance(r.confidence, float)
    return f"status={r.status}, {len(r.evidence)} evidence item(s)"


@check("adapter never concatenates dependency context into the query")
def _():
    # a polluted instruction would show up as a near_miss/no-match
    r = g["run_search_files"]("show me glinerTest.ipynb",
                              {"1": {"summary": "POISON-CONTENT"}})
    assert r.status == "ok", f"context leaked into the query: {r.status}"
    return "context ignored by design"


@check("short stems do not fuzzy-match")
def _():
    # difflib scores these on incidental overlap: 'search' vs 'src' = 0.667,
    # 'def' vs 'ide' = 0.667. Both must be rejected.
    assert g["_match_kind"]("search", "/x/src")[0] is None, g["_match_kind"]("search", "/x/src")
    assert g["_match_kind"]("def", "/x/ide")[0] is None, g["_match_kind"]("def", "/x/ide")
    # a real similarity must still work
    assert g["_match_kind"]("utilities", "/x/utility")[0] == 3, "lost true fuzzy"
    return "src/search + ide/def rejected; utilities/utility still fuzzy"


@check("planner prose does not become a folder hint")
def _():
    it = g["extract_intent"]("Search the codebase for the definition of `safe_resolve`")
    assert "search" not in [k.lower() for k in it["keywords"]], it["keywords"]
    assert "codebase" not in [k.lower() for k in it["keywords"]], it["keywords"]
    assert "definition" not in [k.lower() for k in it["keywords"]], it["keywords"]
    return f"keywords={it['keywords']}"


@check("unsure is inconclusive, not failed")
def _():
    from dataclasses import asdict
    good = AgentResult(status="ok", summary="found it",
                       evidence=[{"path": "/home/princ3kr/Desktop/learnings/glinerTest.ipynb",
                                  "line": 15}],
                       confidence=0.9)
    orig = g["rubric_verify"]
    orig_runner = g["AGENT_RUNNERS"]["searchFiles"]
    g["rubric_verify"] = lambda s, r, use_llm=True: (Verdict.unsure, "rubric unavailable")
    # the AGENT must be stubbed too: a real run for "find it" returns not_found,
    # which fails deterministically and never reaches the rubric
    g["AGENT_RUNNERS"]["searchFiles"] = lambda i, c, **k: good
    try:
        g["PATH_LOG"].clear()
        p = Plan(goal="g", steps=[S(1, instruction="find it")])
        run = g["execute_plan"](p, use_llm_verify=True, replans_left=0)
        s1 = p.by_id(1)
        assert s1.status is StepStatus.inconclusive, s1.status
        assert run["inconclusive"] == [1], run
        assert run["failed"] == [], "unsure was reported as failed"
        a = g["PATH_LOG"][-1]
        assert a.failure_type is g["FailureType"].inconclusive, a.failure_type
        return f"status={s1.status.value}, fail_type={a.failure_type.value}"
    finally:
        g["rubric_verify"] = orig
        g["AGENT_RUNNERS"]["searchFiles"] = orig_runner


@check("verifier never raises on transport failure")
def _():
    def boom(*a, **k):
        raise RuntimeError("simulated 429")
    orig = g["rubric_verify"]
    g["rubric_verify"] = boom
    try:
        v, why = g["verify_step"](S(1), AgentResult(
            status="ok", summary="s", confidence=0.9,
            evidence=[{"path": "/home/princ3kr/Desktop/learnings/glinerTest.ipynb"}]))
        assert v is Verdict.unsure, (v, why)
        return "downgraded to unsure, no exception escaped"
    finally:
        g["rubric_verify"] = orig


@check("evidence formatting leads with path:line")
def _():
    txt = g["_format_evidence"](AgentResult(
        status="ok", summary="s",
        evidence=[{"path": "/home/princ3kr/Desktop/learnings/glinerTest.ipynb",
                   "line": 15, "excerpt": "markdown prose here"}]))
    assert txt.index("glinerTest.ipynb:15") < txt.index("markdown prose"), txt
    assert "PARTIAL" in txt, txt
    return "citations precede excerpts"


@check("dead_ends excludes inconclusive")
def _():
    from dataclasses import asdict
    A = g["Attempt"]
    g["PATH_LOG"].clear()
    g["PATH_LOG"].extend([
        A(attempt_id=1, plan_version=1, step_id=1, agent="searchFiles",
          instruction="real failure", status="not_found", result_summary="",
          verdict=Verdict.fail, failure_reason="no files matched",
          failure_type=g["FailureType"].empty),
        A(attempt_id=2, plan_version=1, step_id=2, agent="searchFiles",
          instruction="never checked", status="ok", result_summary="looks fine",
          verdict=Verdict.unsure, failure_reason="rubric unavailable",
          failure_type=g["FailureType"].inconclusive),
    ])
    de = g["dead_ends"]()
    assert any("real failure" in d for d in de), de
    assert not any("never checked" in d for d in de), \
        "an unverified step was blacklisted as a dead end"
    return f"{len(de)} dead end(s), inconclusive excluded"


@check("ReAct tools route through the jail")
def _():
    tools = g.get("REACT_TOOLS", [])
    if not tools:
        return "skipped (dspy unavailable)"
    names = {t.name for t in tools}
    assert {"find_files", "search_content", "read_file", "survey_files"} <= names, names
    # every tool must be a thin wrapper: calling one with a jail-escaping path
    # must go through safe_resolve, not touch the filesystem
    out = g["_t_read"]("/etc/passwd")
    assert "could not read" in out and "outside jail" in out, out[:120]
    return f"{len(names)} tools; /etc/passwd refused"


@check("ReAct trajectory renders readably")
def _():
    traj = {"thought_0": "look for the file", "tool_name_0": "find_files",
            "tool_args_0": {"pattern": "a.py"}, "observation_0": "[exact] /x/a.py",
            "thought_1": "done", "tool_name_1": "finish",
            "tool_args_1": {}, "observation_1": ""}
    txt = g["render_trajectory"](traj)
    assert "thought :" in txt and "find_files" in txt and "/x/a.py" in txt, txt
    assert g["render_trajectory"]({}) == "(empty trajectory)"
    return "2 iterations rendered"


@check("strategy dispatch routes correctly")
def _():
    calls = []
    orig_ask, orig_react = g["ask"], g.get("run_react")
    g["ask"] = lambda q, **k: calls.append("direct") or "D"
    g["run_react"] = lambda q, **k: calls.append("react") or "R"
    try:
        g["run_task"]("hi", strategy="direct")
        g["run_task"]("hi", strategy="react")
        assert calls == ["direct", "react"], calls
    finally:
        g["ask"], g["run_react"] = orig_ask, orig_react
    return "direct/react dispatched"


@check("unknown strategy is rejected")
def _():
    try:
        g["run_task"]("hi", strategy="teleport")
    except ValueError as e:
        assert "unknown strategy" in str(e), e
        return "ValueError raised"
    raise AssertionError("accepted a bogus strategy")


@check("path log compact() is one line")
def _():
    a = g["Attempt"](attempt_id=1, plan_version=1, step_id=2, agent="searchFiles",
                     instruction="a " * 60, status="not_found", result_summary="",
                     verdict=Verdict.fail, failure_reason="no files matched",
                     failure_type=g["FailureType"].empty)
    c = a.compact()
    assert "\n" not in c, c
    assert "DEAD END" in c and "no files matched" in c, c
    return f"{len(c)} chars, contains reason"


@check("needs_plan gate separates single vs multi-step")
def _():
    single, r1 = g["needs_plan"]("where is safe_resolve defined?")
    multi, r2 = g["needs_plan"]("find safe_resolve in main.py then count images")
    assert single is False, r1
    assert multi is True, r2
    return f"single={single} multi={multi}"


# ---------------------------------------------------------------- report
print(f"{'test':<44} {'result':<8} detail")
print("-" * 92)
for nm, st, d in results:
    print(f"{nm:<44} {st:<8} {d}{'' if st == 'ok' else '  <-- '}")
bad = [r for r in results if r[1] != "ok"]
print("-" * 92)
print(f"{len(results) - len(bad)}/{len(results)} passed")
sys.exit(1 if bad else 0)