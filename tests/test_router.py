"""The router is a MODEL now, and the fast path is not a router.

Two things are being tested, and keeping them apart is the point of the file split.

**`route()` -- one LLM call, three strategies.** These tests drive it with a stubbed
provider, so they are deterministic and assert on the DECISION rather than on prose.
The properties that matter: all three rungs are reachable, the fallback when the model
is unreachable picks the rung that verifies its own work, and a config error is
reported as a config error rather than papered over.

**The fast path.** `detect_query_type` / `extract_intent` decide what to LOOK FOR in a
`direct` call. They must NOT influence strategy -- that coupling is what made a regex
cue list load-bearing. There is a test asserting they are not consulted for routing.

Deliberately absent: any test of the old `needs_plan` boolean, and any test that a
substring fires on a query. Those asserted bugs that no longer exist.
"""
from __future__ import annotations

import inspect
import json

import pytest

from jarvis import router as R
from jarvis.router import (AGGREGATE_CUES, BARE_FILENAME_RE, AGENT_NAME_LITERAL,
                           CATEGORY_WORDS, ESCALATION, STRATEGIES, detect_query_type,
                           extract_intent, route)
from jarvis.trace import Trace


# ------------------------------------------------------------------ the judge
class StubProvider:
    """Replays a scripted reply per `purpose`. No network, no DSPy."""

    def __init__(self, replies: dict | None = None, raise_exc: Exception | None = None):
        self.replies = replies or {}
        self.raise_exc = raise_exc
        self.seen: list[tuple[str, str]] = []

    def complete(self, messages, *, temperature=None, max_tokens=None, trace=None):
        purpose = "route"
        if trace is not None:
            for e in getattr(trace, "events", []):
                purpose = e.action
        self.seen.append((purpose, messages[-1]["content"]))
        if self.raise_exc is not None:
            raise self.raise_exc
        return {"text": self.replies.get("route", '{"strategy":"loop","reason":"x"}'),
                "usage": {"total_tokens": 12}}


def _ctx():
    from jarvis.config import load_settings
    from jarvis.context import Counters, TaskContext
    return TaskContext(query="t", settings=load_settings(), trace=Trace(),
                       counters=Counters())


@pytest.mark.parametrize("want", STRATEGIES)
def test_every_strategy_is_reachable_from_the_judge(want, monkeypatch):
    """All three rungs. The old gate had no cue that could reach the loop, so a whole
    strategy looked wired up and was dead."""
    from jarvis import llm
    monkeypatch.setattr(llm, "call", lambda ctx, m, **kw: {
        "text": json.dumps({"strategy": want, "reason": "because"}),
        "usage": {"total_tokens": 9}})
    r = route("some question", ctx=_ctx())
    assert r.strategy == want
    assert r.source == "llm"


def test_the_judge_costs_exactly_one_call(monkeypatch):
    """One call, before any work. A router that costs more than the work it routes is
    not worth having."""
    calls = []

    def fake(ctx, m, **kw):
        calls.append(kw.get("purpose"))
        return {"text": '{"strategy":"loop","reason":"needs command output"}',
                "usage": {"total_tokens": 9}}

    from jarvis import llm
    monkeypatch.setattr(llm, "call", fake)
    route("which file is >= 100MB", ctx=_ctx())
    assert calls == ["route"], calls


def test_the_judge_prompt_describes_what_each_strategy_can_do(monkeypatch):
    """The prompt is the routing policy. It has to say what each rung is FOR, or the
    model is guessing from three bare words."""
    captured = {}

    def fake(ctx, m, **kw):
        captured["sys"] = m[0]["content"]
        return {"text": '{"strategy":"loop","reason":"x"}', "usage": {}}

    from jarvis import llm
    monkeypatch.setattr(llm, "call", fake)
    route("q", ctx=_ctx())
    sys = captured["sys"].lower()
    for needed in ("size", "count", "system", "command output", "multi-part"):
        assert needed in sys, f"the prompt never mentions {needed!r}"
    assert "you are not answering" in sys


def test_an_unreachable_model_falls_back_to_loop(monkeypatch):
    """Not `direct`. `loop` is the only rung whose failures are self-correcting, and
    the ladder escalates on not_found -- so a degraded router that picks `loop` still
    converges, while one that picks `direct` returns a confident wrong answer."""
    from jarvis import llm
    from jarvis.llm import LLMUnavailable
    monkeypatch.setattr(llm, "call",
                        lambda ctx, m, **kw: (_ for _ in ()).throw(
                            LLMUnavailable("429")))
    r = route("anything", ctx=_ctx())
    assert r.strategy == "loop"
    assert r.source == "fallback"
    assert "unavailable" in r.reason.lower()


def test_a_config_error_is_not_hidden_behind_a_degraded_route(monkeypatch):
    """A config error must not be papered over. An agent that appears to work and
    cannot answer is the worst failure mode for diagnosing it."""
    from jarvis import llm
    from jarvis.llm import LLMConfigError
    monkeypatch.setattr(llm, "call",
                        lambda ctx, m, **kw: (_ for _ in ()).throw(
                            LLMConfigError("model not found")))
    r = route("anything", ctx=_ctx())
    assert "CONFIG ERROR" in r.reason.upper()
    assert r.strategy == "loop"          # still runnable


def test_a_nonsense_strategy_is_not_coerced(monkeypatch):
    """A model naming a strategy that does not exist has not answered. Coercing it
    would let a typo silently pick a rung."""
    from jarvis import llm
    monkeypatch.setattr(llm, "call", lambda ctx, m, **kw: {
        "text": '{"strategy":"direct-but-fast","reason":"x"}', "usage": {}})
    r = route("q", ctx=_ctx())
    assert r.strategy == "loop"
    assert r.source == "fallback"
    assert "unrecognised" in r.reason


def test_prose_around_the_json_is_tolerated(monkeypatch):
    from jarvis import llm
    monkeypatch.setattr(llm, "call", lambda ctx, m, **kw: {
        "text": 'Sure!\n```json\n{"strategy":"plan","reason":"two parts"}\n```\nDone.',
        "usage": {}})
    assert route("q", ctx=_ctx()).strategy == "plan"


def test_routing_is_logged_so_a_bad_route_is_visible(monkeypatch):
    from jarvis import llm
    monkeypatch.setattr(llm, "call", lambda ctx, m, **kw: {
        "text": '{"strategy":"plan","reason":"multi-part"}', "usage": {}})
    ctx = _ctx()
    route("q", ctx=ctx)
    events = [e for e in ctx.trace.events if e.action == "judge"]
    assert events and events[-1].status == "ok"
    assert "plan" in events[-1].detail


def test_the_escalation_order_is_a_ladder():
    """direct -> loop -> plan -> loop. `plan` retries as `loop` rather than stopping,
    because an undecomposable plan is a loop's problem."""
    assert ESCALATION["direct"] == "loop"
    assert ESCALATION["loop"] == "plan"
    assert ESCALATION["plan"] == "loop"
    r = R.Route("direct", "x")
    assert r.escalate_to == "loop"
    assert R.Route("plan", "x").escalate_to == "loop"


def test_the_old_cue_list_is_gone():
    """The plan removes it. Asserted by ABSENCE on the module object, not on its
    source text.

    The text version fails on this very file: the module docstring explains what the
    cue list did and why it went, so `inspect.getsource` still contains the name.
    Checking the source for a string that a comment legitimately mentions tests the
    comment, not the code. `hasattr` is the question actually being asked: can anything
    still reach that decision table?
    """
    assert not hasattr(R, "needs_plan")
    assert not hasattr(R, "PLAN_NEEDED_CUES")
    assert "needs_plan" not in R.__all__

    # `AGGREGATE_CUES` legitimately exists -- it is the FAST PATH deciding whether a
    # query is an aggregate, not the router choosing a strategy. What must not exist is
    # a cue list consulted by `route()`, which `test_no_regex_decides_the_strategy`
    # already checks structurally. Here we only forbid a strategy-named cue table.
    for banned in ("STRATEGY_CUES", "PLAN_CUES", "ROUTE_CUES", "NEEDS_PLAN_CUES"):
        assert not hasattr(R, banned), f"{banned} is back"


def test_no_regex_decides_the_strategy(monkeypatch):
    """Structural. `route()` must not consult any of the fast-path helpers."""
    import inspect
    src = inspect.getsource(R.route)
    for banned in ("AGGREGATE_CUES", "BARE_FILENAME_RE", "FILENAME_RE", "STOPWORDS",
                   "CATEGORY_WORDS", "detect_query_type", "extract_intent"):
        assert banned not in src, f"route() consults {banned}"


# ------------------------------------------------------------------ the fast path
def test_agent_name_literal_is_the_one_spelling():
    """The prompt bundle stamps this so a trace can be traced to the agent."""
    assert AGENT_NAME_LITERAL == "searchFiles"


@pytest.mark.parametrize("query,shape", [
    ("where is safe_resolve defined?", "locate"),
    ("how many images are there?", "aggregate"),
])
def test_query_shape(query, shape):
    assert detect_query_type(query)["shape"] == shape


def test_aggregate_query_recognises_its_category():
    assert "image" in detect_query_type("how many images are there?")["categories"]


def test_aggregate_cues_are_real_phrases():
    assert "how many" in AGGREGATE_CUES


def test_extent_of_extraction_is_names_not_prose():
    """The plan's complaint about the old extractor: instruction prose leaked into
    keywords, and a keyword like `codebase` fuzzy-matched every src/ on disk."""
    got = extract_intent("Search the codebase for the definition of X")
    assert "search" not in got["keywords"]
    assert "codebase" not in got["keywords"]


def test_a_named_symbol_survives_extraction():
    assert "safe_resolve" in extract_intent("where is safe_resolve defined?")["keywords"]


def test_bare_filename_recogniser_matches_extensionless_names():
    """BARE_FILENAME_RE is for names with NO extension -- Dockerfile, Makefile. An
    extension is FILENAME_RE's job. Two overlapping patterns for one job is how a
    query gets matched twice and counted twice."""
    assert BARE_FILENAME_RE.search("Dockerfile")
    assert BARE_FILENAME_RE.search("Makefile")
    assert not BARE_FILENAME_RE.search("notes.txt")


def test_category_words_map_words_to_survey_categories():
    assert CATEGORY_WORDS["image"] == "image"
    assert CATEGORY_WORDS["files"] is None      # "files" must not narrow the survey


def test_extension_query_recognizes_plain_and_dotted_extensions():
    from jarvis.router import extract_extension_query

    assert extract_extension_query("can you find mp4 files?") == ".mp4"
    assert extract_extension_query("find *.MP4 files in Downloads") == ".mp4"
    assert extract_extension_query("is the mp4 codec installed?") is None