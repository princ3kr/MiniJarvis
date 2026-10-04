"""Router tests: the plan gate, the aggregate/locate split, and intent extraction.

Ported from `test_searchfiles.py` (the aggregate routing, intent, and scope tests) and
`test_planner.py` (the needs_plan gate), adapted to the new module layout.

Two of these tests assert that a KNOWN BUG is still present. That is deliberate: they are
the tripwire Phase 1 fires against. When Phase 1 replaces the gate, `test_needs_plan_gate`
is where the new expectation goes, and these two are deleted.
"""
from __future__ import annotations

import pytest

from jarvis import router
from jarvis.router import (AGENT_NAME_LITERAL, AGGREGATE_CUES, BARE_FILENAME_RE,
                           CATEGORY_WORDS, FILENAME_RE, PLAN_NEEDED_CUES, STOPWORDS,
                           detect_query_type, extract_intent, needs_plan)


# --------------------------------------------------------------- the plan gate
def test_needs_plan_gate(ctx):
    single, r1 = needs_plan("where is safe_resolve defined?", ctx=ctx)
    multi, r2 = needs_plan("find safe_resolve in main.py then count images", ctx=ctx)
    assert single is False, r1
    assert multi is True, r2
    assert r1 == "single-step" and "cue=" in r2


def test_needs_plan_counts_multiple_named_files(ctx):
    decided, why = needs_plan("open util.py or main.py", ctx=ctx)
    assert decided is True and why == "multiple files named", why


def test_needs_plan_gate_never_selects_react():
    """KNOWN BUG (removal scheduled, Phase 1): the gate returns a bool, so there is no
    value it can return that reaches the ReAct strategy. `run_task(strategy='auto')`
    can only ever choose direct or plan."""
    # every query shape the gate knows about collapses to a two-valued answer
    for q in ["hello", "find main.py", "count the images then report", "a.py and b.py"]:
        out = needs_plan(q)
        assert isinstance(out[0], bool)
        assert len(out) == 2
    # and the auto branch in run_task only ever assigns two strategies
    from jarvis.engine import run_task
    src = run_task.__doc__
    assert "react  -- dspy.ReAct adaptive loop" in src


def test_needs_plan_cue_is_a_substring_not_a_phrase():
    """KNOWN BUG (removal scheduled, Phase 1): cues are matched with `in`, not as
    phrases, so `both` fires on any query containing those four letters anywhere."""
    assert "both" in PLAN_NEEDED_CUES
    # 'across' is inside 'across_the_pacific_notes.md' -- no plan intent whatsoever
    assert needs_plan("read across_the_pacific_notes.md")[0] is True
    # and 'then' fires on the word 'strengthen'
    assert needs_plan("how do I strengthen the search ranking")[0] is True


def test_needs_plan_logs_to_the_task_trace(ctx):
    needs_plan("find main.py then summarise", ctx=ctx)
    assert any(e.action == "gate" for e in ctx.trace.events)


# --------------------------------------------------------------- aggregate vs locate
def test_aggregate_locate_routing():
    qt = detect_query_type("how many images are there in the system?")
    assert qt["shape"] == "aggregate", qt
    assert qt["categories"] == ["image"], qt
    qt2 = detect_query_type("count the python files")
    assert qt2["shape"] == "aggregate" and qt2["categories"] == ["code"], qt2
    qt3 = detect_query_type("where is safe_resolve defined?")
    assert qt3["shape"] == "locate", qt3
    qt4 = detect_query_type("show me glinerTest.ipynb")
    assert qt4["shape"] == "locate", qt4


def test_detect_query_type_reports_the_cue_that_fired():
    qt = detect_query_type("give me a breakdown of the video files")
    assert qt["cue_hit"] == "breakdown", qt
    assert qt["categories"] == ["video"], qt


def test_generic_file_words_carry_no_category():
    """`file`/`files` map to None on purpose: a query about "files" must not narrow the
    survey to one category, or "how many files?" reports only the code count."""
    assert CATEGORY_WORDS["file"] is None and CATEGORY_WORDS["files"] is None
    qt = detect_query_type("how many files are there?")
    assert qt["shape"] == "aggregate" and qt["categories"] == [], qt


def test_aggregate_cues_cover_counts_and_breakdowns():
    for cue in ["how many", "count", "breakdown", "what type", "statistics"]:
        assert cue in AGGREGATE_CUES
        assert detect_query_type(f"{cue} of the images")["aggregate"] is True


# --------------------------------------------------------------- intent extraction
def test_intent_captures_archives():
    got = extract_intent("what is inside Ask-My-Repo-mainV2.zip?")
    assert got["filenames"] == ["Ask-My-Repo-mainV2.zip"], got["filenames"]
    # the filename fragment must not leak into keywords, or a junk folder search runs
    assert "Ask" not in got["keywords"], got["keywords"]
    assert not any(k.lower() == "ask" for k in got["keywords"]), got["keywords"]


def test_intent_captures_common_types():
    cases = {"read notes.pdf": "notes.pdf",
             "open AI_ML_Roadmap.docx": "AI_ML_Roadmap.docx",
             "show main.py": "main.py",
             "check config.yaml": "config.yaml",
             "summarise data.csv": "data.csv",
             "play clip.mp4": "clip.mp4",
             "run the Dockerfile": "Dockerfile"}
    for q, want in cases.items():
        got = extract_intent(q)["filenames"]
        assert want in got, f"{q!r} -> {got}, expected {want!r}"


def test_intent_never_reports_the_same_filename_twice():
    got = extract_intent("show main.py and then MAIN.PY again")
    assert got["filenames"].count("main.py") == 1, got["filenames"]


def test_planner_prose_does_not_become_a_folder_hint():
    it = extract_intent("Search the codebase for the definition of `safe_resolve`")
    lowered = [k.lower() for k in it["keywords"]]
    assert "search" not in lowered, it["keywords"]
    assert "codebase" not in lowered, it["keywords"]
    assert "definition" not in lowered, it["keywords"]


def test_intent_keeps_the_real_subject():
    it = extract_intent("Search the codebase for the definition of `safe_resolve`")
    assert any("safe_resolve" in k for k in it["keywords"]), it["keywords"]
    assert len(it["keywords"]) <= 5, it["keywords"]


def test_intent_captures_bare_names_without_an_extension():
    assert "Dockerfile" in extract_intent("rebuild the Dockerfile")["filenames"]
    assert "Makefile" in extract_intent("what does the Makefile do")["filenames"]


def test_filename_regex_covers_archives_and_media():
    """The regression: `.zip` was missing from FILENAME_EXTS, so an archive question
    captured no filename at all and fell through to fuzzy-matching a fragment."""
    for ext in ["zip", "tar", "gz", "pdf", "docx", "png", "mp4", "ipynb"]:
        assert FILENAME_RE.search(f"thing.{ext}"), ext


def test_stopwords_cover_the_planner_instruction_vocabulary():
    for word in ["search", "codebase", "definition", "locate", "repository", "folder"]:
        assert word in STOPWORDS, word


def test_agent_name_literal_is_searchfiles():
    assert AGENT_NAME_LITERAL == "searchFiles"
