"""Environment facts, scratchpad, stuck detection, and observation paging.

Each of these exists because of a specific observed loop failure, and the test names
it. They are cheap to get wrong and their absence is invisible in the code -- a loop
with no scratchpad still runs, it just re-derives everything.

`page_observation` gets the most attention because the failure it prevents is the
worst kind: a model shown truncated output reports a smaller number as if it were
complete, confidently, with no way for the user to tell.
"""
from __future__ import annotations

import pytest

from jarvis.scratch import (Scratchpad, StuckDetector, environment_facts,
                            page_observation)


# ------------------------------------------------------------------ facts
def test_environment_facts_are_read_from_the_machine():
    """Not hardcoded. A prompt claiming `rg` exists on a machine without it sends the
    model to a command that 400s."""
    from pathlib import Path
    f = environment_facts(read_roots=(Path("/tmp"),))
    assert "Linux" in f or "Darwin" in f or "Windows" in f
    assert "/tmp" in f
    # Every tool named as available must actually resolve.
    import shutil
    avail = f.split("available ")[1].splitlines()[0]
    for tool in [t.strip() for t in avail.split(",") if t.strip()]:
        assert shutil.which(tool), f"prompt claims {tool} exists; it does not"


def test_environment_facts_say_what_is_absent():
    """The absence is the useful half. A model told what is NOT available stops
    proposing it, and the tools it proposes are the ones that get refused."""
    f = environment_facts()
    assert "absent" in f


def test_environment_facts_explain_the_no_shell_rule():
    """Every early refusal was a model reaching for a pipe, a redirect, or `bash -c`.
    It did not know a shell was unavailable."""
    f = environment_facts().lower()
    assert "no shell" in f
    assert "exclusion" in f, "the model must know the policy adds prune flags for it"


def test_environment_facts_are_capped():
    """A context that grows without bound eventually contains more instruction than
    task, and the task is what gets forgotten."""
    assert len(environment_facts()) <= 2000


# ------------------------------------------------------------------ scratchpad
def test_scratchpad_remembers_facts():
    s = Scratchpad()
    s.learn("there is no .ssh in the read scope")
    s.learn("the largest file is large_module.py")
    assert "no .ssh" in s.render()
    assert "large_module.py" in s.render()


def test_scratchpad_deduplicates():
    s = Scratchpad()
    for _ in range(5):
        s.learn("same fact")
    assert s.render().count("same fact") == 1


def test_scratchpad_records_dead_ends_separately():
    """A dead end is not a fact. Mixing them means the model re-runs the command it
    already learned failed."""
    s = Scratchpad()
    s.learn("the archive is a zip")
    s.dead_end("du -sh . -> refused: needs exclusions")
    r = s.render()
    assert "ESTABLISHED" in r and "DO NOT RETRY" in r
    assert "du -sh" in r.split("DO NOT RETRY")[1]


def test_scratchpad_is_bounded():
    s = Scratchpad()
    for i in range(200):
        s.learn(f"fact number {i}")
    assert len(s.facts) <= 24


def test_scratchpad_is_not_shared_between_loops():
    """A module-level scratchpad would carry one task's filesystem into the next one's
    context -- a privacy leak AND a correctness bug, because task 2 would refuse paths
    task 1 had cleared."""
    from jarvis.context import new_task
    a = Scratchpad()
    b = Scratchpad()
    a.learn("private fact about task a")
    assert "private fact" not in b.render()


# ------------------------------------------------------------------ stuck detection
def test_stuck_needs_three_similar_turns():
    """Two is a normal retry. One is not stuck at all."""
    d = StuckDetector(threshold=3)
    assert not d.observe("fail", "ls -la")
    assert not d.observe("fail", "ls -la")
    assert d.observe("fail", "ls -la")


def test_a_new_command_is_not_stuck():
    """Same verdict, different command: that is progress, not a loop."""
    d = StuckDetector(threshold=3)
    d.observe("fail", "ls -la")
    d.observe("fail", "du -sh .")
    assert not d.observe("fail", "wc -l f")


def test_a_new_verdict_is_not_stuck():
    d = StuckDetector(threshold=3)
    d.observe("fail", "ls -la")
    d.observe("unsure", "ls -la")
    assert not d.observe("pass", "ls -la")


def test_stuck_explains_itself():
    """A stop with no explanation is indistinguishable from a crash."""
    d = StuckDetector(threshold=3)
    for _ in range(3):
        d.observe("fail", "ls -la")
    why = d.explain()
    assert "stuck" in why and "fail" in why


# ------------------------------------------------------------------ paging
def test_paging_states_how_much_was_hidden():
    """THE point. A truncated observation with no count is indistinguishable from a
    complete one, and the model reports the smaller number as the total."""
    text = "\n".join(f"file{i}.txt" for i in range(500))
    out = page_observation(text, limit=10)
    assert "490 more line(s) not shown" in out
    assert "file0.txt" in out
    assert "file499.txt" not in out


def test_paging_tells_the_model_what_to_do_about_it():
    out = page_observation("\n".join(str(i) for i in range(100)), limit=5)
    assert "narrow" in out.lower()


def test_short_output_is_not_paged():
    assert page_observation("a\nb\nc", limit=60) == "a\nb\nc"


def test_paging_never_grows_the_text():
    """The cap must hold. A "summary" longer than the original is worse than none."""
    text = "\n".join(f"line {i}" for i in range(10000))
    assert len(page_observation(text, limit=20)) < len(text)


def test_paging_a_window():
    text = "\n".join(str(i) for i in range(100))
    out = page_observation(text, window=(10, 20))
    assert "10" in out and "19" in out
    assert "80 more line(s)" in out
    assert "10 earlier line(s)" in out