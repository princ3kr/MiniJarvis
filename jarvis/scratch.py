"""What the model is told before it chooses anything: the machine, the rules, the rules
it has already broken.

The plan asks for this, and the reasoning is specific. The loop's first failures were
all the same shape: the model ran `bash -lc '... | xargs ...'` because it did not know
whether a shell was available, tried to add its own prune flags because it did not know
the policy already adds them, and repeated a refused command because nothing in its
context said that command had been refused. All three are missing-context failures, not
reasoning failures.

Three things are assembled here, once per loop:

  * **environment facts.** OS, which tools exist, what the read scope is, what the
    policy does for it, today's date. Facts the model would otherwise guess, and
    guesses about a filesystem are how a path gets invented.
  * **a scratchpad.** Facts it established that are still true, so it does not
    re-derive them. This is also where "I already know there is no X" goes, which is
    what stops it retrying a dead end.
  * **stuck detection.** When several consecutive turns produce the same verdict and
    no new path, the loop stops and says so, rather than spending its budget
    re-deciding the same way.

The facts are read from the live machine, never hardcoded, and are truncated to a
budget. A context that grows without bound is a context that eventually contains more
instruction than task.
"""
from __future__ import annotations

import os
import platform
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

# Tools the loop should know exist. Checked, not assumed.
_WATCHED_TOOLS = ("ls", "cat", "find", "grep", "sed", "awk", "wc", "head", "tail",
                  "du", "df", "stat", "file", "tree", "sort", "uniq", "cut", "tr",
                  "diff", "git", "sha256sum", "basename", "dirname", "realpath",
                  "readlink", "rg", "fd", "jq", "which")

MAX_FACT_CHARS = 1800
MAX_SCRATCH_ITEMS = 24


def environment_facts(*, read_roots: tuple[Path, ...] | None = None,
                      max_chars: int = MAX_FACT_CHARS) -> str:
    """What this machine is, as one compact block. Read live, never hardcoded."""
    have = [t for t in _WATCHED_TOOLS if shutil.which(t)]
    missing = [t for t in _WATCHED_TOOLS if not shutil.which(t)]
    roots = ", ".join(str(r) for r in (read_roots or ())) or "(unset)"
    lines = [
        f"OS        {platform.system()} {platform.release()} ({platform.machine()})",
        f"available {', '.join(have)}",
        f"absent    {', '.join(missing) or '(none)'}",
        f"read scope {roots}",
        f"today     {time.strftime('%Y-%m-%d')}",
        "",
        "How to run things:",
        "  argv is a list, one program per turn. NO shell: pipes, &&, >, <, backticks",
        "  and globs are all rejected, and 'bash -c ...' is an interpreter (approval).",
        "  Use one program and read its output. To filter, narrow the arguments.",
        "  The policy appends deny-list exclusions to any program that walks",
        "  directories. Write 'find . -type f -name \"*.py\"' and it is made safe.",
        "  A command may be REFUSED with a reason. Read it and change approach.",
    ]
    out = "\n".join(lines)
    return out[:max_chars]


@dataclass
class Scratchpad:
    """Facts that are still true, and dead ends that should not be retried.

    Per loop, never per module: a module-level scratchpad would carry one task's
    filesystem into the next one's context, which is both a privacy leak and a
    correctness bug -- the second task would refuse paths the first one had cleared.
    """

    facts: list[str] = field(default_factory=list)
    dead_ends: list[str] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    def learn(self, fact: str) -> None:
        """Record something established. Deduplicated, and capped."""
        f = (fact or "").strip()[:220]
        if not f or f in self._seen:
            return
        self._seen.add(f)
        self.facts.append(f)
        if len(self.facts) > MAX_SCRATCH_ITEMS:
            dropped = self.facts.pop(0)
            self._seen.discard(dropped)

    def dead_end(self, what: str) -> None:
        w = (what or "").strip()[:220]
        if w and w not in self.dead_ends:
            self.dead_ends.append(w)
            if len(self.dead_ends) > MAX_SCRATCH_ITEMS:
                self.dead_ends.pop(0)

    def render(self, max_chars: int = 1200) -> str:
        if not self.facts and not self.dead_ends:
            return ""
        out = []
        if self.facts:
            out.append("ESTABLISHED:")
            out += [f"  - {f}" for f in self.facts[-12:]]
        if self.dead_ends:
            out.append("DO NOT RETRY:")
            out += [f"  - {d}" for d in self.dead_ends[-12:]]
        return "\n".join(out)[:max_chars]


@dataclass
class StuckDetector:
    """Is the loop going anywhere?

    Three consecutive turns with the same verdict AND no new path or command shape is
    stuck. Without this the loop spends its whole budget re-deciding the same way and
    reports `inconclusive` having learned nothing -- which is indistinguishable, from
    the outside, from a genuinely hard question.

    The threshold is 3 because 2 is a normal retry and 4 is a lot of budget to waste.
    It is a parameter, not a constant, because the right value depends on the budget
    and the caller knows that.
    """

    threshold: int = 3
    _verdicts: list[str] = field(default_factory=list)
    _commands: list[str] = field(default_factory=list)

    def observe(self, verdict: str, command: str) -> bool:
        """Record a turn. Returns True when it looks stuck."""
        self._verdicts.append(verdict)
        self._commands.append(command)
        if len(self._verdicts) < self.threshold:
            return False
        window = slice(-self.threshold, None)
        recent_v = self._verdicts[window]
        recent_c = self._commands[window]
        # Same verdict every time, and not one new command in the window.
        return len(set(recent_v)) == 1 and len(set(recent_c)) == 1

    def explain(self) -> str:
        if not self._verdicts:
            return ""
        v = self._verdicts[-1]
        return (f"stuck: the last {self.threshold} turns were all {v} with the same "
                f"command shape, so more turns of the same will not help")


def page_observation(text: str, *, limit: int = 60,
                     window: tuple[int, int] | None = None) -> str:
    """Cap an observation and say how much was left out.

    The count is not decoration. A model shown a truncated `find` output cannot tell
    "there are no more matches" from "the rest was cut off", and it will usually guess
    the flattering one -- reporting 12 files when there were 400. Saying "480 more
    lines not shown" lets the model decide whether to narrow the search, which is the
    behaviour that actually helps.
    """
    if not text:
        return text
    lines = text.splitlines()
    if window is not None:
        lo, hi = window
        shown = lines[max(0, lo):hi]
        before, after = max(0, lo), max(0, len(lines) - hi)
        note = []
        if before:
            note.append(f"{before} earlier line(s)")
        if after:
            note.append(f"{after} more line(s)")
        tail = f"\n... [{', '.join(note)} not shown. Narrow the search to see more.]" \
            if note else ""
        return "\n".join(shown) + tail
    if len(lines) <= limit:
        return text
    hidden = len(lines) - limit
    return ("\n".join(lines[:limit])
            + f"\n... [{hidden} more line(s) not shown. Narrow the search, or use "
              f"-maxdepth / a specific path, to see them.]")