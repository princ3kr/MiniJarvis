"""The trace: one append-only record of what a task actually did.

This replaces the notebook's four module-level globals -- `LOGS`, `PATH_LOG`,
`_GATEWAY_CALLS`, and `_LAST_*`. They were shared mutable state, which meant a
second task in the same kernel silently inherited the first task's budget state and
log, and forced a `gateway_reset()` call that every caller had to remember.

Two record kinds, deliberately separate:
  * Event  -- something happened (tool ran, llm replied, policy said no)
  * Attempt -- the agent TRIED something and it was judged. This is the unit the
    planner's do-not-repeat list is built from.

A Trace belongs to exactly one TaskContext. Nothing here is global.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    pass_ = "pass"
    fail = "fail"
    unsure = "unsure"


class FailureType(str, Enum):
    none = "none"
    empty = "empty"                 # the agent returned nothing usable
    wrong = "wrong"                 # returned something off-target
    error = "error"                 # tool or LLM blew up
    timeout = "timeout"
    contradiction = "contradiction"  # conflicts with a verified earlier result
    repeated = "repeated"           # rejected as a retry of a known dead end
    inconclusive = "inconclusive"   # verifier unsure: not wrong, just unproven


@dataclass
class Event:
    n: int
    kind: str                       # "tool" | "llm" | "policy" | "gateway" | "route"
    action: str
    status: str = "ok"
    detail: str = ""
    duration_ms: float | None = None
    ts: str = field(default_factory=lambda: time.strftime("%H:%M:%S"))
    extra: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        dur = f"{self.duration_ms:.0f}ms" if self.duration_ms is not None else "-"
        return (f"{self.n:>3} {self.kind:<8} {self.action:<22} {self.status:<18} "
                f"{dur:>8}  {self.detail[:60]}")


@dataclass
class Attempt:
    """One agent attempt and its verdict. `compact()` is what a replan sees."""

    attempt_id: int
    plan_version: int
    step_id: int
    agent: str
    instruction: str
    status: str
    result_summary: str
    verdict: Verdict
    failure_reason: str = ""
    failure_type: FailureType = FailureType.none
    evidence_checked: str = ""
    duration_ms: float | None = None

    def compact(self) -> str:
        if self.verdict is Verdict.pass_:
            return f"step {self.step_id} [{self.agent}] OK: {self.result_summary[:70]}"
        return (f"DEAD END step {self.step_id} [{self.agent}] "
                f"{self.instruction[:60]!r} -> {self.failure_type.value}: "
                f"{self.failure_reason[:70]}")

    def signature(self) -> tuple[str, str]:
        """Normalised identity, used to reject a repeat of a known dead end."""
        return (self.agent, re.sub(r"\s+", " ", self.instruction.strip().lower()))


class Trace:
    """Append-only, per task. Cheap to build, cheap to read in a REPL."""

    def __init__(self, keep_excerpts: bool = True) -> None:
        self.events: list[Event] = []
        self.attempts: list[Attempt] = []
        self._keep_excerpts = keep_excerpts

    # ------------------------------------------------------------------ write
    def event(self, kind: str, action: str, *, detail: str = "", status: str = "ok",
              t0: float | None = None, **extra) -> Event:
        dur = round((time.perf_counter() - t0) * 1000, 2) if t0 else None
        ev = Event(n=len(self.events) + 1, kind=kind, action=action, status=status,
                   detail=detail, duration_ms=dur, extra=extra)
        self.events.append(ev)
        return ev

    def attempt(self, *, plan_version: int, step_id: int, agent: str, instruction: str,
                status: str, result_summary: str, verdict: Verdict,
                failure_reason: str = "", failure_type: FailureType = FailureType.none,
                evidence_checked: str = "", duration_ms: float | None = None) -> Attempt:
        a = Attempt(attempt_id=len(self.attempts) + 1, plan_version=plan_version,
                    step_id=step_id, agent=agent, instruction=instruction,
                    status=status, result_summary=result_summary, verdict=verdict,
                    failure_reason=failure_reason, failure_type=failure_type,
                    evidence_checked=evidence_checked, duration_ms=duration_ms)
        self.attempts.append(a)
        self.event("attempt", f"{agent}", status=verdict.value,
                   detail=failure_reason or result_summary,
                   attempt_id=a.attempt_id, step_id=step_id)
        return a

    # ------------------------------------------------------------------- read
    def dead_ends(self) -> list[str]:
        """The compact do-not-repeat list handed to a replanner.

        Only genuine FAILURES belong here. An `inconclusive` attempt was never shown
        to be wrong -- including it would tell the planner "don't repeat this" about
        a step that simply could not be checked, which is how a planner learns to
        avoid correct work.
        """
        return [a.compact() for a in self.attempts if a.verdict is Verdict.fail]

    def signatures(self) -> set[tuple[str, str]]:
        return {a.signature() for a in self.attempts}

    def verified_summary(self) -> str:
        ok = [a for a in self.attempts if a.verdict is Verdict.pass_]
        return "; ".join(f"step {a.step_id}: {a.result_summary[:120]}" for a in ok) or "none"

    def render(self, last: int | None = None) -> str:
        rows = self.events if last is None else self.events[-last:]
        if not rows:
            return "trace is empty."
        out = [f"{'#':>3} {'kind':<8} {'action':<22} {'status':<18} {'time':>8}  detail",
               "-" * 100]
        out += [e.render() for e in rows]
        out.append("-" * 100)
        ok = sum(1 for e in self.events if e.status == "ok")
        out.append(f"{len(rows)} event(s): {ok} ok, {len(rows) - ok} not ok")
        return "\n".join(out)

    def render_attempts(self, last: int | None = None) -> str:
        rows = self.attempts if last is None else self.attempts[-last:]
        if not rows:
            return "no attempts recorded."
        out = [f"{'#':>3} {'plan':>4} {'step':>4} {'agent':<12} {'verdict':<7} "
               f"{'fail_type':<13} detail", "-" * 100]
        for a in rows:
            detail = a.result_summary or a.failure_reason
            out.append(f"{a.attempt_id:>3} v{a.plan_version:<3} {a.step_id:>4} "
                       f"{a.agent:<12} {a.verdict.value:<7} {a.failure_type.value:<13} "
                       f"{str(detail)[:44]}")
        out.append("-" * 100)
        ok = sum(1 for a in rows if a.verdict is Verdict.pass_)
        out.append(f"{len(rows)} attempt(s): {ok} pass, {len(rows) - ok} not")
        return "\n".join(out)