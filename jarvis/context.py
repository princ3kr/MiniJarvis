"""TaskContext: the per-task owner of budgets, counters, and the trace.

What it replaces
----------------
The notebook kept this state in module globals:

    LOGS, PATH_LOG, _GATEWAY_CALLS, APPROVAL_GRANTED, _LAST_EXIT, _LAST_OUT, _LAST_ERR

which caused three concrete bugs:

1. `gateway_reset()` had to be called once per task, and every caller had to
   remember. Forgetting it made a multi-step plan exhaust max_calls on step two.
2. Two tasks in one kernel shared a call budget and a log, so results interleaved.
3. `_LAST_EXIT` / `_LAST_OUT` / `_LAST_ERR` held the previous command's output, so a
   postcondition could be evaluated against the wrong command's stdout.

Every one of those is a missing parameter, not a missing lock. So this is a
parameter: constructed once per task, passed down explicitly, never mutated into a
module.

Budget accounting fails CLOSED. If the wall clock or the token ceiling is spent,
`charge_llm` raises rather than silently truncating -- a loop that cannot afford its
next call must stop and report honestly, not half-answer.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Budget, Settings, load_settings
from .trace import Attempt, Trace, Verdict


class BudgetExhausted(RuntimeError):
    """Raised when a task runs out of a resource. The caller must stop and report.

    Not an error in the agent's reasoning -- a fact about the world. The engine
    surfaces it as `inconclusive` with the trace attached, never as a crash.
    """


@dataclass
class Counters:
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    iterations: int = 0
    replans: int = 0
    # per "agent:tool", replacing the _GATEWAY_CALLS dict
    gateway: dict[str, int] = field(default_factory=dict)
    denied: int = 0
    timeouts: int = 0


@dataclass
class TaskContext:
    query: str
    settings: Settings
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    root: Path | None = None                       # None -> first read root
    budget: Budget = None                          # type: ignore[assignment]
    trace: Trace = field(default_factory=Trace)
    counters: Counters = field(default_factory=Counters)
    # Exact argv fingerprints a human approved. Never "allow all": a grant for
    # `rm build/output` says nothing about `rm -rf /`.
    approvals: set[str] = field(default_factory=set)
    # Scratch for the command currently being evaluated. Was _LAST_EXIT/_OUT/_ERR,
    # which made postconditions read the previous command's output.
    last_exit: str = "0"
    last_stdout: str = ""
    last_stderr: str = ""
    started: float = field(default_factory=time.perf_counter)
    finished_at: float | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.budget is None:
            self.budget = self.settings.budget
        if self.root is None:
            self.root = self.settings.read_roots[0]

    # ------------------------------------------------------------- lifecycle
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started

    def out_of_time(self) -> bool:
        return self.elapsed_s() >= self.budget.wall_clock_s

    def exhausted(self) -> str | None:
        """First exhausted budget, or None. Order matters: time is reported before
        counters because a timeout usually explains the counter overshoot too."""
        if self.out_of_time():
            return (f"wall clock {self.budget.wall_clock_s:.0f}s exceeded "
                    f"({self.elapsed_s():.0f}s)")
        if self.counters.tokens >= self.budget.max_tokens:
            return f"token ceiling {self.budget.max_tokens} reached"
        if self.counters.llm_calls >= self.budget.max_llm_calls:
            return f"llm call ceiling {self.budget.max_llm_calls} reached"
        if self.counters.tool_calls >= self.budget.max_tool_calls:
            return f"tool call ceiling {self.budget.max_tool_calls} reached"
        if self.counters.iterations >= self.budget.max_iterations:
            return f"iteration ceiling {self.budget.max_iterations} reached"
        return None

    def check(self) -> None:
        """Raise if anything is spent. Called before each LLM call and each iteration."""
        why = self.exhausted()
        if why:
            raise BudgetExhausted(why)

    # ---------------------------------------------------------------- charges
    def charge_llm(self, tokens: int = 0) -> None:
        """Account for one LLM call. Raises BEFORE the call if the budget is spent."""
        self.check()
        self.counters.llm_calls += 1
        self.counters.tokens += max(0, int(tokens or 0))
        if self.counters.tokens >= self.budget.max_tokens:
            raise BudgetExhausted(
                f"token ceiling {self.budget.max_tokens} reached")

    def charge_tool(self, name: str, n: int = 1) -> None:
        self.counters.tool_calls += n
        key = f"{name}"
        self.counters.gateway[key] = self.counters.gateway.get(key, 0) + n

    def charge_iteration(self) -> None:
        self.check()
        self.counters.iterations += 1

    def charge_replan(self) -> None:
        self.counters.replans += 1
        if self.counters.replans > self.budget.max_replans:
            raise BudgetExhausted(
                f"replan ceiling {self.budget.max_replans} reached")

    def tool_calls(self, name: str) -> int:
        return self.counters.gateway.get(name, 0)

    # ------------------------------------------------------------- approvals
    @staticmethod
    def fingerprint(argv: list[str]) -> str:
        """Exact-argv grant key.

        Delimited by \x00 rather than spaces on purpose. With spaces, argv
        ["a b"] and ["a", "b"] produce the SAME fingerprint, so an approval for one
        command silently authorises a different one. That was a real probe finding.
        """
        return "\x00".join(argv)

    def approve(self, argv: list[str]) -> None:
        self.approvals.add(self.fingerprint(argv))
        self.trace.event("gateway", "approve", detail=" ".join(argv)[:60],
                         argv=list(argv))

    def is_approved(self, argv: list[str]) -> bool:
        return self.fingerprint(argv) in self.approvals

    # ------------------------------------------------------------ trace sugar
    def log(self, kind: str, action: str, **kw) -> None:
        self.trace.event(kind, action, **kw)

    def attempt(self, **kw) -> Attempt:
        return self.trace.attempt(**kw)

    def note(self, text: str) -> None:
        self.notes.append(text)
        self.trace.event("note", "note", detail=text)

    # ----------------------------------------------------------------- finish
    def close(self, status: str = "ok") -> "TaskContext":
        self.finished_at = time.perf_counter()
        self.trace.event("task", "close", status=status,
                         detail=f"{self.counters.llm_calls} llm, "
                                f"{self.counters.tool_calls} tool, "
                                f"{self.counters.tokens} tok, "
                                f"{self.elapsed_s():.0f}s")
        return self

    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "elapsed_s": round(self.elapsed_s(), 2),
            "llm_calls": self.counters.llm_calls,
            "tool_calls": self.counters.tool_calls,
            "tokens": self.counters.tokens,
            "iterations": self.counters.iterations,
            "replans": self.counters.replans,
            "denied": self.counters.denied,
            "timeouts": self.counters.timeouts,
            "dead_ends": len(self.trace.dead_ends()),
            "budget_stop": self.exhausted(),
        }

    def __enter__(self) -> "TaskContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is BudgetExhausted:
            self.note(f"budget exhausted: {exc}")
            return True                     # a spent budget is an outcome, not a crash
        self.close("error" if exc_type else "ok")
        return False


def new_task(query: str, **kw) -> TaskContext:
    """The one constructor. `settings=` overrides win over the environment."""
    settings = kw.pop("settings", None) or load_settings()
    return TaskContext(query=query, settings=settings, **kw)


def venv_python() -> str | None:
    """The project venv interpreter, if one exists.

    Only used for diagnostics. dspy/pydantic are in `.venv` and NOT in system
    python, which is why the notebook silently degraded to deterministic behaviour
    rather than crashing.
    """
    from .config import PROJECT_ROOT
    p = PROJECT_ROOT / ".venv" / "bin" / "python"
    return str(p) if p.exists() else None


def running_in_venv() -> bool:
    """True when running inside a virtualenv.

    Matters because dspy and pydantic are in `.venv` and NOT in system python: under
    system python the planner and ReAct silently fall back to deterministic
    behaviour. Nothing crashes, you just get a different (worse) agent.
    """
    import sys
    return (Path(sys.prefix) / "pyvenv.cfg").exists()