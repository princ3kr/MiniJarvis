"""The interactive console: ask questions, watch every step, read every command.

This is the primary interface, not a wrapper around a library. Three requirements drove
its shape, and each one is a consequence of the agent being untrusted:

**Complete transparency.** Every LLM call, every tool call, every command with its REAL
argv, every verdict, and the running cost are printed as they happen. Not summarised
at the end. An agent that runs commands on your machine and then shows you a paragraph
is asking for trust it has not earned; showing the raw argv is the cheapest way to let
a person check for themselves. `--quiet` exists for when you do not want it, and it is
opt-in.

**Nothing is hidden behind "unavailable".** A refused command prints the refusal
reason. A verification failure prints what was checked and what disagreed. A `not
found` prints the list of what was tried. Silence is the one thing that would be a lie,
because it is indistinguishable from "it worked".

**Answers are labelled by how they were earned.** A verified answer, an unverified one
from `--no-verify`, and a `not found` look different, because they are different
things. The plan is explicit that unverified text must never be shown as an answer.

Everything is configurable from `.env`; nothing here hardcodes a model, a key, a
scope, or a limit. The REPL itself is a `cmd.Cmd`, so `help`, `trace`, `plan`, `exit`
and friends work as you would expect, and `?` inside a query is not special.
"""
from __future__ import annotations

import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

# ANSI, and disabled when not a tty or when NO_COLOR is set. A log that cannot be read
# because colour codes are interleaved with traceback text is worse than a plain one.
_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


DIM = lambda s: _c("2", s)          # noqa: E731
BOLD = lambda s: _c("1", s)         # noqa: E731
GREEN = lambda s: _c("32", s)       # noqa: E731
YELLOW = lambda s: _c("33", s)      # noqa: E731
RED = lambda s: _c("31", s)         # noqa: E731
CYAN = lambda s: _c("36", s)        # noqa: E731
MAGENTA = lambda s: _c("35", s)     # noqa: E731

STATUS_COLOR = {"ok": GREEN, "blocked": RED, "error": RED, "timeout": YELLOW,
                "denied": RED, "dry_run": YELLOW, "uncertain": YELLOW,
                "not_found": YELLOW, "near_miss": YELLOW, "fallback": YELLOW,
                "config_error": RED, "budget": YELLOW, "parse": YELLOW}


class Console:
    """Rendering. Owns no task state; every method takes what it prints as arguments.

    That is not fussiness. The console is imported by the tests to assert that a
    refusal reason is actually SHOWN, and a renderer that accumulated state between
    calls would make those assertions depend on call order.
    """

    def __init__(self, *, quiet: bool = False, show_trace: bool = True,
                 stream=None) -> None:
        self.quiet = quiet
        self.show_trace = show_trace
        self.out = stream or sys.stdout
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------ output
    def _p(self, s: str = "") -> None:
        print(s, file=self.out, flush=True)

    def info(self, s: str) -> None:
        if not self.quiet:
            self._p(s)

    def rule(self, title: str = "") -> None:
        if self.quiet:
            return
        self._p()
        self._p(BOLD(f"-- {title} " + "-" * max(0, 62 - len(title))) if title
                else DIM("-" * 68))

    def banner(self, version: str = "") -> None:
        self._p()
        self._p(BOLD("  jarvis") + DIM(f"  read-only file agent {version}"))
        self._p(DIM("  ask a question.  /help for commands.  ctrl-d or /exit to quit."))
        self._p()

    # ------------------------------------------------------------------ events
    def route(self, strategy: str, reason: str, fallback: bool = False) -> None:
        tag = CYAN(f"route->{strategy}")
        if fallback:
            tag += YELLOW(" (fallback)")
        self.info(f"{DIM('*')} {tag}  {DIM(reason)}")

    def stage(self, stage: str, outcome: str, reason: str = "") -> None:
        col = STATUS_COLOR.get(outcome, lambda s: s)
        self.info(f"{DIM('*')} {MAGENTA(f'[{stage}]')} {col(outcome)}"
                  + (f"  {DIM(reason)}" if reason else ""))

    def llm(self, purpose: str, model: str, detail: str, tokens: Any = None,
            ms: float | None = None) -> None:
        if self.quiet:
            return
        bits = [DIM(model)]
        if tokens not in (None, "", "?"):
            bits.append(DIM(f"{tokens} tok"))
        if ms is not None:
            bits.append(DIM(f"{ms:.0f}ms"))
        self._p(f"  {CYAN('llm')}  {purpose:<16} {'  '.join(bits)}  {detail}")

    def command(self, argv: list[str], *, status: str, reason: str = "",
                ms: float | None = None, tier: int | None = None,
                output: str = "", truncated: bool = False) -> None:
        """The REAL argv, always. This is the transparency the whole thing rests on."""
        if self.quiet:
            return
        col = STATUS_COLOR.get(status, lambda s: s)
        printable = " ".join(shlex.quote(a) for a in argv)
        head = f"  {GREEN('$')} {BOLD(printable)}"
        bits = [col(status)]
        if tier is not None:
            bits.append(DIM(f"tier {tier}"))
        if ms is not None:
            bits.append(DIM(f"{ms:.0f}ms"))
        if truncated:
            bits.append(YELLOW("truncated"))
        if reason:
            bits.append(DIM(reason))
        self._p(head + "   " + "  ".join(bits))
        if output:
            for line in output.splitlines()[:40]:
                self._p(DIM("      | " + line[:160]))
            n = output.count("\n") + 1
            if n > 40:
                self._p(DIM(f"      | ... {n - 40} more line(s)"))
        self.out.flush()

    def verdict(self, stage: str, verdict: str, reason: str = "") -> None:
        if self.quiet:
            return
        col = {"pass": GREEN, "pass_": GREEN, "unsure": YELLOW, "fail": RED}.get(
            verdict, lambda s: s)
        self._p(f"  {MAGENTA('verdict')} {col(verdict):<8} {DIM(reason)}")

    def note(self, text: str, *, level: str = "info") -> None:
        if self.quiet:
            return
        col = {"info": DIM, "warn": YELLOW, "error": RED}.get(level, DIM)
        self._p(f"  {col(text)}")

    def error(self, text: str) -> None:
        self._p(RED(f"  error: {text}"))

    # ------------------------------------------------------------------ answers
    def answer(self, text: str, *, verified: bool = True, source: str = "",
               cost: dict | None = None) -> None:
        """A verified answer. Green heading, because that is the claim being made."""
        self._p()
        self._p(GREEN(BOLD("ANSWER")) + DIM(f"  ({source})" if source else ""))
        self._p(text)
        if cost:
            self._p(DIM("  " + _fmt_cost(cost)))
        self._p()

    def unverified(self, text: str, *, source: str = "", cost: dict | None = None) -> None:
        """An UNVERIFIED answer. Shown -- you asked, and during development you need
        to see what the loop produced -- but in a box that cannot be mistaken for a
        result. The plan is explicit: unverified text must be labelled as such."""
        self._p()
        self._p(YELLOW(BOLD("UNVERIFIED ANSWER"))
                + DIM("  (--no-verify: structural checks skipped, no rubric run. "
                      "This is debug output, not a result.)"))
        self._p(YELLOW(f"  [{source}]") if source else "")
        self._p(text)
        if cost:
            self._p(DIM("  " + _fmt_cost(cost)))
        self._p()

    def not_found(self, tried: list[str], reason: str = "") -> None:
        """An honest miss. The plan wants "not found" PLUS the list of what was tried,
        because "not found" alone is indistinguishable from "did not try"."""
        self._p()
        self._p(YELLOW(BOLD("NOT FOUND")))
        if reason:
            self._p(YELLOW(f"  {reason}"))
        if tried:
            self._p(DIM("  what was tried:"))
            for t in tried[:20]:
                self._p(DIM(f"    - {t[:150]}"))
            if len(tried) > 20:
                self._p(DIM(f"    ... and {len(tried) - 20} more"))
        else:
            self._p(DIM("  (nothing was tried)"))
        self._p()

    # ------------------------------------------------------------------ reports
    def trace(self, ctx, *, last: int | None = None) -> None:
        """The whole task trace: strategy path, real commands, verdicts, cost."""
        self.rule("trace")
        self._p(ctx.trace.render(last=last))
        self._p()
        self._p(ctx.trace.render_attempts(last=last))
        self._p()
        self._p(DIM(_fmt_cost(ctx.summary(), indent=False)))

    def path_log(self, ctx) -> None:
        """Every path the agent touched, in order. The notebook called this PATH_LOG
        and it was a module global, so a second task inherited the first task's list.
        Here it is `ctx.trace`, which is per task by construction."""
        self.rule("paths touched")
        rows = []
        for a in ctx.trace.attempts:
            for item in a.evidence_checked or []:
                if isinstance(item, str):
                    rows.append(item)
        for ev in ctx.trace.events:
            for key in ("path", "paths"):
                v = (ev.extra or {}).get(key)
                if isinstance(v, str):
                    rows.append(v)
                elif isinstance(v, list):
                    rows += [str(x) for x in v]
        if rows:
            for r in dict.fromkeys(rows):
                self._p(f"  {DIM(r)}")
        else:
            self._p(DIM("  (no paths recorded -- the trace records tool calls and "
                        "commands, not per-path touches)"))


def _fmt_cost(summary: dict, *, indent: bool = True) -> str:
    bits = [
        f"llm_calls={summary.get('llm_calls')}",
        f"tool_calls={summary.get('tool_calls')}",
        f"tokens={summary.get('tokens')}",
        f"elapsed={summary.get('elapsed_s')}s",
    ]
    if summary.get("iterations"):
        bits.append(f"turns={summary['iterations']}")
    if summary.get("denied"):
        bits.append(f"denied={summary['denied']}")
    if summary.get("budget_stop"):
        bits.append(f"budget_stop={summary['budget_stop']}")
    return "  ".join(bits)


# -------------------------------------------------------------------------- events
class TraceBridge:
    """Turns `ctx.trace` events into console output, live.

    Polling rather than a callback because the trace is a plain dataclass and giving it
    an observer would mean every module that writes an event had to know a UI exists.
    This sits beside the task, reads what was appended, and prints it. The cursor
    index is per-bridge, so two tasks on one console cannot interleave into each
    other's output.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self.seen = 0

    def poll(self, ctx) -> None:
        events = ctx.trace.events
        if self.seen > len(events):        # trace cleared (a re-run): restart
            self.seen = 0
        for ev in events[self.seen:]:
            self._show(ev)
        self.seen = len(events)

    def _show(self, ev) -> None:
        c = self.console
        kind = getattr(ev, "kind", "")
        action = getattr(ev, "action", "")
        status = getattr(ev, "status", "")
        detail = str(getattr(ev, "detail", "") or "")
        extra = getattr(ev, "extra", {}) or {}

        if kind == "llm":
            c.llm(action, extra.get("model", "?"), detail,
                  tokens=extra.get("tokens"), ms=getattr(ev, "duration_ms", None)
                  or None)
        elif kind == "gateway":
            # `authorize` logs the real argv it was asked about, which is the single
            # most important line in the whole console for auditing a refusal.
            argv = extra.get("argv")
            if isinstance(argv, list):
                c.command(argv, status=status,
                          reason=detail or extra.get("policy_reasons", [""])[0]
                          if extra.get("policy_reasons") else detail)
            else:
                c.info(f"  {CYAN('gateway')} {action:<14} {STATUS_COLOR.get(status, lambda s: s)(status)}"
                       f"  {DIM(detail)}")
        elif kind == "tool":
            c.info(f"  {CYAN('tool')}  {action:<14} {DIM(detail)}")
        elif kind == "react":
            c.info(f"  {MAGENTA('react')} {action:<14} {detail}")
        elif kind == "route":
            c.route(extra.get("strategy", extra.get("stage", "?")),
                    detail, fallback=(extra.get("source") == "fallback"))
        elif kind == "planner":
            c.info(f"  {MAGENTA('plan')}  {action:<14} {DIM(detail)}")
        elif kind == "verify":
            c.verdict("verify", "pass" if status == "ok" else "fail", detail)
        elif kind == "note":
            c.note(detail, level="warn" if status in ("budget", "error") else "info")


# -------------------------------------------------------------------------- the app
COMMANDS = frozenset({
    "help", "exit", "quit", "q", "trace", "paths", "cost", "plan",
    "quiet", "noverify", "root", "scope", "model",
})


def _is_command(word: str) -> bool:
    """Is this a console command, or the start of a question that happens to begin
    with a slash?

    "which file is at /etc/hosts" is a real question. Intercepting it as `/etc` and
    answering "unknown command" would be refusing to do the job over a punctuation
    accident, so only a word on the command list intercepts.
    """
    return word.strip().lower() in COMMANDS


HELP = """\
ask a question, or a command:

  /help              this
  /trace             the last task's full trace: strategy, commands, verdicts, cost
  /paths             every path the last task touched
  /cost              llm calls, tool calls, tokens, time
  /plan              print the last result's verification chain
  /quiet             toggle live event output
  /noverify          toggle --no-verify (DEBUG: output is labelled UNVERIFIED)
  /root [DIR]        set or show the search root (must be inside the read scope)
  /scope             show the read scope, write scope, and deny-list size
  /model             show the configured model, endpoint, and key status
  /exit              quit   (ctrl-d also works)

  everything else is sent to the agent as a question.

The agent runs shell commands on your machine. Read-only: every command is
classified before it runs, and the real argv is printed as it happens.  /trace
afterwards shows the whole path.
"""


def _interactive(console: Console, settings, strategy: str, no_verify: bool,
                 max_turns: int | None, root: str | None) -> int:
    from .config import load_settings
    from .engine import stage_result

    s = settings or load_settings()
    task_root = root
    strategy = strategy
    no_verify = no_verify
    max_turns = max_turns
    last_ctx = None
    last_result = None

    console.banner()
    if not s.llm.api_key:
        console.error("no API key. set LLM_API_KEY in .env (see .env.example).")
    elif not s.llm.base_url:
        console.error("no LLM_BASE_URL in .env.")
    else:
        console.info(DIM(f"  model {s.llm.model} at {s.llm.base_url}"))
    console.info(DIM(f"  scope {[str(r) for r in s.read_roots]}"))
    console.info(DIM(f"  executor tier={s.executor_enabled_tier} "
                     f"dry_run={s.executor_dry_run}"))
    console.info(DIM(f"  strategy={strategy}"
                     + ("  --no-verify (DEBUG)" if no_verify else "")))

    while True:
        try:
            line = input("\n" + BOLD("you> ") if _COLOR else "\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            console._p()
            console.info(DIM("bye."))
            return 0
        if not line:
            continue

        if line.startswith("/") and _is_command(line[1:].split(" ", 1)[0]):
            cmd, _, rest = line[1:].partition(" ")
            cmd = cmd.lower()
            if cmd in ("exit", "quit", "q"):
                return 0
            elif cmd == "help":
                console._p(HELP)
            elif cmd == "trace":
                if last_ctx is None:
                    console.note("no task yet")
                else:
                    console.trace(last_ctx)
            elif cmd == "paths":
                if last_ctx is None:
                    console.note("no task yet")
                else:
                    console.path_log(last_ctx)
            elif cmd == "cost":
                if last_ctx is None:
                    console.note("no task yet")
                else:
                    console._p(DIM("  " + _fmt_cost(last_ctx.summary())))
            elif cmd == "plan":
                if last_result is None:
                    console.note("no task yet")
                else:
                    console.rule("verification chain")
                    console._p(f"  outcome   {last_result.outcome}")
                    console._p(f"  verified  {last_result.verified}")
                    console._p(f"  reason    {last_result.reason}")
                    if last_result.evidence:
                        console._p(DIM("  evidence:"))
                        for e in last_result.evidence[:6]:
                            cmdv = " ".join(str(x) for x in e.get("command", []))
                            console._p(DIM(f"    - {cmdv[:120]}"))
            elif cmd == "quiet":
                console.quiet = not console.quiet
                console.note(f"live output {'off' if console.quiet else 'on'}")
            elif cmd == "noverify":
                no_verify = not no_verify
                console.note("--no-verify ON (debug; answers will be labelled UNVERIFIED)"
                             if no_verify else "--no-verify off")
            elif cmd == "root":
                if rest.strip():
                    task_root = rest.strip()
                    console.note(f"root = {task_root}")
                else:
                    console.note(f"root = {task_root or '(default: first read root)'}")
            elif cmd == "scope":
                console._p(DIM(f"  read  {[str(r) for r in s.read_roots]}"))
                console._p(DIM(f"  write {s.write_root}"))
                console._p(DIM(f"  deny  {len(s.deny_globs)} patterns"))
                console._p(DIM(f"  skip  {len(s.skip_dirs)} dirs"))
            elif cmd == "model":
                console._p(DIM(f"  base      {s.llm.base_url}"))
                console._p(DIM(f"  model     {s.llm.model}"))
                console._p(DIM(f"  dspy      {s.llm.dspy_model or '(same as model)'}"))
                console._p(DIM(f"  key       {'set' if s.llm.api_key else 'NOT SET'}"))
                console._p(DIM(f"  fallbacks {list(s.llm.fallback_models) or '(none)'}"))
            else:
                console.error(f"unknown command /{cmd}. try /help")
            continue
        elif line.startswith("/"):
            # A slash that is not a command. Asking "which file is at /etc/hosts" is a
            # legitimate question, and answering "unknown command /etc" would be
            # refusing to do the job. Only a real command word intercepts.
            pass

        # ------------------------------------------------------------ a real task
        from .context import new_task
        try:
            ctx = new_task(line, settings=s, root=task_root)
        except Exception as e:
            console.error(f"could not start task: {type(e).__name__}: {e}")
            continue
        last_ctx = ctx
        bridge = TraceBridge(console)
        t0 = time.perf_counter()

        def on_event(kind: str, detail: str, _c=console) -> None:
            if kind == "route":
                _c.info(f"  {DIM('*')} {CYAN(detail)}")
            elif kind == "stage":
                _c.info(f"  {DIM('*')} {detail}")
            elif kind == "verify":
                _c.verdict("verify",
                           "pass" if detail.startswith("verified") else "fail", detail)

        console.rule("thinking")
        try:
            res = stage_result(line, ctx=ctx, strategy=strategy,
                               use_llm_verify=not no_verify,
                               max_turns=max_turns, on_event=on_event)
        except Exception as e:
            console.error(f"{type(e).__name__}: {e}")
            import traceback as _tb
            console._p(DIM(_tb.format_exc()[-1500:]))
            continue
        last_result = res
        bridge.poll(ctx)

        cost = ctx.summary()
        cost["seconds"] = round(time.perf_counter() - t0, 2)
        if res.outcome == "found" and res.text:
            if no_verify:
                console.unverified(res.text, source=res.strategy, cost=cost)
            else:
                console.answer(res.text, verified=True, source=res.strategy,
                               cost=cost)
        elif res.text:
            console.unverified(res.text, source=f"{res.strategy} ({res.outcome})",
                               cost=cost)
        else:
            console.not_found(res.tried, reason=f"{res.outcome}: {res.reason}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m jarvis",
        description="Read-only file agent with a planner, a shell loop, and a trace.")
    ap.add_argument("query", nargs="*", help="ask once and exit (omit for the REPL)")
    ap.add_argument("--strategy", default="auto",
                    choices=["auto", "direct", "loop", "plan"],
                    help="auto = LLM judge, escalating on not_found")
    ap.add_argument("--root", default=None, help="directory to search (must be in scope)")
    ap.add_argument("--max-turns", type=int, default=None, help="cap the loop")
    ap.add_argument("--no-verify", action="store_true",
                    help="DEBUG ONLY: skip the rubric. Output is labelled UNVERIFIED "
                         "and cannot be presented as an answer.")
    ap.add_argument("--quiet", action="store_true", help="suppress live events")
    ap.add_argument("--check", action="store_true", help="preflight and exit")
    args = ap.parse_args(argv)

    from .config import load_settings
    settings = load_settings()
    console = Console(quiet=args.quiet)

    if args.check:
        from . import llm as _llm
        console.rule("preflight")
        out = _llm.preflight(settings, verbose=True)
        from .planner import DSPY_OK
        console.note(f"dspy: {'available' if DSPY_OK else 'MISSING (deterministic fallback)'}")
        console.note(f"executor: tier={settings.executor_enabled_tier} "
                     f"dry_run={settings.executor_dry_run}")
        return 0 if out["ok"] else 1

    if args.query:
        from .context import new_task
        from .engine import stage_result
        q = " ".join(args.query)
        ctx = new_task(q, settings=settings, root=args.root)
        res = stage_result(q, ctx=ctx, strategy=args.strategy,
                           use_llm_verify=not args.no_verify,
                           max_turns=args.max_turns)
        TraceBridge(console).poll(ctx)
        cost = ctx.summary()
        if res.outcome == "found" and res.text:
            (console.unverified if args.no_verify else console.answer)(
                res.text, source=res.strategy, cost=cost)
            return 0
        if res.text:
            console.unverified(res.text, source=f"{res.strategy} ({res.outcome})",
                               cost=cost)
            return 1
        console.not_found(res.tried, reason=f"{res.outcome}: {res.reason}")
        return 1

    return _interactive(console, settings, args.strategy, args.no_verify,
                        args.max_turns, args.root)


if __name__ == "__main__":
    raise SystemExit(main())