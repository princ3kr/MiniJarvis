"""`python -m jarvis "query"` -- the only entry point a user needs.

Two modes, and they are deliberately separated:

    python -m jarvis --check        env preflight. NO network unless the config looks
                                    complete, and even then it is one PONG probe.
    python -m jarvis "query"        one task, then the trace and the attempts.

Why `--check` exists separately: the notebook discovered a bad `LLM_BASE_URL` only after
an agent had already walked a tree and burned a budget. `llm.preflight` separates "wired
wrong" from "service down", which is the distinction that was getting lost.

With no arguments and no `--check`, nothing happens: no network, no filesystem walk. The
agent is read-only, but it is not free.
"""
from __future__ import annotations

import argparse
import sys

from . import llm
from .config import load_settings
from .context import BudgetExhausted, new_task


def _check() -> int:
    """Env preflight. Returns a shell exit code."""
    print("--- LLM preflight ---")
    out = llm.preflight(load_settings(), verbose=True)
    from .planner import DSPY_OK
    print(f"  dspy   : {'available' if DSPY_OK else 'MISSING (deterministic fallback plan)'}")
    return 0 if out["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m jarvis",
        description="Read-only file agent with a planner, a verifier, and a trace.")
    ap.add_argument("query", nargs="?", help="the question to answer")
    ap.add_argument("--check", action="store_true",
                    help="preflight the LLM config and exit")
    ap.add_argument("--strategy", default="auto",
                    choices=["auto", "direct", "plan", "react"],
                    help="auto routes then escalates direct -> plan -> react; "
                         "a pinned strategy does NOT escalate")
    ap.add_argument("--react", action="store_true",
                    help="force the shell loop, and show the full trajectory")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="cap the react loop (default: min(12, budget.max_iterations))")
    ap.add_argument("--no-verify", action="store_true",
                    help="structural-only verification (no rubric, no LLM for the judge)")
    ap.add_argument("--root", default=None, help="directory to search (must be in scope)")
    ap.add_argument("--last", type=int, default=None,
                    help="show only the last N trace events / attempts")
    args = ap.parse_args(argv)

    if args.check:
        return _check()

    if not args.query:
        ap.print_usage()
        # Not "safe_resolve": that function was the notebook's tool-side jail and
        # became `access.resolve` during extraction, so the old name appears
        # nowhere in the package. A hint that points at a function you cannot grep
        # for is worse than no hint.
        print("\nnothing to do. try: python -m jarvis \"where is run_process defined?\"")
        return 2

    settings = load_settings()
    strategy = "react" if args.react else args.strategy
    with new_task(args.query, settings=settings, root=args.root) as ctx:
        from .engine import run_task
        if strategy == "react" and args.max_turns:
            # Pin the turn cap only when asked, so the budget stays the default
            # authority for how long a loop may run.
            ctx.budget = ctx.budget.__class__(
                **{**ctx.budget.__dict__, "max_iterations": args.max_turns})
        try:
            answer = run_task(args.query, ctx=ctx, strategy=strategy,
                              use_llm_verify=not args.no_verify)
        except NotImplementedError as e:
            print(f"\n{e}", file=sys.stderr)
            ctx.close("not_implemented")
            return 3
        except BudgetExhausted as e:
            # TaskContext.__exit__ swallows BudgetExhausted; this is here for the case
            # where the exhaustion happens before the context manager is entered.
            print(f"\nbudget exhausted: {e}", file=sys.stderr)
            ctx.close("budget")
            return 4

        print("\n" + "=" * 100)
        print(ctx.trace.render(last=args.last))
        print()
        print(ctx.trace.render_attempts(last=args.last))
        print()
        print(f"summary: {ctx.summary()}")
        if answer is None:
            # Not a crash: the agent reports honestly rather than guessing.
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
