"""The shell tool: the model proposes argv, the gateway decides, the runner executes.

This is the piece that was missing. The notebook had `run_executor`, which took a
planner step's *instruction text* and used it as literal argv -- so "count the
python files" became an attempt to exec a binary called `count`. That is the bug
the design called out for removal, and the replacement is here:

    model  ->  CommandSpec(argv=[...], why=..., expected_exit=...)
           ->  gateway.authorize   (manifest tool gate, per-agent budget,
           |                        binary resolution, grammar, jail, taint)
           ->  gateway.execute     (tier ceiling, approval, cwd, runner)
           ->  Observation         (status + redacted output)

Three properties that are the whole point:

1. **The model emits argv, never a shell string.** `CommandSpec` rejects pipes,
   semicolons, `&&`, command substitution, backticks and redirections outright. A
   shell would reinterpret the argv that policy validated and run something else.
   There is no shell in this process tree, ever.

2. **Policy decides before approval.** Approval answers "this exact argv, already
   classified legal, may it run NOW" -- it is never asked to authorise something
   nobody has classified. A gateway that asked first would be asking a human to
   audit a command the policy engine has not seen.

3. **A refusal is a result, not an exception.** The model gets `status="denied"`
   with the reason and can adapt. Raising would kill the loop, and a loop that dies
   on the first refusal cannot be steered away from refusals.

Taint is handled here rather than with a blanket flag (the notebook's bug, which
escalated every command whose argv merely mentioned a path to approval). Only an
argument whose *content came from a file we read* is tainted, because that is the
only case where the argv is an attacker's text rather than the model's own.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .access import AccessPolicy
from .context import TaskContext
from .gateway import CommandResult, execute
from .manifests import AgentManifest, build_manifests
from .policy import CommandSpec
from .tools.observation import Observation

__all__ = ["CommandTool", "REACT_TOOL_NAMES", "parse_command_spec"]

REACT_TOOL_NAMES = ("run_command", "find", "grep", "read", "survey")


def absolutise(argv: list[str]) -> list[str]:
    """Make argv[0] absolute if it is a bare name. Leaves `./x` and `/x` alone.

    Ergonomics only, and deliberately so. `policy.resolve_binary` rejects any
    relative argv[0] outright, which is correct -- a relative binary can be swapped
    for a lookalike -- but it means every caller has to write `/usr/bin/ls`. A model
    that gets refused for that stops trying, so the path is filled in here and
    policy then re-checks the result against `TRUSTED_BIN_DIRS`. Resolving through
    PATH therefore cannot authorise anything policy would otherwise refuse.
    """
    import shutil

    if not argv:
        return argv
    head = str(argv[0])
    if head.startswith("/") or head.startswith("."):
        return [str(a) for a in argv]
    found = shutil.which(head)
    return ([found, *[str(a) for a in argv[1:]]] if found
            else [str(a) for a in argv])


def _pydantic_reason(e: Exception, limit: int = 200) -> str:
    """The actual reason out of a pydantic ValidationError.

    pydantic's message ends with a documentation URL, so `splitlines()[-1]` -- the
    obvious way to get "the reason" -- returns the URL. The reason is in the first
    line, after `Value error, `. Falls back to the exception's own first line so a
    non-pydantic error still says something.
    """
    msg = str(e)
    # pydantic's layout is: a summary line ("1 validation error for CommandSpec"),
    # then the FIELD name ("argv"), then the reason indented ("Value error, ..."),
    # then a docs URL. The reason is the only line worth showing.
    for line in msg.splitlines():
        if "Value error," in line:
            return line.split("Value error,", 1)[1].split("[type=")[0].strip()[:limit]
    # Not pydantic: the first non-empty line is the message.
    for line in msg.splitlines():
        if line.strip():
            return line.strip()[:limit]
    return type(e).__name__


def parse_command_spec(raw: str, *, cwd: Path | None = None) -> CommandSpec:
    """Turn a model's reply into a validated CommandSpec.

    The model is asked for a JSON object. It will sometimes wrap it in prose or a
    fence, so this tolerates that -- but only that. A loose parser here is how
    `{"argv": ["rm", "-rf", "/"]}` gets through with a field nobody validated.
    """
    from .llm import extract_json

    # `llm.extract_json(expect="object")`, not the planner's `_extract_json`: the
    # planner's expects an ARRAY (plan steps) and this reply is an OBJECT. Asking for
    # an array against `{"argv": [...]}` returns the first nested array -- the argv --
    # and the fields are then read off a list. The original import also named a
    # function the planner no longer exports, so this path raised ImportError on
    # every call.
    data = extract_json(raw, expect="object")
    if not isinstance(data, list) or not data:
        # A bare object is the common shape; wrap it.
        if isinstance(data, dict):
            data = [data]
        else:
            data = extract_json("[" + raw.strip() + "]")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ValueError("expected a JSON object describing the command")
    d: dict[str, Any] = data[0]

    argv = d.get("argv")
    if isinstance(argv, str):
        # A model that answers with a command line rather than argv. Split it --
        # shlex, so quoting is honoured -- rather than taking it as one token, which
        # would try to exec a file literally named "ls -la".
        import shlex
        argv = shlex.split(argv)
    if not argv or not isinstance(argv, list):
        raise ValueError("argv must be a non-empty list of strings")
    argv = [str(a) for a in argv]

    return CommandSpec(
        argv=absolutise(argv),
        cwd=str(cwd) if cwd else None,
        why=str(d.get("why") or "model proposed"),
        expected_exit=int(d.get("expected_exit", 0) or 0),
        expected_effects=[str(e) for e in (d.get("expected_effects") or [])][:20],
    )


@dataclass
class CommandTool:
    """The `run_command` action, as the ReAct loop sees it.

    Holds no mutable state: the manifest, tier ceiling and dry-run flag all come from
    `Settings`, and the per-task command count lives on `ctx`. That is what lets two
    tasks share a process without sharing an executor's history.
    """

    ctx: TaskContext
    manifest: AgentManifest
    # Argv fragments the model read out of a FILE this task opened. Only these are
    # tainted. A path the model chose itself is the model's own text.
    tainted_from_disk: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tainted_from_disk is None:
            self.tainted_from_disk = set()

    # ------------------------------------------------------------------ helpers
    def note_read_text(self, text: str) -> None:
        """Record that some bytes came off disk, so an argv containing them is tainted.

        Coarse on purpose. It cannot tell whether a given argv fragment really came
        from the file; it only knows these strings might have. That is the safe
        direction -- a false positive costs an approval prompt, a false negative lets
        a file say "run this".
        """
        if not text:
            return
        for frag in text.split():
            frag = frag.strip("\"'`(),;:")
            if len(frag) >= 8 and not frag.startswith("/"):
                self.tainted_from_disk.add(frag)

    def _is_tainted(self, argv: list[str]) -> bool:
        return any(a in self.tainted_from_disk for a in argv)

    # ------------------------------------------------------------------ action
    def __call__(self, argv: list[str] | str, *, why: str = "",
                 expected_exit: int = 0, cwd: str | None = None) -> Observation:
        """Run one command. Always returns an Observation; never raises.

        CONTRACT.md rule 5: a tool crash must degrade to an observation, not an
        exception that kills the loop. Every branch below returns.
        """
        t0 = time.perf_counter()

        if isinstance(argv, str):
            import shlex
            try:
                argv = shlex.split(argv)
            except ValueError as e:
                return self._obs("unparseable command: %s" % e, t0, status="error")
        argv = [str(a) for a in argv]
        if not argv:
            return self._obs("empty command", t0, status="error")
        # Same normalisation the parser does. Without it every model-written `ls` is
        # refused for having a relative argv[0] -- technically right, and useless.
        argv = absolutise(argv)

        # Default cwd is the TASK ROOT, not the workspace. The workspace is where a
        # writer would go, and it is almost always empty; a model told "find the
        # big files under src/" that runs `cat src/main.py` and gets exit 1 from an
        # empty directory has been given an answer it cannot interpret. For a
        # read-only agent the tree it is searching IS the cwd.
        target_cwd = cwd or (str(self.ctx.root) if self.ctx.root else None)

        # Taint is applied AFTER parsing and BEFORE the gateway, so the gateway sees
        # the real flag and makes the real decision.
        spec_kwargs: dict[str, Any] = {}
        if self._is_tainted(argv):
            spec_kwargs["tainted"] = True

        try:
            spec = CommandSpec(argv=argv, cwd=target_cwd, why=why or "model proposed",
                               expected_exit=int(expected_exit or 0),
                               **spec_kwargs)
        except Exception as e:
            # Schema rejection. Reported as blocked with the REASON so the model can
            # see what to change -- a silent refusal teaches it to give up.
            #
            # pydantic's ValidationError ends with a docs URL, and taking the last
            # line produced "For further information visit https://errors.pydantic.dev/
            # 2.13/v/value_error" as the error text. That tells the model nothing and
            # tells a human nothing. The reason is in the FIRST line, after
            # "Value error, ".
            return self._obs(f"rejected by policy schema: {_pydantic_reason(e)}",
                             t0, status="blocked")

        try:
            result = execute(spec, self.manifest, ctx=self.ctx)
        except Exception as e:                       # never expected; never fatal
            return self._obs(f"gateway error: {type(e).__name__}: {str(e)[:120]}",
                             t0, status="error")

        return self._obs_from(result, t0)

    def _obs_from(self, r: CommandResult, t0: float) -> Observation:
        body = ""
        if r.stdout_tail:
            body += r.stdout_tail
        if r.stderr_tail:
            body += ("\n[stderr]\n" + r.stderr_tail if body else "[stderr]\n"
                     + r.stderr_tail)
        if not body:
            body = f"(no output; status={r.status}"
            if r.reason:
                body += f", {r.reason}"
            body += ")"
        status = {
            "ok": "ok",
            "nonzero_exit": "ok",           # a non-zero exit is INFORMATION, not a failure
            "timeout": "timeout",
            "denied": "blocked",
            "declined": "blocked",
            "dry_run": "blocked",
            "error": "error",
        }.get(r.status, "error")

        return Observation(
            tool="run_command", status=status,
            items=[{"argv": r.argv, "tier": r.tier, "exit_code": r.exit_code,
                    "status": r.status, "reason": r.reason,
                    "policy_reasons": r.policy_reasons,
                    "truncated": r.truncated, "redacted": r.redacted,
                    "checks": [{"kind": c.kind, "ok": c.ok, "note": c.note}
                               for c in r.checks]}],
            text=body, total_lines=body.count("\n") + 1,
            n_lines=body.count("\n") + 1, start_line=1,
            truncated=r.truncated,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            reason=r.reason or (r.stdout_tail or "")[:200],
        )

    def _obs(self, reason: str, t0: float, *, status: str = "error") -> Observation:
        return Observation(tool="run_command", status=status, items=[], text=reason,
                           total_lines=1, n_lines=1, start_line=1, truncated=False,
                           duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                           reason=reason)


def executor_manifest(settings, roots=None) -> AgentManifest:
    """The manifest the shell tool runs under.

    `max_tier=0` is the important number. The discovery agent has no `run_command`
    tool at all, and this one has the tool but can only reach provably read-only
    commands: `ls`, `cat`, `grep`, `find`, `du`, `df`, `git status`, `wc`, `stat`.

    So on first contact the agent can run shell commands against this machine and
    still cannot write a file, cannot run an interpreter, and cannot reach the
    network. `probe` is what makes that claim checkable rather than aspirational.
    Raising it is a deliberate edit to this line and to `executor_enabled_tier`.
    """
    r = roots or tuple(str(x) for x in settings.read_roots)
    return build_manifests(r, str(settings.write_root))["loop"]