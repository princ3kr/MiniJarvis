"""The gateway: the one place a command may be decided, refused, or run.

Everything the notebook decided inside `run_command`, plus the four things that
function did not do. The word "choke point" is the design claim: there is no path
from an agent to a subprocess that does not pass through `authorize` and then
`execute`, so policy cannot be bypassed by a new tool that forgets to check.

Order is load-bearing and is preserved from the notebook:

    classify  ->  ENABLED_TIER ceiling  ->  approval  ->  dry-run  ->  run

Policy decides BEFORE approval, so an approval can never authorise something policy
denies; approval only answers "this exact argv, which is already legal, may it run
now". A gateway that asked for approval first would be asking the human to audit a
command nobody has classified yet.

Four things this adds, each a specific failure:

1. **Dry-run defaults ON.** `executor_enabled_tier` is 0 and `executor_dry_run` is
   true, so a fresh checkout runs nothing that writes or executes. The notebook had
   these as module globals read at import, meaning a test could not change them and a
   config change needed a restart. They are settings, and they are passed in.

2. **Approvals are per-task.** `APPROVAL_GRANTED` was a module set: an approval
   granted by one task in one kernel was visible to every later task. Approvals now
   live on `ctx.approvals`, keyed by a `\x00`-joined argv. The `\x00` matters --
   `" ".join` makes `["a b"]` and `["a", "b"]` the same key, so approving one command
   silently approves a different one. That was a real probe finding.

3. **Postconditions go through the jail.** The notebook called
   `Path(c.path).expanduser().exists()` with no scope check, so a check of kind
   `file_exists` and path `/root/.ssh/id_rsa` was a side channel: the exit status
   told the model whether a private key exists, with no read of its contents. A
   postcondition is an assertion about the world, and an assertion that can only be
   checked inside the scope is enforced.

4. **`exit_code` is compared exactly.** The notebook did
   `str(c.value) in _LAST_EXIT`, a substring test: an expectation of "1" is satisfied
   by an actual exit of "10", and an expectation of "0" is satisfied by "20". A
   postcondition that passes for the wrong reason is worse than one that fails.

The verifier's rule applies here too: `authorize` never raises. A classifier fault
degrades to DENIED, because "the policy engine crashed" and "this command is safe"
must not look alike.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .access import AccessDenied, AccessPolicy
from .config import Settings, load_settings
from .context import TaskContext, new_task
from .manifests import build_manifests
from .policy import CommandSpec, PolicyVerdict, Tier, classify
from .runner import ProcessResult, argv_preview, run_process

__all__ = ["Gateway", "CommandResult", "CheckResult", "authorize", "execute",
           "main"]

# The highest tier permitted to reach the runner. Deliberately duplicated here as a
# default only: `Settings.executor_enabled_tier` is what actually applies, and it is
# read per call, not at import.
DEFAULT_ENABLED_TIER = 0


@dataclass
class CheckResult:
    kind: str
    value: str
    ok: bool
    note: str = ""


@dataclass
class CommandResult:
    status: Literal["ok", "denied", "declined", "nonzero_exit", "timeout",
                    "postcondition_failed", "dry_run", "error"]
    argv: list[str]
    cwd: str
    tier: int
    exit_code: int | None = None
    duration_ms: float | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    timed_out: bool = False
    changed_files: list[str] = field(default_factory=list)
    undo_token: dict[str, Any] | None = None
    reason: str = ""
    redacted: int = 0
    checks: list[CheckResult] = field(default_factory=list)
    # Additive, so existing field access is unchanged.
    policy_reasons: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def authorize(spec: CommandSpec, manifest, *, ctx: TaskContext,
              enabled_tier: int | None = None,
              settings: Settings | None = None) -> CommandResult | None:
    """Decide whether `spec` may run. Returns a refusal, or None when it may.

    Returning `None` for "allowed" is deliberate and slightly unusual: it makes
    `execute` unable to proceed without an explicit success from this function, so
    adding a new early-return here cannot accidentally mean "allowed".
    """
    s = settings or ctx.settings
    tier_ceiling = s.executor_enabled_tier if enabled_tier is None else enabled_tier
    t0 = time.perf_counter()
    cwd_str = spec.cwd or str(ctx.settings.write_root)

    # The manifest gates the TOOL, not just the tier. A max_tier of 0 stops a
    # command that would have been dangerous; it does not stop `echo`. Without
    # this check a read-only discovery agent -- whose whole guarantee is that it
    # executes nothing -- can still run every tier-0 binary, because the manifest
    # never had a way to say "this agent has no exec".
    if "run_command" not in manifest.tools:
        reason = (f"agent {manifest.name!r} is not permitted to execute commands "
                  f"(tools: {list(manifest.tools) or 'none'})")
        ctx.log("gateway", "authorize", detail=reason, status="denied", t0=t0)
        return CommandResult(status="denied", argv=list(spec.argv), cwd=cwd_str,
                             tier=int(Tier.DENIED), reason=reason)

    # Per-agent command budget. A model-driven loop calls run_command until
    # something stops it, and each call may spawn a process. The notebook counted
    # these in a module dict (`_GATEWAY_CALLS`) which was therefore shared across
    # tasks and needed an explicit `gateway_reset()`; here the count lives on the
    # task, so it is per-task by construction and there is nothing to reset.
    key = f"{manifest.name}:run_command"
    used = ctx.counters.gateway.get(key, 0)
    if used >= manifest.limits.max_calls:
        ctx.counters.denied += 1
        reason = (f"command budget exhausted: {manifest.name} has used "
                  f"{used}/{manifest.limits.max_calls} run_command calls")
        ctx.log("gateway", "authorize", detail=reason, status="denied", t0=t0)
        return CommandResult(status="denied", argv=list(spec.argv), cwd=cwd_str,
                             tier=int(Tier.DENIED), reason=reason)

    policy = AccessPolicy.from_settings(s, cwd=ctx.root)
    try:
        verdict = classify(spec, manifest, access=policy, settings=s)
    except Exception as e:
        # Fail closed. A classifier fault is not evidence of safety.
        reason = f"classifier error: {type(e).__name__}: {str(e)[:80]}"
        ctx.log("gateway", "authorize", detail=reason, status="denied", t0=t0)
        return CommandResult(status="denied", argv=list(spec.argv), cwd=cwd_str,
                             tier=int(Tier.DENIED), reason=reason)

    ctx.log("gateway", "authorize",
            detail=f"tier {int(verdict.tier)} {argv_preview(spec.argv)}",
            status="allowed", t0=t0, binary=verdict.binary,
            policy_reasons=verdict.reasons)

    if verdict.denied:
        return CommandResult(status="denied", argv=list(spec.argv), cwd=cwd_str,
                             tier=int(verdict.tier), reason=verdict.reason,
                             policy_reasons=verdict.reasons)

    if int(verdict.tier) > tier_ceiling:
        reason = (f"tier {int(verdict.tier)} exceeds executor_enabled_tier="
                  f"{tier_ceiling}")
        return CommandResult(status="dry_run", argv=list(spec.argv), cwd=cwd_str,
                             tier=int(verdict.tier), reason=reason,
                             policy_reasons=verdict.reasons)

    # Approval. Scoped to this task, keyed by exact argv. The refusal repeats WHY,
    # not just that approval is needed: a model told only "needs approval" cannot
    # decide what to change, and re-proposes the same command.
    if verdict.tier is Tier.APPROVAL and not ctx.is_approved(spec.argv):
        why = "; ".join(verdict.reasons) or "tier 2 command"
        return CommandResult(status="declined", argv=list(spec.argv), cwd=cwd_str,
                             tier=int(verdict.tier),
                             reason=f"needs approval ({why})",
                             policy_reasons=verdict.reasons)

    return None


def execute(spec: CommandSpec, manifest, *, ctx: TaskContext, approve: bool = False,
            dry_run: bool | None = None,
            timeout_s: int | None = None) -> CommandResult:
    """Authorize and run one command. Returns a value; never raises."""
    s = ctx.settings
    dry = s.executor_dry_run if dry_run is None else dry_run
    limit_s = int(timeout_s or manifest.limits.timeout_s)
    t0 = time.perf_counter()
    cwd_str = spec.cwd or str(ctx.settings.write_root)

    if approve:
        ctx.approve(spec.argv)

    refusal = authorize(spec, manifest, ctx=ctx)
    if refusal is not None:
        return refusal
    ctx.charge_tool(f"{manifest.name}:run_command")   # charge only what policy allowed

    # Dry-run applies to anything that is not provably read-only. Tier 0 commands
    # still run, because a system that cannot grep its own workspace is not useful.
    policy = AccessPolicy.from_settings(s, cwd=ctx.root)
    verdict = classify(spec, manifest, access=policy, settings=s)
    if dry and verdict.tier is not Tier.READ_ONLY:
        reason = "dry-run: nothing that writes or executes was run"
        ctx.log("gateway", "execute", detail=reason, status="dry_run", t0=t0)
        return CommandResult(status="dry_run", argv=list(spec.argv), cwd=cwd_str,
                             tier=int(verdict.tier), reason=reason,
                             policy_reasons=verdict.reasons)

    # Resolve cwd INSIDE the read scope, by containment, not by string prefix.
    # Write scope is additionally required when the command can write, and `classify`
    # has already decided that -- so ask it rather than duplicating the rule here.
    try:
        cwd = policy.resolve(spec.cwd or ctx.settings.write_root, for_write=False)
    except AccessDenied as e:
        reason = f"cwd rejected: {e}"
        ctx.log("gateway", "execute", detail=reason, status="denied", t0=t0)
        return CommandResult(status="denied", argv=list(spec.argv),
                             cwd=str(spec.cwd or ""), tier=int(verdict.tier),
                             reason=reason)

    before = _snapshot_paths(cwd) if int(verdict.tier) >= 1 else None

    proc: ProcessResult = run_process(
        spec.argv, cwd=cwd, timeout_s=limit_s,
        max_output_bytes=int(manifest.limits.max_output_bytes),
        limits=s.limits, expect_exit=spec.expected_exit)

    # The context carries the last command's output for postconditions and for a
    # model asked "what did that print?". Per-task, never module-level.
    ctx.last_exit = str(proc.exit_code)
    ctx.last_stdout = proc.stdout
    ctx.last_stderr = proc.stderr

    checks = run_postconditions(spec, cwd, proc=proc, policy=policy)
    changed = _changed_paths(before) if before is not None else []

    status: str = proc.status
    reason = proc.reason
    if status == "ok" and any(not c.ok for c in checks):
        status = "postcondition_failed"
        reason = f"{sum(1 for c in checks if not c.ok)} of {len(checks)} checks failed"
    # Side effects the model did not declare are a red flag even on exit 0: the
    # command did something it did not say it would do.
    if status == "ok" and changed and not set(spec.expected_effects) & set(changed):
        reason = f"exit 0 but undeclared files changed: {changed[:3]}"

    ctx.log("gateway", "execute",
            detail=f"{status} {argv_preview(spec.argv)}", status=status, t0=t0,
            exit_code=proc.exit_code, truncated=proc.truncated)

    return CommandResult(
        status=status, argv=list(spec.argv), cwd=str(cwd), tier=int(verdict.tier),
        exit_code=proc.exit_code, duration_ms=proc.duration_ms,
        stdout_tail=proc.stdout[-4000:], stderr_tail=proc.stderr[-4000:],
        timed_out=proc.timed_out, changed_files=changed, truncated=proc.truncated,
        reason=reason, redacted=proc.redacted, checks=checks,
        policy_reasons=verdict.reasons)


def run_postconditions(spec: CommandSpec, cwd: Path, *, proc: ProcessResult,
                       policy: AccessPolicy) -> list[CheckResult]:
    """Machine-checkable postconditions. "exit 0" is never accepted alone.

    Every path is resolved through the jail first. A check that cannot be verified
    inside the scope FAILS rather than passing vacuously -- an unverifiable
    assertion reported as true is exactly how a model learns to trust this output.
    """
    out: list[CheckResult] = []
    for c in spec.checks:
        try:
            if c.kind in ("file_exists", "file_absent", "dir_listed"):
                if not c.path:
                    out.append(CheckResult(c.kind, c.value, False, "check has no path"))
                    continue
                # for_write=True for existence checks: we are asking "does the write
                # land here", which is the write question, not the read question.
                target = policy.resolve(c.path, for_write=True, base=cwd)
                exists = target.exists()
                ok = exists if c.kind in ("file_exists", "dir_listed") else not exists
                if c.kind == "dir_listed":
                    ok = target.is_dir()
                out.append(CheckResult(c.kind, c.value, bool(ok), str(target)))

            elif c.kind == "exit_code":
                # Exact comparison. The notebook used `str(c.value) in _LAST_EXIT`,
                # so expecting "1" was satisfied by an exit of "10".
                ok = str(proc.exit_code) == str(c.value).strip()
                out.append(CheckResult(c.kind, c.value, ok,
                                       f"actual exit {proc.exit_code}"))

            elif c.kind == "stdout_contains":
                ok = c.value in proc.stdout
                out.append(CheckResult(c.kind, c.value, ok))
            elif c.kind == "stderr_contains":
                ok = c.value in proc.stderr
                out.append(CheckResult(c.kind, c.value, ok))
            else:
                out.append(CheckResult(c.kind, c.value, False, "unknown check kind"))

        except AccessDenied as e:
            # Refused to even look. That is a FAIL, not a pass and not an error:
            # the model asked about something outside the scope and the answer is no.
            out.append(CheckResult(c.kind, c.value, False, f"out of scope: {e}"))
        except Exception as e:
            out.append(CheckResult(c.kind, c.value, False,
                                   f"{type(e).__name__}: {str(e)[:40]}"))
    return out


# --------------------------------------------------------------------------- change detect
def _snapshot_paths(cwd: Path, limit: int = 4000) -> dict[str, tuple[int, float]]:
    """(size, mtime) for every file under cwd, for tier>=1 side-effect detection."""
    import os
    snap: dict[str, tuple[int, float]] = {}
    for root, dirs, files in os.walk(cwd):
        dirs[:] = dirs[:50]
        for f in files:
            p = Path(root) / f
            try:
                st = p.stat()
            except OSError:
                continue
            snap[str(p)] = (st.st_size, st.st_mtime)
            if len(snap) >= limit:
                return snap
    return snap


def _changed_paths(before: dict[str, tuple[int, float]]) -> list[str]:
    if not before:
        return []
    after = _snapshot_paths(Path(next(iter(before))).parent)
    changed = [p for p, sig in after.items()
               if p not in before or before[p] != sig]
    return sorted(changed)[:20]


def main(argv: list[str] | None = None) -> int:
    """CLI: ask the policy about one command. `python -m jarvis.gateway <cmd...>`

    The point is that this is a way to ASK the policy a question, not a way around
    it. `--tier` classifies and exits; without it the command runs, but only through
    the same `authorize` path an agent would take -- including the enabled-tier
    ceiling, the dry-run default, and per-argv approval.
    """
    import argparse
    import shlex
    import shutil

    ap = argparse.ArgumentParser(prog="python -m jarvis.gateway",
                                 description="classify or run one command under policy")
    ap.add_argument("--tier", action="store_true", help="classify only, do not run")
    ap.add_argument("--approve", action="store_true", help="grant this exact argv")
    ap.add_argument("--agent", default="executor")
    ap.add_argument("--dry-run", action="store_true",
                    help="force dry-run even when the setting allows execution")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given")

    # Split when the caller passed a single string, so both `-- ls -la` and
    # `-- "ls -la"` work.
    if len(cmd) == 1 and " " in cmd[0]:
        cmd = shlex.split(cmd[0])

    # A bare `ls` is refused, correctly, because argv[0] must be absolute. That is
    # the right refusal for a model but the wrong ergonomics for a human at a
    # terminal, so the CLI resolves bare names itself. It uses `which`, which
    # searches PATH -- including user-writable directories. Policy then re-checks
    # that the result is in a trusted directory, so resolving through PATH here
    # cannot authorise anything it would otherwise refuse; it only saves the user
    # from having to type /usr/bin/ls.
    head = cmd[0]
    if not head.startswith("/") and not head.startswith("."):
        found = shutil.which(head)
        if found:
            cmd = [found, *cmd[1:]]
        else:
            print(f"no such executable: {head}")
            return 3

    settings = load_settings()
    try:
        spec = CommandSpec(argv=[str(a) for a in cmd], why="cli")
    except Exception as e:
        print(f"rejected by schema: {str(e).splitlines()[-1]}")
        return 3

    ctx = new_task("cli", settings=settings)
    manifest = build_manifests(tuple(str(r) for r in settings.read_roots),
                               str(settings.write_root))[args.agent]

    if args.tier:
        policy = AccessPolicy.from_settings(settings, cwd=ctx.root)
        try:
            v = classify(spec, manifest, access=policy, settings=s)
        except Exception as e:
            print(f"classifier error: {type(e).__name__}")
            return 3
        print(f"tier {int(v.tier)}  {v.binary or argv_preview(cmd)}")
        for r in v.reasons:
            print(f"        {r}")
        return 0 if not v.denied else 1

    r = execute(spec, manifest, ctx=ctx, approve=args.approve,
                dry_run=True if args.dry_run else None)
    print(f"{r.status}  tier {r.tier}  {argv_preview(cmd)}")
    if r.reason:
        print(f"  reason: {r.reason}")
    for c in r.policy_reasons:
        print(f"  policy: {c}")
    if r.stdout_tail:
        print(r.stdout_tail[-4000:])
    if r.stderr_tail:
        print(f"[stderr] {r.stderr_tail[-2000:]}")
    # denied is a refusal (3), declined is a policy decision awaiting a human (2),
    # everything else is an outcome the caller asked for.
    return {"denied": 3, "declined": 2, "error": 1}.get(r.status, 0)


class Gateway:
    """Object wrapper, for callers that want the choke point as a dependency.

    Holds no mutable state of its own: the enabled tier, the dry-run flag and the
    approvals all live in `Settings` and `TaskContext`, which is what makes two
    concurrent tasks independent.
    """

    def __init__(self, *, settings: Settings | None = None,
                 enabled_tier: int | None = None) -> None:
        self.settings = settings or load_settings()
        self.enabled_tier = enabled_tier

    def authorize(self, spec: CommandSpec, manifest, *, ctx: TaskContext):
        return authorize(spec, manifest, ctx=ctx, settings=self.settings,
                         enabled_tier=self.enabled_tier)

    def execute(self, spec: CommandSpec, manifest, *, ctx: TaskContext,
                approve: bool = False, dry_run: bool | None = None,
                timeout_s: int | None = None) -> CommandResult:
        return execute(spec, manifest, ctx=ctx, approve=approve, dry_run=dry_run,
                       timeout_s=timeout_s)

    def run(self, spec: CommandSpec, manifest, *, ctx: TaskContext) -> CommandResult:
        return self.execute(spec, manifest, ctx=ctx)


if __name__ == "__main__":
    raise SystemExit(main())
