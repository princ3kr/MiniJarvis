"""The policy probe: verify the executor on your own commands.

This is the notebook's `PROBE_RESULTS` cell, ported. It belongs in the package
rather than in a notebook because the thing it measures is the safety property, and
a safety property that only exists inside one notebook is a property nobody runs.

Run it:

    python -m jarvis.probe            # everything
    python -m jarvis.probe --quiet    # summary only
    python -m jarvis.probe --no-live  # skip the section that executes commands

Verdicts
--------
``PASS``       behaved as a safe policy should
``PERMISSIVE`` allowed MORE than it should -- these are the findings
``STRICTER``   stricter than documented (safe, but docs and policy disagree)

**Zero PERMISSIVE is the bar.** Not zero findings: zero *under*-refusals. A probe
that reports STRICTER is fine, and quietly fixing those by loosening policy would
make the probe worse.

One thing changed from the notebook probe, and it is the difference between a
measurement and a green light. The notebook passed bare names (`["cat", "/etc/
passwd"]`) to `classify`, which looked the binary up in a table. This package
requires an absolute `argv[0]` in a trusted directory, so a bare `cat` is DENIED
before any path is ever examined. Pointing the unchanged case list at that would
report "cat outside read roots -> PASS" for the wrong reason -- it would pass because
the name was unknown, not because the path was refused. A probe that goes green by
denying everything is worse than no probe.

So `_absolutise()` rewrites each bare name to its real absolute path via `PATH`, and
the two cases that are *about* untrusted binaries (`./cat`, `/tmp/git`) are left
exactly as written. The case list is otherwise identical, so the numbers are
comparable with the notebook baseline of 32 PASS / 20 PERMISSIVE / 1 STRICTER.

Sections 1, 2, 3 and 5 execute nothing. Section 4 runs tier-0 reads against a
synthetic scratch tree containing a decoy key and a decoy `.ssh/id_rsa`, then
deletes it.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .access import AccessPolicy
from .config import load_settings
from .context import new_task
from .gateway import Gateway, run_postconditions
from .manifests import build_manifests
from .policy import Check, CommandSpec, Tier, classify
from .redaction import redact, scrubbed_env
from .runner import run_process

RESULTS: list[dict[str, Any]] = []


def _record(section: str, label: str, verdict: str, detail: str = "") -> None:
    RESULTS.append({"section": section, "label": label, "verdict": verdict,
                    "detail": detail})
    print(f"  {verdict:<11} {label:<44} {detail[:76]}")


def _judge(expect: str, tier: int) -> str:
    if expect == "T0":
        return "PASS" if tier == 0 else "STRICTER"
    if expect == "T1":
        return "PASS" if tier == 1 else ("PERMISSIVE" if tier < 1 else "STRICTER")
    if expect == "GE1":
        return "PASS" if tier >= 1 else "PERMISSIVE"
    if expect == "GE2":
        return "PASS" if tier >= 2 else "PERMISSIVE"
    if expect == "DENY":
        return "PASS" if tier == 3 else "PERMISSIVE"
    return "PERMISSIVE"


# A case whose binary is not installed on this machine cannot be measured. Recording
# it as PASS would be exactly the false pass this module exists to prevent -- the
# notebook printed "binary missing here -> denied (inconclusive)" as a note and still
# counted it, which is how a case can look green while testing nothing. It gets its
# own verdict, is excluded from the PERMISSIVE bar, and is reported so the reader can
# go and install the binary.
INCONCLUSIVE = "INCONCLUSIVE"


def _absolutise(argv: list[str]) -> list[str]:
    """Bare binary names -> their absolute trusted path. Everything else untouched.

    See the module docstring: without this the probe measures name resolution rather
    than path scoping, and reports PASS for the wrong reason.
    """
    if not argv:
        return argv
    first = argv[0]
    if first.startswith("/") or first.startswith("."):
        return list(argv)            # ./cat and /tmp/git are the point of those cases
    found = shutil.which(first)
    return [found or first, *argv[1:]]


def _classify(argv: list[str], manifest, policy: AccessPolicy,
              cwd: str | None = None) -> tuple[int, str]:
    """(tier, reason). A schema rejection counts as tier 3, i.e. blocked."""
    argv = _absolutise(argv)
    try:
        spec = CommandSpec(argv=list(argv), cwd=cwd, why="probe")
    except Exception as e:
        return 3, f"schema reject: {str(e).splitlines()[-1][:70]}"
    try:
        v = classify(spec, manifest, access=policy)
    except Exception as e:
        return 3, f"classifier raised {type(e).__name__}"
    return int(v.tier), v.reason


# --------------------------------------------------------------------------- 1
def section_classify(ws: str, home: str, manifest, policy: AccessPolicy) -> None:
    print("== 1. classification (nothing executed) ==")
    cases = [
        # controls: prove we are not simply denying everything
        ("control: ls", ["ls", "-la"], None, "T0"),
        ("control: git status", ["git", "status"], None, "T0"),
        ("control: python3 --version", ["python3", "--version"], None, "T0"),
        ("control: touch inside workspace", ["touch", f"{ws}/_p.txt"], None, "T1"),
        ("control: python3 -c (interpreter)", ["python3", "-c", "print(1)"], None, "GE2"),
        ("control: find -exec", ["find", ".", "-exec", "ls", "{}", "+"], None, "DENY"),
        ("control: xargs", ["xargs", "ls"], None, "DENY"),
        ("control: cat ~/.ssh/id_rsa", ["cat", "~/.ssh/id_rsa"], None, "DENY"),
        # flag escape hatches
        ("rg --pre=cmd (equals form)", ["rg", "--pre=./x", "foo", "."], None, "GE2"),
        ("rg --hostname-bin=cmd", ["rg", "--hostname-bin=./x", "foo"], None, "GE2"),
        ("find -fprint0 (writes a file)", ["find", ".", "-fprint0", "/tmp/out"], None, "GE2"),
        ("cp --target-directory=/etc", ["cp", "--target-directory=/etc",
                                        f"{ws}/a", f"{ws}/b"], None, "GE2"),
        # tier-0 reads not confined to read_roots
        ("cat outside read roots", ["cat", "/etc/passwd"], None, "GE2"),
        ("ls /", ["ls", "/"], None, "GE2"),
        ("diff two files outside roots", ["diff", "/etc/passwd", "/etc/hosts"], None, "GE2"),
        # recursion walks into deny-listed dirs
        ("grep -r over $HOME", ["grep", "-r", "password", home], None, "GE2"),
        ("rg over $HOME", ["rg", "password", home], None, "GE2"),
        # binary identity by basename
        ("./cat (basename trick)", ["./cat", "x"], None, "GE2"),
        ("/tmp/git (basename trick)", ["/tmp/git", "status"], None, "GE2"),
        # git: everything used to be tier 0
        ("git commit (mutates)", ["git", "commit", "-m", "x"], None, "GE1"),
        ("git add (mutates)", ["git", "add", "."], None, "GE1"),
        ("git branch -D", ["git", "branch", "-D", "x"], None, "GE2"),
        ("git config alias (code exec)", ["git", "config", "alias.x", "!sh"], None, "GE2"),
        ("git diff --output=file (write)", ["git", "diff", "--output=/tmp/o"], None, "GE2"),
        ("git --git-dir= (retarget)", ["git", "--git-dir=/tmp/x", "status"], None, "GE2"),
        ("git -C /tmp (retarget)", ["git", "-C", "/tmp", "status"], None, "GE2"),
        # relative paths resolved against the wrong base
        ("relative ../.ssh/config, cwd=WS", ["cat", "../.ssh/config"], ws, "GE2"),
        ("cwd outside workspace", ["ls"], "/", "GE2"),
        # documented-vs-policy
        ("pip show (network)", ["pip", "show", "numpy"], None, "GE2"),
        ("env wrapper around python", ["env", "python3", "x.py"], None, "GE2"),
        # new in this package: bundling and writers
        ("tar -cf (bundled short flags)", ["tar", "-cf", "/tmp/o.tar", ws], None, "GE1"),
        ("mkdir -p (writer, tier 1)", ["mkdir", "-p", f"{ws}/sub"], None, "T1"),
    ]
    for label, argv, cwd, expect in cases:
        bare = argv[0] if argv else ""
        missing = bool(bare) and not bare.startswith(("/", ".")) and not shutil.which(bare)
        tier, reason = _classify(argv, manifest, policy, cwd)
        if missing:
            _record("classify", label, INCONCLUSIVE,
                    f"{bare} not installed here -> DENY, which proves nothing | "
                    f"want {expect}")
            continue
        _record("classify", label, _judge(expect, tier),
                f"T{tier} want {expect} | {reason}")


# --------------------------------------------------------------------------- 2
def section_schema() -> None:
    print("\n== 2. schema rejects ==")
    bad = [
        ("pipe", ["ls", "|", "cat"]),
        ("semicolon", ["ls;rm"]),
        ("and-and", ["ls", "&&", "ls"]),
        ("subshell", ["echo", "$(id)"]),
        ("backtick", ["echo", "`id`"]),
        ("redirect", ["echo", "x", ">", "f"]),
        ("newline", ["ls\nrm"]),
        ("flag as argv[0]", ["-rf", "/"]),
        ("empty argv", []),
        ("300-item argv", ["ls"] * 300),
    ]
    for label, argv in bad:
        try:
            CommandSpec(argv=argv, why="probe")
            _record("schema", label, "PERMISSIVE", "accepted")
        except Exception:
            _record("schema", label, "PASS", "rejected")


# --------------------------------------------------------------------------- 3
def section_gateway(ws: Path, settings, manifests, gw: Gateway) -> None:
    print("\n== 3. gateway ==")
    ls = CommandSpec(argv=[str(shutil.which("ls"))], why="probe")

    ctx = new_task("probe", settings=settings, root=ws)
    r = gw.execute(ls, manifests["discovery"], ctx=ctx)
    _record("gateway", "discovery calling run_command",
            "PASS" if r.status == "denied" else "PERMISSIVE",
            f"{r.status}: {r.reason[:50]}")

    class _Fake:
        name = "nonexistent_agent"
        tools = ("run_command",)
        max_tier = 3
        limits = manifests["executor"].limits
    # An agent name nobody declared must not resolve to anything executable.
    # The notebook tested this by passing a fake manifest; here the property is
    # that the REGISTRY refuses the name, because a manifest constructed ad hoc
    # with `max_tier=3` is exactly what a caller would do by mistake.
    from .manifests import Registry
    try:
        Registry(manifests)["nonexistent_agent"]
        reg_ok = False
    except KeyError as e:
        reg_ok = "nonexistent_agent" in str(e)
    _record("gateway", "unknown agent name does not resolve",
            "PASS" if reg_ok else "PERMISSIVE",
            "registry raised KeyError" if reg_ok else "resolved to something")

    ctx3 = new_task("probe", settings=settings, root=ws)
    ex = manifests["executor"]
    outs = [gw.execute(ls, ex, ctx=ctx3) for _ in range(ex.limits.max_calls + 1)]
    _record("gateway", f"executor call #{ex.limits.max_calls + 1} (budget)",
            "PASS" if (outs[-1].status == "denied"
                       and all(o.status in ("ok", "nonzero_exit") for o in outs[:-1]))
            else "PERMISSIVE",
            f"{outs[-1].status}: {outs[-1].reason[:44]}")
    _record("gateway", "budget is per task, not a module global",
            "PASS" if ctx3.counters.gateway.get("executor:run_command")
            and ctx.counters.gateway.get("executor:run_command", 0) == 0 else "PERMISSIVE",
            "second task starts at zero")

    # Approval is per exact argv AND per task. Two independent things, and both are
    # checked here: an approval in one task must not authorise the same argv in
    # another, and it must not authorise a DIFFERENT argv in the same task.
    #
    # This section needs a manifest that PERMITS tier 2. `executor` is max_tier=0 --
    # the safe default -- so a tier-2 command is DENIED by the manifest cap before
    # approval is ever consulted, and the case would fail for the wrong reason. The
    # approval machinery only exists above tier 0, so testing it needs a manifest that
    # gets there. This is not a loosening of what ships: it is a test fixture.
    approving = manifests["executor"].model_copy(update={"name": "executor_approving",
                                                         "max_tier": 3})
    ctx4 = new_task("probe", settings=settings, root=ws)
    a = CommandSpec(argv=["/usr/bin/env", "/usr/bin/python3", "-c", "print(1)"],
                    why="probe")
    ctx5 = new_task("probe", settings=settings, root=ws)
    b = CommandSpec(argv=["/usr/bin/env", "/usr/bin/python3", "-c", "print(2)"],
                    why="probe")
    r1 = gw.execute(a, approving, ctx=ctx4)               # tier 2, no approval yet
    r2 = gw.execute(a, approving, ctx=ctx4, approve=True)  # approved, same argv
    r3 = gw.execute(a, approving, ctx=ctx5)               # same argv, DIFFERENT task
    r4 = gw.execute(b, approving, ctx=ctx4)               # different argv, SAME task
    _record("gateway", "approval does not leak to another task",
            "PASS" if (r1.status == "declined" and r3.status == "declined"
                       and r2.status != "declined") else "PERMISSIVE",
            f"fresh={r1.status} approved={r2.status} other-task={r3.status}")
    _record("gateway", "approval is per exact argv, not per task",
            "PASS" if r4.status == "declined" else "PERMISSIVE",
            f"different argv in an approved task -> {r4.status}")

    # And the tier ceiling must be what stops a tier-2 command when nobody approves.
    r5 = gw.execute(a, ex, ctx=ctx4)
    _record("gateway", "default executor cannot reach tier 2 at all",
            "PASS" if r5.status == "denied" else "PERMISSIVE",
            f"max_tier=0 -> {r5.status}: {r5.reason[:44]}")


# --------------------------------------------------------------------------- 5
def section_pure(ws: Path, settings, policy: AccessPolicy) -> None:
    print("\n== 5. pure-function checks ==")
    ctx = new_task("probe", settings=settings, root=ws)
    _record("pure", "approval fingerprint collision",
            "PERMISSIVE" if ctx.fingerprint(["a b"]) == ctx.fingerprint(["a", "b"])
            else "PASS", 'argv ["a b"] vs ["a","b"]')

    from .runner import run_process as _rp
    proc = _rp(["/bin/echo", "x"], cwd=ws, timeout_s=5, max_output_bytes=1000,
               limits=settings.limits)
    proc.exit_code = 10
    chk = run_postconditions(
        CommandSpec(argv=["/bin/echo", "x"], why="p",
                    checks=[Check(kind="exit_code", value="1")]),
        ws, proc=proc, policy=policy)
    _record("pure", "exit_code check is exact (1 vs 10)",
            "PERMISSIVE" if chk and chk[0].ok else "PASS",
            "substring match" if chk and chk[0].ok else "exact compare")

    chk = run_postconditions(
        CommandSpec(argv=["/bin/echo", "x"], why="p",
                    checks=[Check(kind="file_exists", value="x", path="/etc/passwd")]),
        ws, proc=proc, policy=policy)
    _record("pure", "file_exists check is jailed",
            "PERMISSIVE" if chk and chk[0].ok else "PASS",
            chk[0].note[:52] if chk else "")

    fake = ("gsk_" + "A" * 30, "sk-" + "B" * 24, "AKIA" + "C" * 16,
            "-----BEGIN RSA PRIVATE KEY-----\nxyz\n-----END RSA PRIVATE KEY-----")
    txt, n = redact(" | ".join(fake))
    leaked = [f[:8] for f in fake[:3] if f in txt]
    _record("pure", "redact() masks known key shapes",
            "PASS" if (n >= 4 and not leaked) else "PERMISSIVE",
            f"{n} redactions, leaked={leaked}")

    old = os.environ.get("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = "probe-value"
    ok = "GROQ_API_KEY" not in scrubbed_env(ws)
    if old is None:
        os.environ.pop("GROQ_API_KEY", None)
    else:
        os.environ["GROQ_API_KEY"] = old
    _record("pure", "GROQ_API_KEY not in child env", "PASS" if ok else "PERMISSIVE", "")


# --------------------------------------------------------------------------- 4
def section_live(ws: Path, settings, manifests, gw: Gateway) -> None:
    print("\n== 4. live (tier 0 only, synthetic scratch tree) ==")
    scratch = ws / "_probe_tmp"
    try:
        (scratch / ".ssh").mkdir(parents=True, exist_ok=True)
        (scratch / ".ssh" / "id_rsa").write_text("decoy-marker-8f3a9c\n")
        (scratch / "notes.txt").write_text("hello\napi_key=gsk_" + "A" * 30 + "\n")
        (scratch / "log.txt").write_text("line\n")

        def _live(argv, **kw):
            ctx = new_task("probe", settings=settings, root=ws)
            spec = CommandSpec(argv=_absolutise(argv), why="probe",
                               **kw.pop("spec_kw", {}))
            return gw.execute(spec, manifests["executor"], ctx=ctx,
                              dry_run=False, **kw)

        r = _live(["cat", str(scratch / "notes.txt")])
        _record("live", "cat output is redacted",
                "PASS" if ("gsk_" + "A" * 30) not in r.stdout_tail else "PERMISSIVE",
                f"status={r.status} redacted={r.redacted}")

        r = _live(["cat", str(scratch / ".ssh" / "id_rsa")])
        _record("live", "cat of deny-listed file refused",
                "PASS" if r.status in ("denied", "declined") else "PERMISSIVE",
                f"status={r.status}")

        r = _live(["grep", "-r", "decoy-marker", str(scratch)])
        _record("live", "grep -r reaches deny-listed file",
                "PERMISSIVE" if "decoy-marker-8f3a9c" in r.stdout_tail else "PASS",
                f"status={r.status} leaked={'decoy-marker-8f3a9c' in r.stdout_tail}")

        t0 = time.time()
        r = _live(["tail", "-f", str(scratch / "log.txt")], timeout_s=2)
        dt = time.time() - t0
        _record("live", "timeout kills the process group",
                "PASS" if (r.status == "timeout" and dt < 8) else "PERMISSIVE",
                f"status={r.status} after {dt:.1f}s")

        r = _live(["grep", "nomatch-xyz", str(scratch / "notes.txt")],
                  spec_kw={"expected_exit": 1})
        _record("live", "expected_exit=1 is honoured",
                "PASS" if r.status == "ok" else "PERMISSIVE",
                f"status={r.status} (grep exits 1 on no match)")

        # NEW: an unbounded flood must not exhaust memory or hang the runner.
        # `yes` never stops. Before the runner was given a byte cap this buffered
        # without limit; now it is truncated and the group is killed.
        #
        # Approved, not tier-0: `yes` is an unknown binary, so policy demands an
        # explicit per-argv approval -- which also makes this a second check that
        # `approve=True` really authorises exactly one argv.
        t0 = time.time()
        ctx_y = new_task("probe", settings=settings, root=ws)
        r = gw.execute(CommandSpec(argv=["/usr/bin/yes"], why="probe"),
                       manifests["executor"], ctx=ctx_y, approve=True,
                       dry_run=False, timeout_s=6)
        dt = time.time() - t0
        _record("live", "unbounded output is capped, not buffered",
                "PASS" if (r.truncated and dt < 12) else "PERMISSIVE",
                f"status={r.status} truncated={r.truncated} {dt:.1f}s, "
                f"tail {len(r.stdout_tail)}B "
                f"(cap {settings.limits.exec_max_output_bytes}B)")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"  cleaned up {scratch}")


# --------------------------------------------------------------------------- 6
def section_my_commands(ws: str, manifest, policy: AccessPolicy, settings) -> None:
    print("\n== 6. your commands ==")
    print(f"  settings: executor_enabled_tier={settings.executor_enabled_tier} "
          f"executor_dry_run={settings.executor_dry_run}")
    for cmd in MY_COMMANDS:
        try:
            argv = shlex.split(cmd)
        except ValueError as e:
            print(f"  [unparseable] {cmd!r}: {e}")
            continue
        tier, reason = _classify(argv, manifest, policy)
        if tier == 3:
            action = "refused"
        elif tier > settings.executor_enabled_tier:
            action = "dry-run only (tier above executor_enabled_tier)"
        elif tier == 0:
            action = "would RUN (tier 0 runs even in dry-run)"
        elif tier == 2:
            action = "needs approval"
        else:
            action = "dry-run" if settings.executor_dry_run else "would run after snapshot"
        print(f"  T{tier}  {cmd:<40} -> {action}")
        print(f"        {reason[:96]}")


# Put your own commands here (one string per command).
MY_COMMANDS = [
    "ls -la",
    "git status",
    "git commit -m wip",
    "rg --pre=./x foo .",
    "cat ../.ssh/config",
    "python3 train.py",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jarvis.probe")
    ap.add_argument("--no-live", action="store_true", help="skip the live section")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args(argv)

    import contextlib
    import io
    devnull = io.StringIO()

    settings = load_settings()
    # The probe needs a real scope to measure against, so it uses a private scratch
    # workspace rather than the configured roots. Nothing here reads $HOME.
    import tempfile
    ws = Path(tempfile.mkdtemp(prefix="jarvis_probe_"))
    settings = settings.replaced(read_roots=(ws,), write_root=ws,
                                 executor_enabled_tier=3, executor_dry_run=False)
    manifests = build_manifests((str(ws),), str(ws))
    exec_m = manifests["executor"]
    policy = AccessPolicy.from_settings(settings, cwd=ws)
    gw = Gateway(settings=settings, enabled_tier=3)
    home = str(Path.home())

    try:
        if args.quiet:
            with contextlib.redirect_stdout(devnull):
                section_classify(str(ws), home, exec_m, policy)
                section_schema()
                section_gateway(ws, settings, manifests, gw)
                section_pure(ws, settings, policy)
                if not args.no_live:
                    section_live(ws, settings, manifests, gw)
        else:
            section_classify(str(ws), home, exec_m, policy)
            section_schema()
            section_gateway(ws, settings, manifests, gw)
            section_pure(ws, settings, policy)
            if not args.no_live:
                section_live(ws, settings, manifests, gw)

        print("\n== 6. your commands ==")
        if args.quiet:
            with contextlib.redirect_stdout(devnull):
                section_my_commands(str(ws), exec_m, policy, settings)
        else:
            section_my_commands(str(ws), exec_m, policy, settings)
    finally:
        shutil.rmtree(ws, ignore_errors=True)

    print("\n== summary ==")
    cnt = Counter(r["verdict"] for r in RESULTS)
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    perm = [r["label"] for r in RESULTS if r["verdict"] == "PERMISSIVE"]
    if perm:
        print("  PERMISSIVE (fix these before raising executor_enabled_tier):")
        for p in perm:
            print("   -", p)
    else:
        print("  no PERMISSIVE findings: the policy refused everything it should have.")

    st = [r["label"] for r in RESULTS if r["verdict"] == "STRICTER"]
    if st:
        print("  STRICTER (safe, but stricter than documented):")
        for s in st:
            print("   -", s)

    inc = [r["label"] for r in RESULTS if r["verdict"] == INCONCLUSIVE]
    if inc:
        print("  INCONCLUSIVE (binary not installed here; unmeasured, not passing):")
        for i in inc:
            print("   -", i)

    return 1 if perm else 0


if __name__ == "__main__":
    sys.exit(main())
