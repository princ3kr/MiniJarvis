"""The shared subprocess path for the read-only tools.

Every tool in this package starts a process, and CONTRACT.md rule 3 says they all go
through `runner.run_process` rather than calling `subprocess` themselves. That is not
tidiness. The notebook had three tools each calling `subprocess.run` with a hand-built
command, and each one independently:

  * built its own `--exclude-dir` list, from a slightly different subset of the
    skip-dirs, so two tools walked different trees
  * inherited the full parent environment, including every API key, into a child
  * had no byte cap, so `grep` over a directory containing a huge file returned
    whatever fitted in memory
  * did not process-group kill, so a slow tool could outlive its own timeout

Four copies of "run a command carefully" is four chances to get one detail wrong.
One function, one set of guarantees.

The read-only tools deliberately do NOT go through the gateway. Policy and approval
exist to decide whether a command may touch the machine; a tool the agent already
holds has its scope decided by `AccessPolicy` when it resolves paths, and routing
`find` through a tier table would mean an approval prompt for every search. The
runner still applies scrubbed env, rlimits, process-group kill, and the output cap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..runner import ProcessResult, run_process


def run_tool(argv: list[str], *, cwd: Path, timeout_s: int,
             max_output_bytes: int | None = None,
             limits=None,
             stdout_record_callback: Callable[[bytes], None] | None = None,
             stdout_record_separator: bytes = b"\n",
             count_stdout_lines: bool = False) -> ProcessResult:
    """Run a read-only tool command with the runner's guarantees.

    Returns a `ProcessResult` and never raises: a tool that cannot start is an
    observation with status `error`, per CONTRACT.md rule 5. Callers read
    `.stdout` / `.stderr` and branch on `.status`.

    The return type changed from `subprocess.CompletedProcess` to `ProcessResult`.
    That was the point of the swap -- the old shape could not express `truncated`,
    and `except subprocess.TimeoutExpired` in each tool became a status check.
    """
    from ..config import load_settings
    lim = limits if limits is not None else load_settings().limits
    return run_process(
        argv,
        cwd=cwd,
        timeout_s=timeout_s,
        max_output_bytes=max_output_bytes or lim.exec_max_output_bytes,
        limits=lim,
        stdout_record_callback=stdout_record_callback,
        stdout_record_separator=stdout_record_separator,
        count_stdout_lines=count_stdout_lines,
        # Tools report their own status; a non-zero exit is meaningful (grep's "no
        # match" is 1) and must NOT be conflated with a runner failure. So the
        # expected exit is 0 here and any other code is surfaced as `nonzero_exit`,
        # which the callers already handle.
        expect_exit=0,
    )


def exclusion_flags(policy, *, binary: str = "grep") -> list[str]:
    """The `--exclude-dir` list a recursive search must carry.

    Derived from the policy rather than restated per tool, so the policy and the
    tools cannot disagree about what is skipped -- which is how the notebook ended
    up with three tools that walked three slightly different trees. The gateway
    requires these on any recursive command (`policy._check_recursion`), so a grep
    built here satisfies the gateway without a special case.
    """
    dirs = set(policy.skip_dirs) | {".ssh", ".aws", ".gnupg", ".kube"}
    if binary == "rg":
        return [f"!/**/{d}/**" for d in sorted(dirs)]
    return [f"--exclude-dir={d}" for d in sorted(dirs)]