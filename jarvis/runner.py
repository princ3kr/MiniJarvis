"""The single place a child process is ever started.

Everything the notebook did inline in one cell, ported here, because the contract
says no module may call `subprocess.run` directly. Concentrating it means the
hygiene below is auditable in one file instead of scattered across agents.

Six guarantees, each fixing a specific failure:

1. **No shell, ever.** `shell=False` and `argv` only. A shell reinterprets `;`,
   `|`, `*` and `$()`; policy validated argv and would have validated something the
   shell never ran.

2. **Scrubbed environment.** A subprocess that merely prints its own environment
   would otherwise hand back `*_API_KEY` in its stdout, and that stdout goes to a
   remote model.

3. **Own process group, and kill the whole group on timeout.** `start_new_session=True`
   puts the child in a new session and process group. On timeout the notebook called
   `proc.kill()`, which kills only the direct child: a command like
   `sh -c 'sleep 999'` leaves grandchildren alive holding the pipes open, so
   `communicate()` never returns and the timeout hangs forever. `killpg` on the
   child's group takes the tree.

4. **rlimits, from settings rather than literals.** The notebook hard-coded 1 GiB of
   address space and 4 MiB per file inside a closure, so `Limits` was a decorative
   dataclass. They are read from `limits` here.
   `RLIMIT_NPROC` is deliberately NOT set: it is per-UID rather than per-process, so
   a low value breaks every command for a user who already runs more than N
   processes. That reasoning is preserved because it is easy to "fix" by adding it.

5. **Bounded output capture.** The notebook called `communicate()`, which buffers
   an unbounded amount in memory. A command that emits 10 GB -- or `yes` -- exhausts
   memory and takes the agent down. Output is read through a selector with a hard
   byte cap; past the cap the group is killed and `truncated` is set.

6. **Redaction on the way out.** stdout/stderr go to a model, so they are masked
   before anyone can read them here.

`ProcessResult.status` is one of `ok` | `nonzero_exit` | `timeout` | `error`.
It never raises for a command that ran badly: the caller needs a status to reason
about, and an exception across the gateway boundary would kill the loop.

KNOWN LIMITATION, not a KNOWN BUG: rlimits are applied with `preexec_fn`, which
CPython documents as unsafe in the presence of threads. Nothing in the package runs
plans in threads today. If `budget.max_parallel_plans` ever becomes real, this must
move to applying limits after spawn or to a wrapper binary -- `preexec_fn` would
deadlock under load, which is the same class of hang as #3.
"""
from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import Limits
from .redaction import redact, scrubbed_env

__all__ = ["ProcessResult", "run_process", "argv_preview"]

# Read granularity for the selector loop. Small enough that a timeout is noticed
# promptly, large enough that we are not syscall-bound.
_CHUNK = 65536

# Bytes retained in each of stdout/stderr after the cap is hit. Enough to show the
# tail of an error, not enough to flood the next prompt.
_TAIL_BYTES = 4000

# After the byte cap is hit, how long to keep reading in case the command is
# finishing rather than flooding. Long enough for a short computation to complete,
# short enough that `yes` does not hold the runner for its whole timeout.
_FLOOD_GRACE_S = 0.4


@dataclass
class ProcessResult:
    """What came back. `status` is the field callers should branch on."""

    status: str                       # ok | nonzero_exit | timeout | error
    exit_code: int | None
    duration_ms: float
    stdout: str
    stderr: str
    timed_out: bool
    command: list[str] = field(default_factory=list)
    # Additive, default-valued, so the CONTRACT signature is unchanged.
    truncated: bool = False
    redacted: int = 0
    reason: str = ""
    streamed_records: int = 0
    stdout_line_count: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def render(self, limit: int = 2000) -> str:
        s = f"[{self.status}] exit={self.exit_code} {self.duration_ms:.0f}ms"
        if self.reason:
            s += f" -- {self.reason}"
        if self.stdout:
            s += "\n" + self.stdout[-limit:]
        if self.stderr:
            s += "\n" + self.stderr[-limit:]
        return s


def argv_preview(argv: list[str], n: int = 80) -> str:
    """One truncated line for a trace. Never the full argv: it may carry file content
    that was read from a file, and a trace is rendered into prompts."""
    s = " ".join(argv)
    return s if len(s) <= n else s[:n] + "..."


def _preexec(limits: Limits, timeout_s: int):
    """Build the child-side rlimit setter.

    A closure rather than a module global so the values are explicit and a test can
    read what they will be. Runs in the child after fork, before exec.
    """
    cpu = int(limits.cpu_rlimit_s)
    as_bytes = int(limits.address_space_bytes)
    fsize = int(limits.file_size_bytes)

    def _apply() -> None:  # pragma: no cover - runs in the forked child
        import resource
        # CPU: soft = timeout, hard = timeout+1. Hard one second higher so a process
        # killed by SIGXCPU at the soft limit can still flush; both caps bound the
        # wall clock independently of the parent's `communicate` timeout.
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        # RLIMIT_NPROC deliberately unset -- per-UID, not per-process.

    return _apply


def _kill_tree(proc: subprocess.Popen) -> None:
    """SIGKILL the child's whole process group.

    Falls back to killing only the direct child if the group is already gone, which
    happens when the child exited between the timeout and this call.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def run_process(argv: list[str], *, cwd: Path, timeout_s: int,
                max_output_bytes: int, limits: Limits,
                expect_exit: int = 0,
                stdout_record_callback: Callable[[bytes], None] | None = None,
                stdout_record_separator: bytes = b"\n",
                count_stdout_lines: bool = False) -> ProcessResult:
    """Run `argv` under full hygiene and return a status. Never raises.

    `expect_exit` only affects `status`: an exit code other than it is
    `nonzero_exit`. It never makes a bad command look good, because the exit code
    itself is always reported.
    """
    argv = [str(a) for a in argv]
    t0 = time.perf_counter()
    if not argv:
        return ProcessResult(status="error", exit_code=None, duration_ms=0.0,
                             stdout="", stderr="", timed_out=False, command=argv,
                             reason="empty argv")

    cap = int(max_output_bytes) if max_output_bytes and max_output_bytes > 0 else 1 << 20

    try:
        proc = subprocess.Popen(                      # noqa: S603 - argv only, no shell
            argv,
            cwd=str(cwd),
            env=scrubbed_env(workspace=cwd),          # no API key inherited
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # No `text=` and no `errors=`: CPython wraps the pipes in a TextIOWrapper
            # if EITHER is passed, so passing `errors="replace"` with `text=False`
            # silently gives you str anyway. Keeping the pipes binary is what makes
            # `max_output_bytes` an actual byte cap.
            text=False,
            start_new_session=True,                   # own process group
            preexec_fn=_preexec(limits, timeout_s),   # rlimits before exec
        )
    except FileNotFoundError as e:
        return _fail(argv, f"no such executable: {e}", t0)
    except PermissionError as e:
        return _fail(argv, f"not executable: {e}", t0)
    except OSError as e:
        return _fail(argv, f"{type(e).__name__}: {e}", t0)

    out_b, err_b, truncated, timed_out, streamed_records, stdout_line_count = _drain_bounded(
        proc, cap, timeout_s, stdout_record_callback, stdout_record_separator,
        count_stdout_lines)

    # Reap. After a group kill the direct child is gone too, but `wait` is still
    # required to avoid a zombie.
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:                 # pragma: no cover - defensive
        _kill_tree(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    stdout = out_b.decode("utf-8", errors="replace")
    stderr = err_b.decode("utf-8", errors="replace")
    stdout, n1 = redact(_cap_tail(stdout, cap))
    stderr, n2 = redact(_cap_tail(stderr, cap))

    exit_code = proc.returncode
    dt = round((time.perf_counter() - t0) * 1000, 1)

    if timed_out:
        status, reason = "timeout", f"exceeded {timeout_s}s; process group killed"
    elif exit_code is None:                          # pragma: no cover - defensive
        status, reason = "error", "no exit code"
    elif exit_code != expect_exit:
        status, reason = "nonzero_exit", f"exit {exit_code}, expected {expect_exit}"
    else:
        status, reason = "ok", ""

    return ProcessResult(status=status, exit_code=exit_code, duration_ms=dt,
                         stdout=stdout, stderr=stderr, timed_out=timed_out,
                         command=argv, truncated=truncated, redacted=n1 + n2,
                         reason=reason, streamed_records=streamed_records,
                         stdout_line_count=stdout_line_count if count_stdout_lines else None)


def _cap_tail(text: str, cap: int) -> str:
    """Keep the TAIL, not the head.

    When a command produces too much, the end is where the error is. Keeping the
    head would keep the first 20 KB of an infinite stream and drop the message.
    """
    if len(text) <= cap:
        return text
    return f"... [truncated, showing last {cap} chars]\n" + text[-cap:]


def _drain_bounded(proc: subprocess.Popen, cap: int,
                   timeout_s: float,
                   stdout_record_callback: Callable[[bytes], None] | None = None,
                   stdout_record_separator: bytes = b"\n",
                   count_stdout_lines: bool = False
                   ) -> tuple[bytes, bytes, bool, bool, int, int]:
    """Read both pipes until EOF, the byte cap, or the deadline.

    Returns (stdout_bytes, stderr_bytes, truncated, timed_out).

    The notebook used `communicate()`, which buffers without limit. This reads
    through a selector instead: no deadlock from a full pipe (we drain
    continuously), no unbounded memory (we stop at the cap), and a deadline that is
    actually checked rather than delegated to `communicate`.
    """
    out_parts: list[bytes] = []
    err_parts: list[bytes] = []
    out_len = 0
    err_len = 0
    truncated = False
    timed_out = False

    sel = selectors.DefaultSelector()
    assert proc.stdout is not None and proc.stderr is not None
    src = {proc.stdout: (out_parts, "o"), proc.stderr: (err_parts, "e")}
    os.set_blocking(proc.stdout.fileno(), False)
    os.set_blocking(proc.stderr.fileno(), False)
    for fh in src:
        sel.register(fh, selectors.EVENT_READ)

    deadline = time.monotonic() + timeout_s
    open_files = 2
    # A ring per stream, holding the LAST `cap` bytes rather than the first.
    #
    # This is why the cap is not simply "stop appending": a command that floods
    # forever (`yes`) never reaches EOF, so the first `cap` bytes are the only
    # bytes there ever are, and an error printed at the end of a long output would
    # be dropped. Keeping a rolling tail means the cap bounds memory AND keeps the
    # part of the output that carries information.
    tails: dict[str, bytearray] = {"o": bytearray(), "e": bytearray()}
    pending_record = bytearray()
    streamed_records = 0
    stdout_line_count = 0
    # Once the ring is full, how much longer to keep reading before deciding the
    # command is flooding rather than finishing.
    #
    # Two cases look identical from the outside and need opposite treatment:
    #   * `yes` -- will never stop; waiting the full timeout defeats the cap.
    #   * a loop printing 5000 lines and then an error line -- 40ms from done, and
    #     the error is the only part worth reading.
    # EOF separates them. A bounded command reaches it almost immediately, so it is
    # read to the end and exits normally; only a command that is STILL producing
    # data after the grace period is killed.
    flood_deadline: float | None = None
    try:
        while open_files:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                break
            if flood_deadline is not None and now >= flood_deadline:
                break                       # still talking after the grace period
            wait = min(0.05 if flood_deadline is not None else 0.25,
                       deadline - now, max(0.001, (flood_deadline or deadline) - now))
            for key, _ in sel.select(timeout=max(0.001, wait)):
                fh = key.fileobj
                try:
                    chunk = fh.read(_CHUNK)          # type: ignore[union-attr]
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    chunk = b""
                if not chunk:                       # EOF on this pipe
                    sel.unregister(fh)
                    open_files -= 1
                    continue
                _, which = src[fh]
                if which == "o" and count_stdout_lines:
                    stdout_line_count += chunk.count(b"\n")
                if which == "o" and stdout_record_callback is not None:
                    pending_record.extend(chunk)
                    while True:
                        split_at = pending_record.find(stdout_record_separator)
                        if split_at < 0:
                            break
                        record = bytes(pending_record[:split_at])
                        del pending_record[:split_at + len(stdout_record_separator)]
                        stdout_record_callback(record)
                        streamed_records += 1
                buf = tails[which]
                buf.extend(chunk)
                # Keep only the tail. The per-stream share is half the cap so one
                # noisy stream cannot starve the other's error message.
                share = max(1024, cap // 2)
                if len(buf) > share:
                    del buf[:len(buf) - share]
                    truncated = True
            streaming = stdout_record_callback is not None or count_stdout_lines
            if truncated and not streaming and flood_deadline is None:
                # Start the clock ONCE, on the first overflow, and do not extend it
                # however much more data arrives. Restarting it on each chunk is the
                # bug this fixes: `yes` produces data forever, so a clock that
                # resets on data never expires and `yes` runs the full timeout.
                # Data arriving is not evidence that the command is nearly done --
                # it is evidence that the ring is still full, which is the whole
                # reason we are here.
                flood_deadline = time.monotonic() + _FLOOD_GRACE_S
    finally:
        sel.close()

    # Either we hit the cap or the command overran: stop it, then take whatever is
    # still buffered in the pipes.
    streaming = stdout_record_callback is not None or count_stdout_lines
    if (truncated and not streaming) or timed_out:
        _kill_tree(proc)
        # Let the tree die before draining, or the read below can block.
        time.sleep(0.05)
        for fh, which in ((proc.stdout, "o"), (proc.stderr, "e")):
            try:
                rest = fh.read(_CHUNK)               # type: ignore[union-attr]
            except (OSError, ValueError):
                rest = b""
            if rest:
                buf = tails[which]
                buf.extend(rest)
                share = max(1024, cap // 2)
                if len(buf) > share:
                    del buf[:len(buf) - share]

    if stdout_record_callback is not None and pending_record:
        stdout_record_callback(bytes(pending_record))
        streamed_records += 1

    for fh in (proc.stdout, proc.stderr):
        try:
            fh.close()
        except (OSError, ValueError):
            pass

    # `status == "timeout"` is reserved for the deadline. Hitting the byte cap is
    # its own outcome: the command may well have exited 0 with a lot to say.
    return (bytes(tails["o"]), bytes(tails["e"]), truncated,
            timed_out and not truncated, streamed_records, stdout_line_count)


def _fail(argv: list[str], reason: str, t0: float) -> ProcessResult:
    return ProcessResult(status="error", exit_code=None,
                         duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                         stdout="", stderr="", timed_out=False, command=argv,
                         reason=reason)
