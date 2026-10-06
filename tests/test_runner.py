"""Runner tests. The hygiene guarantees are the product, so each is pinned.

Every test here is a property the notebook either got wrong or got right by
accident. The comments say which, because "why is there a test that kills a process
group" is not answerable from the test alone.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.config import Limits
from jarvis.runner import argv_preview, run_process

L = Limits()


def go(argv, *, t=10, cap=20_000, expect=0):
    return run_process(argv, cwd=Path("/tmp"), timeout_s=t, max_output_bytes=cap,
                       limits=L, expect_exit=expect)


def test_returns_ok_and_captures_stdout():
    r = go(["/bin/echo", "hello"])
    assert r.status == "ok" and r.exit_code == 0
    assert r.stdout == "hello\n"
    assert r.ok


def test_nonzero_exit_is_a_status_not_an_exception():
    """The caller needs to branch on this, so it must arrive as a value. A tool
    that raises here kills the loop instead of reporting 'no match'."""
    r = go(["/bin/sh", "-c", "exit 3"])
    assert r.status == "nonzero_exit" and r.exit_code == 3
    assert "expected 0" in r.reason


def test_expected_exit_makes_the_same_run_ok():
    """grep exits 1 on no match. Without this a correct 'nothing found' looks broken."""
    r = go(["/bin/sh", "-c", "exit 1"], expect=1)
    assert r.status == "ok"


def test_missing_binary_is_an_error_not_a_crash():
    r = go(["/bin/definitely-not-here"])
    assert r.status == "error" and r.exit_code is None


def test_empty_argv_is_an_error():
    assert run_process([], cwd=Path("/tmp"), timeout_s=5, max_output_bytes=100,
                       limits=L).status == "error"


# ------------------------------------------------------------------ env scrubbing
def test_child_inherits_no_api_key(tmp_workspace, monkeypatch):
    """A subprocess that merely prints its own environment would otherwise hand the
    key back in stdout, and that stdout goes to a remote model."""
    monkeypatch.setenv("GROQ_API_KEY", "secret-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-value")
    r = go(["/usr/bin/env"], cap=100_000)
    assert r.status == "ok"
    assert "GROQ_API_KEY" not in r.stdout
    assert "AWS_SECRET_ACCESS_KEY" not in r.stdout
    assert "secret-value" not in r.stdout


def test_child_env_is_minimal_not_empty():
    """Scrubbing must not produce an env with no PATH, or nothing runs at all."""
    r = go(["/usr/bin/env"], cap=100_000)
    assert "PATH=" in r.stdout and "HOME=" in r.stdout


# ------------------------------------------------------------------ timeouts
def test_timeout_kills_the_process():
    r = go(["/bin/sleep", "30"], t=2)
    assert r.status == "timeout" and r.timed_out
    assert "process group killed" in r.reason


def test_timeout_kills_the_whole_process_group():
    """THE regression. `proc.kill()` kills only the direct child, so
    `sh -c 'sleep 999'` leaves a grandchild holding the pipes open and
    `communicate()` never returns -- the timeout does not fire, it hangs.

    If this test ever hangs, the group kill regressed. The outer timeout is a
    safety net, not the assertion.
    """
    t0 = time.perf_counter()
    r = go(["/bin/sh", "-c", "sleep 999 & echo started"], t=2)
    dt = time.perf_counter() - t0
    assert r.status == "timeout"
    assert dt < 8, f"took {dt:.1f}s; the group kill is not working"
    assert "started" in r.stdout       # we got what it wrote before the deadline


def test_timeout_reports_partial_output():
    """Output produced before the deadline is evidence and must not be discarded."""
    r = go(["/bin/sh", "-c", "echo before; sleep 30"], t=2)
    assert r.status == "timeout"
    assert "before" in r.stdout


# ------------------------------------------------------------------ rlimits
def test_address_space_rlimit_comes_from_settings():
    """The notebook hard-coded 1 GiB inside a closure, which made `Limits` decorative.
    So the test asserts the limit EQUALS the configured value rather than a constant."""
    r = go(["/usr/bin/python3", "-c",
            "import resource;print(resource.getrlimit(resource.RLIMIT_AS)[0])"])
    assert r.stdout.strip() == str(L.address_space_bytes)


def test_address_space_rlimit_is_configurable():
    tiny = Limits(address_space_bytes=1 << 24)
    r = run_process(["/usr/bin/python3", "-c",
                     "import resource;print(resource.getrlimit(resource.RLIMIT_AS)[0])"],
                    cwd=Path("/tmp"), timeout_s=10, max_output_bytes=1000, limits=tiny)
    assert r.stdout.strip() == str(1 << 24)


def test_file_size_rlimit_stops_a_giant_write(tmp_workspace):
    """RLIMIT_FSIZE turns a disk-filling command into a SIGXFSZ at 4 MiB. Without
    it, one runaway command can fill a filesystem."""
    target = tmp_workspace / "big.bin"
    r = go(["/usr/bin/python3", "-c",
            f"open({str(target)!r},'wb').write(b'x'*9000000)"], t=10)
    assert r.status != "ok"
    assert target.stat().st_size <= L.file_size_bytes


def test_rlimits_do_not_break_ordinary_commands():
    """A limit that breaks normal work gets removed, so it needs a test too."""
    assert go(["/bin/echo", "still fine"]).status == "ok"
    assert go(["/usr/bin/python3", "-c", "print(1+1)"]).stdout.strip() == "2"


# ------------------------------------------------------------------ output cap
def test_unbounded_output_is_truncated_at_the_cap():
    """`yes` never stops. `communicate()` would buffer all of it."""
    r = go(["/usr/bin/yes"], t=6, cap=5000)
    assert r.truncated
    assert len(r.stdout) <= 5000


def test_streaming_line_count_drains_past_output_cap():
    r = run_process(["/usr/bin/seq", "1", "10000"], cwd=Path("/tmp"),
                    timeout_s=10, max_output_bytes=64, limits=L,
                    count_stdout_lines=True)
    assert r.status == "ok"
    assert r.truncated
    assert len(r.stdout) <= 128
    assert r.stdout.endswith("10000\n")
    assert r.stdout_line_count == 10_000


def test_truncation_keeps_the_tail():
    """When a command produces too much, the END is where the error is. Keeping the
    head keeps the first 2 KB and drops the message the user needed.

    The command produces its output then prints the error, so this measures the
    runner's retention policy rather than `yes` never exiting (which is what the
    flood tests above cover).
    """
    r = go(["/bin/sh", "-c",
            "i=0; while [ $i -lt 5000 ]; do echo \"line $i padding padding\"; "
            "i=$((i+1)); done; echo THE_ERROR_AT_THE_END"], t=10, cap=3000)
    assert r.truncated
    assert "THE_ERROR_AT_THE_END" in r.stdout


def test_both_stdout_and_stderr_are_drained_before_return():
    """The drain loop must finish both pipes before returning. Returning after the
    first stream close drops the second stream and makes a command appear to have
    said less than it did.
    """
    r = run_process(["/bin/sh", "-c", "echo STDOUT; echo STDERR >&2"],
                    cwd=Path("/tmp"), timeout_s=10, max_output_bytes=20_000,
                    limits=L)
    assert r.status == "ok"
    assert "STDOUT" in r.stdout
    assert "STDERR" in r.stderr


def test_drain_reads_every_stream_before_return(monkeypatch):
    """The bounded-drain helper must walk every pipe before returning. Returning
    after the first stream is closed drops the remainder of the second stream, which
    is the same bug as a one-sided timeout path.
    """
    import jarvis.runner as runner_mod

    class FakeFile:
        def __init__(self, name, chunks):
            self.name = name
            self.chunks = list(chunks)
            self.closed = False

        def fileno(self):
            return 1 if self.name == "o" else 2

        def read(self, n):
            if not self.chunks:
                return b""
            return self.chunks.pop(0)

        def close(self):
            self.closed = True

    class FakeSelector:
        def __init__(self, stream):
            self.stream = stream

        def register(self, fh, event):
            pass

        def unregister(self, fh):
            pass

        def close(self):
            pass

        def select(self, timeout=None):
            if self.stream is None:
                return []
            out = self.stream
            self.stream = None
            return [(SimpleNamespace(fileobj=out), None)]

    stdout = FakeFile("o", [b"x" * 4096])
    stderr = FakeFile("e", [b"ERR\n"])
    proc = SimpleNamespace(stdout=stdout, stderr=stderr, pid=123)

    monkeypatch.setattr(runner_mod.selectors, "DefaultSelector",
                        lambda: FakeSelector(stdout))
    monkeypatch.setattr(runner_mod, "_kill_tree", lambda proc: None)

    out, err, truncated, timed_out, _, _ = runner_mod._drain_bounded(
        proc, cap=1024, timeout_s=1, count_stdout_lines=False
    )

    assert truncated is True
    assert b"ERR\n" in err
    assert out.startswith(b"x")


def test_output_cap_stops_the_command_instead_of_waiting_for_the_timeout():
    """The point of capping is to end the run. Draining the pipe and ignoring the
    excess -- the first version -- let `yes` run the full 30s, which is exactly
    what a caller relying on the cap is trying to avoid."""
    t0 = time.perf_counter()
    r = go(["/usr/bin/yes"], t=30, cap=5000)
    dt = time.perf_counter() - t0
    assert dt < 10, f"took {dt:.1f}s of a 30s budget; the cap did not stop it"
    assert r.truncated


def test_byte_cap_is_not_reported_as_a_timeout():
    """They are different outcomes. A command that exited 0 with a lot to say was
    not 'timed out', and a caller counting timeouts would be misled."""
    r = go(["/usr/bin/yes"], t=30, cap=5000)
    assert r.status != "timeout"
    assert not r.timed_out
    assert r.truncated


def test_a_normal_large_output_is_not_truncated():
    """A cap that fires on ordinary output gets turned off by whoever hits it, so it
    has to be generous enough for a real result."""
    r = go(["/bin/sh", "-c", "seq 1 2000"], cap=20_000)
    assert not r.truncated
    assert r.stdout.strip().splitlines()[-1] == "2000"


def test_truncation_is_reported_so_a_caller_can_ask_for_less():
    r = go(["/usr/bin/yes"], t=5, cap=5000)
    assert r.truncated is True


# ------------------------------------------------------------------ no shell
def test_metacharacters_are_not_interpreted():
    """shell=False. There is no shell here, so these are literal arguments."""
    r = go(["/bin/echo", "; rm -rf /"])
    assert r.stdout.strip() == "; rm -rf /"


def test_output_is_redacted():
    r = go(["/bin/echo", "gsk_" + "A" * 30])
    assert "gsk_" + "A" * 30 not in r.stdout
    assert r.redacted >= 1


def test_argv_preview_truncates():
    """A trace is rendered into prompts, so it must not carry file content."""
    assert len(argv_preview(["x" * 500])) <= 83


def test_argv_preview_keeps_short_argv_intact():
    assert argv_preview(["ls", "-la"]) == "ls -la"


def test_cwd_is_honoured(tmp_workspace):
    (tmp_workspace / "marker.txt").write_text("here")
    r = run_process(["/bin/sh", "-c", "ls"], cwd=tmp_workspace, timeout_s=10,
                    max_output_bytes=5000, limits=L)
    assert "marker.txt" in r.stdout