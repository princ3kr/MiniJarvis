from __future__ import annotations

import functools
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from ..redaction import scrubbed_env

# TODO: Swap this thin helper for jarvis.runner.run_process once that lands.
# runner signature (expected):
#   run_process(argv: list[str], *, cwd: Path, timeout_s: int, max_output_bytes: int,
#               limits: Limits, expect_exit: int = 0) -> ProcessResult


@functools.lru_cache(maxsize=32)
def _seen_executable(basename: str) -> bool:
    """Cache of already-seen executable basenames (pure function of basename)."""
    return True


def run_subprocess(
    argv: list[str],
    *,
    cwd: Path,
    timeout_s: int,
    max_output_bytes: int | None = None,
) -> subprocess.CompletedProcess:
    """Thin local subprocess helper.
    
    MUST use shell=False, pass scrubbed env, and capture_output=True.
    Calls jarvis.runner.run_process eventually - TODO.
    """
    _ = _seen_executable(os.path.basename(argv[0]) if argv else "")
    env = scrubbed_env(workspace=cwd)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            timeout=timeout_s,
            env=env,
            cwd=cwd,
            text=True,
        )
    except subprocess.TimeoutExpired as e:
        raise
    return proc
