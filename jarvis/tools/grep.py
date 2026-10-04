from __future__ import annotations

import time
from pathlib import Path

from ..access import AccessPolicy, AccessDenied
from ..config import NO_TEXT_EXT, Settings
from ..context import TaskContext
from .base import run_tool
from .observation import Observation


def _build_grep_cmd(query: str, base: Path, policy: AccessPolicy, settings: Settings,
                    regex: bool = False, max_per_file: int = 3,
                    case_sensitive: bool = False, recursive: bool = True) -> list[str]:
    # `-H` is not optional. grep omits the filename prefix when given a single file
    # and prints `LINE:TEXT`; with `-r DIR` it prints `PATH:LINE:TEXT`. The parser
    # below splits on path/line/body, so the single-file form silently produced zero
    # hits and the tool reported `not_found` for a file that plainly matched.
    # Forcing the prefix makes the output format identical in both cases.
    #
    # `-r` and `-I` are bundled as `-rI` in one token, which means a caller cannot
    # strip recursion by filtering argv -- the first attempt at single-file support
    # did exactly that and silently kept recursing. The flag is a parameter here
    # instead.
    cmd = ["grep", "-rI" if recursive else "-I", "-H", "-n", "-m", str(max_per_file)]
    if not regex:
        cmd.append("-F")
    else:
        cmd.append("-E")
    if not case_sensitive:
        cmd.append("-i")
    # Exclude skip dirs and also deny-listed directory basenames
    # Enforce scope on recursion: use --exclude-dir for skip dirs and for deny basenames
    for d in sorted(policy.skip_dirs):
        cmd.append(f"--exclude-dir={d}")
    # Deny-listed dirs like .ssh, .aws, .gnupg, .kube as per requirement
    for deny in (".ssh", ".aws", ".gnupg", ".kube"):
        if deny not in policy.skip_dirs:  # avoid duplicates if present
            cmd.append(f"--exclude-dir={deny}")
    for e in sorted(NO_TEXT_EXT):
        cmd.append(f"--exclude=*{e}")
    # ORDER MATTERS: the PATTERN must precede the PATH. `grep ... PATH PATTERN`
    # makes grep treat the pattern as a second filename, exit 2 with
    # "No such file or directory", and return zero hits -- a confident false
    # "nothing matched". The notebook had this right; the extraction lost it.
    cmd.append(query)
    cmd.append(str(base))
    return cmd


def search_content(query: str, *, ctx: TaskContext, root: Path | str | None = None,
                   max_results: int | None = None, regex: bool = False,
                   max_per_file: int | None = None, case_sensitive: bool = False) -> Observation:
    t0 = time.perf_counter()
    policy = AccessPolicy.from_settings(ctx.settings, cwd=ctx.root or Path.cwd())
    try:
        base = policy.resolve(root or ctx.root or ctx.settings.read_roots[0])
    except AccessDenied as e:
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="grep", status="blocked", reason=str(e), duration_ms=dur)

    limits = ctx.settings.limits
    max_res = max_results if max_results is not None else limits.grep_max_results
    max_pf = max_per_file if max_per_file is not None else limits.grep_max_per_file

    # A root that does not exist is a caller error, and reporting it as such beats
    # letting Popen fail: a missing cwd surfaces as FileNotFoundError from the
    # spawn, which the runner words as "no such executable" and names the BINARY --
    # which exists. The message would be actively misleading about what went wrong.
    if not base.exists():
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="grep", status="not_found",
                           reason=f"no such path: {base}", duration_ms=dur)

    # A root that is a FILE is a legitimate request ("search this one file"), but
    # `grep -r` needs a directory as its cwd. Without this check the runner raises
    # NotADirectoryError and -- because the tool layer is contractually forbidden from
    # raising across an agent boundary -- that used to become `not_found`: a
    # confident "nothing matches" for a file that does match. So: run non-recursively
    # against the file, which is both correct and faster.
    single_file = base.is_file()

    use_rg = ctx.settings.use_rg
    cmd = None
    # Try ripgrep if requested and available
    if use_rg is not False:
        # Check if rg exists
        import shutil
        if shutil.which("rg"):
            cmd = ["rg", "--line-number", "--no-heading", "--max-count", str(max_pf)]
            if not regex:
                cmd.append("--fixed-strings")
            if not case_sensitive:
                cmd.append("--ignore-case")
            # Exclude patterns
            for d in sorted(policy.skip_dirs):
                cmd.extend(["--glob", f"!**/{d}/**"])
            for deny in (".ssh", ".aws", ".gnupg", ".kube"):
                if deny not in policy.skip_dirs:
                    cmd.extend(["--glob", f"!**/{deny}/**"])
            cmd.append(query)
            cmd.append(str(base))
    if cmd is None:
        cmd = _build_grep_cmd(query, base, policy, ctx.settings, regex=regex,
                              max_per_file=max_pf, case_sensitive=case_sensitive,
                              recursive=not single_file)

    # The child's cwd must be a DIRECTORY. When the target is a single file, the
    # file's own parent is used and the recursion flag is dropped -- `grep -r` with a
    # file as cwd raises NotADirectoryError, and a tool that cannot raise across the
    # agent boundary would turn that into a false `not_found`.
    workdir = base.parent if single_file else base

    out = run_tool(cmd, cwd=workdir, timeout_s=limits.timeout_s)
    if out.status == "timeout":
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="grep", status="error", reason="timeout", duration_ms=dur)
    if out.status == "error":
        # Report the failure. Falling through to `not_found` here would assert
        # "nothing matched" when in fact nothing ran, which is the one thing a
        # search tool must never do.
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="grep", status="error", reason=out.reason or "could not run",
                           duration_ms=dur)

    hits: list[dict] = []
    seen: dict[str, int] = {}
    # Parse based on tool
    if cmd and cmd[0] == "rg":
        for line in out.stdout.splitlines():
            if len(hits) >= max_res:
                break
            # format: path:line:content (or path:line:col:content depending on flags; we didn't add --column)
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            raw_path, line_str, body = parts[0], parts[1], parts[2]
            if not line_str.isdigit():
                continue
            line_no = int(line_str)
            try:
                p = policy.resolve(raw_path, base=workdir)
            except AccessDenied:
                continue
            if not policy.in_scope(p):
                continue
            seen_key = str(p)
            seen[seen_key] = seen.get(seen_key, 0) + 1
            if seen[seen_key] > max_pf:
                continue
            hits.append({"path": str(p), "line_no": line_no, "text": body.strip()[:300]})
    else:
        for line in out.stdout.splitlines():
            if len(hits) >= max_res:
                break
            if ":" not in line:
                continue
            raw_path, _, rest = line.partition(":")
            parts = rest.split(":", 1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            line_no, body = int(parts[0]), parts[1]
            try:
                p = policy.resolve(raw_path, base=workdir)
            except AccessDenied:
                continue
            if not policy.in_scope(p):
                continue
            seen[raw_path] = seen.get(raw_path, 0) + 1
            if seen[raw_path] > max_pf:
                continue
            hits.append({"path": str(p), "line_no": line_no, "text": body.strip()[:300]})

    dur = (time.perf_counter() - t0) * 1000
    status = "ok" if hits else "not_found"
    return Observation(
        tool="grep",
        status=status,
        items=hits,
        text="",
        total_lines=len(hits),
        n_lines=len(hits),
        start_line=1,
        truncated=len(hits) >= max_res,
        duration_ms=dur,
    )
