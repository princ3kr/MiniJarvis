from __future__ import annotations

import subprocess
import time
from pathlib import Path

from ..access import AccessPolicy, AccessDenied
from ..config import NO_TEXT_EXT, Settings
from ..context import TaskContext
from .observation import Observation


def _build_grep_cmd(query: str, base: Path, policy: AccessPolicy, settings: Settings,
                    regex: bool = False, max_per_file: int = 3, case_sensitive: bool = False) -> list[str]:
    cmd = ["grep", "-rI", "-n", "-m", str(max_per_file)]
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
    cmd.append(str(base))
    cmd.append(query)
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
                              max_per_file=max_pf, case_sensitive=case_sensitive)

    try:
        from ..redaction import scrubbed_env
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=limits.timeout_s, check=False,
                             env=scrubbed_env(workspace=base), shell=False)
    except subprocess.TimeoutExpired:
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="grep", status="error", reason="timeout", duration_ms=dur)

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
                p = policy.resolve(raw_path)
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
                p = policy.resolve(raw_path)
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
