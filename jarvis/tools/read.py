from __future__ import annotations

import time
import zipfile
import tarfile
from pathlib import Path

from ..access import AccessPolicy, AccessDenied
from ..config import NO_TEXT_EXT, LISTABLE_ARCHIVE_EXT, Settings
from ..context import TaskContext
from .observation import Observation


def _list_archive(p: Path, t0: float, max_archive_members: int) -> Observation:
    ext = p.suffix.lower()
    try:
        if ext == ".zip":
            with zipfile.ZipFile(p) as z:
                names = [f"{i.filename}  ({i.file_size:,}B)" for i in z.infolist()
                         if not i.is_dir()]
        else:
            mode = "r:gz" if ext == ".gz" else ("r:bz2" if ext == ".bz2" else
                                                 ("r:xz" if ext == ".xz" else "r"))
            with tarfile.open(p, mode) as t:
                names = [f"{m.name}  ({m.size:,}B)" for m in t.getmembers() if m.isfile()]
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError) as e:
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="read", status="error", reason=f"could not list archive: {type(e).__name__}: {e}",
                           path=str(p), duration_ms=dur)

    truncated = len(names) > max_archive_members
    body = "\n".join(f"{i:>4} | {n}" for i, n in enumerate(names[:max_archive_members], 1))
    text = (f"ARCHIVE: {p.name}  ({len(names)} file(s), "
            f"{p.stat().st_size:,}B compressed){'  [TRUNCATED]' if truncated else ''}\n" + body)
    dur = (time.perf_counter() - t0) * 1000
    return Observation(tool="read", status="ok", path=str(p), start_line=1,
                       n_lines=min(len(names), max_archive_members), total_lines=len(names),
                       truncated=truncated, text=text,
                       meta={"kind": "archive_listing", "members": len(names)},
                       duration_ms=dur)


def read_file(path=None, *, ctx: TaskContext, start_line: int = 1, max_lines: int | None = None,
              max_bytes: int | None = None, handle: str | None = None) -> Observation:
    t0 = time.perf_counter()
    policy = AccessPolicy.from_settings(ctx.settings, cwd=ctx.root or Path.cwd())

    # Handle paging via handle if provided
    resolved_path = None
    start = start_line
    if handle is not None:
        # Parse handle - format roughly like "tool:abs_path:start_line"
        # But be self-contained and recheck against policy
        try:
            parts = handle.split(":", 2)
            if len(parts) >= 2:
                _toolh, pathh = parts[0], parts[1]
                if len(parts) == 3 and parts[2].isdigit():
                    start = int(parts[2])
                resolved_path = policy.resolve(pathh)
            else:
                resolved_path = policy.resolve(handle)
        except AccessDenied as e:
            dur = (time.perf_counter() - t0) * 1000
            return Observation(tool="read", status="blocked", reason=str(e), duration_ms=dur)
        except Exception as e:
            dur = (time.perf_counter() - t0) * 1000
            return Observation(tool="read", status="error", reason=str(e), duration_ms=dur)
    else:
        if path is None:
            dur = (time.perf_counter() - t0) * 1000
            return Observation(tool="read", status="error", reason="path or handle required", duration_ms=dur)
        try:
            resolved_path = policy.resolve(path)
        except AccessDenied as e:
            dur = (time.perf_counter() - t0) * 1000
            return Observation(tool="read", status="blocked", reason=str(e), duration_ms=dur)

    p = resolved_path
    limits = ctx.settings.limits
    max_l = max_lines if max_lines is not None else limits.read_max_lines
    max_b = max_bytes if max_bytes is not None else limits.read_max_bytes
    binary_sniff = limits.binary_sniff

    if not p.is_file():
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="read", status="not_found", path=str(p), reason="not a file", duration_ms=dur)

    ext = p.suffix.lower()
    try:
        size = p.stat().st_size
    except OSError as e:
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="read", status="error", path=str(p), reason=str(e), duration_ms=dur)

    if ext in LISTABLE_ARCHIVE_EXT:
        return _list_archive(p, t0, limits.max_archive_members)

    if ext in NO_TEXT_EXT:
        dur = (time.perf_counter() - t0) * 1000
        hint = {".pdf": "install pypdf", ".docx": "install python-docx",
                ".doc": "install antiword/libreoffice",
                ".xlsx": "install openpyxl"}.get(ext, f"{ext} is binary")
        return Observation(tool="read", status="skipped", path=str(p), reason=hint,
                           meta={"size_bytes": size, "ext": ext}, duration_ms=dur)

    if size > max_b * 4:
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="read", status="skipped", path=str(p),
                           reason=f"file is {size:,}B, too large to read blindly",
                           meta={"size_bytes": size, "ext": ext}, duration_ms=dur)

    try:
        with open(p, "rb") as fh:
            head = fh.read(binary_sniff)
            if b"\x00" in head:
                dur = (time.perf_counter() - t0) * 1000
                return Observation(tool="read", status="binary", path=str(p),
                                   reason="contains NUL bytes", duration_ms=dur)
            fh.seek(0)
            raw = fh.read(max_b)
    except OSError as e:
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="read", status="error", path=str(p), reason=str(e), duration_ms=dur)

    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception as e:
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="read", status="error", path=str(p), reason=str(e), duration_ms=dur)

    lines = text.splitlines()
    start = max(1, int(start))
    window = lines[start - 1: start - 1 + max_l]
    shown = "\n".join(f"{start + i:>5} | {ln}" for i, ln in enumerate(window))

    truncated = len(window) < (len(lines) - start + 1)
    handle_res = f"read:{str(p)}:{start + len(window)}" if truncated else None
    dur = (time.perf_counter() - t0) * 1000
    return Observation(tool="read", status="ok", path=str(p), start_line=start,
                       n_lines=len(window), total_lines=len(lines),
                       truncated=truncated, text=shown, handle=handle_res,
                       duration_ms=dur)
