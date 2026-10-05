from __future__ import annotations

import time
from pathlib import Path

from ..access import AccessPolicy, AccessDenied, SECRET_DIRS
from ..config import CATEGORIES, Settings
from ..context import TaskContext
from .observation import Observation


def _survey_command(base: Path, policy: AccessPolicy, *,
                    include_skipped_dirs: bool,
                    count_only: bool = False,
                    count_extensions: set[str] | None = None) -> list[str]:
    """Build a native find walk with secret directories pruned before descent."""
    prune_paths: set[str] = set()
    file_denials: list[tuple[str, str]] = []
    for pattern in policy.deny_globs:
        expanded = str(Path(pattern).expanduser()) if pattern.startswith("~") else pattern
        if expanded.endswith("/*"):
            prune_paths.add(expanded[:-2])
        elif "/" in expanded:
            prune_paths.add(expanded)
            file_denials.append(("-path", expanded))
        else:
            file_denials.append(("-name", expanded))

    cmd = ["find", str(base), "(", "-type", "d", "("]
    directory_names = set(SECRET_DIRS)
    if not include_skipped_dirs:
        directory_names.update(policy.skip_dirs)
    predicates: list[list[str]] = [["-name", name] for name in sorted(directory_names)]
    predicates.extend([["-path", path] for path in sorted(prune_paths)])
    for index, predicate in enumerate(predicates):
        if index:
            cmd.append("-o")
        cmd.extend(predicate)
    cmd += [")", "-prune", ")", "-o", "(", "-type", "f"]
    if count_extensions:
        cmd.append("(")
        for index, extension in enumerate(sorted(count_extensions)):
            if index:
                cmd.append("-o")
            cmd += ["-iname", f"*{extension}"]
        cmd.append(")")
    for kind, pattern in file_denials:
        cmd += ["!", kind, pattern]
    if count_only:
        cmd += ["-printf", r"\n", ")"]
    else:
        cmd += ["-print0", ")"]
    return cmd


def survey_files(*, ctx: TaskContext, root: Path | str | None = None,
                 categories: list[str] | str | None = None, extensions: list[str] | None = None,
                 include_skipped_dirs: bool = False,
                 count_only: bool = False) -> Observation:
    t0 = time.perf_counter()
    policy = AccessPolicy.from_settings(ctx.settings, cwd=ctx.root or Path.cwd())
    try:
        base = policy.resolve(root or ctx.root or ctx.settings.read_roots[0])
    except AccessDenied as e:
        dur = (time.perf_counter() - t0) * 1000
        return Observation(tool="survey", status="blocked", reason=str(e), duration_ms=dur)

    selected_categories = ([categories] if isinstance(categories, str)
                           else list(categories or []))
    cats_list = selected_categories or list(CATEGORIES.keys())
    extension_filter = {
        e.lower() if e.startswith(".") else f".{e.lower()}" for e in (extensions or [])
    }
    wanted: set[str] = set()
    for c in selected_categories or ([] if extension_filter else cats_list):
        wanted |= CATEGORIES.get(c, set())
    wanted |= extension_filter
    if count_only and len(selected_categories) > 1:
        count_only = False
    count_extensions = (wanted if count_only and (selected_categories or extensions)
                        else None)

    by_ext: dict[str, int] = {}
    by_cat: dict[str, int] = {c: 0 for c in cats_list}
    by_top: dict[str, int] = {}
    sample: list[str] = []
    seen = 0
    matched = 0
    walk_errors: list[str] = []

    def count_path(raw_path: bytes) -> None:
        nonlocal seen, matched
        p = Path(raw_path.decode("utf-8", errors="surrogateescape"))
        seen += 1
        ext = p.suffix.lower()
        by_ext[ext or "(none)"] = by_ext.get(ext or "(none)", 0) + 1
        try:
            rel_parts = p.relative_to(base).parts
        except ValueError:
            rel_parts = p.parts[-2:] if len(p.parts) >= 2 else p.parts
        top = rel_parts[0] if rel_parts else "(root)"
        by_top[top] = by_top.get(top, 0) + 1
        if ext in wanted:
            matched += 1
            for category in cats_list:
                if ext in CATEGORIES.get(category, set()):
                    by_cat[category] += 1
            if len(sample) < 200:
                sample.append(str(p))

    from .base import run_tool
    result = run_tool(
        _survey_command(base, policy, include_skipped_dirs=include_skipped_dirs,
                        count_only=count_only, count_extensions=count_extensions),
        cwd=base, timeout_s=ctx.settings.limits.survey_timeout_s,
        stdout_record_callback=None if count_only else count_path,
        stdout_record_separator=b"\0",
        count_stdout_lines=count_only)
    if result.status != "ok":
        walk_errors.append(result.reason or result.stderr.strip() or result.status)
    elif count_only:
        count = result.stdout_line_count or 0
        matched = count
        if count_extensions is None:
            seen = count
        elif len(selected_categories) == 1:
            by_cat[selected_categories[0]] = count

    dur = (time.perf_counter() - t0) * 1000
    complete = not walk_errors
    return Observation(
        tool="survey",
        status="ok" if complete else "error",
        items=[],
        text="",
        total_lines=seen,
        n_lines=seen,
        meta={
            "root": str(base),
            "files_seen": None if count_only and count_extensions is not None else seen,
            "by_extension": dict(sorted(by_ext.items(), key=lambda kv: -kv[1])[:25]),
            "by_category": by_cat,
            "by_toplevel": dict(sorted(by_top.items(), key=lambda kv: -kv[1])[:15]),
            "matched_count": matched,
            "complete": complete,
            "included_skipped_dirs": include_skipped_dirs,
            "count_only": count_only,
            "sample": sample[:12],
            "excluded_dirs": sorted(policy.skip_dirs),
            "note": ("Counts include performance-skipped directories; denied paths "
                     "remain excluded." if include_skipped_dirs else
                     "Counts exclude pruned and denied paths."),
        },
        reason="; ".join(walk_errors[:3]),
        duration_ms=dur,
    )
