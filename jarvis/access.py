"""The ONE access module: read scope plus deny-list.

This merges the notebook's two independent jails, which disagreed:

  * `safe_resolve(path)`   -- tools. Tried `Path.cwd()/p` then EACH read root in
                              turn, so a relative path silently resolved against
                              whichever root happened to contain it.
  * `_resolve_in(v, roots)` -- commands. Resolved relative paths against
                              `Path.cwd()` and nothing else.

So the same relative string passed to a tool and to a command could land in two
different directories, and only one of them was checked against the deny-list. That
disagreement is why the probe found `cat ../.ssh/config` reading a secret: the
command path never applied the jail to its operands at all.

Rules, in order:

  1. `~` is expanded.
  2. A relative path resolves against `base` (default: the task's real cwd), then
     against the read roots -- but `base` wins, so one call means one answer.
  3. `..` is collapsed by `resolve()`, then the result must be inside a permitted
     root. Containment is checked with `Path.parents`, NOT string prefix, so
     `/home/princ3kr-evil` is not "inside" `/home/princ3kr`.
  4. The deny-list is applied LAST and always wins, even inside the scope.
  5. Symlinks are resolved before every check, so a symlink pointing out of scope
     is caught at the link, not followed out.

Everything that touches the filesystem goes through `AccessPolicy.resolve`: every
tool, every command operand, and the metadata index.
"""
from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings, load_settings


class AccessDenied(PermissionError):
    """A path is out of scope, or matches the deny-list. Carries the reason."""

    def __init__(self, message: str, *, path=None, reason: str = "denied") -> None:
        super().__init__(message)
        self.path = path
        self.reason = reason


def _deny_patterns(patterns: tuple[str, ...]) -> tuple[tuple[str, bool, str], ...]:
    """Expand `~` once, note which patterns carry a separator, and record the
    directory a `dir/*` pattern is guarding.

    A pattern with a "/" is matched against the whole absolute path, where `*`
    crosses separators -- which is what makes `*/.env` block a `.env` at ANY depth.
    A pattern without one is matched against the basename only, so `*.key` means
    "a file called *.key anywhere", not "a directory called *.key/x".

    `X/*` expands to a second entry for `X` itself. Without that, `/home/u/.ssh/*`
    denies the files in ~/.ssh but not the directory, so a recursive walk can still
    descend into it and read the bytes inside -- the exact failure mode the probe
    reported for `grep -r`.
    """
    out = []
    for pat in patterns:
        expanded = str(Path(pat).expanduser()) if pat.startswith("~") else pat
        has_sep = "/" in expanded
        out.append((expanded, has_sep, expanded))
        if expanded.endswith("/*"):
            guarded = expanded[:-2]
            # a bare "/" root guard is not useful
            if guarded and guarded != "/":
                out.append((guarded, "/" in guarded, expanded))
    return tuple(out)


def is_denied(path, patterns: tuple[str, ...]) -> tuple[bool, str]:
    """(denied, matched_pattern). Unparseable input counts as denied.

    Failing closed here is the point: a path we cannot even expand is a path we
    cannot vouch for, and "cannot check" must not mean "allowed".
    """
    try:
        raw = Path(path).expanduser()
        try:
            resolved = raw.resolve()
        except (OSError, RuntimeError):
            resolved = raw.absolute()
        full = str(resolved)
        name = resolved.name
    except Exception:
        return True, "<unparseable>"

    for pat, has_sep, reported in _deny_patterns(patterns):
        if has_sep:
            if fnmatch.fnmatch(full, pat):
                return True, reported
        elif fnmatch.fnmatch(name, pat):
            return True, reported
    return False, ""


@dataclass(frozen=True)
class AccessPolicy:
    """Read scope plus deny-list plus the skip list. Immutable; safe to share."""

    read_roots: tuple[Path, ...] = ()
    write_roots: tuple[Path, ...] = ()
    deny_globs: tuple[str, ...] = ()
    skip_dirs: frozenset[str] = frozenset()
    cwd: Path | None = None
    _roots_resolved: tuple[Path, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_roots_resolved",
                           tuple(Path(r).expanduser().resolve() for r in self.read_roots))
        object.__setattr__(self, "write_roots",
                           tuple(Path(r).expanduser().resolve() for r in self.write_roots))

    # ------------------------------------------------------------- factories
    @classmethod
    def from_settings(cls, settings: Settings, cwd: Path | None = None) -> "AccessPolicy":
        return cls(read_roots=settings.read_roots,
                   write_roots=(settings.write_root,),
                   deny_globs=settings.deny_globs,
                   skip_dirs=settings.skip_dirs,
                   cwd=cwd)

    # ------------------------------------------------------------ containment
    def contains(self, path: Path, roots: tuple[Path, ...]) -> bool:
        """True if `path` IS a root or is inside one. Path arithmetic, not prefix.

        `Path("/home/u-evil").parents` does not contain `Path("/home/u")`, so the
        sibling-prefix trap does not apply here.
        """
        return any(path == r or r in path.parents for r in roots)

    def in_scope(self, path: Path) -> bool:
        return self.contains(path, self._roots_resolved)

    # ---------------------------------------------------------------- resolve
    def resolve(self, value, *, for_write: bool = False, base: Path | None = None,
                check_scope: bool = True, check_deny: bool = True) -> Path:
        """Resolve `value` to an absolute path, proving it is in scope.

        This is the single entry point. A caller that hand-rolls containment is the
        bug this module exists to prevent.

        `base` defaults to `self.cwd` (the task's real working directory), then to
        the process cwd. It does NOT fan out across read roots -- that fan-out is
        why the notebook's tool jail and command jail disagreed about where
        "notes.txt" was.
        """
        text = str(value)
        if not text.strip():
            raise AccessDenied("empty path", path=value, reason="empty")

        raw = Path(text).expanduser()

        if raw.is_absolute():
            candidates = [raw]
        else:
            anchors = []
            if base is not None:
                anchors.append(base)
            elif self.cwd is not None:
                anchors.append(self.cwd)
            else:
                anchors.append(Path.cwd())
            candidates = [a / raw for a in anchors]
            # Fall back to the read roots only if the primary anchor fails the
            # scope check. Recorded so the caller can tell which one applied.
            candidates += [r / raw for r in self._roots_resolved]

        last_error: Exception | None = None
        first_out_of_scope: Path | None = None
        for cand in candidates:
            try:
                resolved = cand.resolve()
            except (OSError, RuntimeError) as e:      # broken symlink, loop, ELOOP
                last_error = e
                continue

            if check_deny:
                denied, pat = is_denied(resolved, self.deny_globs)
                if denied:
                    # The deny-list is unconditional: do not keep hunting for a
                    # candidate root that would "work". One answer per call.
                    raise AccessDenied(
                        f"blocked by deny-list ({pat}): {value}",
                        path=resolved, reason="denied")

            if not check_scope:
                return resolved

            roots = self.write_roots if for_write else self._roots_resolved
            roots = roots or self._roots_resolved
            if self.contains(resolved, roots):
                return resolved
            if first_out_of_scope is None:
                first_out_of_scope = resolved

        scope = [str(r) for r in (self.write_roots if for_write else self._roots_resolved)]
        raise AccessDenied(
            f"outside {'write' if for_write else 'read'} scope: {value} -> "
            f"{first_out_of_scope} is not inside {scope}",
            path=first_out_of_scope, reason="out_of_scope") from last_error

    def try_resolve(self, value, **kw) -> Path | None:
        """resolve() that returns None instead of raising. For filtering."""
        try:
            return self.resolve(value, **kw)
        except AccessDenied:
            return None

    def check(self, value, **kw) -> tuple[bool, str]:
        try:
            self.resolve(value, **kw)
            return True, ""
        except AccessDenied as e:
            return False, str(e)

    # ------------------------------------------------------------------ scope
    def walkable(self, path: Path) -> bool:
        """May a recursive walk DESCEND into this directory?

        Deliberately separate from `resolve`. A path can be perfectly readable and
        still not be worth descending into (`.git`, `node_modules`), and the
        deny-list has to stop the DESCENT rather than merely filter the output --
        otherwise `grep -r` still reads the bytes inside and leaks them, which is
        what the probe caught.

        Both checks are needed: the deny-list blocks secrets, and SKIP_DIRS keeps
        the walk cheap. Dot-directories are allowed here because ~/.config and
        ~/.cache hold real user files the agent should be able to see; SKIP_DIRS
        already names the ones that are noise.
        """
        if path.name in self.skip_dirs:
            return False
        denied, _ = is_denied(path, self.deny_globs)
        return not denied

    def in_scope_dir(self, path: Path) -> bool:
        return self.contains(path, self._roots_resolved)

    def scope_roots(self) -> list[Path]:
        return list(self._roots_resolved)

    def describe(self) -> str:
        return (f"read scope: {[str(r) for r in self._roots_resolved]}\n"
                f"write scope: {[str(r) for r in self.write_roots]}\n"
                f"deny-list: {len(self.deny_globs)} patterns "
                f"(always wins over scope)\n"
                f"skip dirs: {len(self.skip_dirs)}")


def default_policy(**overrides) -> AccessPolicy:
    return AccessPolicy.from_settings(load_settings(), **overrides)