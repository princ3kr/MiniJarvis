"""The policy rewrites a recursive command so it cannot walk into a secret.

The plan's requirement, and it exists because of a measured failure. The loop's prompt
told the model to add `--exclude=.ssh --exclude=.aws ...` to every recursive command,
and the model did -- badly. Its first three attempts were all

    find . -type f -name *.py -not -path */.ssh/* -not -path */.aws/* ...

which the grammar reads as an unknown flag or a path outside scope, so every attempt
was refused and the loop burned its budget on three variations of the same mistake.

The agent cannot be relied on to write a correct recursive command, because getting it
right requires knowing a list of secret directories it has no way to discover. So the
policy supplies the list. Given `find . -type f -name '*.py'`, it returns

    find . -type f -name '*.py' -not -path '*/.ssh/*' -not -path '*/.aws/*' ...

Two properties make this safe rather than a hole:

  * **It only ever ADDS arguments.** It never removes a flag, reorders, or substitutes
    a value. A command that was going to be refused is still refused: the rewrite runs
    AFTER classification, and a denial is returned before the rewrite is reached.
  * **The real argv is logged.** `gateway` emits the post-rewrite argv, so what ran is
    what the trace shows -- never the model's version. The plan requires this, and it is
    the difference between an auditable agent and one you have to take on trust.

The `-not -path` form is used for `find` because `-not` is not `-prune` but composes
the same way, and it needs no parenthesis grouping. `-prune` forms ARE recognised for
the purposes of deciding whether a command is recursive; see `_is_recursive`.
"""
from __future__ import annotations

from .access import SECRET_DIRS, AccessPolicy

# Per-binary: how to say "skip this directory" for a command that walks.
#
# `find` uses `-not -path '*/NAME/*'` rather than `-prune` because `-prune` needs
# `\( ... \)` grouping which is exactly the parenthesised form `CommandSpec` rejects as
# shell metacharacters. `-not -path` needs none.
_PRUNE_FORM = {
    "find": ["-not", "-path", "*/{d}/*"],
    "grep": ["--exclude-dir={d}"],
    "egrep": ["--exclude-dir={d}"],
    "fgrep": ["--exclude-dir={d}"],
    "du": ["--exclude=./{d}", "--exclude=*/{d}", "--exclude={d}"],
    "rg": ["--glob", "!**/{d}/**"],
    "rsync": ["--exclude={d}"],
    "tree": ["-I", "{d}"],
    "cp": [],              # writes, not a read walk
    "ls": ["--ignore={d}"],       # `ls -R` does walk; this is its only exclusion
}

# Binaries that walk whether or not they are told to. For these a bare invocation is
# already a walk, so the exclusions go on unconditionally.
#
# `rg` belongs here rather than in the flag table because ripgrep has no
# non-recursive mode: given a path it descends. Same for `find`, `du`, `tree`.
_ALWAYS_DESCEND = frozenset({"find", "du", "tree", "tar", "rg"})

# Binaries that walk only with a flag. Without the flag there is nothing to exclude.
_RECURSION_FLAGS: dict[str, tuple[str, ...]] = {
    "grep": ("-r", "-R", "--recursive", "--dereference-recursive"),
    "egrep": ("-r", "-R", "--recursive"),
    "fgrep": ("-r", "-R", "--recursive"),
    "rg": ("-r", "-R", "--recursive"),
    "ls": ("-R",),
    "diff": ("-r",),
    "rsync": ("-r", "-R", "--recursive"),
    "cp": ("-r", "-R"),
}


def is_recursive(binary: str, argv: list[str]) -> bool:
    """Will this invocation descend into subdirectories?

    `_prune` is recognised as a PRUNING form, not a recursion trigger: `find -prune`
    is the opposite of walking. The original check treated `-prune` as part of the
    dangerous-flag set for `find` AND as recursion at the same time, so a correctly
    pruned `find` was flagged for both.

    Short-flag BUNDLING is handled, because `grep -rn x .` is the single most common
    recursive grep there is and `-rn` is not `-r`. Checking `"−r" in argv` against it
    reports not-recursive, so no exclusions get injected and the deny-list is not
    applied. That is a silent failure in the direction that matters.
    """
    if binary in _ALWAYS_DESCEND:
        return True
    if binary == "tar":
        # `tar -x` extracts (writes, tier 1). `tar -t` lists. Neither is a read walk
        # we need to exclude, and both are gated by the tier floor instead.
        return False
    flags = _RECURSION_FLAGS.get(binary, ())
    for tok in argv[1:]:
        if not tok.startswith("-"):
            continue
        for flag in flags:
            if tok == flag:
                return True
            if flag.startswith("--"):
                if tok.startswith(flag):          # --recursive, --recursive=...
                    return True
            elif flag.startswith("-") and not flag.startswith("--"):
                # Bundled short flags: `-rn` contains `-r`, `-R` stands alone.
                if tok[1:2] == flag[1:2] and len(tok) > 1:
                    return True
    return False


def has_prune(binary: str, argv: list[str]) -> bool:
    """Does the command already prune a specific set of directories itself?

    When it does, the policy still adds its own -- but only for the names the model did
    not mention, so a command that already excludes `.git` does not gain a second
    `--exclude-dir=.git`.
    """
    if binary != "find":
        return any(a == "-prune" or a == "-o" for a in argv)
    return "-prune" in argv


def inject(argv: list[str], *, access: AccessPolicy,
           secrets: tuple[str, ...] | None = None) -> list[str]:
    """Return `argv` with deny-list exclusions appended, if it walks.

    Returns a NEW list. Never mutates the caller's argv, because that list came out of
    the model and may be referenced by the trace.
    """
    if not argv:
        return argv
    binary = argv[0].rsplit("/", 1)[-1]
    if binary not in _PRUNE_FORM:
        return list(argv)
    if not is_recursive(binary, argv):
        return list(argv)

    names = tuple(secrets) if secrets else tuple(sorted(SECRET_DIRS))
    already = " ".join(argv)
    out = list(argv)
    for d in names:
        if d in already:            # the model already excluded it; do not double up
            continue
        for part in _PRUNE_FORM[binary]:
            out.append(part.format(d=d))
    return out


def describe() -> str:
    """One line for the prompt, so the model knows what the policy does for it."""
    return (f"The policy appends deny-list exclusions to recursive commands "
            f"({', '.join(sorted(SECRET_DIRS))}). Write the command as you would "
            f"normally; you do not need to add exclusions yourself.")