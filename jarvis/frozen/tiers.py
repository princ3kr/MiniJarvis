"""Tier floors. Frozen: see `jarvis/frozen/__init__.py` for why this is a package.

A grammar validates FLAGS. It does not describe what a binary DOES with legal flags,
so `tar -cf out.tar /home` is a clean, fully-validated, category-2 command. The
notebook had this knowledge -- a tier in the same table as the flag validator -- and
the split into separate grammar tables lost it, which is why it is restored here as an
explicit floor rather than as a grammar entry.

Tier 1 writes inside the workspace, reversible by snapshot.
Tier 2 builds or fetches: network, package managers, compilers, debuggers.
"""
from __future__ import annotations

from pathlib import Path

from ..access import SECRET_DIRS
from . import enabled

# Writes inside the workspace. Reversible by snapshot once a sandbox exists.
TIER1_FLOOR: frozenset[str] = frozenset({
    "mkdir", "touch", "cp", "mv", "ln", "tar", "unzip", "zip", "install", "patch",
    "rm", "chmod", "chown", "truncate", "dd", "shred",
})

# Builds or fetches. Where network and package managers live.
TIER2_FLOOR: frozenset[str] = frozenset({
    "awk", "make", "cmake", "ninja", "cargo", "go", "java", "javac", "gradle",
    "mvn", "pip", "pip3", "npm", "npx", "yarn", "pnpm", "apt", "apt-get", "dnf",
    "yum", "pacman", "brew", "rsync", "screen", "tmux", "gdb", "strace",
    "curl", "wget", "ssh", "scp", "nc", "socat", "ftp", "telnet",
})

# Interpreters run arbitrary code that no argument check can see inside. Kept separate
# from TIER2_FLOOR because the rule is different: an interpreter is not "riskier tier
# 2", it is "unclassifiable", and no ceiling makes it safe without a sandbox.
INTERPRETERS: frozenset[str] = frozenset({
    "python", "python3", "node", "bash", "sh", "zsh", "perl", "ruby", "php",
    "deno", "bun", "julia", "Rscript", "tclsh", "wish",
})


def floor_for(binary: str) -> int | None:
    """The minimum tier for `binary`, or None if it has no floor.

    Returned as an int rather than a `Tier` so this module does not import `policy` --
    `policy` imports this module, and a cycle here would be a genuinely unpleasant
    thing to debug.
    """
    if binary in INTERPRETERS:
        return 2
    if binary in TIER2_FLOOR:
        return 2
    if binary in TIER1_FLOOR:
        return 1
    return None


def apply(settings, binary: str, tier, reasons: list[str]):
    """Raise `tier` to `binary`'s floor. Returns `tier` unchanged when frozen.

    The `enabled()` check is the guard: with the ceiling at 0 -- the default, and the
    shipped configuration -- this returns immediately and the frozen sets are never
    consulted. They are still IMPORTED, which is why the sets live in this module
    rather than being inlined here: importing a frozenset of 30 strings is cheap and
    harmless, and not consulting it is the meaningful part.
    """
    if not enabled(settings):
        return tier
    want = floor_for(binary)
    if want is None:
        return tier
    if binary in INTERPRETERS:
        reasons.append(
            f"{binary} runs arbitrary code; isolation is 'none', so no ceiling makes "
            f"this safe without a sandbox")
        want = 2
    elif want == 2:
        reasons.append(f"{binary} builds or fetches -> tier 2 minimum")
    else:
        reasons.append(f"{binary} writes -> tier 1 minimum")
    return tier if int(tier) >= want else want


def required_exclusions(access) -> set[str]:
    """Directories a recursive walk must exclude. `SECRET_DIRS` only.

    NOT `skip_dirs`. `.git`, `node_modules` and `__pycache__` are NOISE, not secrets,
    and requiring a model to pass thirty `--exclude-dir` flags for them made `du` and
    `grep -r` unusable -- the agent got refused, could not construct a command that
    would pass, and burned turns guessing. The secrecy requirement is short enough to
    state in one error message, which is what an agent can actually act on.
    """
    return (set(SECRET_DIRS)
            | {Path(g).name for g in access.deny_globs
               if Path(g).name and "*" not in Path(g).name
               and Path(g).name in SECRET_DIRS})