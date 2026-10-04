# jarvis/policy.py
"""Policy: per-binary allowlist grammars and risk classification.

The shell metacharacter rejection is a blunt instrument and STAYS — ported
verbatim. Unknown flags escalate to APPROVAL, never to tier 0. Absolute
binary resolution must go through a TRUSTED DIRECTORY allowlist.
"""
from __future__ import annotations

import os
import re
import shutil
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .access import AccessDenied, AccessPolicy
from .context import TaskContext  # for reference
from .redaction import redact


class Tier(IntEnum):
    READ_ONLY = 0
    REVERSIBLE = 1
    APPROVAL = 2
    DENIED = 3


TRUSTED_BIN_DIRS = (
    "/usr/bin",
    "/bin",
    "/usr/local/bin",
    "/sbin",
    "/usr/sbin",
    "/usr/local/sbin",
)


# Keep this reasoning in a comment — "no argument check can see inside it"
INTERPRETERS = {
    "python",
    "python3",
    "node",
    "bash",
    "sh",
    "zsh",
    "perl",
    "ruby",
    "pytest",
    "make",
    "cargo",
    "go",
    "java",
    "deno",
    "bun",
    "php",
    "nodejs",
}


NETWORK_BINARIES = {
    "curl",
    "wget",
    "ssh",
    "scp",
    "sftp",
    "nc",
    "ncat",
    "telnet",
    "ftp",
    "rsync",
    "pip",
    "pip3",
    "npm",
    "yarn",
    "apt",
    "apt-get",
    "dnf",
    "yum",
    "pacman",
    "brew",
    "git-remote-http",
}


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "file_exists",
        "file_absent",
        "exit_code",
        "stdout_contains",
        "stderr_contains",
        "dir_listed",
    ]
    value: str = Field(max_length=400)
    path: str | None = None


class CommandSpec(BaseModel):
    """What the LLM proposes. Never a shell string -- argv only."""

    model_config = ConfigDict(extra="forbid")

    argv: list[str] = Field(min_length=1, max_length=256)
    cwd: str | None = None
    why: str = Field(min_length=1, max_length=500)
    expected_exit: int = Field(default=0, ge=-9, le=125)
    expected_effects: list[str] = Field(default_factory=list, max_length=20)
    checks: list[Check] = Field(default_factory=list, max_length=10)
    # set by the orchestrator when the command's arguments echo text we READ.
    # A file can say "run this to fix it"; taint forces approval even at tier 0/1.
    tainted: bool = False

    @field_validator("argv")
    @classmethod
    def _no_shell_tokens(cls, argv):
        """v1: single command only. Reject shell metacharacters outright rather
        than trying to reason about them.

        Keep the comment and behavior verbatim.
        """
        joined = " ".join(argv)
        bad = [
            t
            for t in ("|", ";", "&&", "||", ">", ">>", "<", "$(", "`", "\n", "&")
            if t in joined
        ]
        if bad:
            raise ValueError(f"shell metacharacters rejected in v1: {bad}")
        if argv and argv[0].startswith("-"):
            raise ValueError("argv[0] must be a binary, not a flag")
        return argv


class PolicyVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: Tier
    reason: str
    binary: str
    binary_resolved: str = ""
    reasons: list[str] = Field(default_factory=list)

    @property
    def denied(self) -> bool:
        return self.tier is Tier.DENIED


# Grammar types
FlagValueType = Literal["none", "path", "number", "size", "string", "any"]


class FlagSpec:
    def __init__(self, name: str, takes: FlagValueType = "none", aliases: tuple[str, ...] = ()):
        self.name = name
        self.takes = takes
        self.aliases = aliases


class Grammar:
    def __init__(self, binary: str, flags: list[FlagSpec] | None = None, allow_unknown: bool = False):
        self.binary = binary
        self.flags = flags or []
        self.allow_unknown = allow_unknown
        # build lookup
        self._map: dict[str, FlagSpec] = {}
        for f in self.flags:
            self._map[f.name] = f
            for a in f.aliases:
                self._map[a] = f


def _split_flag(token: str) -> tuple[str, str | None]:
    if token.startswith("--"):
        if "=" in token:
            return token.split("=", 1)
        return token, None
    if token.startswith("-") and len(token) > 1 and "=" in token:
        return token.split("=", 1)
    return token, None


def _is_flag_like(token: str) -> bool:
    if not token:
        return False
    if token == "-":
        return False
    if token.startswith("-"):
        return True
    return False


class GrammarResult:
    def __init__(self, status: Literal["ok", "deny", "unknown"], reason: str = "", path_positions: list[int] | None = None, **kw):
        self.status = status
        self.reason = reason
        self.path_positions = path_positions or []
        for k, v in kw.items():
            setattr(self, k, v)


DANGEROUS_FIND_HEADS_EXACT = {
    "-exec",
    "-execdir",
    "-ok",
    "-okdir",
    "-delete",
    "-fls",
    "-fprintf",
}


def _has_dangerous_find_flag(argv: list[str]) -> tuple[bool, str]:
    for tok in argv[1:]:
        head = _head_of(tok)
        if head in DANGEROUS_FIND_HEADS_EXACT:
            return True, f"dangerous find flag: {tok}"
        if head.startswith("-fprint"):
            return True, f"dangerous find flag: {tok}"
    return False, ""


def _head_of(token: str) -> str:
    if token.startswith("--"):
        if "=" in token:
            return token.split("=", 1)[0]
        return token
    if token.startswith("-") and "=" in token:
        return token.split("=", 1)[0]
    return token


def validate_grammar(grammar: Grammar, argv: list[str]) -> GrammarResult:
    """Validate argv against grammar. Returns status ok/deny/unknown."""
    if len(argv) == 0:
        return GrammarResult("deny", "empty argv")

    i = 1  # skip binary
    path_positions: list[int] = []
    n = len(argv)
    while i < n:
        tok = argv[i]
        if not _is_flag_like(tok):
            path_positions.append(i)
            i += 1
            continue
        head, val_eq = _split_flag(tok)
        spec = grammar._map.get(head)
        if spec is None:
            # POSIX short-option bundling: `-cf`, `-rf`, `-la` are ONE token, not
            # three. Without this, `tar -cf out.tar .` classifies as "unknown flag
            # -cf" and escalates to APPROVAL for a command that is perfectly
            # ordinary -- the refusal is caused by the policy's own wiring, not by
            # anything the model did wrong.
            step = _short_bundle(tok, grammar, argv, i)
            if step is None:
                return GrammarResult("unknown", f"unknown flag {head}", path_positions)
            # A value-taking final letter consumes the next token, which may be a
            # PATH operand and must be scope-checked like any other.
            if step.get("consumes"):
                if i + 1 >= n:
                    return GrammarResult("deny",
                                         f"flag {head} requires a value", path_positions)
                nxt = argv[i + 1]
                if _is_flag_like(nxt):
                    return GrammarResult("deny",
                                         f"flag {head} requires a value, got flag {nxt}",
                                         path_positions)
                if step["takes"] == "path":
                    path_positions.append(i + 1)
            i += step["advance"]
            continue
        if spec.takes == "none":
            if val_eq is not None:
                return GrammarResult("deny", f"flag {head} takes no value")
            i += 1
            continue
        if val_eq is not None:
            i += 1
            continue
        if i + 1 >= n:
            return GrammarResult("deny", f"flag {head} requires a value")
        next_tok = argv[i + 1]
        if _is_flag_like(next_tok):
            return GrammarResult("deny", f"flag {head} requires a value, got flag {next_tok}")
        i += 2
        continue
    return GrammarResult("ok", "", path_positions)


def _short_bundle(token: str, grammar: Grammar, argv: list[str],
                  i: int) -> dict | None:
    """`-cf` -> {'advance': 2, 'consumes': True, 'takes': 'path'} if it is a valid bundle.

    POSIX getopt: in a bundle, only the LAST letter may take a value. So `-cf out.tar`
    is `-c -f out.tar`, while `-fc out.tar` is not a legal bundle under that rule.
    Returns None when the token is not a bundle at all, so the caller can report
    "unknown flag" rather than a misleading parse error.
    """
    if not token.startswith("-") or token.startswith("--") or len(token) < 3:
        return None
    if "=" in token:
        return None
    letters = token[1:]
    last_takes = "none"
    for pos, ch in enumerate(letters):
        spec = grammar._map.get(f"-{ch}")
        if spec is None:
            return None
        if pos < len(letters) - 1 and spec.takes != "none":
            return None          # only the final letter may take a value
        last_takes = spec.takes
    return {"advance": 2 if last_takes != "none" else 1,
            "consumes": last_takes != "none",
            "takes": last_takes}


# Per-binary grammars
def make_find_grammar() -> Grammar:
    flags = [
        FlagSpec("-name", "string"),
        FlagSpec("-iname", "string"),
        FlagSpec("-path", "string"),
        FlagSpec("-ipath", "string"),
        FlagSpec("-regex", "string"),
        FlagSpec("-iregex", "string"),
        FlagSpec("-type", "string"),
        FlagSpec("-not", "none"),
        FlagSpec("-or", "none"),
        FlagSpec("-o", "none"),
        FlagSpec("-and", "none"),
        FlagSpec("-a", "none"),
        FlagSpec("-maxdepth", "number"),
        FlagSpec("-mindepth", "number"),
        FlagSpec("-depth", "none"),
        FlagSpec("-prune", "none"),
        FlagSpec("-print", "none"),
        FlagSpec("-print0", "none"),
        FlagSpec("-empty", "none"),
        FlagSpec("-true", "none"),
        FlagSpec("-false", "none"),
        FlagSpec("-size", "size"),
        FlagSpec("-mtime", "number"),
        FlagSpec("-mmin", "number"),
        FlagSpec("-ctime", "number"),
        FlagSpec("-cmin", "number"),
        FlagSpec("-atime", "number"),
        FlagSpec("-amin", "number"),
        FlagSpec("-perm", "string"),
        FlagSpec("-user", "string"),
        FlagSpec("-group", "string"),
        FlagSpec("-nouser", "none"),
        FlagSpec("-nogroup", "none"),
        FlagSpec("-newer", "path"),
        FlagSpec("-newerXY", "string"),
        FlagSpec("-fstype", "string"),
        FlagSpec("-mount", "none"),
        FlagSpec("-xdev", "none"),
        FlagSpec("-follow", "none"),
        FlagSpec("-readable", "none"),
        FlagSpec("-writable", "none"),
        FlagSpec("-executable", "none"),
        FlagSpec("-samefile", "path"),
    ]
    return Grammar("find", flags)


def _make_basic_read_grammar(names: list[str]) -> Grammar:
    flags = [
        FlagSpec("-n", "number"),
        FlagSpec("-c", "number"),
        FlagSpec("-b", "number"),
        FlagSpec("-q", "none"),
        FlagSpec("-v", "none"),
        FlagSpec("-r", "none"),
        FlagSpec("-R", "none"),
        FlagSpec("-h", "none"),
        FlagSpec("--help", "none"),
        FlagSpec("--version", "none"),
        FlagSpec("-z", "none"),
        FlagSpec("-s", "none"),
        FlagSpec("-0", "none"),
    ]
    return Grammar("basic", flags)


GREP_FLAGS = [
    FlagSpec("-E", "none"),
    FlagSpec("-F", "none"),
    FlagSpec("-G", "none"),
    FlagSpec("-P", "none"),
    FlagSpec("-e", "string"),
    FlagSpec("-f", "path"),
    FlagSpec("-i", "none"),
    FlagSpec("-v", "none"),
    FlagSpec("-w", "none"),
    FlagSpec("-x", "none"),
    FlagSpec("-c", "none"),
    FlagSpec("-l", "none"),
    FlagSpec("-L", "none"),
    FlagSpec("-n", "none"),
    FlagSpec("-o", "none"),
    FlagSpec("-r", "none"),
    FlagSpec("-R", "none"),
    FlagSpec("-q", "none"),
    FlagSpec("-h", "none"),
    FlagSpec("-H", "none"),
    FlagSpec("-a", "none"),
    FlagSpec("-I", "none"),
    FlagSpec("-B", "number"),
    FlagSpec("-A", "number"),
    FlagSpec("-C", "number"),
    FlagSpec("--max-count", "number"),
    FlagSpec("--max-count=", "number"),
    FlagSpec("--line-number", "none"),
    FlagSpec("--files-with-matches", "none"),
    FlagSpec("--files-without-match", "none"),
    FlagSpec("--color", "string"),
    FlagSpec("--no-color", "none"),
    FlagSpec("--version", "none"),
    FlagSpec("--help", "none"),
    FlagSpec("-m", "number"),
    FlagSpec("--mmap", "none"),
    FlagSpec("--null", "none"),
    FlagSpec("-Z", "none"),
    FlagSpec("--include", "string"),
    FlagSpec("--exclude", "string"),
    FlagSpec("--exclude-dir", "string"),
    FlagSpec("--include-dir", "string"),
]
# reject --pre, --pre-glob by treating as unknown? Or catch in grammar - easier to check head
GREPLIKE_BAD_HEADS = {"--pre", "--pre-glob"}


def make_grep_grammar() -> Grammar:
    g = Grammar("grep", list(GREP_FLAGS))
    return g


def make_rg_grammar() -> Grammar:
    g = make_grep_grammar()
    return g


def make_ls_grammar() -> Grammar:
    flags = [
        FlagSpec("-l", "none"),
        FlagSpec("-a", "none"),
        FlagSpec("-A", "none"),
        FlagSpec("-h", "none"),
        FlagSpec("-t", "none"),
        FlagSpec("-r", "none"),
        FlagSpec("-S", "none"),
        FlagSpec("-d", "none"),
        FlagSpec("-1", "none"),
        FlagSpec("-R", "none"),
        FlagSpec("-F", "none"),
        FlagSpec("--color", "string"),
        FlagSpec("--long", "none"),
        FlagSpec("--all", "none"),
        FlagSpec("--almost-all", "none"),
        FlagSpec("--human-readable", "none"),
        FlagSpec("--reverse", "none"),
        FlagSpec("--sort", "string"),
        FlagSpec("--time-style", "string"),
        FlagSpec("-Z", "none"),
    ]
    return Grammar("ls", flags)


def make_cat_grammar() -> Grammar:
    flags = [
        FlagSpec("-n", "none"),
        FlagSpec("-b", "none"),
        FlagSpec("-s", "none"),
        FlagSpec("-v", "none"),
        FlagSpec("-A", "none"),
        FlagSpec("-E", "none"),
        FlagSpec("-T", "none"),
        FlagSpec("-u", "none"),
        FlagSpec("-t", "none"),
        FlagSpec("-e", "none"),
        FlagSpec("-z", "none"),
        FlagSpec("--show-all", "none"),
        FlagSpec("--number", "none"),
        FlagSpec("--number-nonblank", "none"),
        FlagSpec("--squeeze-blank", "none"),
    ]
    return Grammar("cat", flags)


def make_basic_noextra() -> Grammar:
    return Grammar("basic", [])


def make_echo_date_uname_id_pwd() -> Grammar:
    flags = [
        FlagSpec("-n", "none"),
        FlagSpec("-e", "none"),
        FlagSpec("-E", "none"),
        FlagSpec("--help", "none"),
        FlagSpec("--version", "none"),
    ]
    return Grammar("simple", flags)


def make_diff_grammar() -> Grammar:
    flags = [
        FlagSpec("-u", "none"),
        FlagSpec("-c", "none"),
        FlagSpec("-p", "none"),
        FlagSpec("-r", "none"),
        FlagSpec("-q", "none"),
        FlagSpec("-w", "none"),
        FlagSpec("-b", "none"),
        FlagSpec("-B", "none"),
        FlagSpec("-a", "none"),
        FlagSpec("-N", "none"),
        FlagSpec("--unified", "number"),
        FlagSpec("--version", "none"),
        FlagSpec("--help", "none"),
    ]
    return Grammar("diff", flags)


def make_sort_uniq_tr_cut_which() -> Grammar:
    flags = [
        FlagSpec("-n", "none"),
        FlagSpec("-r", "none"),
        FlagSpec("-k", "string"),
        FlagSpec("-t", "string"),
        FlagSpec("-f", "none"),
        FlagSpec("-d", "none"),
        FlagSpec("-i", "none"),
        FlagSpec("-u", "none"),
        FlagSpec("-c", "none"),
        FlagSpec("-D", "none"),
        FlagSpec("-z", "none"),
        FlagSpec("-s", "none"),
        FlagSpec("--version", "none"),
        FlagSpec("--help", "none"),
    ]
    return Grammar("util", flags)


# NOTE: head/tail/wc/stat/file and du/df/sha/realpath/readlink/tree grammars are
# defined once below, after tar. An earlier draft defined them here too; the second
# definition shadowed the first (including a `-f` PATH flag that only the later copy
# had), so the first copies were removed rather than left as misleading dead code.


# sed
SED_BAD_HEADS = {"-i", "--in-place"}


def make_sed_grammar() -> Grammar:
    flags = [
        FlagSpec("-n", "none"),
        FlagSpec("-E", "none"),
        FlagSpec("-r", "none"),
        FlagSpec("-u", "none"),
        FlagSpec("-l", "number"),
        FlagSpec("--posix", "none"),
    ]
    return Grammar("sed", flags)


# tar
def make_writer_grammar(binary: str) -> Grammar:
    """Grammars for the tier-1 workspace writers.

    These exist so the writers get their notebook tier (1, reversible by snapshot)
    instead of falling through "unknown binary -> APPROVAL" (2). The difference
    matters: a tier-1 command runs under the enabled-tier ceiling once someone
    raises it deliberately, while tier 2 additionally demands a per-argv approval.

    `mkdir -p a/b` and `touch -d yesterday f` are ordinary usage and are accepted.
    `install` and `patch` are here for the same reason; both write inside the
    workspace and neither is on its own an exfiltration.
    """
    flags = [
        FlagSpec("-p", "none"),
        FlagSpec("-v", "none"),
        FlagSpec("-f", "none"),
        FlagSpec("-r", "none"),
        FlagSpec("-R", "none"),
        FlagSpec("-a", "none"),
        FlagSpec("-i", "none"),
        FlagSpec("-n", "none"),
        FlagSpec("-t", "none"),
        FlagSpec("-m", "number"),
        FlagSpec("-d", "string"),
        FlagSpec("--no-clobber", "none"),
        FlagSpec("--parents", "none"),
        FlagSpec("--help", "none"),
        FlagSpec("--version", "none"),
    ]
    if binary == "patch":
        flags = [
            FlagSpec("-p", "none"), FlagSpec("-R", "none"), FlagSpec("-N", "none"),
            FlagSpec("--dry-run", "none"), FlagSpec("--forward", "none"),
            FlagSpec("--strip", "number"), FlagSpec("-d", "string"),
            FlagSpec("--backup", "string"), FlagSpec("--help", "none"),
        ]
    return Grammar(binary, flags)


def make_tar_grammar() -> Grammar:
    flags = [
        FlagSpec("-c", "none"),
        FlagSpec("-x", "none"),
        FlagSpec("-t", "none"),
        FlagSpec("-v", "none"),
        FlagSpec("-f", "path"),
        FlagSpec("-C", "path"),
        FlagSpec("-z", "none"),
        FlagSpec("-j", "none"),
        FlagSpec("-J", "none"),
        FlagSpec("-k", "none"),
        FlagSpec("-p", "none"),
        FlagSpec("--force-local", "none"),
        FlagSpec("--directory", "path"),
        FlagSpec("--warning", "string"),
        FlagSpec("--help", "none"),
        FlagSpec("--version", "none"),
    ]
    return Grammar("tar", flags)


def make_head_tail_wc_stat_file_grammar() -> Grammar:
    flags = [
        FlagSpec("-n", "number"),
        FlagSpec("-c", "number"),
        FlagSpec("-q", "none"),
        FlagSpec("-v", "none"),
        FlagSpec("-f", "path"),
        FlagSpec("--help", "none"),
        FlagSpec("--version", "none"),
        FlagSpec("-z", "none"),
    ]
    return Grammar("util", flags)


def make_du_df_sha_realpath_readlink_tree() -> Grammar:
    flags = [
        FlagSpec("-h", "none"),
        FlagSpec("--human-readable", "none"),
        FlagSpec("-k", "none"),
        FlagSpec("-m", "none"),
        FlagSpec("-b", "none"),
        FlagSpec("-c", "none"),
        FlagSpec("-s", "none"),
        FlagSpec("-x", "none"),
        FlagSpec("-P", "none"),
        FlagSpec("-H", "none"),
        FlagSpec("--version", "none"),
        FlagSpec("--help", "none"),
    ]
    return Grammar("util", flags)


# git
GIT_READ_SUBS = {"status", "diff", "log", "show", "rev-parse", "ls-files", "blame", "describe"}
GIT_MUTATING_SUBS = {"add", "commit", "branch", "checkout", "reset", "stash", "rm", "mv", "tag"}
GIT_NETWORK_SUBS = {"push", "pull", "clone", "fetch", "remote", "submodule"}


def _check_git(argv: list[str]) -> GrammarResult:
    sub = None
    for a in argv[1:]:
        if not a.startswith("-"):
            sub = a
            break
    if sub is None:
        return GrammarResult("deny", "git needs a subcommand")
    if sub in GIT_NETWORK_SUBS:
        return GrammarResult("deny", f"git {sub} reaches network or mutates remotes")
    for f in argv[1:]:
        head = _head_of(f)
        if head == "-c" or head == "--exec-path":
            return GrammarResult("deny", "git -c / --exec-path can execute arbitrary code")
        if head == "--git-dir" or head == "--work-tree" or head == "-C":
            return GrammarResult("deny", "git retargeting flags (--git-dir/--work-tree/-C) not allowed")
        if head == "--output" or head == "-o":
            return GrammarResult("deny", "git output redirection/write flag not allowed")
    if sub not in GIT_READ_SUBS and sub not in GIT_MUTATING_SUBS:
        return GrammarResult("deny", f"git subcommand not allowed: {sub}")
    if sub == "config":
        for a in argv[1:]:
            as_ = str(a)
            if "alias" in as_:
                return GrammarResult("deny", "git config alias.* is not allowed")
    return GrammarResult("ok", "", subcommand=sub)


def make_git_grammar() -> Grammar:
    flags = [
        FlagSpec("--version", "none"),
        FlagSpec("--help", "none"),
        FlagSpec("--no-pager", "none"),
        FlagSpec("--paginate", "none"),
        FlagSpec("-p", "none"),
        FlagSpec("-q", "none"),
        FlagSpec("--quiet", "none"),
        FlagSpec("--verbose", "none"),
        FlagSpec("-v", "none"),
        FlagSpec("--no-color", "none"),
    ]
    return Grammar("git", flags)


# git-specific checks
def _check_git(argv: list[str]) -> GrammarResult:
    # find subcommand (not starting with -)
    sub = None
    for a in argv[1:]:
        if not a.startswith("-"):
            sub = a
            break
    if sub is None:
        return GrammarResult("deny", "git needs a subcommand")
    if sub in GIT_NETWORK_SUBS:
        return GrammarResult("deny", f"git {sub} reaches network or mutates remotes")
    # reject -c / --exec-path and retargeting / write flags on read subs
    for f in argv[1:]:
        head = _head_of(f)
        if head == "-c" or head == "--exec-path":
            return GrammarResult("deny", "git -c / --exec-path can execute arbitrary code")
        if head == "--git-dir" or head == "--work-tree" or head == "-C":
            # retargeting - needs careful handling; reject as dangerous for read subs
            return GrammarResult("deny", "git retargeting flags (--git-dir/--work-tree/-C) not allowed")
        if head == "--output" or head == "-o":
            return GrammarResult("deny", "git output redirection/write flag not allowed")
        if head == "--file" or head == "-f":
            pass
    if sub not in GIT_READ_SUBS and sub not in GIT_MUTATING_SUBS:
        return GrammarResult("deny", f"git subcommand not allowed: {sub}")
    if sub == "config":
        for a in argv[1:]:
            if "alias." in a:
                return GrammarResult("deny", "git config alias.* is not allowed")
    return GrammarResult("ok", "", subcommand=sub)


# attach subcommand info
def _check_git_with_sub(argv: list[str]):
    res = _check_git(argv)
    sub = None
    for a in argv[1:]:
        if not a.startswith("-"):
            sub = a
            break
    res.subcommand = sub  # type: ignore
    return res


# ---------------------------------------------------------------------------
# env: the wrapper hole
# ---------------------------------------------------------------------------
ENV_VALUE_FLAGS = {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}
ENV_ALLOWED_FLAGS = {"-i", "--ignore-environment", "-0", "--null", "--debug",
                     "--help", "--version", "--default-signal", "--list-signal"}


def _env_payload(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split `env` argv into (leading KEY=VALUE assignments, the wrapped command).

    `env python3 train.py` was the last PERMISSIVE finding: `env` had the generic
    no-flags grammar, so every argument fell through as a "path", the grammar
    returned ok, and the command classified TIER 0 -- the same tier as `env` with no
    command at all. `env` is not a reader, it is a launcher, and the thing it
    launches has to be checked.
    """
    assignments: list[str] = []
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ENV_VALUE_FLAGS:
            i += 2                       # the value is consumed with the flag
            continue
        if a.startswith("-") and "=" in a:
            i += 1                       # --unset=NAME
            continue
        if a.startswith("-"):
            i += 1                       # a value-less flag
            continue
        if "=" in a and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", a):
            assignments.append(a)
            i += 1
            continue
        return assignments, argv[i:]     # first bare token starts the command
    return assignments, []


# ---------------------------------------------------------------------------
# Recursion: the deny-list is about paths the agent may READ, and -r walks them
# ---------------------------------------------------------------------------
# Which binaries recurse, and which flag of theirs turns that on.
RECURSIVE_FLAGS: dict[str, tuple[str, ...]] = {
    "grep": ("-r", "-R", "--recursive", "--dereference-recursive"),
    "rg": ("--recursive", "-r", "-R"),
    "find": (),          # find always recurses unless -maxdepth 0
    "tar": (),
    "diff": ("-r",),
    "ls": ("-R",),
    "cp": ("-R", "-r"),
    "rsync": ("-r", "-R", "--recursive"),
    "du": (),            # du always descends
    "tree": (),
}

# Of the above, these have no non-recursive mode: `find` with no -maxdepth descends
# the whole tree, `du` always does, and `tar -x` writes everything it finds. So for
# them a bare invocation is already a walk and no flag is needed to trigger it.
_ALWAYS_DESCEND = frozenset({"find", "du", "tree", "tar"})

# Basenames that must be excluded, regardless of what `policy.deny_globs` says.
# The globs are path-shaped (`**/.env`), and a walk needs directory names to
# prune on, so these are named explicitly.
_ALWAYS_PRUNE_DIRS = (".ssh", ".aws", ".gnupg", ".kube", ".docker", ".gnome")


def _required_exclusions(access: AccessPolicy) -> set[str]:
    return (set(access.skip_dirs) | set(_ALWAYS_PRUNE_DIRS)
            | {Path(g).name for g in access.deny_globs
               if Path(g).name and "*" not in Path(g).name})


def _check_recursion(binary: str, argv: list[str], access: AccessPolicy) -> tuple[bool, str]:
    """A recursive walk must carry exclusions for everything deny-listed.

    This was the probe's last live finding, and it is the same shape as `find
    -exec`: the PATH OPERAND is in scope, so every path check passes, and then the
    command walks into a deny-listed subdirectory and returns its contents. Path
    validation is blind to what a command does *between* operands.

    The requirement is stated positively rather than by inspecting the tool's own
    flags, because a blocklist of dangerous flag combinations is exactly the blind
    spot that made the notebook's validator permissive in the first place: it
    enumerates the attacks it has heard of. Requiring the exclusions to be PRESENT
    inverts that -- a new recursive flag, or a new tool, is refused by default
    rather than allowed by omission.

    It also keeps the tool layer honest: `tools/grep.py` already emits
    `--exclude-dir` for every skip dir and for `.ssh/.aws/.gnupg/.kube`, so a
    grep the agent built itself satisfies this without changes.
    """
    if binary not in RECURSIVE_FLAGS:
        return True, ""

    # Binaries that descend whether or not they are told to, versus those that need
    # a flag. Keeping the two lists apart is the point: "grep" without -r reads one
    # file and is harmless, while "find" has no non-recursive mode at all.
    if binary in _ALWAYS_DESCEND:
        recursing = True
    else:
        recursing = any(f in argv for f in RECURSIVE_FLAGS[binary])
        # `--recursive`/`-R` may be spelled with a value (`-m3`) or with `=`; the
        # prefix forms are accepted so the check cannot be defeated by a spelling
        # the table happens not to list.
        if not recursing:
            recursing = any(a.startswith(("--recursive", "--dereference-recursive"))
                            for a in argv)
    if not recursing:
        return True, ""

    required = _required_exclusions(access)
    present = {a.split("=", 1)[-1].lstrip("-").split("/", 1)[-1]
               for a in argv if a.startswith("-")}
    missing = sorted(d for d in required if d not in present)
    if missing:
        return False, (f"{binary} recurses but does not exclude deny-listed "
                       f"directories: missing {' '.join(missing[:6])}"
                       + (" ..." if len(missing) > 6 else ""))
    return True, ""



# ---------------------------------------------------------------------------
# Grammar registry: binary name -> grammar. Assembled once, at import.
# ---------------------------------------------------------------------------
def _build_grammars() -> dict[str, Grammar]:
    """binary name -> Grammar. Anything absent is an unknown binary, which
    escalates to APPROVAL (fail closed)."""
    g: dict[str, Grammar] = {}
    g["find"] = make_find_grammar()
    g["grep"] = make_grep_grammar()
    g["egrep"] = make_grep_grammar()
    g["fgrep"] = make_grep_grammar()
    g["rg"] = make_rg_grammar()
    g["ls"] = make_ls_grammar()
    g["cat"] = make_cat_grammar()
    g["git"] = make_git_grammar()
    g["diff"] = make_diff_grammar()
    # sed and tar have real grammars. Wiring them to `make_basic_noextra` (which has
    # NO flags) means every `sed -n` and `tar -cf` classifies as "unknown flag" and
    # escalates to APPROVAL -- a refusal caused by the policy's own wiring, not by
    # anything the model did.
    g["sed"] = make_sed_grammar()
    g["tar"] = make_tar_grammar()
    for b in ("head", "tail", "wc", "stat", "file"):
        g[b] = make_head_tail_wc_stat_file_grammar()
    for b in ("du", "df", "sha256sum", "realpath", "readlink", "tree"):
        g[b] = make_du_df_sha_realpath_readlink_tree()
    for b in ("sort", "uniq", "tr", "cut", "which"):
        g[b] = make_sort_uniq_tr_cut_which()
    for b in ("echo", "date", "uname", "id", "pwd"):
        g[b] = make_echo_date_uname_id_pwd()
    for b in ("basename", "dirname", "true", "false", "env"):
        g[b] = make_basic_noextra()
    for b in ("mkdir", "touch", "cp", "mv", "ln", "install", "patch"):
        g[b] = make_writer_grammar(b)
    for b in ("python", "python3", "node", "bash", "sh", "zsh", "perl", "ruby", "pytest", "make", "cargo", "go", "java", "deno", "bun", "php", "nodejs"):
        g[b] = make_basic_noextra()
    return g


# A resolved binary is not always the name we reasoned about: on Debian
# /bin/awk is a symlink to gawk, /bin/sh to dash. Classifying on the resolved name
# means the allowlist entry for `awk` never fires and `gawk` falls through to
# "unknown binary -> APPROVAL". That fails safe, but for the wrong reason, and it
# means adding a tier floor for `awk` silently has no effect. So the resolved name
# is mapped back to its canonical spelling before any set lookup.
BINARY_ALIASES: dict[str, str] = {
    "gawk": "awk", "mawk": "awk", "nawk": "awk",
    "ripgrep": "rg",
    "dash": "sh",
    "busybox": "sh",          # multi-call binary; treat as an interpreter
    "pgrep": "grep",          # pgrep is fgrep-compatible
    "egrep": "grep", "fgrep": "grep",
}

# Version-suffixed interpreters. On this machine `python3` on PATH is a symlink to
# `python3.14`, so resolving to an absolute path -- which the policy now requires --
# yields `python3.14`. That name matches no grammar and no interpreter set, so the
# control case "python3 --version should be tier 0" reads as unknown-binary and the
# interpreter rule silently stops applying. Stripping a trailing version keeps the
# allowlist keyed on the name a person would actually type.
_VERSION_SUFFIX = re.compile(r"^(?P<stem>.+?)\.(?P<ver>\d+(?:\.\d+)*)$")


def _known_names() -> set[str]:
    """Every name any allowlist set is keyed on. Rebuilt each call, not cached:
    the sets are assembled at import and a cache here would be a module global
    holding derived state, which is exactly what the contract forbids."""
    return (set(INTERPRETERS) | set(GRAMMARS) | set(NETWORK_BINARIES)
            | set(ALWAYS_DENIED) | set(TIER1_FLOOR) | set(TIER2_FLOOR)
            | set(BINARY_ALIASES))


def canonical_binary(name: str) -> str:
    """Map a resolved executable name onto the name the allowlists are keyed on.

    Two normalisations, both because resolving to an absolute path is now required
    and that changes the name we get:

      * aliases -- `awk` resolves to `/usr/bin/gawk` on Debian, `sh` to `dash`.
      * version suffixes -- `python3` resolves to `python3.14`.
    """
    if name in BINARY_ALIASES:
        return BINARY_ALIASES[name]
    m = _VERSION_SUFFIX.match(name)
    if not m:
        return name
    stem = m.group("stem")
    known = _known_names()
    if stem in known:
        return stem
    # `pip3.14` -> `pip`: some tools number the major version into the name.
    for candidate in (stem + "3", stem + "2"):
        if candidate in known:
            return candidate
    return name


GRAMMARS: dict[str, Grammar] = _build_grammars()


# ---------------------------------------------------------------------------
# Absolute binary resolution through a TRUSTED DIRECTORY allowlist.
# ---------------------------------------------------------------------------
def resolve_binary(argv0: str) -> tuple[str | None, str]:
    """(absolute_path, reason).

    The notebook keyed policy on `os.path.basename(argv[0])`, so `./cat x` and
    `/tmp/git status` classified as `cat` and `git`. Anyone who can drop a file
    named `cat` in a directory on PATH then gets it classified as the real cat.

    So: reject relative argv[0] entirely, and resolve the rest against the trusted
    directory list rather than PATH. PATH itself is not trusted because it includes
    user-writable directories (~/.local/bin, ~/bin).
    """
    if not argv0:
        return None, "empty binary"
    p = Path(argv0)
    if not p.is_absolute():
        return None, (f"argv[0] must be an absolute path, got {argv0!r}; "
                      f"a relative binary can be swapped for a lookalike")

    resolved = p.resolve()
    for d in TRUSTED_BIN_DIRS:
        try:
            root = Path(d).resolve()
        except OSError:
            continue
        if resolved.parent == root:
            return str(resolved), ""
    return None, (f"binary {resolved} is not in a trusted directory "
                  f"({', '.join(TRUSTED_BIN_DIRS)}); PATH is not trusted because it "
                  f"contains user-writable directories")


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------
# Bound on `env` recursion. `env env env ...` would otherwise recurse until the
# stack gives out, and a policy engine that can be crashed by a long argv is a
# policy engine that will be.
_MAX_ENV_DEPTH = 4

# Assignments that change how the child loads or interprets code. Setting these is
# not "setting a variable" the way `LANG=C` is.
_DANGEROUS_ENV_KEYS = {
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH", "BASH_ENV", "ENV", "IFS", "PYTHONPATH", "PYTHONSTARTUP",
    "PERL5OPT", "NODE_OPTIONS", "GIT_SSH", "GIT_EXTERNAL_DIFF",
}


def classify(spec: CommandSpec, manifest, *, access: AccessPolicy,
             _depth: int = 0) -> PolicyVerdict:
    """Decide the risk tier for one command. Fails closed.

    `access` is a REQUIRED keyword argument, not an optional default. Policy cannot
    be evaluated without knowing the scope -- that omission is exactly what let
    `cat /etc/passwd` classify as tier 0 in the notebook.

    Order:
      1. resolve the binary through a trusted directory allowlist
      2. always-denied and network-capable binaries
      3. unknown binary -> APPROVAL (never tier 0)
      4. grammar check: unknown flags escalate, dangerous flags deny
      5. every PATH operand must resolve inside the read scope
      6. interpreters without a sandbox -> APPROVAL
      7. mutating git subcommands -> at least REVERSIBLE
      8. cwd and write targets -> write scope
      9. a wrapper (`env`) takes the tier of whatever it wraps
     10. tainted -> APPROVAL
     11. tier floor: a clean grammar does not make a writer read-only
     12. the manifest caps everything
    """
    argv = spec.argv
    reasons: list[str] = []

    if not argv or not argv[0]:
        return PolicyVerdict(tier=Tier.DENIED, reason="empty argv", binary="",
                             reasons=["empty argv"])

    binary_path, why = resolve_binary(argv[0])
    if binary_path is None:
        return PolicyVerdict(tier=Tier.DENIED, reason=why, binary=argv[0],
                             reasons=[why])
    binary = canonical_binary(Path(binary_path).name)

    # 2. always denied
    if binary in ALWAYS_DENIED:
        r = f"{binary} is always denied"
        return PolicyVerdict(tier=Tier.DENIED, reason=r, binary=binary,
                             binary_resolved=binary_path, reasons=[r])

    # 3. network
    if binary in NETWORK_BINARIES and not manifest.network:
        r = f"{binary} can reach the network and no sandbox is active"
        return PolicyVerdict(tier=Tier.DENIED, reason=r, binary=binary,
                             binary_resolved=binary_path, reasons=[r])

    # 4. unknown binary -> fail closed to APPROVAL, never tier 0
    if binary not in GRAMMARS:
        r = f"unknown binary {binary!r} -> tier 2 minimum"
        return PolicyVerdict(tier=Tier.APPROVAL, reason=r, binary=binary,
                             binary_resolved=binary_path, reasons=[r])

    # 5. grammar
    grammar = GRAMMARS[binary]
    # A floor contributed by a wrapper (currently only `env`), applied after the
    # grammar check assigns `tier`.
    wrapper_tier: Tier | None = None

    # 5a. `env` is a launcher, not a reader. Classify the command it wraps and take
    #     the worse of the two, so `env python3 train.py` cannot borrow `env`'s tier.
    if binary == "env":
        assignments, nested = _env_payload(argv)
        for assign in assignments:
            key = assign.split("=", 1)[0]
            # LD_PRELOAD and friends are code-loading primitives. `CommandSpec`
            # already rejects shell metacharacters, but a preload survives that check
            # and runs whatever shared object it names.
            if key in _DANGEROUS_ENV_KEYS:
                r = f"env assignment to {key} can change how the child loads code"
                return PolicyVerdict(tier=Tier.DENIED, reason=r, binary=binary,
                                     binary_resolved=binary_path, reasons=[r])
        if nested:
            if _depth >= _MAX_ENV_DEPTH:
                r = f"env nested {_depth} deep"
                return PolicyVerdict(tier=Tier.DENIED, reason=r, binary=binary,
                                     binary_resolved=binary_path, reasons=[r])
            inner = classify(
                CommandSpec(argv=nested, cwd=spec.cwd, why=spec.why,
                            expected_exit=spec.expected_exit,
                            expected_effects=spec.expected_effects,
                            checks=spec.checks, tainted=spec.tainted),
                manifest, access=access, _depth=_depth + 1)
            if inner.denied:
                r = f"env wraps a denied command: {inner.reason}"
                return PolicyVerdict(tier=Tier.DENIED, reason=r, binary=binary,
                                     binary_resolved=binary_path,
                                     reasons=[f"env -> {inner.binary}: {inner.reason}"])
            wrapper_tier = inner.tier
            reasons.append(f"env wraps {inner.binary} (tier {int(inner.tier)})")
            reasons.extend(inner.reasons)

    # 5b. recursion into deny-listed directories. Path operands can be in scope and
    #     the walk still reaches a secret, so this is checked before the grammar
    #     result is used, not after.
    ok, detail = _check_recursion(binary, argv, access)
    if not ok:
        return PolicyVerdict(tier=Tier.DENIED, reason=detail, binary=binary,
                             binary_resolved=binary_path, reasons=[detail])

    if binary == "find":
        danger, detail = _has_dangerous_find_flag(argv)
        if danger:
            return PolicyVerdict(tier=Tier.DENIED, reason=detail, binary=binary,
                                 binary_resolved=binary_path, reasons=[detail])

    if binary in ("grep", "egrep", "fgrep", "rg"):
        for tok in argv[1:]:
            head = _head_of(tok)
            if head in GREPLIKE_BAD_HEADS:
                r = f"{head} runs a preprocessor command"
                return PolicyVerdict(tier=Tier.DENIED, reason=r, binary=binary,
                                     binary_resolved=binary_path, reasons=[r])

    if binary == "sed":
        for tok in argv[1:]:
            if _head_of(tok) in SED_BAD_HEADS:
                r = "sed in-place edits files"
                return PolicyVerdict(tier=Tier.DENIED, reason=r, binary=binary,
                                     binary_resolved=binary_path, reasons=[r])

    g = validate_grammar(grammar, argv)
    if g.status == "deny":
        return PolicyVerdict(tier=Tier.DENIED, reason=g.reason, binary=binary,
                             binary_resolved=binary_path, reasons=[g.reason])
    if g.status == "unknown":
        # Escalate, never deny: an unrecognised flag is not known-bad, it is
        # unknown, and a blocklist's blind spot is precisely the unknown flag.
        reasons.append(f"unknown flag {g.reason.replace('unknown flag ', '')} -> approval")
        tier = Tier.APPROVAL
    else:
        tier = Tier.READ_ONLY

    # 6. interpreters. `--version` stays tier 0; anything else runs arbitrary code
    #    that no argument check can see inside.
    if binary in INTERPRETERS:
        if argv[1:2] in (["--version"], ["-V"]):
            return PolicyVerdict(tier=Tier.READ_ONLY, reason=f"{binary} --version",
                                 binary=binary, binary_resolved=binary_path)
        if manifest.isolation == "none":
            r = (f"{binary} runs arbitrary code and isolation is 'none'; "
                 f"no argument check can see inside it")
            return PolicyVerdict(tier=Tier.APPROVAL, reason=r, binary=binary,
                                 binary_resolved=binary_path,
                                 reasons=["interpreter without sandbox requires approval"])
        tier = max(tier, Tier.REVERSIBLE)

    # 7. git: tier by subcommand
    if binary == "git":
        gres = _check_git_with_sub(argv)
        if gres.status != "ok":
            return PolicyVerdict(tier=Tier.DENIED, reason=gres.reason, binary=binary,
                                 binary_resolved=binary_path, reasons=[gres.reason])
        sub = getattr(gres, "subcommand", None)
        if sub in GIT_MUTATING_SUBS:
            tier = max(tier, Tier.REVERSIBLE)
            reasons.append(f"git {sub} mutates the working tree -> tier 1 minimum")

    # 8. every PATH operand must be in scope
    for i in g.path_positions:
        operand = argv[i]
        # a glob is a pattern, not a path; resolve its literal prefix
        probe = operand.split("*", 1)[0].rstrip("/") or operand
        try:
            access.resolve(probe, for_write=False)
        except AccessDenied as e:
            if e.reason == "denied":
                return PolicyVerdict(tier=Tier.DENIED,
                                     reason=f"sensitive path argument: {operand}",
                                     binary=binary, binary_resolved=binary_path,
                                     reasons=[f"deny-list: {operand}"])
            # out of scope: escalate, do not deny. The command is not malicious,
            # it is just reading somewhere it should not.
            reasons.append(f"path outside read scope: {operand}")
            tier = max(tier, Tier.APPROVAL)

    # 9. cwd
    if spec.cwd:
        try:
            access.resolve(spec.cwd, for_write=True)
        except AccessDenied as e:
            if e.reason == "denied":
                return PolicyVerdict(tier=Tier.DENIED,
                                     reason=f"cwd blocked by deny-list: {spec.cwd}",
                                     binary=binary, binary_resolved=binary_path,
                                     reasons=[str(e)])
            reasons.append(f"cwd outside write scope: {spec.cwd}")
            tier = max(tier, Tier.APPROVAL)

    # 10. write targets
    if binary in ("cp", "mv", "mkdir", "touch", "ln") and manifest.write_roots:
        for a in argv[1:]:
            if _is_flag_like(a):
                continue
            try:
                access.resolve(a, for_write=True)
            except AccessDenied:
                reasons.append(f"write target outside workspace: {a}")
                tier = max(tier, Tier.APPROVAL)

    # 11. tainted: args echo content we read. A file can say "run this".
    if spec.tainted:
        reasons.append("arguments echo content read from a file (possible injection)")
        tier = max(tier, Tier.APPROVAL)

    # 12. tier floor. A clean grammar does not make a writer read-only.
    tier = apply_tier_floor(binary, tier, reasons)
    if wrapper_tier is not None:
        tier = max(tier, wrapper_tier)

    # 13. the manifest caps everything
    if int(tier) > manifest.max_tier:
        return PolicyVerdict(
            tier=Tier.DENIED,
            reason=f"tier {int(tier)} exceeds {manifest.name} max_tier "
                   f"{manifest.max_tier}",
            binary=binary, binary_resolved=binary_path,
            reasons=[*reasons, "exceeds agent max_tier"])

    return PolicyVerdict(tier=Tier(int(tier)),
                         reason="; ".join(reasons) or f"tier {int(tier)}",
                         binary=binary, binary_resolved=binary_path, reasons=reasons)


# Always denied regardless of manifest. Separate from the notebook's DANGER_ALWAYS
# int so a typo cannot silently drop one of these.
ALWAYS_DENIED = {
    "sudo", "su", "dd", "mkfs", "shutdown", "reboot", "halt", "poweroff",
    "chmod", "chown", "chgrp", "kill", "killall", "pkill", "systemctl",
    "mount", "umount", "xargs", "nohup", "less", "more", "man", "vim", "vi",
    "nano", "emacs", "at", "crontab", "insmod", "rmmod", "modprobe",
    "iptables", "ufw", "swapoff", "fdisk", "parted", "mkfs.ext4",
}

# ---------------------------------------------------------------------------
# Tier floors.
# ---------------------------------------------------------------------------
# A grammar says which FLAGS are acceptable. It says nothing about what the binary
# DOES when the flags are fine, so a read-only grammar alone will happily classify
# `tar -cf out.tar /home` as tier 0 -- which it is, flag-wise, and categorically is
# not. The notebook expressed this as a tier in the same table as the flag
# validator; the grammar split lost it, so it is restored here as an explicit floor.
#
# Tier 1 = writes inside the workspace, reversible by snapshot.
# Tier 2 = builds or fetches, which is where network and package managers live.
TIER1_FLOOR = {
    "mkdir", "touch", "cp", "mv", "ln", "tar", "unzip", "zip", "install", "patch",
}
TIER2_FLOOR = {
    "awk", "make", "cmake", "ninja", "cargo", "go", "java", "javac", "gradle",
    "mvn", "pip", "pip3", "npm", "npx", "yarn", "pnpm", "apt", "apt-get", "dnf",
    "yum", "pacman", "brew", "rsync", "screen", "tmux", "gdb", "strace",
}


def apply_tier_floor(binary: str, tier: Tier, reasons: list[str]) -> Tier:
    """Raise `tier` to the floor for `binary`. Never lowers it."""
    if binary in TIER2_FLOOR:
        if int(tier) < int(Tier.APPROVAL):
            reasons.append(f"{binary} builds or fetches -> tier 2 minimum")
            return Tier.APPROVAL
        return tier
    if binary in TIER1_FLOOR:
        if int(tier) < int(Tier.REVERSIBLE):
            reasons.append(f"{binary} writes -> tier 1 minimum")
            return Tier.REVERSIBLE
        return tier
    return tier
