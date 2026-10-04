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
            return GrammarResult("unknown", f"unknown flag {head}", path_positions)
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


def make_head_tail_wc_stat_file_grammar() -> Grammar:
    flags = [
        FlagSpec("-n", "number"),
        FlagSpec("-c", "number"),
        FlagSpec("-q", "none"),
        FlagSpec("-v", "none"),
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
            pass  # allow file for config maybe? but we restrict subs; config needs care
    if sub not in GIT_READ_SUBS and sub not in GIT_MUTATING_SUBS:
        return GrammarResult("deny", f"git subcommand not allowed: {sub}")
    if sub == "config":
        # config alias.* executes code - reject
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

