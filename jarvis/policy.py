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
FlagValueType = Literal["none", "path", "number", "size", "string"]


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
    def __init__(self, status: Literal["ok", "deny", "unknown"], reason: str = "", path_positions: list[int] | None = None):
        self.status = status
        self.reason = reason
        self.path_positions = path_positions or []


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

