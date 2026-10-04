"""Redaction and environment scrubbing for everything that leaves the process.

Two separate concerns that both exist because tool output goes to a remote model:

  * `redact` masks secret shapes in command stdout/stderr and in file excerpts.
  * `scrubbed_env` builds a MINIMAL environment for child processes, so a
    subprocess cannot read GROQ_API_KEY out of its own environment.

The redaction patterns are shape-based, which means they catch a key the agent has
never seen, at the cost of occasionally masking a harmless string. That trade is
deliberate: a false positive costs a redacted token, a false negative leaks a
credential to a third-party API.
"""
from __future__ import annotations

import re

# Ordered: specific shapes before generic `key=value`, so `api_key=gsk_...` is
# masked by the gsk rule first and does not fall through to the generic one.
REDACT_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?"
                r"-----END [A-Z ]*PRIVATE KEY-----", re.S), "<PRIVATE KEY REDACTED>"),
    (re.compile(r"gsk_[A-Za-z0-9_\-]{20,}"), "gsk_<REDACTED>"),
    (re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{16,}"), "sk-<REDACTED>"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"), "gh?_<REDACTED>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA<REDACTED>"),
    (re.compile(r"xox[abposr]-[A-Za-z0-9-]{10,}"), "xox?<REDACTED>"),
    # generic assignment, with several separators tolerated
    (re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|passphrase|token"
                r"|access[_-]?key|private[_-]?key|auth[_-]?token|bearer)"
                r"(\"?\s*[:=]\s*\"?)([^\s\"',;]{4,})"),
     r"\1\2<REDACTED>"),
    # `Authorization: Bearer xyz` header form
    (re.compile(r"(?i)(authorization\s*:\s*bearer\s+)(\S+)"), r"\1<REDACTED>"),
)


def redact(text: str, limit: int = 20_000) -> tuple[str, int]:
    """Mask known secret shapes. Returns (text, n_redactions).

    Never raises: redaction sits on the output path of every command, so a bug here
    would turn a working search into a crash. A pattern that errors on its own input
    is skipped.
    """
    if not text:
        return "", 0
    n = 0
    for pat, repl in REDACT_PATTERNS:
        try:
            text, k = pat.subn(repl, text)
        except Exception:
            continue
        n += k
    if limit and len(text) > limit:
        text = text[:limit] + f"\n... [truncated at {limit} bytes]"
    return text, n


def scrubbed_env(workspace=None) -> dict[str, str]:
    """A MINIMAL environment for child processes.

    The point is that no API key is inherited. A subprocess that reaches the
    network is already refused by policy, but a subprocess that merely prints its
    own environment would otherwise hand back the key in its stdout.
    """
    home = str(workspace) if workspace is not None else "/tmp"
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": home,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        # deliberately absent: every API key, token, and cloud credential
    }


def contains_secret(text: str) -> bool:
    """Would `redact` change this text? Used by tests and the live probe."""
    if not text:
        return False
    for pat, _ in REDACT_PATTERNS:
        try:
            if pat.search(text):
                return True
        except Exception:
            continue
    return False