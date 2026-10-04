"""The repo must not contain a credential. This test is the reason that stays true.

A `.gitignore` entry is a promise about the future; it does nothing about the 264
commits already written. Two real keys were in history here when this test was added,
which is the normal outcome of developing an agent that reads `.env` and then having
a bad afternoon.

So this scans the WORKING TREE (fast, deterministic, runs in CI) and the test next to
it scans every reachable git object (slow, marked, run before a push).

Both scan for SHAPE, not for a known key. A test that knows one specific secret only
catches that secret; a shape test catches the next one too.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Shapes, not values. `redaction.REDACT_PATTERNS` knows these too, but that module is
# for masking output on its way to a model; this one is for failing a build.
#
# Two compiled forms: text for the working tree, bytes for git blobs. `cat-file`
# hands back raw bytes and a str pattern raises `TypeError` -- which is how the first
# version of this test "passed" by crashing rather than by scanning.
KEY_SHAPES = re.compile(
    r"""(
        gsk_[A-Za-z0-9_\-]{20,}                    # groq
      | sk-(?:proj-)?[A-Za-z0-9_\-]{20,}           # openai / anthropic style
      | gh[pousr]_[A-Za-z0-9]{20,}                 # github
      | AKIA[0-9A-Z]{16}                           # aws access key id
      | xox[abposr]-[A-Za-z0-9\-]{20,}             # slack
      | AIza[0-9A-Za-z_\-]{35}                     # google
      | eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}   # jwt
    )""",
    re.X,
)

# An assignment whose VALUE looks like a secret. Catches the shapes above does not
# name -- a database URL with a password in it, say.
ASSIGNMENT = re.compile(
    r"""(?i)\b(api[_-]?key|secret|token|password|passwd|auth|credential)
        \s*[:=]\s*["']?([^\s"',;]{16,})""",
    re.X,
)

# Values that are obviously not credentials. Without this the test fires on its own
# fixtures -- `api_key=<REDACTED>` and `password = "hunter2"` are both 16+ chars.
KEY_SHAPES_B = re.compile(KEY_SHAPES.pattern.encode(), re.X)

BENIGN = re.compile(
    r"""(?ix)^(
        <[a-z_]+>            |   # <REDACTED>, <your-key-here>
        \.?\.?x+             |   # xxxxxxxx in test fixtures
        redacted|placeholder|example|changeme|dummy|fake|dummyvalue
        |\$\{|\$\(|%s|%d     |   # unexpanded template
        |os\.|env\.|
        |none|null|true|false|
        test|dummy|notarealkey
    )$""",
)


def _is_benign(value: str) -> bool:
    """Would a scanner stop here, or keep going?

    The list of "not a literal" forms is long because the alternative is a test that
    fires on every attribute access and gets switched off. `api_key=s.llm.api_key` is
    the shape that pushed this from four entries to a dozen: passing a config field
    around is not leaking it.
    """
    v = value.strip().strip("\"'")
    if not v or BENIGN.match(v):
        return True
    # An env-var reference, a call, or an attribute access -- never a literal.
    if v.startswith(("$", "%", "(", "{")):
        return True
    if re.match(r"^[A-Za-z_][\w.]*\s*[\(\[{]", v):      # settings.llm, load(...)
        return True
    if re.match(r"^[A-Za-z_][\w.]*$", v):              # s.llm.api_key, self.key
        return True
    # A call on something: re.search(...).group(1)
    if "()." in v or v.endswith(")"):
        return True
    # A long token that is all identifier-ish characters and contains an underscore
    # is far more likely to be a Python name than a key; real keys are mixed-case
    # base62 or have a prefix.
    if re.match(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*$", v):
        return True
    return False


SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
             ".mypy_cache", "dist", "build", ".ipynb_checkpoints", "legacy"}


def _iter_files():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            p = Path(root) / f
            # Binary files cannot contain a readable key literal, and reading them
            # to prove it is how a test suite starts eating a gigabyte.
            if p.suffix.lower() in (".png", ".jpg", ".zip", ".gz", ".pyc", ".so",
                                    ".pdf", ".woff", ".woff2", ".ico"):
                continue
            yield p


def test_no_credential_shape_in_the_working_tree():
    """The fast one. Runs in CI, catches a paste into a source file."""
    hits: list[str] = []
    for p in _iter_files():
        # `.env` itself is expected to hold the live key and is git-ignored.
        if p.name == ".env" or p.name.endswith(".env.backup"):
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for m in KEY_SHAPES.finditer(text):
            hits.append(f"{p.relative_to(REPO)}: literal key shape {m.group(0)[:12]}...")
        for m in ASSIGNMENT.finditer(text):
            if not _is_benign(m.group(2)):
                hits.append(f"{p.relative_to(REPO)}: {m.group(1)}= looks like a secret")
    assert not hits, "credential-shaped strings in the tree:\n  " + "\n  ".join(hits)


def test_env_file_is_ignored_by_git():
    """`.env` holds the live key. It must never be tracked, and this asserts the
    rule rather than trusting that someone remembered it."""
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                             text=True).stdout.split()
    assert ".env" not in tracked, ".env is tracked; it holds a live credential"
    for name in tracked:
        assert not name.endswith(".env.backup"), f"{name} is tracked and holds a key"


@pytest.mark.slow
def test_no_credential_shape_in_git_history():
    """The slow one. Every reachable blob, not just the working tree.

    Marked slow because it walks the object store. Run it before pushing -- this is
    the test that would have caught the two keys that were in this repo's history.
    """
    objs = subprocess.run(
        ["git", "rev-list", "--all", "--objects"], cwd=REPO,
        capture_output=True, text=True).stdout.splitlines()
    blobs = []
    for line in objs:
        sha = line.split(" ", 1)[0]
        kind = subprocess.run(["git", "cat-file", "-t", sha], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
        if kind == "blob":
            blobs.append(sha)
    assert blobs, "no objects found; is this a git repo?"

    hits: list[str] = []
    for sha in blobs:
        data = subprocess.run(["git", "cat-file", "blob", sha], cwd=REPO,
                              capture_output=True).stdout
        m = KEY_SHAPES_B.search(data)
        if m:
            hits.append(f"{sha[:10]}: {m.group(0)[:12].decode(errors='replace')}...")
    assert not hits, ("credential shapes found in git history -- rotate the key, "
                      "then `git filter-repo --blob-callback`:\n  "
                      + "\n  ".join(hits[:10]))


def test_the_scanner_would_catch_a_real_looking_key():
    """The counter-test. A scanner with no false positives is worthless; prove it
    fires on a planted value before trusting it to pass on a real one."""
    for probe in ("gsk_" + "A" * 40, "sk-" + "B" * 32, "AKIA" + "C" * 16):
        assert KEY_SHAPES.search(probe), f"scanner missed {probe[:8]}"
    for benign in ("gsk_<REDACTED>", "api_key = os.environ['X']", "token = ${TOKEN}",
                   "password = 'hunter2'", "key = None",
                   "api_key=s.llm.api_key", "api_key=config.api_key",
                   "token = ctx.approvals", "secret = load_secret()"):
        m = ASSIGNMENT.search(benign)
        if m:
            assert _is_benign(m.group(2)), f"false positive on {benign!r}"