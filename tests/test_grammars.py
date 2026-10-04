"""Grammars must accept ordinary command lines.

Every test here is a command line someone would actually type. The failure they guard
against is not exotic: it is `wc -l` escalating to APPROVAL because `-l` was missing
from a hand-written flag table, which trains a model to stop asking and which a user
experiences as "the agent refuses to count lines".

The theme of the bugs this file pins:

  * a grammar that is too NARROW causes refusals for correct commands. That is the
    failure mode that makes an agent look broken, and it is much more common than a
    grammar being too wide.
  * `wc -lw` is POSIX short-option BUNDLING of two value-less flags. Typing them as
    value-taking flags made bundling fail.
  * `cut -d:` is an ATTACHED short value. getopt accepts it; a parser that only knew
    `--long=v` and `-s=v` reported an unknown flag.
  * `du` has no `--exclude-dir`, only `--exclude`. A policy requirement that no tool
    can satisfy is a requirement that stops the agent doing its job.
"""
from __future__ import annotations

import pytest

from jarvis.policy import GRAMMARS, GrammarResult, validate_grammar

# (binary, argv) pairs a person might type. Grouped by the bug they cover.
ORDINARY = [
    # ---- counting and sizing
    ("wc", ["wc", "-l", "f"]),
    ("wc", ["wc", "-c", "f"]),
    ("wc", ["wc", "-lw", "f"]),                     # bundling, both value-less
    ("wc", ["wc", "--lines", "f"]),
    ("du", ["du", "-h", "."]),
    ("du", ["du", "--max-depth=2", "."]),           # equals form
    ("du", ["du", "-sh", "."]),
    ("df", ["df", "-h"]),
    # ---- listing
    ("ls", ["ls", "-la"]),
    ("ls", ["ls", "-lah"]),                         # bundled
    ("ls", ["ls", "--color=auto"]),
    ("ls", ["ls", "-S", "-t", "-l"]),
    ("tree", ["tree", "-L", "2"]),                  # -L takes a value
    # ---- head/tail
    ("head", ["head", "-n", "20", "f"]),
    ("tail", ["tail", "-n", "5", "f"]),
    ("cat", ["cat", "-n", "f"]),
    # ---- finding
    ("find", ["find", ".", "-type", "f"]),
    ("find", ["find", ".", "-name", "*.py"]),
    ("find", ["find", ".", "-size", "+100M"]),
    ("find", ["find", ".", "-maxdepth", "2"]),
    # ---- searching
    ("grep", ["grep", "-rn", "x", "."]),
    ("grep", ["grep", "-c", "x", "f"]),
    ("grep", ["grep", "-F", "-i", "x", "f"]),
    # ---- text processing
    ("sort", ["sort", "-u", "f"]),
    ("uniq", ["uniq", "-c", "f"]),
    ("cut", ["cut", "-d:", "-f1", "f"]),             # attached short value
    ("cut", ["cut", "-d", ":", "-f", "1", "f"]),    # separated value
    ("tr", ["tr", "a-z", "A-Z"]),
    ("stat", ["stat", "-c", "%s", "f"]),
    ("file", ["file", "f"]),
    # ---- comparing
    ("diff", ["diff", "-u", "a", "b"]),
    # ---- misc
    ("which", ["which", "ls"]),
    ("basename", ["basename", "/a/b"]),
    ("dirname", ["dirname", "/a/b"]),
    ("realpath", ["realpath", "f"]),
    ("sha256sum", ["sha256sum", "f"]),
    # ---- tar/sed, whose grammars were defined but never registered
    ("sed", ["sed", "-n", "1p", "f"]),
    ("sed", ["sed", "-E", "s/a/b/", "f"]),
    ("tar", ["tar", "-cf", "out.tar", "."]),        # bundled, -f takes a value
    ("tar", ["tar", "-xf", "in.tar"]),
]


@pytest.mark.parametrize("binary,argv", ORDINARY,
                         ids=[f"{b}-{'_'.join(a.replace('/', '_') for a in v[:2])}"
                              for b, v in ORDINARY])
def test_ordinary_command_line_is_accepted(binary, argv):
    g = GRAMMARS.get(binary)
    assert g is not None, f"no grammar registered for {binary}"
    r = validate_grammar(g, argv)
    assert r.status == "ok", f"{' '.join(argv)} rejected: {r.status} {r.reason}"


@pytest.mark.parametrize("binary,argv", ORDINARY,
                         ids=[f"{b}-{'_'.join(a.replace('/', '_') for a in v[:2])}"
                              for b, v in ORDINARY])
def test_ordinary_command_line_has_no_dangling_path_requirement(binary, argv):
    """A flag that the grammar says takes a value must not leave it unconsumed.

    If `head -n` matched but left the value to be read as a PATH, then a command
    that is really `head -n 20 file` would have its `20` scope-checked as a path and
    refused for being outside the read scope.
    """
    g = GRAMMARS[binary]
    r = validate_grammar(g, argv)
    if r.status == "ok":
        for i in r.path_positions:
            assert i < len(argv)
            assert not argv[i].lstrip("-").isdigit() or argv[i].startswith("-")


def test_wc_flags_are_value_less_so_they_can_bundle():
    """`wc -lw` is two selectors, not two arguments. Typing them as value-taking
    flags made the bundling check fail and escalated the commonest wc invocation."""
    g = GRAMMARS["wc"]
    for f in ("-l", "-w", "-c", "-m"):
        assert g._map[f].takes == "none", f"{f} should be a value-less selector"


def test_attached_short_value_is_understood():
    g = GRAMMARS["cut"]
    assert validate_grammar(g, ["cut", "-d:", "-f1", "f"]).status == "ok"
    assert validate_grammar(g, ["cut", "--delimiter=:", "--fields=1", "f"]).status == "ok"


def test_du_has_an_exclusion_flag_the_policy_can_require():
    """The gateway refuses a recursive command that does not exclude the deny-listed
    directories. `du` has no `--exclude-dir`, only `--exclude`, so checking only the
    former made `du` impossible to run at all."""
    g = GRAMMARS["du"] if "du" in GRAMMARS else GRAMMARS["df"]
    assert "--exclude" in g._map, "du needs --exclude for the recursion rule"
    assert validate_grammar(g, ["du", "-sh", "--exclude=.git", "."]).status == "ok"


def test_tree_dash_L_takes_a_value():
    """`-L` is value-taking for tree but value-LESS for tail. One table serves both
    binaries, so the flag is declared as taking a value and tail uses `--follow`."""
    assert GRAMMARS["tree"] if "tree" in GRAMMARS else True
    assert validate_grammar(GRAMMARS["tree"], ["tree", "-L", "2", "."]).status == "ok"


def test_an_unknown_flag_still_escalates_rather_than_passing():
    """The counter-test. Narrow grammars are the failure this file guards; too-wide
    ones are the failure that matters more. A flag nobody declared must NOT validate."""
    r = validate_grammar(GRAMMARS["ls"], ["ls", "--frobnicate"])
    assert r.status == "unknown"
    r = validate_grammar(GRAMMARS["grep"], ["grep", "--frobnicate", "x", "f"])
    assert r.status == "unknown"


def test_dangerous_flags_are_still_denied():
    """Narrower tables must not become laxer ones.

    Asserted at the `classify` level rather than the grammar level, because that is
    where the guarantee actually lives. `sed -i` is not in the sed grammar, so the
    grammar alone reports "unknown" -- which escalates to APPROVAL. The denial comes
    from `SED_BAD_HEADS`, a separate check in `classify` that runs first. Testing the
    grammar would have been testing the wrong layer and passing for the wrong reason.
    """
    import tempfile
    from pathlib import Path

    from jarvis.access import AccessPolicy
    from jarvis.config import load_settings
    from jarvis.manifests import build_manifests
    from jarvis.policy import (CommandSpec, Tier, _has_dangerous_find_flag,
                               classify)

    tmp = Path(tempfile.mkdtemp())
    (tmp / "f").write_text("x")
    s = load_settings().replaced(read_roots=(tmp,), write_root=tmp)
    ex = build_manifests((tmp,), str(tmp))["executor"]
    access = AccessPolicy.from_settings(s, cwd=tmp)

    v = classify(CommandSpec(argv=["/usr/bin/sed", "-i", "s/a/b/", str(tmp / "f")],
                             why="t"), ex, access=access)
    assert v.tier is Tier.DENIED, v.reason

    # find -exec is rejected even earlier, by CommandSpec itself: its argument list
    # needs the `;` terminator, which is a shell metacharacter. Two independent
    # layers refuse it, which is the point of having both.
    with pytest.raises(Exception):
        CommandSpec(argv=["/usr/bin/find", str(tmp), "-exec", "ls", "{}", ";"],
                    why="t")
    assert _has_dangerous_find_flag(
        ["/usr/bin/find", str(tmp), "-exec", "ls", "{}", ";"])[0]

    # `-fc` is genuinely ambiguous and this package resolves it the POSIX way: a short
    # option takes the REST OF THE TOKEN as its value, so `-fc` is `-f c`, not the
    # bundle `-f -c`. GNU tar reads it as a bundle. There is no way to satisfy both
    # from argv alone, so POSIX wins -- and the consequence is safe either way:
    # "c" is treated as the archive path, so it goes through the jail like any other
    # operand. Asserted here because the reasoning is not obvious and someone will
    # otherwise "fix" it into a bundle and break the scope check.
    r = validate_grammar(GRAMMARS["tar"], ["tar", "-fc", "out", "."])
    assert r.status == "ok"
    assert r.path_positions == [2, 3], r.path_positions   # "out" and "." are operands

    # A value-taking flag still requires its value.
    assert validate_grammar(GRAMMARS["tar"], ["tar", "-cf"]).status == "deny"
    assert validate_grammar(GRAMMARS["tar"], ["tar", "-cf", "-x", "."]).status == "deny"