"""The jail. `access.AccessPolicy` is the only thing allowed to say yes to a path.

Everything here is a containment property, and most of it is a way a naive
implementation fails. The deny-globs, the read/write split, and the distinction
between "may I descend here" and "may I name this" each had a real failure behind
them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.access import SECRET_DIRS, AccessDenied, AccessPolicy, is_denied


@pytest.fixture
def policy(settings, tmp_workspace):
    return AccessPolicy.from_settings(settings, cwd=tmp_workspace)


# ------------------------------------------------------------------ scope
def test_a_path_inside_a_read_root_resolves(policy, tmp_workspace):
    f = tmp_workspace / "main.py"
    f.write_text("x")
    assert policy.resolve(f) == f.resolve()


def test_a_path_outside_every_root_is_refused(policy):
    with pytest.raises(AccessDenied) as e:
        policy.resolve("/etc/passwd")
    assert e.value.reason == "out_of_scope"


def test_refusal_says_which_reason_it_was(policy, tmp_workspace):
    """`denied` and `out_of_scope` mean different things: one is a secret, the other
    is simply somewhere else. A caller that cannot tell them apart cannot decide
    whether to escalate or to re-scope."""
    with pytest.raises(AccessDenied) as e:
        policy.resolve(str(tmp_workspace / ".ssh" / "id_rsa"))
    assert e.value.reason == "denied"


def test_empty_path_is_refused(policy):
    with pytest.raises(AccessDenied) as e:
        policy.resolve("")
    assert e.value.reason == "empty"


# ------------------------------------------------------------------ containment
def test_prefix_is_not_containment(tmp_workspace):
    """`/home/u-evil` starts with the string `/home/u` and is NOT inside it.

    This is why containment walks `Path.parents` instead of `str.startswith`. A
    string comparison reads as obviously wrong once written down, which is exactly
    why it keeps getting written.
    """
    base = tmp_workspace / "u"
    evil = tmp_workspace / "u-evil"
    base.mkdir()
    evil.mkdir()
    (evil / "secret.txt").write_text("x")
    s = None
    from jarvis.config import load_settings
    s = load_settings().replaced(read_roots=(base,))
    p = AccessPolicy.from_settings(s, cwd=base)
    with pytest.raises(AccessDenied):
        p.resolve(evil / "secret.txt")


def test_try_resolve_is_the_filtering_variant(policy, tmp_workspace):
    """It returns None instead of raising, which is what makes it usable in a filter
    over a search result. It does NOT check existence: that is the tools' job, and
    conflating the two turns "no such file" into "no such path"."""
    assert policy.try_resolve("/etc/passwd") is None        # out of scope
    assert policy.try_resolve(str(tmp_workspace / "nope")) is not None  # just absent


def test_check_returns_a_reason_instead_of_raising(policy):
    ok, why = policy.check("/etc/passwd")
    assert ok is False and why


# ------------------------------------------------------------------ relative paths
def test_relative_paths_resolve_against_the_task_cwd_not_the_process_cwd(policy,
                                                                       tmp_workspace):
    """The notebook had two jails that disagreed about where `notes.txt` was: one
    tried `Path.cwd()` and the other tried each root in turn. That fan-out made a
    file reachable from one code path and unreachable from another."""
    (tmp_workspace / "notes.txt").write_text("x")
    assert policy.resolve("notes.txt").name == "notes.txt"


def test_parent_traversal_cannot_escape(policy, tmp_workspace):
    with pytest.raises(AccessDenied):
        policy.resolve(str(tmp_workspace / ".." / ".." / "etc" / "passwd"))


# ------------------------------------------------------------------ symlinks
def test_a_symlink_out_of_scope_is_refused(policy, tmp_workspace):
    """Otherwise `resolve` is trivially bypassed: make a symlink inside the scope
    that points at /etc/shadow, and read it as if it were in scope."""
    link = tmp_workspace / "innocent.txt"
    try:
        link.symlink_to("/etc/shadow")
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(AccessDenied):
        policy.resolve(link)


def test_a_symlink_inside_scope_still_resolves(policy, tmp_workspace):
    real = tmp_workspace / "real.txt"
    real.write_text("x")
    link = tmp_workspace / "link.txt"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert policy.resolve(link).name == "real.txt"


# ------------------------------------------------------------------ deny-list
def test_deny_list_wins_over_read_scope(policy, tmp_workspace):
    """A tier-0 read must not reach a secret even though the secret is inside the
    read scope. Deny is checked last and always wins, because every other check is
    about permission and this one is about the thing itself."""
    secret = tmp_workspace / ".env"
    secret.write_text("KEY=v")
    with pytest.raises(AccessDenied):
        policy.resolve(secret)


@pytest.mark.parametrize("name", [".env", ".env.local", ".env.production", ".netrc"])
def test_deny_globs_cover_variants(policy, tmp_workspace, name):
    f = tmp_workspace / name
    f.write_text("x")
    with pytest.raises(AccessDenied) as e:
        policy.resolve(f)
    assert e.value.reason == "denied"


def test_deny_list_matches_at_any_depth(policy, tmp_workspace):
    deep = tmp_workspace / "a" / "b" / "c"
    deep.mkdir(parents=True)
    k = deep / "id_rsa"
    k.write_text("x")
    with pytest.raises(AccessDenied):
        policy.resolve(k)


def test_is_denied_reports_which_pattern_matched():
    denied, pattern = is_denied("/home/u/.ssh/id_rsa", ("*/id_rsa*",))
    assert denied and pattern == "*/id_rsa*"


# ------------------------------------------------------------------ SECRET_DIRS
def test_secret_directories_are_not_walkable(policy, tmp_workspace):
    """A private-key directory must never be entered.

    The deny-globs are path patterns (`~/.ssh/*`), which match the key FILE but not
    the folder holding it, so `is_denied("~/.ssh")` was False. `resolve` still
    refused the key inside, so no contents leaked -- but a walk that descends into
    `.ssh` counts its files, and "3 files in ~/.ssh" is itself the disclosure.
    """
    for d in (".ssh", ".gnupg", ".kube", ".aws"):
        p = tmp_workspace / d
        p.mkdir(exist_ok=True)
        assert not policy.walkable(p), f"{d} is walkable"
    keyring = tmp_workspace / ".gnome" / "keyring"
    keyring.mkdir(parents=True, exist_ok=True)
    assert not policy.walkable(keyring)


def test_secret_directories_are_not_nameable(policy, tmp_workspace):
    """Naming one is a disclosure too, and costs nothing to prevent. A folder
    summary that says `.ssh/` exists has already told the model something."""
    for d in (".ssh", ".gnupg", ".kube", ".aws"):
        p = tmp_workspace / d
        p.mkdir(exist_ok=True)
        assert not policy.nameable(p), f"{d} is nameable"


def test_noisy_directories_are_still_nameable(policy, tmp_workspace):
    """The counter-test, and the reason `nameable` exists separately.

    `node_modules` and `.config` are noise, not secrets. Hiding them makes "what is
    in this folder" answer with absence, which the model reads as "you have nothing
    here" rather than "we filtered this". Conflating noise with secrecy is how a
    jail starts lying to its own user.
    """
    for d in ("node_modules", "__pycache__"):
        p = tmp_workspace / d
        p.mkdir(exist_ok=True)
        assert policy.nameable(p), f"{d} is not nameable"
        assert not policy.walkable(p), f"{d} should still not be walked"


def test_nameable_is_never_stricter_than_walkable(policy, tmp_workspace):
    """Nameable must never deny something walkable allows. If it did, a directory
    could be descended into but not mentioned, which is incoherent."""
    for d in sorted(SECRET_DIRS):
        p = tmp_workspace / d
        p.mkdir(parents=True, exist_ok=True)
        assert policy.walkable(p) or not policy.nameable(p), d
    for d in ("node_modules", "src", "__pycache__", "docs"):
        p = tmp_workspace / d
        p.mkdir(exist_ok=True)
        if policy.walkable(p):
            assert policy.nameable(p), d


def test_a_dot_directory_is_nameable_when_it_is_not_a_secret(policy, tmp_workspace):
    """`access.walkable`'s docstring says dot-directories are allowed, because
    ~/.config and ~/.cache hold real files the user should be able to search. So a
    blanket "hide anything starting with a dot" rule would contradict the jail's own
    stated policy and hide real work."""
    d = tmp_workspace / ".config" / "myapp"
    d.mkdir(parents=True)
    assert policy.nameable(d)
    assert policy.walkable(d)


# ------------------------------------------------------------------ read vs write
def test_a_read_only_path_is_not_writable(policy, tmp_workspace):
    """Read scope and write scope are different questions. A path being readable
    says nothing about whether the agent may modify it."""
    from jarvis.config import load_settings
    s = load_settings().replaced(read_roots=(tmp_workspace,),
                                 write_root=tmp_workspace / "allowed")
    p = AccessPolicy.from_settings(s, cwd=tmp_workspace)
    f = tmp_workspace / "f.txt"
    f.write_text("x")
    assert p.resolve(f)                       # readable
    with pytest.raises(AccessDenied):
        p.resolve(f, for_write=True)          # not writable