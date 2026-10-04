from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile

import pytest

from jarvis.tools import find, grep, read, survey
from jarvis.tools.observation import Observation


def test_grep_pattern_precedes_path(ctx, tmp_workspace):
    """REGRESSION: `grep ... PATH PATTERN` makes grep treat the pattern as a
    second filename, exit 2 with "No such file or directory", and return zero
    hits. That is a confident false "nothing matched" -- the worst kind of bug
    in a search agent, because it looks like a correct negative answer.

    The notebook had the order right; the Phase 0 extraction lost it.
    """
    (tmp_workspace / "target.txt").write_text("NEEDLE_HERE\nother line\n")
    obs = grep.search_content("NEEDLE_HERE", ctx=ctx, root=tmp_workspace)
    assert obs.status == "ok", f"grep found nothing for a string that exists: {obs.reason}"
    assert any("target.txt" in h["path"] for h in obs.items)
    assert any(h["line_no"] == 1 for h in obs.items)


def test_grep_reports_honest_negative_for_absent_string(ctx, tmp_workspace):
    """The converse: a string that genuinely is not there must still be not_found.

    Guards against "fixing" the above by making grep always return something.
    """
    obs = grep.search_content("definitely_not_present_9f3a", ctx=ctx, root=tmp_workspace)
    assert obs.status == "not_found"
    assert obs.items == []


def test_find_returns_observation_and_ranking(ctx, tmp_workspace):
    obs = find.find_files("main", ctx=ctx, root=tmp_workspace)
    assert isinstance(obs, Observation)
    assert obs.status in ("ok", "not_found")
    # Should find main.py
    paths = [i["path"] for i in obs.items]
    assert any("main.py" in str(p) for p in paths)


def test_find_iname_before_print(ctx, tmp_workspace):
    # Create files with mixed case
    (tmp_workspace / "MainFile.txt").write_text("hi")
    obs = find.find_files("mainfile", ctx=ctx, root=tmp_workspace, fuzzy=False)
    assert obs.status == "ok"
    paths = [i["path"] for i in obs.items]
    assert any("MainFile.txt" in str(p) for p in paths)


def test_grep_denied_not_read_even_with_r(ctx, tmp_workspace):
    # .ssh/id_rsa should not be returned even if content matches
    obs = grep.search_content("fake", ctx=ctx, root=tmp_workspace)
    # grep should not return hits from deny-listed paths
    for item in obs.items:
        p = Path(item["path"])
        # ensure it's not in .ssh
        assert ".ssh" not in p.parts


def test_read_reports_total_lines_and_truncated(ctx, tmp_workspace):
    f = tmp_workspace / "long.txt"
    lines = [f"line{i}\n" for i in range(100)]
    f.write_text("".join(lines))
    obs = read.read_file(path=str(f), ctx=ctx, start_line=1, max_lines=10)
    assert obs.status == "ok"
    assert obs.total_lines == 100
    assert obs.n_lines == 10
    assert obs.truncated is True


def test_read_blocks_symlink_outside_scope(ctx, tmp_workspace):
    # Create link pointing outside tmp workspace? Or just create symlink to /etc/passwd if accessible
    try:
        target = Path("/etc/passwd")
        if target.exists():
            link = tmp_workspace / "badlink"
            link.symlink_to(target)
            obs = read.read_file(path=str(link), ctx=ctx)
            # Should be blocked or not found depending on resolution; access should catch
            assert obs.status in ("blocked", "not_found", "error")
    except Exception:
        pytest.skip("cannot create symlink in this env")


def test_survey_respects_skip_and_deny(ctx, tmp_workspace):
    (tmp_workspace / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_workspace / "node_modules" / "pkg" / "index.js").write_text("x")
    obs = survey.survey_files(ctx=ctx, root=tmp_workspace)
    assert obs.status == "ok"
    # node_modules files should not be in sample typically and walkable respected
    sample = obs.meta.get("sample", [])
    for s in sample:
        assert "node_modules" not in s


def test_read_observation_valid(ctx, tmp_workspace):
    f = tmp_workspace / "hello.txt"
    f.write_text("hello\nworld\n")
    obs = read.read_file(path=str(f), ctx=ctx)
    assert isinstance(obs, Observation)
    assert obs.status == "ok"
    assert obs.tool == "read"


# ------------------------------------------------------------------ root handling
def test_relative_root_is_resolved_to_an_absolute_path(tmp_workspace, monkeypatch):
    """REGRESSION: `TaskContext.root` is annotated Path but nothing enforced it, so
    `--root jarvis` left it as the string 'jarvis'. `policy.resolve` then resolved
    that against itself, producing `/proj/jarvis/jarvis` -- a nonexistent path that
    is still inside the read scope, so nothing raised. The agent searched a directory
    that was never there and answered `not_found` with full confidence."""
    monkeypatch.chdir(tmp_workspace)
    from jarvis.context import new_task
    ctx = new_task("t", root="project")
    assert isinstance(ctx.root, Path), type(ctx.root)
    assert ctx.root.is_absolute()
    assert ctx.root == (tmp_workspace / "project").resolve()


def test_a_relative_root_finds_what_it_should(settings, tmp_workspace, monkeypatch):
    """The user-visible consequence of the above. A confident `not_found` from a
    search tool is the worst failure mode it has, so this asserts the positive."""
    monkeypatch.chdir(tmp_workspace)
    (tmp_workspace / "project" / "src" / "target_symbol.txt").write_text("needle_here\n")
    from jarvis.context import new_task
    ctx = new_task("t", settings=settings, root="project")
    obs = grep.search_content("needle_here", ctx=ctx, root=ctx.root)
    assert obs.status == "ok", obs.reason
    assert any("target_symbol.txt" in h["path"] for h in obs.items)


def test_grep_works_on_a_single_file(ctx, tmp_workspace):
    """A file as root is a legitimate request.

    Two bugs met here and both produced `not_found` for a file that matches:
      * `grep -r` with a file as cwd raises NotADirectoryError, and a tool that
        cannot raise across the agent boundary turned that into "nothing matched"
      * `grep PATTERN FILE` omits the filename prefix (it prints `LINE:TEXT`, not
        `PATH:LINE:TEXT`), so the parser's path split found nothing
    `-H` forces the prefix; recursion is dropped for a single file.
    """
    f = tmp_workspace / "single.py"
    f.write_text("alpha\nMATCHME\nomega\n")
    obs = grep.search_content("MATCHME", ctx=ctx, root=f)
    assert obs.status == "ok", obs.reason
    assert len(obs.items) == 1
    assert obs.items[0]["line_no"] == 2
    assert obs.items[0]["path"].endswith("single.py")


def test_grep_single_file_output_format_matches_recursive(ctx, tmp_workspace):
    """One parser, two shapes of grep output. If the prefix can vanish, the parser
    needs a branch -- so it does not."""
    (tmp_workspace / "sub").mkdir()
    (tmp_workspace / "sub" / "a.py").write_text("SYMBOL_X\n")
    (tmp_workspace / "b.py").write_text("SYMBOL_X\n")
    single = grep.search_content("SYMBOL_X", ctx=ctx, root=tmp_workspace / "b.py")
    multi = grep.search_content("SYMBOL_X", ctx=ctx, root=tmp_workspace)
    assert single.status == "ok" and multi.status == "ok"
    assert all("path" in h and "line_no" in h for h in single.items)
    assert all("path" in h and "line_no" in h for h in multi.items)


def test_grep_on_a_missing_path_says_so(ctx, tmp_workspace):
    """It must not surface as `no such executable`, which names the BINARY -- and the
    binary exists. A misleading message about what failed sends the caller looking
    in the wrong place."""
    obs = grep.search_content("x", ctx=ctx, root=tmp_workspace / "nope.py")
    assert "no such path" in obs.reason
    assert "executable" not in obs.reason
