from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile

import pytest

from jarvis.tools import find, grep, read, survey
from jarvis.tools.observation import Observation


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
