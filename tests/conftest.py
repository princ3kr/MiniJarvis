import tempfile
from pathlib import Path

import pytest

from jarvis.config import Settings, Limits, Budget, LLMConfig
from jarvis.context import TaskContext, Counters, Trace
from jarvis.access import AccessPolicy


@pytest.fixture
def tmp_workspace():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "project" / "src").mkdir(parents=True)
        (p / "project" / "tests").mkdir(parents=True)
        (p / "docs").mkdir(parents=True)
        (p / "deep" / "nested" / "levels").mkdir(parents=True)
        (p / ".ssh").mkdir(parents=True)

        (p / "project" / "src" / "main.py").write_text("def main():\n    print('hello')\n    return 0\n")
        (p / "project" / "src" / "util.py").write_text("import os\n# search_term_here\n")
        (p / "project" / "tests" / "test_main.py").write_text("def test_main():\n    assert True\n")
        (p / "README.md").write_text("# Project\n\nSome text\n")
        (p / "notes.txt").write_text("notes\n")
        (p / "data.json").write_text('{"key": 1}\n')
        (p / "binary.bin").write_bytes(b"\x00\x01\x02")
        (p / "LICENSE").write_text("MIT\n")
        (p / "deep" / "nested" / "levels" / "deep.txt").write_text("deep content\n")
        (p / ".env").write_text("SECRET=123\n")  # file, not dir
        (p / ".ssh" / "id_rsa").write_text("fake key\n")
        import zipfile
        zpath = p / "archive.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            z.writestr("file1.txt", "hello")
            z.writestr("file2.txt", "world")
        yield p


@pytest.fixture
def settings(tmp_workspace):
    return Settings(
        limits=Limits(find_max_results=200, grep_max_results=20, grep_max_per_file=3,
                      read_max_bytes=50000, read_max_lines=200, timeout_s=15),
        budget=Budget(),
        llm=LLMConfig(),
        read_roots=(tmp_workspace,),
        write_root=tmp_workspace / "w",
        # A realistic deny-list, not a token one. A fixture with three globs makes
        # every containment test pass for the wrong reason -- it never asks whether
        # `.env.production` or a nested `id_rsa.pub` is covered, and those are the
        # variants that actually appear on disk.
        deny_globs=(
            "*/id_rsa*", "*/id_ed25519*", "*.pem", "*.key",
            "*/.env", "*/.env.*", "*/.netrc", "*_rsa",
        ),
        skip_dirs=frozenset({".git", "node_modules", "__pycache__"}),
    )


@pytest.fixture
def policy(settings, tmp_workspace):
    return AccessPolicy.from_settings(settings, cwd=tmp_workspace)


@pytest.fixture
def ctx(settings, tmp_workspace):
    return TaskContext(
        query="test",
        settings=settings,
        root=tmp_workspace,
        trace=Trace(),
        counters=Counters(),
    )
