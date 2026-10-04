"""Index tests. The privacy guarantees are the point, so they are tested first."""
from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.index import FileIndex


@pytest.fixture
def indexed(settings, tmp_workspace):
    """A small scanned index over the fixture tree."""
    idx = FileIndex(settings=settings, db_path=tmp_workspace / ".idx" / "i.sqlite")
    from jarvis.context import TaskContext
    idx.scan(TaskContext(query="t", settings=settings, root=tmp_workspace))
    return idx


def test_scan_records_files(indexed, tmp_workspace):
    rows = indexed.query("SELECT path, size FROM entries WHERE is_dir=0")
    names = {Path(r["path"]).name for r in rows}
    assert "main.py" in names
    assert "data.json" in names


def test_index_never_records_deny_listed_paths(indexed):
    """The index must not know that a secret exists.

    An index containing ~/.ssh/id_rsa leaks its existence through a row COUNT even
    when no content is ever read. So the deny-list is enforced when writing.
    """
    rows = indexed.query("SELECT path FROM entries")
    for r in rows:
        assert "id_rsa" not in r["path"], f"secret indexed: {r['path']}"
        assert not r["path"].endswith("/.env"), f".env indexed: {r['path']}"


def test_index_does_not_ingest_itself(indexed):
    """The db lives inside the read scope by default; scanning it would record
    its own WAL files and report phantom entries."""
    rows = indexed.query("SELECT path FROM entries")
    for r in rows:
        assert ".sqlite" not in r["path"]
        assert not r["path"].endswith(".idx")


def test_sql_tool_rejects_writes(indexed):
    for bad in ("DROP TABLE entries", "DELETE FROM entries", "UPDATE entries SET size=0",
                "INSERT INTO entries VALUES (1)", "ATTACH DATABASE '/tmp/x' AS y",
                "PRAGMA table_info(entries)", "CREATE TABLE t (a)"):
        with pytest.raises(ValueError):
            indexed.query(bad)


def test_sql_tool_enforces_limit(indexed):
    assert len(indexed.query("SELECT * FROM entries", limit=2)) <= 2


def test_sql_tool_respects_explicit_limit(indexed):
    rows = indexed.query("SELECT * FROM entries LIMIT 3")
    assert len(rows) <= 3


def test_staleness_is_reported_not_hidden(indexed):
    st = indexed.stats()
    assert st.entries > 0
    assert not st.is_stale(max_age_s=900)          # just built
    assert st.is_stale(max_age_s=0.0)              # any positive age is stale at 0
    assert "built" in st.describe()


def test_search_names_finds_substring(indexed):
    rows = indexed.search_names("main")
    assert any("main.py" in r["path"] for r in rows)


def test_search_names_is_case_insensitive(indexed):
    assert any("MAIN.PY" in r["path"].upper() for r in indexed.search_names("MAIN"))


def test_missing_index_reports_empty_not_error(settings, tmp_workspace):
    idx = FileIndex(settings=settings, db_path=tmp_workspace / "nope.sqlite")
    st = idx.stats()
    assert st.entries == 0
    assert st.age_s() == float("inf")


def test_invalidate_clears_entries(indexed):
    indexed.invalidate()
    assert indexed.stats().entries == 0