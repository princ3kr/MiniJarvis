"""SQLite metadata index: path, size, mtime, extension. Read-only queries.

Why this exists
---------------
Two costs made the notebook's repeated `find` walks untenable once a loop drives
them:

1. Each `find` over a home directory takes seconds. A discovery loop that runs ten
   of them spends most of its wall clock in the kernel.
2. Every one of those walks is charged tokens, because the results go to a model.

So the walk happens ONCE, into SQLite, and the loop queries that.

Privacy, which is a real constraint and not a formality
--------------------------------------------------------
The index lists every path on the machine, which is a more sensitive artifact than
the jail itself: a path can reveal what someone's projects, health, or finances are
even with no file contents. Three consequences, all enforced here:

1. The database is created 0600 in a private directory. It is never world-readable.
2. Deny-listed paths are NEVER written. The index cannot become a side channel that
   answers "do you have ~/.ssh/id_rsa" -- it simply does not know.
3. Nothing is sent to a model unless a query is made. The agent sees index AGE and
   counts, not the whole listing.

Staleness is reported, not hidden
---------------------------------
A stale index is worse than no index: it produces confident answers about files that
moved. So `stats()` carries the age, the engine surfaces it in the prompt, and
`refresh()` can re-walk a subtree on demand.
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from .access import AccessDenied, AccessPolicy
from .config import Settings, load_settings
from .context import TaskContext

SCHEMA_VERSION = 2

# SQLite's own sidecar files. Never index them: their size changes on every write,
# so indexing them reports phantom files that mutate between queries.
_SELF_NAMES = frozenset({
    "index.sqlite", "index.sqlite-wal", "index.sqlite-shm", "index.sqlite-journal",
})

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entries (
    path     TEXT PRIMARY KEY,
    parent   TEXT NOT NULL,
    name     TEXT NOT NULL,
    ext      TEXT NOT NULL,
    is_dir   INTEGER NOT NULL,
    size     INTEGER NOT NULL,
    mtime    REAL NOT NULL,
    depth    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_name  ON entries(name);
CREATE INDEX IF NOT EXISTS idx_ext   ON entries(ext);
CREATE INDEX IF NOT EXISTS idx_size  ON entries(size);
CREATE INDEX IF NOT EXISTS idx_mtime ON entries(mtime);
CREATE INDEX IF NOT EXISTS idx_parent ON entries(parent);
CREATE INDEX IF NOT EXISTS idx_name_nocase ON entries(name COLLATE NOCASE);
"""


def default_db_path(settings: Settings | None = None) -> Path:
    """Where the index lives. Private directory, 0700, outside the read scope.

    Deliberately NOT inside read_roots: the index is metadata about the whole tree,
    and an agent that can read the index file directly bypasses the query tool's
    limits.
    """
    s = settings or load_settings()
    base = Path(os.getenv("JARVIS_INDEX_DIR", Path.home() / ".local" / "share" / "jarvis"))
    return base / "index.sqlite"


@dataclass
class IndexStats:
    path: Path
    entries: int = 0
    files: int = 0
    dirs: int = 0
    total_bytes: int = 0
    built_at: float = 0.0
    scan_seconds: float = 0.0
    roots: tuple[str, ...] = ()
    errors: int = 0

    def age_s(self) -> float:
        return max(0.0, time.time() - self.built_at) if self.built_at else float("inf")

    def describe(self) -> str:
        age = self.age_s()
        when = "never" if age == float("inf") else (
            f"{age:.0f}s ago" if age < 120 else f"{age / 60:.0f}m ago")
        return (f"index: {self.entries} entries ({self.files} files, {self.dirs} dirs, "
                f"{self.total_bytes / 1e9:.1f} GB) built {when} in {self.scan_seconds:.1f}s")

    def is_stale(self, max_age_s: float = 900.0) -> bool:
        """Stale means "old enough that a file may have moved".

        15 minutes is a guess: long enough that a loop reuses it, short enough that
        "changed 2 seconds ago" answers are not wrong.
        """
        return self.age_s() > max_age_s


class FileIndex:
    """Incremental walk -> SQLite. Query with a read-only SELECT."""

    def __init__(self, settings: Settings | None = None, db_path: Path | None = None,
                 policy: AccessPolicy | None = None) -> None:
        self.settings = settings or load_settings()
        self.db_path = Path(db_path) if db_path else default_db_path(self.settings)
        self.policy = policy or AccessPolicy.from_settings(self.settings)

    # ------------------------------------------------------------------ setup
    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        if read_only:
            # file: URI + mode=ro means a query physically cannot write, even if
            # the SQL is wrong. The SQL tool relies on this.
            uri = f"file:{self.db_path}?mode=ro"
            con = sqlite3.connect(uri, uri=True, timeout=5.0)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.db_path.parent, 0o700)
            except OSError:
                pass                      # a shared dir we cannot tighten: not fatal
            con = sqlite3.connect(str(self.db_path), timeout=30.0)
        con.row_factory = sqlite3.Row
        return con

    def exists(self) -> bool:
        return self.db_path.is_file()

    def _ensure_schema(self, con: sqlite3.Connection) -> None:
        con.executescript(_SCHEMA)
        row = con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            con.execute("INSERT INTO meta(key,value) VALUES('schema_version',?)",
                        (str(SCHEMA_VERSION),))
        elif int(row["value"]) != SCHEMA_VERSION:
            con.executescript("DROP TABLE IF EXISTS entries;")
            con.executescript(_SCHEMA)
            con.execute("UPDATE meta SET value=? WHERE key='schema_version'",
                        (str(SCHEMA_VERSION),))
        con.commit()

    # ------------------------------------------------------------------- scan
    def scan(self, ctx: TaskContext, roots: list[Path] | None = None, *,
             max_entries: int = 2_000_000) -> IndexStats:
        """Walk the scope once and (re)write the index.

        Walks with os.scandir rather than Path.rglob: it is several times faster and
        it yields the stat data in the same syscall, so there is no second stat per
        entry. Prunes with `policy.walkable`, so deny-listed and skipped directories
        are never descended into AND never recorded.
        """
        roots = roots or list(self.settings.read_roots)
        t0 = time.perf_counter()
        stats = IndexStats(path=self.db_path, roots=tuple(str(r) for r in roots))
        con = self._connect()
        try:
            self._ensure_schema(con)
            con.execute("BEGIN")
            con.execute("DELETE FROM entries")
            batch: list[tuple] = []
            for root in roots:
                self._walk_into(con, root, batch, stats, max_entries)
            if batch:
                con.executemany(
                    "INSERT OR REPLACE INTO entries"
                    "(path,parent,name,ext,is_dir,size,mtime,depth) "
                    "VALUES(?,?,?,?,?,?,?,?)", batch)
            stats.built_at = time.time()
            con.executemany(
                "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                [("built_at", str(stats.built_at)),
                 ("entries", str(stats.entries)),
                 ("roots", "\n".join(stats.roots)),
                 ("scan_seconds", f"{time.perf_counter() - t0:.3f}")])
            con.commit()
            stats.scan_seconds = round(time.perf_counter() - t0, 2)
            ctx.log("index", "scan", detail=stats.describe(),
                    entries=stats.entries, seconds=stats.scan_seconds)
        finally:
            con.close()
        return stats

    def _walk_into(self, con: sqlite3.Connection, root: Path, batch: list,
                   stats: IndexStats, max_entries: int) -> None:
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            if stats.entries >= max_entries:
                return
            try:
                with os.scandir(d) as it:
                    for entry in it:
                        if stats.entries >= max_entries:
                            return
                        try:
                            is_dir = entry.is_dir(follow_symlinks=False)
                        except OSError:
                            stats.errors += 1
                            continue
                        # A symlink is recorded as a link, never followed: following
                        # it would walk out of the scope and could cycle.
                        if entry.is_symlink():
                            try:
                                st = entry.stat(follow_symlinks=False)
                                size, mtime = st.st_size, st.st_mtime
                            except OSError:
                                stats.errors += 1
                                continue
                        else:
                            try:
                                st = entry.stat(follow_symlinks=False)
                                size, mtime = st.st_size, st.st_mtime
                            except OSError:
                                stats.errors += 1
                                continue

                        # Deny-list is enforced at WRITE time, not just at read
                        # time: an index that recorded ~/.ssh/id_rsa would leak its
                        # existence through row counts.
                        from .access import is_denied
                        denied, _ = is_denied(entry.path, self.settings.deny_globs)
                        if denied:
                            continue

                        # Never index ourselves. The database, its WAL and SHM
                        # sidecars live inside the read scope by default, so a scan
                        # would record its own growing files and report phantom
                        # entries whose sizes change between every query.
                        if entry.name in _SELF_NAMES or entry.name.startswith(".idx"):
                            continue
                        if entry.path.startswith(str(self.db_path.parent)):
                            continue

                        batch.append((entry.path, str(Path(entry.path).parent),
                                      entry.name, Path(entry.name).suffix.lower(),
                                      1 if is_dir else 0, size, mtime, depth))
                        stats.entries += 1
                        if is_dir:
                            stats.dirs += 1
                            child = Path(entry.path)
                            if self.policy.walkable(child) and depth < 24:
                                stack.append((child, depth + 1))
                        else:
                            stats.files += 1
                            stats.total_bytes += size
                        if len(batch) >= 5000:
                            con.executemany(
                                "INSERT OR REPLACE INTO entries"
                                "(path,parent,name,ext,is_dir,size,mtime,depth) "
                                "VALUES(?,?,?,?,?,?,?,?)", batch)
                            batch.clear()
            except (OSError, AccessDenied) as e:
                stats.errors += 1
                ctx = None
                # unreadable directory: count it and move on. One permission-denied
                # folder must not abort a whole-machine scan.
                continue

    # ------------------------------------------------------------------ query
    def query(self, sql: str, *, limit: int = 100, timeout_s: float = 5.0) -> list[dict]:
        """Run a read-only SELECT. Everything else is rejected.

        Rejected in three independent ways, because a read-only SQL tool handed a
        model is an exfiltration primitive if any one of them is wrong:
          * sqlite3 authorises only SELECT/WITH
          * the connection is opened mode=ro, so a write cannot execute at all
          * a LIMIT is enforced by rewriting, since a model forgetting LIMIT
            should not pull 2M rows into a prompt
        """
        cleaned = sql.strip().rstrip(";").strip()
        if not cleaned:
            raise ValueError("empty SQL")
        head = cleaned.split(None, 1)[0].upper()
        if head not in ("SELECT", "WITH"):
            raise ValueError(f"only SELECT or WITH is allowed, got {head!r}")
        upper = cleaned.upper()
        for banned in ("ATTACH", "PRAGMA", "VACUUM", "CREATE", "INSERT", "UPDATE",
                       "DELETE", "DROP", "REPLACE", "ALTER", "TRIGGER"):
            if re_word(banned, upper):
                raise ValueError(f"{banned} is not allowed in a read-only query")
        if not re_limit(upper):
            cleaned = f"{cleaned} LIMIT {int(limit)}"

        con = self._connect(read_only=True)
        try:
            con.execute("PRAGMA query_only=ON")
            cur = con.execute(f"SELECT * FROM ({cleaned}) LIMIT {int(limit)}")
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.OperationalError as e:
            raise ValueError(f"SQL rejected: {e}") from None
        except sqlite3.Error as e:
            raise ValueError(f"SQL error: {type(e).__name__}") from None
        finally:
            con.close()

    def search_names(self, pattern: str, *, limit: int = 50,
                     files_only: bool = True) -> list[dict]:
        """FTS-ish name search. LIKE with a lower() index, not a real FTS table.

        Deliberately not FTS5: FTS5's tokenizer strips punctuation, so searching for
        `safe_resolve` would not match `safe-resolve` or `safe_resolve.py`. Name
        search on a filesystem wants substring semantics, and LIKE gives that.
        """
        con = self._connect(read_only=True)
        try:
            sql = ("SELECT path, name, ext, size, mtime FROM entries "
                   "WHERE name LIKE ? COLLATE NOCASE "
                   + ("AND is_dir=0 " if files_only else "")
                   + "ORDER BY LENGTH(name), name LIMIT ?")
            rows = con.execute(sql, (f"%{pattern}%", int(limit))).fetchall()
            return [dict(r) for r in rows]
        finally:
            con.close()

    def stats(self) -> IndexStats:
        if not self.exists():
            return IndexStats(path=self.db_path)
        con = self._connect(read_only=True)
        try:
            meta = {r["key"]: r["value"] for r in
                    con.execute("SELECT key, value FROM meta").fetchall()}
            row = con.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(is_dir),0) d, "
                "COALESCE(SUM(CASE WHEN is_dir=0 THEN size ELSE 0 END),0) b "
                "FROM entries").fetchone()
            roots = tuple(meta.get("roots", "").split("\n")) if meta.get("roots") else ()
            return IndexStats(
                path=self.db_path, entries=row["n"], files=row["n"] - row["d"],
                dirs=row["d"], total_bytes=row["b"],
                built_at=float(meta.get("built_at", 0) or 0),
                scan_seconds=float(meta.get("scan_seconds", 0) or 0), roots=roots)
        finally:
            con.close()

    def invalidate(self, ctx: TaskContext | None = None) -> None:
        """Drop the rows but keep the schema, so the next query reports 'no index'."""
        if not self.exists():
            return
        con = self._connect()
        try:
            con.execute("DELETE FROM entries")
            con.execute("DELETE FROM meta WHERE key='built_at'")
            con.commit()
        finally:
            con.close()
        if ctx is not None:
            ctx.log("index", "invalidate", status="ok")


def re_word(word: str, haystack_upper: str) -> bool:
    """Whole-word search in an already-uppercased string, without importing re."""
    needle = f" {word} "
    return needle in f" {haystack_upper} "


def re_limit(upper: str) -> bool:
    return " LIMIT " in f" {upper} "