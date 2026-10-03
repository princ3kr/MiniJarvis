"""Regression tests for the searchFiles agent in glinerTest.ipynb.

Run:  python3 test_searchfiles.py

These live outside the notebook so the pipeline stays lean. They are the tests that
caught real bugs while building it:
  - `find -iname` placed AFTER `-print`, so find ignored it and returned every file
  - the folder histogram counted direct children AND rglob, double-counting them
  - a near-miss similarity floor that provably could not separate the two cases
  - .zip/.pdf hidden from *search* instead of only from reading
  - LLMConfigError vs LLMUnavailable conflation (a 404 model reported as "no key")
  - folders invisible to `find -type f`, so every project folder was unsearchable
  - .zip missing from FILENAME_RE, so archive questions fell through to junk matches
"""
import json
import os
import pathlib
import sys
import tempfile

NB = pathlib.Path(__file__).with_name("glinerTest.ipynb")


def load_pipeline() -> dict:
    """Exec the notebook's code cells (minus the live-run demo) into one namespace."""
    nb = json.loads(NB.read_text())
    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    g = {"__name__": "searchfiles"}
    # the last code cell is the ask(...) demo; skip it to avoid a network call
    for i, c in enumerate(codes[:-1]):
        exec(compile(c, f"<cell{i}>", "exec"), g)
    return g


# The pipeline lives in its OWN namespace (exec'd by load_pipeline). Functions it
# defined resolve globals there, NOT in this module -- so mutating pipeline config
# must go through PIPE, never globals().
PIPE: dict = {}

results = []


def check(name, fn):
    try:
        results.append((name, "ok", str(fn())[:78]))
    except Exception as e:
        results.append((name, "RAISED", f"{type(e).__name__}: {e}"))


def check(name, fn):
    try:
        results.append((name, "ok", str(fn())[:78]))
    except Exception as e:
        results.append((name, "RAISED", f"{type(e).__name__}: {e}"))


def t_escape():
    for bad in ["/etc/passwd", "/home/princ3kr/../../etc/passwd"]:
        try:
            safe_resolve(bad); raise AssertionError(f"LEAK {bad}")
        except PermissionError:
            pass
    return "blocked absolute + traversal"


def t_allowlist():
    try:
        call_tool("delete_everything"); raise AssertionError("breach")
    except ValueError:
        return "unknown tool refused"


def t_missing():
    r = read_file("/home/princ3kr/nope_12345.txt")
    assert r["status"] == "not_found", r
    return r["status"]


def t_binary():
    import tempfile, os, shutil
    d = tempfile.mkdtemp(dir=ROOTS[0], prefix="_smoke_")
    try:
        fd_, tmp = tempfile.mkstemp(suffix=".dat", dir=d)
        os.write(fd_, b"\x00\x01\x02" * 100); os.close(fd_)
        r = read_file(tmp)
        assert r["status"] == "binary", r
        return r["status"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_outside():
    import tempfile, os
    fd_, tmp = tempfile.mkstemp(suffix=".dat", dir="/tmp")
    os.write(fd_, b"hi"); os.close(fd_)
    try:
        r = read_file(tmp)
        assert r["status"] == "blocked", r
        return r["status"]
    finally:
        os.unlink(tmp)


def t_caps():
    r = read_file("/home/princ3kr/Desktop/learnings/glinerTest.ipynb")
    assert len(r.get("text", "")) <= MAX_FILE_BYTES * 1.2
    return f"{r['n_lines']}/{r['total_lines']} lines"


def t_grep():
    hits = search_content("ipynb", root="/home/princ3kr/Desktop")
    assert hits and hits[0]["line_no"] > 0
    return f"{len(hits)} hit(s), first L{hits[0]['line_no']}"


def t_notfound():
    import tempfile, shutil
    d = tempfile.mkdtemp(dir=ROOTS[0], prefix="_empty_")
    try:
        # built at RUNTIME so this notebook's own source cannot match it
        needle = "zzz" + "qqx" + "nomatch" + "99871"
        r = run_agent(needle, root=d)
        assert r["status"] == "not_found" and r["confidence"] == 0.0, r
        assert "not_found" in build_prompt_bundle(r)
        return "graceful not_found"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_root_override():
    try:
        find_files("*.py", root="/etc"); raise AssertionError("leak")
    except PermissionError:
        return "root= blocked"


# --- v2: exhaustive similarity recall
def t_similarity():
    import tempfile, shutil, pathlib
    d = pathlib.Path(tempfile.mkdtemp(dir=ROOTS[0], prefix="_sim_"))
    try:
        for n in ["utils.py", "utils_test.py", "my_utils.py", "util.py", "unrelated.py"]:
            (d / n).write_text("x = 1\n")
        got = {f["path"].name: f["kind"] for f in find_files("utils.py", root=d)}
        must = {"utils.py", "utils_test.py", "my_utils.py", "util.py"}
        assert must <= set(got), f"missed: {must - set(got)}"
        assert "unrelated.py" not in got, f"false positive: {got}"
        assert got["utils.py"] == "exact" and got["util.py"] == "fuzzy", got
        return "4/4 recalled, 1 rejected, kinds correct"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- v4: FOLDERS must be findable by name (the -type f bug)
def t_folders():
    import tempfile, shutil, pathlib
    d = pathlib.Path(tempfile.mkdtemp(dir=ROOTS[0], prefix="_dirs_"))
    try:
        (d / "PracticeML" / "ML").mkdir(parents=True)
        (d / "PracticeML" / "ML" / "HPP.ipynb").write_text("{}\n")
        (d / "WebDev").mkdir()
        hits = find_files("practice", root=d, fuzzy=False)
        names = {h["path"].name for h in hits}
        assert "PracticeML" in names, f"folder not found, got {names}"
        assert any(h["is_dir"] for h in hits), "folder not flagged is_dir"
        return "folder found by name (was 0 hits before)"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- v4: a folder must reach the LLM as a readable summary, not a failed read
def t_folder_evidence():
    import tempfile, shutil, pathlib
    d = pathlib.Path(tempfile.mkdtemp(dir=ROOTS[0], prefix="_ev_"))
    try:
        (d / "RustProj" / "src").mkdir(parents=True)
        (d / "RustProj" / "src" / "main.rs").write_text("fn main(){}\n")
        (d / "RustProj" / "README.md").write_text("# rust\n")
        r = run_agent("RustProj", root=d)
        assert r["status"] == "ok", r
        ev = r["evidence"][0]
        assert ev["kind"] == "folder" and ev["read_status"] == "ok", ev
        ex = ev["excerpt"]
        assert "src/" in ex, ex
        # the histogram must be built from the whole subtree, and must not
        # double-count direct children (README.md appears once, not twice)
        assert ".rs x1" in ex, f"nested type missing or miscounted: {ex}"
        assert ".md x1" in ex, f"direct child double-counted: {ex}"
        assert "2 file(s) in total" in ex, ex
        assert "FOLDER:" in build_prompt_bundle(r), "folder missing from bundle"
        return "folder summarised (incl. nested counts) + in bundle"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- v4: the jail must still span the whole home, and reach outside Desktop
# --- v5: archives must be FINDABLE (they were invisible: .zip was in the skip list,
#         so find reported not_found for a file that demonstrably existed)
def t_archive_findable():
    import tempfile, shutil, pathlib, zipfile
    d = pathlib.Path(tempfile.mkdtemp(dir=ROOTS[0], prefix="_zip_"))
    try:
        z = d / "Ask-My-Repo-mainV2.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("README.md", "# ask my repo\n")
            zf.writestr("src/main.py", "print('hi')\n")
        hits = find_files("Ask-My-Repo-mainV2.zip", root=d, fuzzy=False)
        assert len(hits) == 1, f"zip not found: {hits}"
        assert hits[0]["path"] == z, hits
        # and it must be LISTABLE with stdlib alone
        r = read_file(z)
        assert r["status"] == "ok", r
        assert "README.md" in r["text"] and "main.py" in r["text"], r["text"]
        assert r["meta"]["members"] == 2, r["meta"]
        return "zip found by name + members listed"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- v5: a binary must be findable but reported honestly as unreadable
def t_binary_findable_not_readable():
    import tempfile, shutil, pathlib
    d = pathlib.Path(tempfile.mkdtemp(dir=ROOTS[0], prefix="_bin_"))
    try:
        p = d / "diagram.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        hits = find_files("diagram.png", root=d, fuzzy=False)
        assert len(hits) == 1, f"png invisible: {hits}"
        r = read_file(p)
        assert r["status"] in ("binary", "skipped"), r
        assert r["text"] == "", "binary content leaked into evidence"
        return "png findable, read refused with reason"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- v5: the FILENAME_RE must recognise archives and docs. When `zip` was missing,
#         "what is inside Ask-My-Repo-mainV2.zip?" captured NO filename and fell back
#         to fuzzy-matching the fragment "Ask" against unrelated folders.
def t_intent_captures_archives():
    got = extract_intent("what is inside Ask-My-Repo-mainV2.zip?")
    assert got["filenames"] == ["Ask-My-Repo-mainV2.zip"], got["filenames"]
    # the filename fragment must not leak into keywords, or a junk folder search runs
    assert "Ask" not in got["keywords"], got["keywords"]
    assert not any(k.lower() == "ask" for k in got["keywords"]), got["keywords"]
    return "zip captured as filename, fragment not a keyword"


def t_intent_common_types():
    cases = {"read notes.pdf": "notes.pdf",
             "open AI_ML_Roadmap.docx": "AI_ML_Roadmap.docx",
             "show main.py": "main.py",
             "check config.yaml": "config.yaml",
             "summarise data.csv": "data.csv",
             "play clip.mp4": "clip.mp4",
             "run the Dockerfile": "Dockerfile"}
    for q, want in cases.items():
        got = extract_intent(q)["filenames"]
        assert want in got, f"{q!r} -> {got}, expected {want!r}"
    return f"{len(cases)} filename types captured"


# --- v5: end-to-end -- the zip must be FOUND and its members reach the LLM
def t_zip_end_to_end():
    import tempfile, shutil, pathlib, zipfile
    d = pathlib.Path(tempfile.mkdtemp(dir=ROOTS[0], prefix="_e2e_"))
    try:
        z = d / "Ask-My-Repo-mainV2.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("README.md", "# Ask My Repo\n")
            zf.writestr("src/index.py", "print('x')\n")
        r = run_agent("what is inside Ask-My-Repo-mainV2.zip?", root=d)
        assert r["status"] == "ok", r
        paths = [e["path"] for e in r["evidence"]]
        assert str(z) in paths, f"zip not in evidence: {paths}"
        bundle = build_prompt_bundle(r)
        assert "README.md" in bundle and "index.py" in bundle, "members missing from bundle"
        assert "unrelated" not in bundle
        return "zip found, members listed, in LLM bundle"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- v6: AGGREGATE questions must count, not pattern-match a folder name.
#         Regression: "how many images" matched folders literally named "images".
def t_aggregate_routing():
    qt = detect_query_type("how many images are there in the system?")
    assert qt["shape"] == "aggregate", qt
    assert qt["categories"] == ["image"], qt
    qt2 = detect_query_type("count the python files")
    assert qt2["shape"] == "aggregate" and qt2["categories"] == ["code"], qt2
    qt3 = detect_query_type("where is safe_resolve defined?")
    assert qt3["shape"] == "locate", qt3
    qt4 = detect_query_type("show me glinerTest.ipynb")
    assert qt4["shape"] == "locate", qt4
    return "aggregate/locate split correct on 4 probes"


def t_survey_counts():
    import tempfile, shutil, pathlib
    d = pathlib.Path(tempfile.mkdtemp(dir=ROOTS[0], prefix="_surv_"))
    try:
        (d / "sub").mkdir()
        for n in ["a.png", "b.png", "c.jpg", "d.py", "e.py", "f.py",
                  "g.md", "sub/h.png"]:
            (d / n).write_text("x")
        s = survey_files(root=d, categories=["image"])
        assert s["by_category"]["image"] == 4, s["by_category"]   # 3 + nested 1
        assert s["matched_count"] == 4, s
        s2 = survey_files(root=d, categories=["code"])
        assert s2["by_category"]["code"] == 3, s2["by_category"]
        assert s2["files_seen"] == 8, s2["files_seen"]
        assert ".png" in s2["by_extension"], s2["by_extension"]
        return "image=4 (incl. nested), code=3, seen=8"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_aggregate_bundle_states_scope():
    sv = survey_files(root=Path("/home/princ3kr"), categories=["image"])
    b = build_prompt_bundle({
        "status": "ok", "query": "how many images?", "route": "aggregate",
        "searches": [{"by": "survey", "categories": ["image"]}],
        "survey": sv, "evidence": [], "similar_files": [], "near_misses": [],
        "summary": "", "confidence": 0.85, "elapsed_ms": 1})
    assert "EXCLUDE" in b, "bundle does not state the counting caveat"
    assert ".cache" in b, "excluded dirs not listed"
    assert "TOTAL" in b, "no total emitted"
    return f"caveat + total present ({sv['matched_count']} images visible)"


# --- v6: name mismatch must yield near misses, not a dead end
def t_near_miss():
    import tempfile, shutil, pathlib
    d = pathlib.Path(tempfile.mkdtemp(dir=ROOTS[0], prefix="_nm_"))
    try:
        # config/settings stem ratio is only 0.31 -- below any sane string-similarity
        # floor -- yet settings.yaml is almost certainly what the user meant.
        # Extension agreement is what makes this recoverable.
        for n in ["settings.yaml", "setting.yml"]:
            (d / n).write_text("x = 1\n")

        # an exact hit must never be reported as a near miss
        assert run_agent("open settings.yaml", root=d)["status"] == "ok", "exact -> not ok"

        r = run_agent("open config.yaml", root=d)
        assert r["status"] == "near_miss", (
            f"{r['status']} ev={[e['path'] for e in r['evidence']]}")
        assert r["near_misses"], "no near misses offered"
        assert r["confidence"] < 0.5, r["confidence"]
        b = build_prompt_bundle(r)
        assert "NO EXACT MATCH" in b and "NEAR MISSES" in b, b[:200]
        assert r["near_misses"], "no near misses offered"
        assert r["confidence"] < 0.5, r["confidence"]
        b = build_prompt_bundle(r)
        assert "NO EXACT MATCH" in b and "NEAR MISSES" in b, b[:200]
        return f"{len(r['near_misses'])} near miss(es), bundle flags no-exact-match"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_true_not_found_stays_honest():
    import tempfile, shutil, pathlib
    d = pathlib.Path(tempfile.mkdtemp(dir=ROOTS[0], prefix="_tnf_"))
    try:
        # Case A: the extension exists in the tree but the name does not. Offering the
        # nearest same-extension file is the DESIRED behaviour ("did you mean...?"),
        # and the bundle must not claim it is a match.
        (d / "totally_unrelated_file.md").write_text("x")
        r = run_agent("zzzzqqq_nonexistent_thing_9999.md", root=d)
        assert r["status"] == "near_miss", r["status"]
        assert "NO EXACT MATCH" in build_prompt_bundle(r), "framing missing"

        # Case B: the extension is absent entirely -> empty pool -> honest not_found.
        r2 = run_agent("main.rs", root=d)          # no .rs file exists
        assert r2["status"] == "not_found", (
            f"{r2['status']} near={[c['path'].name for c in r2.get('near_misses', [])]}")
        assert r2["confidence"] == 0.0
        assert not r2["near_misses"], "invented near misses with no pool"
        return "same-ext -> near_miss; absent-ext -> honest not_found"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_no_hidden_extensions():
    import tempfile, shutil, pathlib
    d = pathlib.Path(tempfile.mkdtemp(dir=ROOTS[0], prefix="_ext_"))
    try:
        names = ["a.zip", "b.pdf", "c.docx", "d.png", "e.mp4", "f.parquet"]
        for n in names:
            (d / n).write_bytes(b"\x00\x01binary")
        missing = [n for n in names
                   if not find_files(n, root=d, fuzzy=False)]
        assert not missing, f"hidden from search: {missing}"
        return f"all {len(names)} binary types findable by name"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_whole_home_scope():
    assert ROOTS[0] == Path("/home/princ3kr"), ROOTS
    # a path under home but outside Desktop must resolve
    p = safe_resolve("/home/princ3kr/Downloads")
    assert p.is_relative_to(ROOTS[0])
    # and the whole-tree walk must not be truncated by timeout
    t0 = time.perf_counter()
    raw = _find_raw(None, ROOTS[0], kind="d")
    assert raw, "no dirs found"
    assert (time.perf_counter() - t0) < TIMEOUT_S, "whole-home walk timed out"
    return f"{len(raw)} dirs under home in {round((time.perf_counter()-t0)*1000)}ms"


# --- v2: full loop -> natural language, with the transport stubbed
def t_llm_loop():
    orig_post = requests.post
    captured = {}

    class Resp:
        status_code = 200
        def json(self):
            return {"model": "stub-llm",
                    "choices": [{"message": {"content":
                        "safe_resolve is defined in glinerTest.ipynb and enforces the jail."}}],
                    "usage": {"total_tokens": 42}}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update({"url": url, "headers": headers, "payload": json})
        return Resp()

    requests.post = fake_post
    try:
        out = synthesize(run_agent("where is safe_resolve defined?"))
        assert out is not None, "synthesize returned None"
        assert "safe_resolve" in out["text"], out
        assert out["usage"]["total_tokens"] == 42, out
        # the outgoing request must be a well-formed OpenAI payload
        assert captured["url"].endswith("/chat/completions"), captured["url"]
        assert captured["payload"]["model"] and captured["payload"]["temperature"] == 0.0
        assert captured["payload"]["messages"][0]["role"] == "system"
        assert "Authorization" in captured["headers"]
        return f"natural language via stub ({out['model']})"
    finally:
        requests.post = orig_post


# --- v3: a CONFIG fault must raise loudly, not degrade silently.
#     This is the regression test for the 404-model-masquerading-as-missing-key bug.
def t_config_error_loud():
    orig_post = requests.post

    class Resp404:
        status_code = 404
        text = "model does not exist"
        def json(self):
            return {"error": {"message": "The model `nope` does not exist"}}

    requests.post = lambda *a, **k: Resp404()
    try:
        try:
            synthesize(run_agent("where is safe_resolve defined?"))
        except LLMConfigError as e:
            assert "404" in str(e), e
            return "404 model -> LLMConfigError (loud)"
        raise AssertionError("404 was swallowed instead of raised")
    finally:
        requests.post = orig_post


def t_transient_quiet():
    orig_post = requests.post

    class Resp500:
        status_code = 503
        text = "upstream busy"

    requests.post = lambda *a, **k: Resp500()
    try:
        assert synthesize(run_agent("where is safe_resolve defined?")) is None
        return "503 -> None (quiet degrade)"
    finally:
        requests.post = orig_post


def t_no_key_config_error():
    orig = PIPE["_INLINE_KEY"]
    PIPE["_INLINE_KEY"] = ""
    old = {k: os.environ.pop(k, None) for k in ("GROQ_API_KEY", "LLM_API_KEY")}
    try:
        assert api_key() == "", "key still visible after blanking"
        try:
            synthesize(run_agent("where is safe_resolve defined?"))
        except LLMConfigError:
            return "missing key -> LLMConfigError"
        raise AssertionError("missing key did not raise")
    finally:
        PIPE["_INLINE_KEY"] = orig
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v

SUITE = [
    ("jail: escapes blocked", t_escape),
    ("allowlist fails closed", t_allowlist),
    ("missing file", t_missing),
    ("binary detection", t_binary),
    ("outside jail refused", t_outside),
    ("size caps", t_caps),
    ("grep + line numbers", t_grep),
    ("graceful not_found", t_notfound),
    ("root override blocked", t_root_override),
    ("exhaustive similarity", t_similarity),
    ("folders findable", t_folders),
    ("folder -> evidence", t_folder_evidence),
    ("archives findable", t_archive_findable),
    ("intent sees .zip", t_intent_captures_archives),
    ("intent sees common types", t_intent_common_types),
    ("zip end-to-end", t_zip_end_to_end),
    ("binary findable, not read", t_binary_findable_not_readable),
    ("no hidden extensions", t_no_hidden_extensions),
    ("aggregate/locate routing", t_aggregate_routing),
    ("survey counts correctly", t_survey_counts),
    ("aggregate states scope", t_aggregate_bundle_states_scope),
    ("near miss on mismatch", t_near_miss),
    ("true miss stays honest", t_true_not_found_stays_honest),
    ("whole-home scope", t_whole_home_scope),
    ("LLM loop -> prose", t_llm_loop),
    ("404 model -> LOUD", t_config_error_loud),
    ("503 -> quiet degrade", t_transient_quiet),
    ("missing key -> LOUD", t_no_key_config_error),
]

def main() -> int:
    pipe = load_pipeline()
    globals().update(pipe)
    # rebind, don't copy: pipeline functions resolve `_INLINE_KEY` in THIS dict,
    # so a copy would leave the real one untouched and the test would be a no-op.
    globals()["PIPE"] = pipe
    results.clear()
    for name, fn in SUITE:
        check(name, fn)

    print(f"{'test':<28} {'result':<8} detail")
    print("-" * 80)
    for nm, st, d in results:
        print(f"{nm:<28} {st:<8} {d}{'' if st == 'ok' else '  <-- '}")
    bad = [r for r in results if r[1] != "ok"]
    print("-" * 80)
    print(f"{len(results) - len(bad)}/{len(results)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
