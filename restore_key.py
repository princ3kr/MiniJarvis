"""Restore the inline API key into the notebook from the newest available copy.

Safety: never prints the key. Idempotent. Run after any rebuild if
`./run.sh check` reports 'api key available: NONE SET'.
"""
import json
import pathlib
import re

ROOT = pathlib.Path("/home/princ3kr/Desktop/learnings")
NB = ROOT / "glinerTest.ipynb"
SOURCES = [
    NB,
    ROOT / ".ipynb_checkpoints" / "glinerTest-checkpoint.ipynb",
    pathlib.Path("/tmp/opencode/glinerTest.backup.ipynb"),
    pathlib.Path("/tmp/opencode/before_planner.ipynb"),
]
PAT = re.compile(r'(?:_INLINE_KEY|LLM_API_KEY)\s*=\s*"(gsk_[^"]+)"')


def find_key():
    for p in SOURCES:
        if not p.exists():
            continue
        try:
            src = "".join("".join(c["source"]) for c in
                          json.loads(p.read_text())["cells"] if c["cell_type"] == "code")
        except Exception:
            continue
        m = PAT.search(src)
        if m:
            return m.group(1), p.name
    return None, None


key, src_name = find_key()
if not key:
    print("No key found in any source. Set it with:")
    print('  export GROQ_API_KEY="gsk_..."')
    raise SystemExit(1)

nb = json.loads(NB.read_text())
changed = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    text = "".join(c["source"])
    if "_INLINE_KEY =" not in text:
        continue
    # match the assignment whether it holds a key, an empty string, or a placeholder
    new = re.sub(r'(_INLINE_KEY\s*=\s*")[^"]*(")',
                 lambda m: m.group(1) + key + m.group(2), text)
    if new != text:
        c["source"] = new.splitlines(keepends=True)
        changed += 1

NB.write_text(json.dumps(nb, indent=1))
print(f"restored key ({len(key)} chars) from {src_name}; {changed} cell(s) updated")
print("verify with: ./run.sh check")