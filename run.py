"""Launcher: verify the environment, then hand off to Jupyter Lab.

    ./run.sh          -> check env, then open Jupyter Lab
    ./run.sh check    -> env check only
    ./run.sh react    -> headless ReAct demo (no browser)
    ./run.sh plan     -> headless plan-execute demo
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV = ROOT / ".venv"
VENV_PY = VENV / "bin" / "python"
NOTEBOOK = ROOT / "glinerTest.ipynb"


def check() -> int:
    ok = True

    def row(label, good, detail=""):
        nonlocal ok
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {label:<34} {detail}")

    print("\n=== environment ===")
    row("venv exists", VENV.is_dir(), str(VENV_PY))
    row("running inside venv", sys.prefix == str(VENV),
        f"sys.prefix={sys.prefix}")

    for mod in ("dspy", "pydantic", "requests"):
        try:
            __import__(mod)
            row(f"import {mod}", True)
        except ImportError:
            row(f"import {mod}", False, "MISSING - planner/ReAct disabled")

    try:
        import dspy
        row("dspy version", True, dspy.__version__ if hasattr(dspy, "__version__")
            else "?")
    except Exception:
        pass

    row("notebook present", NOTEBOOK.is_file(), NOTEBOOK.name)

    # The key can come from the env OR be inlined in the notebook (_INLINE_KEY).
    # Checking only the env would report a false failure, since the notebook is
    # the primary way it is supplied here.
    import json
    import re
    env_key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
    inline_key = ""
    if NOTEBOOK.is_file():
        nb_src = "".join("".join(c["source"]) for c in
                         json.loads(NOTEBOOK.read_text())["cells"]
                         if c["cell_type"] == "code")
        m = re.search(r'_INLINE_KEY\s*=\s*"(gsk_[^"]+)"', nb_src)
        inline_key = m.group(1) if m else ""
    row("api key available", bool(env_key or inline_key),
        "env" if env_key else ("inline in notebook" if inline_key else "NONE SET"))
    if inline_key and not env_key:
        print("        note: key is in PLAINTEXT in the notebook. Prefer:")
        print("              export GROQ_API_KEY=\"gsk_...\"  and blank _INLINE_KEY")

    print(f"\n=== usage ===\n  DSPY_MODEL (planner/ReAct): {os.getenv('DSPY_MODEL', 'qwen/qwen3.8-27b')}")
    print(f"  LLM_MODEL  (answers)     : {os.getenv('LLM_MODEL', 'openai/gpt-oss-120b')}")
    print("  Groq's daily token limit is PER MODEL -- keep these aligned or the plan")
    print("  will execute fine and then the final answer will 429.")
    print("  notebook: open glinerTest.ipynb -> pick kernel 'Agent (venv: dspy)'")
    print("            -> Restart Kernel & Run All")
    print("            -> last cell runs a demo; or use run_task() below\n")

    if ok:
        print("=== next ===")
        print("  1. jupyter lab      (select the 'Agent (venv: dspy)' kernel)")
        print("  2. Restart Kernel & Run All")
        print("  3. in a new cell:")
        print('       run_task("how many images are in my Downloads folder?")')
        print('       run_task("where is safe_resolve defined?", strategy="react")')
        print('       show_path_log()')
        return 0
    print("\nSome checks failed - fix those before running the planner.")
    return 1


def run_headless(target: str) -> int:
    """Execute the notebook's cells in-process, then run one demo."""
    import json

    nb = json.loads(NOTEBOOK.read_text())
    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    g = {"__name__": "headless"}
    for i, c in enumerate(codes):
        if c.lstrip().startswith("# ===== live run"):
            print(f"--- skipping demo cell {i}")
            continue
        exec(compile(c, f"<cell{i}>", "exec"), g)

    if target == "react":
        g["run_react"]("where is safe_resolve defined?")
    else:
        plan = g["Plan"](goal="demo", steps=[
            g["Step"](id=1, agent="searchFiles",
                      instruction="where is safe_resolve defined",
                      depends_on=[],
                      expected_output="list of file paths with line numbers"),
            g["Step"](id=2, agent="searchFiles",
                      instruction="count all images in the system",
                      depends_on=[1], expected_output="counts per extension"),
        ])
        g["show_plan"](plan)
        run = g["execute_plan"](plan, use_llm_verify=False, replans_left=1)
        print(f"\nRUN v{run['version']}: done={run['done']} failed={run['failed']} "
              f"inconclusive={run['inconclusive']}")
        g["show_path_log"]()
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "check":
        sys.exit(check())
    if arg in ("react", "plan"):
        sys.exit(run_headless(arg))
    sys.exit(check())