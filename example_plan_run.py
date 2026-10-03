"""Structural-only end-to-end run: no LLM needed for plan execution/verification."""
import json
import pathlib

NB = pathlib.Path("/home/princ3kr/Desktop/learnings/glinerTest.ipynb")
nb = json.loads(NB.read_text())
codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
g = {"__name__": "nb"}
for i, c in enumerate(codes):
    if c.lstrip().startswith("# ===== live run"):
        continue
    exec(compile(c, f"<cell{i}>", "exec"), g)
globals().update(g)

plan = Plan(goal="find safe_resolve, then count images", steps=[
    Step(id=1, agent="searchFiles",
         instruction="where is safe_resolve defined",
         depends_on=[], expected_output="list of file paths with line numbers"),
    Step(id=2, agent="searchFiles",
         instruction="count all images in the system",
         depends_on=[1], expected_output="counts per file extension"),
    Step(id=3, agent="searchFiles",
         instruction="open the zip archive contents of Ask-My-Repo-mainV2.zip",
         depends_on=[1], expected_output="list of archive members"),
])
print("\nPLAN")
show_plan(plan)

run = execute_plan(plan, use_llm_verify=False, replans_left=1)
print(f"\nRUN v{run['version']}: done={run['done']} failed={run['failed']} "
      f"inconclusive={run['inconclusive']} skipped={run['skipped']} "
      f"({run['attempts']} attempts)")
show_plan(plan)
print("\nPATH LOG")
show_path_log()
print("\nDEAD ENDS (planner's do-not-repeat list):", g["dead_ends"]() or "none")