# searchFiles agent + planner + ReAct

A read-only agent over your filesystem, with a planner, a verifier, and a ReAct loop.
Everything lives in `glinerTest.ipynb`.

## Run it

```bash
./run.sh          # check env, then open Jupyter Lab
./run.sh check    # env check only
./run.sh plan     # headless plan-execute demo (no browser)
./run.sh react    # headless ReAct demo
```

Then in Jupyter: **pick the `Agent (venv: dspy)` kernel** → **Restart Kernel → Run All**.

## In a notebook cell

```python
run_task("how many images are in my Downloads folder?")           # auto-routed
run_task("where is safe_resolve defined?", strategy="react")      # adaptive loop
run_task("find X then count Y", strategy="plan")                 # plan + verify

show_path_log()      # every attempt: what ran, what failed, why
show_plan(plan)      # step-by-step status
ask("show me glinerTest.ipynb")   # skip the planner, one agent call
```

## The trap that will bite you

**dspy and pydantic are only in `.venv`, not system python.** Run everything with
`./run.sh` or `.venv/bin/python`, and use the `Agent (venv: dspy)` kernel. Under system
python the planner and ReAct silently fall back to deterministic behaviour — nothing
crashes, you just get a different (worse) agent.

`./run.sh check` tells you which mode you're in.

## Environment

| | |
|---|---|
| venv | `.venv` — dspy 3.4.0, pydantic 2.13.5 |
| model | `qwen/qwen3.8-27b` via Groq (OpenAI-compatible) |
| jail | `/home/princ3kr` — nothing outside it is readable |

**Groq's daily token limit is per-model.** `gpt-oss-120b` and `qwen/qwen3.8-27b` have
separate buckets, so one being exhausted doesn't lock out the other. `run.sh` keeps
`DSPY_MODEL` and `LLM_MODEL` aligned — if they differ, the plan executes fine and then the
final answer hits a 429.

## Tests

```bash
.venv/bin/python test_planner.py       # 32 tests
.venv/bin/python test_searchfiles.py   # 28 tests
```

They exec the notebook's own cells, so they test what's actually in the file.

## What's here

| piece | what it does |
|---|---|
| `searchFiles` | 4 tools: `find_files` (files + folders, 4-tier similarity), `search_content`, `read_file` (capped), `survey_files` (counts) |
| planner | `dspy.Predict` + Pydantic `Plan`/`Step`, DAG-validated |
| verifier | structural checks → LLM rubric → `pass` / `fail` / `unsure` |
| orchestrator | plain Python: ready-queue, repeat rejection, replan cap |
| ReAct | `dspy.ReAct` loop over the same 4 jailed tools, `max_iters=6` |

Result contract, uniform across agents:

| field | meaning |
|---|---|
| `status` | `ok` / `near_miss` / `not_found` |
| `route` | `aggregate` / `locate` |
| `summary` | one line, no dumps |
| `evidence` | citable `path:line` excerpts |
| `confidence` | explicit, so the LLM hedges instead of inventing |

## Known gaps

- **`webSearch` is unimplemented.** Steps targeting it fail visibly rather than pretend.
  ReAct is where it would matter most — the check that a module is missing locally
  belongs in the tool wrapper, not the prompt.
- **ReAct answers are unverified.** Planned steps go through `verify_step`; a ReAct run
  doesn't. Reusing `rubric_verify` on the trajectory is the obvious fix.
- **Your API key is in plaintext in the notebook.** Prefer
  `export GROQ_API_KEY="gsk_..."` and blank `_INLINE_KEY`.
- **The jail spans your whole home**, so `~/Downloads/AMRv2_api_key.txt` is currently
  readable by the agent. There's no `SENSITIVE_NAMES` deny-list yet.
- **Rust `target/` is not pruned** — 4857 of the ~5130 visible files are build artifacts,
  which skews every count.