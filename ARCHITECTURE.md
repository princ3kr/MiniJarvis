# Architecture: what runs, in what order

Phase 0 status. Read this to understand the flow; the code is the authority when the
two disagree.

## The shape of a task

```
query
  │
  ├─ ROUTER (needs_plan) ──→ direct ──→ searchFiles agent ──→ answer
  │        [regex gate]      │
  │                         └─ plan ──→ Plan ──→ orchestrator ──┐
  │                                    (DAG, ready queue)        │
  │                                                              ▼
  └────────────────────────────────────────────────────► VERIFIER
                                                            │
                                       ┌────────────────────┴─────────────┐
                                       │ pass/unsure                       │ fail
                                       ▼                                   ▼
                                    answer                        dead_ends() ──→ replan
                                                                          (≤2)
```

Inner loop (`run_agent` → tools): for a locate query it recalls by name, then by
folder hint, then by content, then reads the top N. For an aggregate query it
surveys once. There is **no model-driven iteration** yet — that is Phase 3.

## Layers, and who may depend on whom

```
                    ┌──────────────────────────┐
                    │  engine   router         │  workflow (may change per phase)
                    │  planner  verifier       │
                    │  agents   prompts        │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  gateway  policy  tools  │  the safety core
                    │  runner                  │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  llm                     │  the seam
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  context  trace          │  per-task state
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  access  redaction       │  the jail
                    │  config                  │
                    └──────────────────────────┘
```

Dependencies point **downward only**. Nothing in the lower four layers imports the
upper four, which is what makes the safety core testable in isolation.

## The per-task contract

Every entry point takes a `TaskContext` and nothing else. There is no global state
to reset between tasks — that was the notebook's central defect.

```python
from jarvis.context import new_task
from jarvis.engine import run_task

ctx = new_task("files over 100MB in ~/Downloads")
answer = run_task(ctx, strategy="auto")
print(ctx.trace.render_attempts())
print(ctx.summary())          # llm_calls, tool_calls, tokens, budget_stop
```

`TaskContext` owns:

| | replaces |
|---|---|
| `budget`, `check()`, `charge_*()` | the absence of any limit |
| `counters` (incl. `gateway` per tool) | `_GATEWAY_CALLS` + `gateway_reset()` |
| `trace` (`Event` + `Attempt`) | `LOGS` + `PATH_LOG` |
| `approvals`, `fingerprint()` | `APPROVAL_GRANTED` |
| `last_exit/stdout/stderr` | `_LAST_EXIT/_LAST_OUT/_LAST_ERR` |

Two tasks in one process are independent. That is the property that makes the
notebook's `gateway_reset()` hack unnecessary.

## The jail

`access.AccessPolicy` is the **only** thing permitted to decide whether a path is
readable. Tools, command operands, and the index all call `policy.resolve(...)`.

Order of checks, and why:

1. expand `~`
2. resolve relative against the task's real cwd (not the process cwd, and never
   "try each root until one works" — that fan-out is why the notebook's tool jail
   and command jail disagreed about where `notes.txt` was)
3. containment by `Path.parents`, not string prefix, so `/home/u-evil` is not
   inside `/home/u`
4. **deny-list last, and it always wins** — a tier-0 read cannot reach a secret
   even though the secret is inside the read scope

`policy.walkable(dir)` is separate from `resolve`, because a recursive walk must
stop the **descent**, not just filter the output. Filtering output is not enough:
`grep -r` still reads the bytes on the way past, which is a probe finding.

## Budgets

A model-driven loop is unbounded by construction, so limits are not optional:

```
max_llm_calls 12 · max_tool_calls 40 · max_iterations 8
max_tokens 200_000 · wall_clock_s 300 · max_replans 2
```

`ctx.check()` raises `BudgetExhausted` **before** spending anything. That is an
outcome, not a crash: `TaskContext.__exit__` catches it, records a note, and the
engine reports what was found and what was not. `ctx.summary()["budget_stop"]`
names the ceiling that stopped it.

## Verifier semantics

Three verdicts, and the middle one is the one that gets confused:

| verdict | means | effect |
|---|---|---|
| `pass` | checked, satisfied | step `done` |
| `unsure` | **unproven, not wrong** | step `inconclusive`, reported separately |
| `fail` | checked, wrong | step `failed`, added to `dead_ends()` |

`unsure` becoming `fail` was a real bug: a rate-limited rubric made a working step
look broken, and `inconclusive` results were fed to the planner as dead ends — which
teaches a planner to avoid correct work.

`dead_ends()` contains **failures only**, for that reason.

## Phase 0 known bugs (carried forward deliberately)

These are preserved on purpose so later phases can remove them one at a time with a
test proving each removal. Marked `# KNOWN BUG (removal scheduled)` in the source.

| bug | where | removed in |
|---|---|---|
| `needs_plan` substring cues misfire; never picks ReAct | `router` | Phase 4 (classifier + ladder) |
| `run_executor` takes step text as literal argv | `engine` | Phase 1 (typed `CommandSpec`) |
| blocklist validators miss `=`-attached flags | `policy` | Phase 1 (flag allowlists) |
| policy keys on `basename(argv[0])` | `policy` | Phase 1 (absolute resolution) |
| blanket `tainted` flag blocks legitimate discovery operands | `policy` | Phase 1 (position-aware taint) |
| regex intent detection + huge extension list | `agents` | Phase 3 (index + fast-path only) |
| `webSearch`/`assistant` in the planner's vocabulary, unimplemented | `planner` | Phase 5 |
| snapshot/undo, tier 1/2, approvals, interpreters | frozen | Phase 6+ |

## Test layers

```
tests/test_access.py    the jail: escapes, symlinks, deny-list depth
tests/test_tools.py     find/grep/read/survey against a tmp fixture tree
tests/test_policy.py    the probe as pytest — zero PERMISSIVE is the bar
tests/test_planner.py   DAG, JSON repair, replan id remap, dead ends
tests/test_verifier.py  verdict semantics, especially unsure-vs-fail
tests/test_engine.py    ready queue, repeat rejection, budget exhaustion
```

No test may touch the network or walk a real tree. The `network`, `live`, and
`slow` markers exist for the cases that legitimately need to.