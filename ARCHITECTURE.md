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
| `MAX_STEPS_PER_PLAN` is decorative; the cap is a literal elsewhere | `engine` | Phase 1 (one setting) |
| `use_planner` branch is unreachable | `engine` | Phase 4 |
| regex intent detection + huge extension list | `agents` | Phase 3 (index + fast-path only) |
| `webSearch` declared in the planner's vocabulary, unimplemented | `manifests` | Phase 5 |
| snapshot/undo, tier 1/2, approvals, interpreters | frozen | Phase 6+ |

### Removed, and proved removed

Each of these was a probe finding against the notebook's executor. They are fixed, and
`python -m jarvis.probe` is the regression net — **zero PERMISSIVE is the bar**.

| finding | fix |
|---|---|
| `find -fprint0` wrote a file at tier 0 | dangerous-flag set in `policy` |
| `cat`/`ls`/`diff`/`grep` read outside the read roots | every PATH operand resolves through the jail |
| policy keyed on `basename(argv[0])`, so `./cat` and `/tmp/git` classified as the real tools | `argv[0]` must be absolute **and** in `TRUSTED_BIN_DIRS`; `PATH` is not trusted because it contains `~/.local/bin` |
| `git commit`/`add`/`branch -D` were all tier 0 | git subcommand tier split; `config alias.*` denied |
| `git diff --output=`, `--git-dir=`, `-C` retargeted or wrote | each denied explicitly |
| approval key was `" ".join(argv)`, so `["a b"]` matched `["a","b"]` | `ctx.fingerprint` is `\x00`-joined |
| approvals lived in a module set, shared across tasks | per-task `ctx.approvals` |
| `exit_code` postcondition did `str(v) in _LAST_EXIT`, so "1" matched "10" | exact comparison |
| `file_exists` postcondition called `Path(p).exists()` unscoped, probing `/root/.ssh` through the exit status | postconditions resolve through the jail; unresolvable is a FAIL, not a pass |
| `grep -r` walked into deny-listed dirs (`decoy-marker` leaked) | `_check_recursion` **requires** the exclusions to be present in argv |
| `env python3 x.py` classified tier 0 | `env` is a launcher: the wrapped command is re-classified, `LD_PRELOAD`-class assignments denied |
| `pip show` reached the network at tier 0 | network binaries denied without a sandbox |
| unbounded output buffered without limit | bounded ring in `runner`; overflow kills the group |
| `communicate()` on timeout left grandchildren holding the pipes | `killpg` on the child's own session |
| executor never re-checked `max_calls` | charged and enforced per task |

Two more things the probe found that were not findings but would have been. `sed` and
`tar` had real grammars that were never wired into the registry, so every `sed -n` and
`tar -cf` escalated to APPROVAL for no reason — a refusal caused by the policy's own
wiring, which trains a model to stop asking. And POSIX short-flag bundling (`-cf`) was
unknown, so ordinary `tar` syntax was treated as suspicious.

## Test layers

```
tests/test_access.py    the jail: escapes, symlinks, deny-list depth
tests/test_index.py     deny-list at WRITE time; read-only SQL; staleness reported
tests/test_tools.py     find/grep/read/survey against a tmp fixture tree
tests/test_runner.py    env scrubbing, group kill, rlimits, the output cap
tests/test_router.py    query shape + intent extraction
tests/test_planner.py   DAG, JSON repair, replan id remap, dead ends
```

The probe is `python -m jarvis.probe`, a module rather than a notebook cell: a safety
property that lives only inside a notebook is one nobody runs. It exits non-zero on any
PERMISSIVE finding. Three cases are reported **INCONCLUSIVE** rather than PASS — `rg`
is not installed on this machine, so the policy refuses it for being unknown, which
proves nothing about `rg`.

No test may touch the network or walk a real tree.