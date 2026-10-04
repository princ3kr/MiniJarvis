# jarvis

A read-only file agent with a planner, a verifier, and a policy-enforced executor.
Every command runs inside a jail; nothing that reaches a model has been through one
unguarded.

This was a 4,600-line notebook. It is now an importable package with tests, because
the safety properties are the product and a safety property that lives only inside a
notebook is one nobody runs.

## Run it

Use `./run.sh`, which finds the venv for you:

```bash
./run.sh                                  # needs a query, prints usage
./run.sh "where is run_process defined?"
./run.sh --check                          # is the config sane?
./run.sh probe                            # the policy self-test
./run.sh test                             # the test suite
./run.sh -- ls -la                        # ask the policy about one command
./run.sh --tier -- cat /etc/passwd        # classify without running
```

Or call the module directly. The venv matters: `dspy` and `pydantic` are not in
system python.

```bash
.venv/bin/python -m jarvis "where is run_process defined?"
.venv/bin/python -m jarvis --check
.venv/bin/python -m jarvis.probe --quiet
```

Useful flags on a query:

| flag | effect |
|---|---|
| `--root DIR` | search somewhere specific (must be inside the read scope) |
| `--strategy direct` | skip the plan/replan machinery; one agent, one pass |
| `--no-verify` | structural checks only, no rubric, no LLM judge |
| `--last 20` | only the tail of the trace and attempts |

`--no-verify` is the one to reach for when you are iterating. It drops the per-step
LLM judge, so a query costs one call instead of several.

As a library:

```python
from jarvis.context import new_task
from jarvis.engine import run_task

ctx = new_task("how many python files are under src/")
run_task(ctx.query, ctx=ctx, strategy="direct", use_llm_verify=False)
print(ctx.summary())
```

## Verify it

```bash
.venv/bin/python -m pytest tests/ -q     # 110 tests, no network
.venv/bin/python -m jarvis.probe         # zero PERMISSIVE is the bar
```

The probe is the interesting one. It feeds the executor the notebook's own list of
attacks — untrusted binaries, paths outside the scope, recursive walks into
deny-listed directories, `env` wrapping an interpreter, an output flood — and
reports anything the policy allowed that it should not have. It exits non-zero on a
single finding. Baseline was 20; it is now 0.

## The shape

```
engine ──── router → planner → agents → verifier        the loop
   │
   └── gateway → policy → runner                        every command
          │
access ─┴── the jail. the only thing allowed to say yes to a path
```

`jarvis/CONTRACT.md` is the interface contract. Read it before changing anything.
`ARCHITECTURE.md` explains why the layering is the way it is, what is deliberately
broken, and which findings have been closed.

Three rules the codebase actually enforces:

1. **One jail.** `AccessPolicy.resolve` is the only way to turn a path into a real
   path. Tools, command operands, postconditions, and the index all go through it.
2. **One process spawner.** `runner.run_process` is the only place a child starts.
   Scrubbed env, process-group kill, rlimits, bounded output.
3. **No module-level mutable state.** No counters, no caches, no `gateway_reset()`.
   Everything mutable is a `TaskContext` field, so two tasks in one process are
   independent.

## Configuration

`.env`, untracked because it holds live credentials:

```
LLM_BASE_URL = ...
LLM_MODEL    = ...
DSPY_MODEL   = ...
```

`jarvis/config.py` parses it. Environment variables win, then explicit overrides.

The executor is off by default and stays off until deliberately raised:

```
executor_enabled_tier = 0     # 0 = read-only commands only
executor_dry_run     = true   # nothing that writes or executes, even at tier 1+
```

Raising either is a decision to make after reading the probe output.

## What is frozen

Snapshot/undo, tier 1 and 2, approvals, and interpreters are implemented and
present but gated off. `manifests.discovery` is `max_tier=0` with no `run_command`
tool, so command execution is unreachable from a normal task.

## Known issues

Preserved deliberately, each marked `# KNOWN BUG (removal scheduled)` with the phase
that removes it. The table in `ARCHITECTURE.md` lists all of them with locations.