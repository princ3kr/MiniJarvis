# jarvis

A read-only file agent with a planner, a verifier, and a policy-enforced executor.
Every command runs inside a jail; nothing that reaches a model has been through one
unguarded.

This was a 4,600-line notebook. It is now an importable package with tests, because
the safety properties are the product and a safety property that lives only inside a
notebook is one nobody runs.

## Run it

```bash
.venv/bin/python -m jarvis "where is safe_resolve defined?"
```

The venv matters: `dspy` and `pydantic` are not in system python.

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