# Architecture

Read this after the code, not before. It records why the layering is the way it is,
what was measured, and what is deliberately still wrong.

## The pipeline

```
query
  -> router.route            ONE model call: direct | loop | plan + a reason
  -> pipeline.plan_path      the stage runs
       direct   one tool call, downgraded to near_miss if only fuzzy hits
       loop     react.ReactLoop, bounded by a share of the iteration budget
       plan     planner v2: sub-goals, each run by a bounded loop
  -> pipeline.synthesize_verified
       code first   verify_paths, _unsupported_numbers, verify_disk.recheck_disk_claims
       rubric last  the LLM, because it is the only part that understands prose
  -> StageResult(found | near_miss | not_found | error)
  -> escalate direct -> loop -> plan on near_miss / not_found
  -> text, or NOT FOUND plus what was tried
```

### Why a ladder and not a classifier

A three-way classifier has to be right about every query on the first try. A ladder
only has to be roughly right, because a stage that cannot settle the question is
itself the signal to try the next one. Guessing low costs one tool call; guessing
wrong costs one turn.

`error` deliberately does NOT escalate. If the endpoint is down, the next rung fails
identically, and three strategies' worth of budget to reach the same 401 is worse
than reporting the error.

### Why the router is a model

The decision is not lexical:

    "which file is >= 100MB"        needs `find -size`, i.e. a command
    "how many images are there"     one tool call
    "compare a.py and b.py"         two searches and a join

The first and second differ by one word. A cue list that separated them would be a
coincidence, and the previous one (`PLAN_NEEDED_CUES`) had no cue that could ever
select the loop, so `strategy="auto"` had two reachable rungs out of three.

The fast path — `detect_query_type`, `extract_intent` — stays, as an exact-lookup
helper. It is structurally barred from `route()`; `test_no_regex_decides_the_strategy`
reads `route()`'s source and fails if it consults any fast-path name.

## Layering

```
cli -> pipeline -> router / react / plan_stage
                      |
                   engine (scheduler, planner orchestration)
                      |
              gateway -> policy -> runner
                      |
                  access  (the jail)
```

Dependencies point downward only. `access.py` depends on nothing in the package.
`runner.py` depends on `redaction` and `config` and not on `access`, `policy` or
`gateway` — it is told where to run, not whether it may.

## The jail

`access.AccessPolicy` is the only thing permitted to decide whether a path is
readable. Tools, command operands, postconditions, the index and the verification
disk-recheck all call `policy.resolve(...)`.

Three predicates, not one, because they answer different questions:

| | question | scope |
|---|---|---|
| `resolve(path, for_write=)` | may I read/write this exact path? | read roots / write roots + deny-list |
| `walkable(dir)` | may a recursive walk DESCEND here? | deny-list + `SECRET_DIRS` + skip-dirs |
| `nameable(dir)` | may this be NAMED in output? | deny-list + `SECRET_DIRS` |

The split between `walkable` and `nameable` was a bug fix. The folder summary filtered
its child listing by `skip_dirs` while its recursive histogram did not, so
`__pycache__` was hidden from one half and counted in the other. Fixing it surfaced
the sharper problem: the filter was a *name* check, not the jail, and `.ssh` is not
in `skip_dirs`, so the listing announced `.ssh/` as a child. No contents leaked —
`resolve` refused the key inside — but publishing the existence of a private-key
directory to a remote model is a disclosure of its own, and preventing it is free.

So: `node_modules` is noise (nameable, not walkable) and `.ssh` is a secret (neither).
Same call site, opposite answers, which is why they are separate methods.

## Policy

`classify(spec, manifest, access, settings)` runs in a fixed order, and the order is
load-bearing:

1. resolve `argv[0]` — absolute, and in `TRUSTED_BIN_DIRS`. PATH is not trusted
   because it contains `~/.local/bin`.
2. always-denied and network-capable binaries
3. unknown binary → APPROVAL. Never tier 0.
4. `env` re-classifies whatever it wraps. It is a launcher, not a reader.
5. dangerous flags (`find -exec`, `sed -i`, `grep --pre`, git retargeting)
6. grammar: unknown flags escalate, never pass
7. interpreters → APPROVAL when `isolation == "none"`
8. git subcommand tier split
9. every PATH operand through the jail
10. cwd through the jail — READ scope, not write. A read-only agent must be able to
    run `ls` in the tree it was asked to search.
11. taint — but only for tier ≥ 1 (see below)
12. tier floor
13. the manifest caps everything

### Short-flag handling

Three things a hand-written flag table gets wrong, all found by feeding it real
command lines:

- **Bundling.** `wc -lw`, `tar -cf out.tar .`, `ls -lah`. `wc`'s `-l`/`-w` are
  value-LESS selectors; typed as value-taking they made the commonest `wc` invocation
  fail the bundle check.
- **Attached values.** `cut -d:`, `-f1`, `-n5`. getopt accepts these; a parser that
  only knew `--long=v` and `-s=v` reported "unknown flag".
- **Illegal bundles.** `-fc` is genuinely ambiguous — POSIX reads `-fc` as `-f c`,
  GNU tar reads it as `-f -c`. POSIX wins, and the consequence is safe either way
  because the "value" becomes a path operand and goes through the jail.

`tests/test_grammars.py` pins 40 ordinary invocations plus counter-tests that an
unknown flag still escalates and `sed -i` still denies. A narrow grammar makes the
agent look broken; a wide one is the failure that matters. Both directions are tested.

### The policy injects prune arguments

The loop does not write exclusions. `jarvis/prune.inject` appends them.

This was measured: the prompt told the model to add `--exclude=.ssh` to every
recursive command, and it produced `-not -path */.ssh/*` three turns running, which
the grammar rejects. Every attempt was refused and the budget ran out.

Three properties make it a control rather than a hole:

- it only ever ADDS arguments, and runs AFTER authorization, so a command that was
  going to be refused is still refused
- `_unpruned()` re-derives the guarantee on the POST-injection argv — a guarantee
  checked by the function that implements it is not a check
- the post-rewrite argv is what runs AND what the trace shows

## Verification

The order is code first, rubric last, and each step earns its place by catching
something the others cannot.

| check | catches | why code |
|---|---|---|
| `verify_paths.unverified_paths` | a fabricated absolute path | a model asked "is this path in the evidence?" says yes often enough to be useless |
| `react._unsupported_numbers` | `ls -l`'s `total 376` restated as "37K" | the number IS in the output, in the wrong column; a rubric says PASS |
| `verify_disk.recheck_disk_claims` | evidence that is internally consistent and wrong | `stat` is cheaper than any model and cannot be talked into agreeing |
| `pipeline.admits_failure` | "I don't have the information for any file" printed under a green ANSWER | it is a *truthful* sentence about the evidence, so every other check passes |
| rubric | prose claims | the only part that understands prose |

Two of these exist because the failure they catch passed everything else.

`_unsupported_numbers` matches on TOKEN boundaries. Substring matching passed on
exactly the bug it was written for, because `total 376` contains "37" — the same
failure as the approval fingerprint that joined argv with spaces.

The checks run again on the SYNTHESIS, not just the evidence. Verifying the evidence
and then letting the model write freely from it leaves the gap this closes.

## Frozen

`jarvis/frozen/` holds tiers 1 and 2, taint escalation, and the interpreter rule. It
is gated on `Settings.executor_enabled_tier`, which ships at 0.

One subtlety worth recording: `frozen.enabled(None)` returns **True**. Unknown
settings must fail safe. The first version defaulted to 0, concluded the frozen tiers
were unreachable, and classified `touch` as tier 0 — the probe reported PERMISSIVE for
correct behaviour. Applying a floor can only ever make a command *more* restricted, so
assuming the worst is free.

## Runner

One place a child process starts. Six guarantees:

- no shell, argv only
- scrubbed env, so a subprocess printing its own environment cannot return a key
- `start_new_session` + `killpg`. `proc.kill()` kills only the direct child, so
  `sh -c 'sleep 999'` leaves a grandchild holding the pipes and the timeout *hangs*
  instead of firing.
- rlimits read from `Limits`. The notebook hard-coded 1 GiB in a closure, which made
  `Limits` decorative. `RLIMIT_NPROC` still unset — per-UID, not per-process.
- bounded ring buffer keeping the TAIL. `communicate()` buffered without limit, and
  the first capped version let `yes` run the full timeout anyway.
- redaction on the way out

## Removed, and proved removed

`python -m jarvis.probe` is the regression net. **Zero PERMISSIVE is the bar.**

| finding | fix |
|---|---|
| `find -fprint0` wrote a file at tier 0 | dangerous-flag set |
| reads outside the read roots | every PATH operand through the jail |
| `basename(argv[0])` let `./cat` and `/tmp/git` pass as the real tools | absolute + trusted dir |
| `git commit`/`add`/`branch -D` all tier 0 | subcommand tier split |
| `git config alias.*` executed code | denied |
| approval key was `" ".join(argv)` — `["a b"]` matched `["a","b"]` | `\x00`-joined |
| approvals in a module set, shared across tasks | per-task `ctx.approvals` |
| `exit_code` check did `str(v) in _LAST_EXIT` — "1" matched "10" | exact comparison |
| `file_exists` postcondition called `Path(p).exists()` unscoped | jail-checked |
| `grep -r` walked into deny-listed dirs | policy-injected exclusions |
| `env python3 x.py` was tier 0 | `env` re-classifies what it wraps |
| `pip show` reached the network at tier 0 | network binaries denied |
| unbounded output buffered without limit | bounded ring |
| `communicate()` left grandchildren on timeout | `killpg` |
| executor never re-checked `max_calls` | charged and enforced per task |

Two more that were not findings but would have become them: `sed` and `tar` had real
grammars that were never registered, so every `sed -n` and `tar -cf` escalated for no
reason — a refusal caused by the policy's own wiring, which trains a model to stop
asking.

## Known bugs, preserved

| bug | where | removed in |
|---|---|---|
| `MAX_PLAN_STEPS` is a literal not a setting | `planner` | when a setting is wanted |
| `needs_plan` shim kept for API compatibility | `router` | with its last caller |
| `run.sh` carries a fallback model default | `run.sh` | never; see README on `.env` |

## Contract

`jarvis/CONTRACT.md` is the interface contract. Read it before changing a signature.