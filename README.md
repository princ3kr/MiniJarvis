# jarvis

A read-only file agent with a planner, a shell loop, and a trace you can audit.

You ask a question. A model decides whether one cheap tool call settles it, a bounded
shell loop, or a decomposed plan. Commands run through a policy that classifies each
one before it executes. Answers are verified by code first and shown only if they hold
up — otherwise you get `NOT FOUND` and the list of what was tried.

This was a 4,600-line notebook. It is a package with tests because the safety
properties are the product and a property that lives in a notebook is one nobody runs.

## Run it

```bash
cp .env.example .env      # then fill in LLM_API_KEY and the model
./run.sh                  # interactive console
```

```
you> how many files are >= 100MB?
  * route->loop  needs command output; size is not a tool call
  llm  openai/gpt-oss-20b  732 tok  534ms  -> tool_call run_shell
  $ /usr/bin/find . -type f -size +100M   ok  tier 0  41ms
       | ./project/big.iso
  verdict pass   the output answers the question
  llm  react_verify  -> PASS: the find output lists a 4MB file
  llm  react_decide  -> tool_call run_shell
  $ /usr/bin/stat -c %s ./project/big.iso   ok  tier 0  3ms
       | 4194304
  verdict pass   confirms the size
ANSWER  (loop)
  ./project/big.iso is 4194304 bytes (4.0 MiB).
  llm_calls=5 tool_calls=2 tokens=1980 elapsed=4.1s turns=2
```

Three things worth noticing in that trace: the **real argv** is printed, including the
exclusions the policy appended; `denied: direct` in the headline the command, not
`blocked: policy`; and every LLM call is a separate line. Nothing is summarised after
the fact.

### Console commands

| | |
|---|---|
| `/trace` | the last task's full trace: strategy path, commands, verdicts, cost |
| `/paths` | every path the last task touched |
| `/cost` | llm calls, tool calls, tokens, time |
| `/plan` | the verification chain: what was checked and what disagreed |
| `/root [DIR]` | set or show the search root (must be in scope) |
| `/scope` | read scope, write scope, deny-list size |
| `/model` | endpoint, model, dspy model, key status, fallbacks |
| `/noverify` | toggle `--no-verify` — **debug only**, output is labelled UNVERIFIED |
| `/quiet` | toggle live events |

A slash that is not a command word is passed through as a question, so
`which file is at /etc/hosts` works.

### One-shot

```bash
.venv/bin/python -m jarvis "how many lines of python are in this directory?"
.venv/bin/python -m jarvis --react --max-turns 6 "which file is >= 100MB?"
.venv/bin/python -m jarvis --check          # is the config sane?
```

## Verify it

```bash
./run.sh test        # 301 tests, no network
./run.sh probe       # the policy self-test; exits 1 on any finding
.venv/bin/python -m jarvis.eval        # labelled queries, gates, cost
```

**The probe is the one that matters.** It feeds the executor the notebook's own list
of attacks — untrusted binaries, paths outside the scope, recursive walks into
deny-listed directories, `env` wrapping an interpreter, an output flood, a
hand-crafted approval fingerprint — and reports anything policy allowed that it
should not have. Baseline was 20 findings. It is now **0 PERMISSIVE**, with 3 cases
reported INCONCLUSIVE rather than passing: `rg` is not installed on this machine, so
the policy refuses it for being an unknown binary, which proves nothing about `rg`.

The eval harness builds a tree designed to break agents (a readable decoy private
key, prompt injection in a filename *and* its contents, a near-duplicate of a name a
query mentions, a newline in a filename) and reports success, routing accuracy,
hallucinated paths, policy violations and cost. Its gates are numbers, so CI can
assert on them.

## How it works

```
query
  -> LLM JUDGE          one call: direct | loop | plan, with a reason
  -> stage              direct: one tool call    loop: think/act/verify    plan: sub-goals
  -> VERIFY             code first: paths, numbers, stat. then the rubric.
  -> SYNTHESIZE         from verified evidence only; checks re-run on the text
  -> or NOT FOUND       plus the list of what was tried
```

Escalation is a **ladder, not a classifier**: `direct → loop → plan` when a stage
returns `not_found`. The judge only has to be roughly right, because a wrong guess
costs one turn and the ladder recovers the rest.

Only verified text reaches you. `NOT FOUND` is a real outcome and is not dressed up.

## Configuration

Everything is in `.env`; nothing is hardcoded. `.env.example` lists every key.

| | |
|---|---|
| `LLM_BASE_URL` `LLM_MODEL` `LLM_API_KEY` | the endpoint. Any OpenAI-compatible one. |
| `DSPY_MODEL` | planner and router model. Defaults to `LLM_MODEL`. |
| `LLM_FALLBACK_MODELS` | tried in order on 429 / transient failure |
| `LLM_REASONING_EFFORT` | `low` by default. **Correctness, not cost** — see below. |
| `JARVIS_READ_ROOTS` | the read scope. Comma-separated. |
| `EXECUTOR_ENABLED_TIER` | `0`. Raising it is the consequential decision. |
| `EXECUTOR_DRY_RUN` | `1`. Refuses anything not provably read-only. |
| `JARVIS_MAX_ITERATIONS` / `MAX_LLM_CALLS` | loop budget, divided across stages |
| `LIMIT_*` | any `Limits` field, e.g. `LIMIT_EXEC_MAX_OUTPUT_BYTES` |

Two things that are not obvious and were both found the hard way:

- **`LLM_REASONING_EFFORT=low` is a correctness setting.** A reasoning model spends
  `max_tokens` on hidden reasoning *before* producing anything. Measured on
  `gpt-oss-20b`: one turn consumed 2498 of 2500 tokens on reasoning and returned
  HTTP 200 with empty content, which is indistinguishable from a model choosing
  silence. At `low` the same turn used 35 tokens.
- **A tool-calling model refuses to be prompted for JSON.** `gpt-oss` answers a
  request that describes running things with a real function call, and the endpoint
  rejects it: `HTTP 400: Tool choice is none, but model called a tool`. Neither
  `tools: []` nor `tool_choice: "none"` works around it. So the loop sends a real
  `tools` array, and falls back to JSON-in-text for models without tool support.

## What it will not do

- Write a file. `executor_enabled_tier` is 0 and the executor manifest is `max_tier=0`,
  so the agent can run shell commands and cannot write, execute code, or reach the
  network. Tiers 1/2, approvals and interpreters are in `jarvis/frozen/` and are
  unreachable until a sandbox exists.
- Read a secret. One jail, `access.AccessPolicy`, is the only thing allowed to say yes
  to a path. `/proc`, `/sys`, `/dev`, browser profiles, keyrings and other agents'
  private state are denied.
- Show an unverified answer. `--no-verify` is debug-only and its output is boxed and
  labelled.
- Guess a filename. `difflib` near-misses are returned as `near_miss`, which
  escalates rather than being presented as the answer.

## Known issues

Preserved deliberately, each marked `# KNOWN BUG (removal scheduled)` with the phase
that removes it. `ARCHITECTURE.md` has the full table.