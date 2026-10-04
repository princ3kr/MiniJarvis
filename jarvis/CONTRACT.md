# Internal interface contract (Phase 0 extraction)

Read this before writing any module. Every module below this line is already written
and tested. Do NOT change these signatures; work with them.

## config.py
```python
load_settings(**overrides) -> Settings     # env/.env, then explicit overrides win
SETTINGS                                 # module-level default, avoid using directly

@dataclass(frozen=True) Settings
  limits: Limits          budget: Budget        llm: LLMConfig
  read_roots: tuple[Path]  write_root: Path     deny_globs: tuple[str]
  skip_dirs: frozenset[str]
  executor_enabled_tier: int   executor_dry_run: bool
  executor_max_calls: int  executor_timeout_s: int
  planner_cot: bool   verifier_cot: bool   verifier_rubric: bool
  use_rg: bool|None        react_max_turns: int|None   # None -> from the budget
  .replaced(**kw) -> Settings

Every field is settable from `.env`. `Limits` fields are generated from the
dataclass as `LIMIT_<FIELD_UPPER>` so adding one cannot be forgotten here.
See `.env.example`.

Limits fields: find_max_results, dir_hint_max, fuzzy_pool_cap, fuzzy_cutoff,
  fuzzy_min_len, near_miss_min, near_miss_max, grep_max_results, grep_max_per_file,
  read_max_bytes, read_max_lines, read_context, binary_sniff,
  max_archive_members, max_results, timeout_s, exec_timeout_s,
  exec_max_output_bytes, cpu_rlimit_s, address_space_bytes, file_size_bytes

Budget fields: max_llm_calls, max_tool_calls, max_iterations, max_tokens,
  wall_clock_s, max_replans, max_parallel_plans

LLMConfig fields: base_url, model, dspy_model, api_key, timeout_s, temperature,
  max_tokens, fallback_models, reasoning_effort, reasoning_format
  # `reasoning_effort` is a CORRECTNESS setting, not a cost one: a reasoning model
  # spends max_tokens on hidden reasoning before producing anything, and a tight
  # budget returns HTTP 200 with empty content. Measured at 2498/2500 on gpt-oss-20b.
```
Also module constants: `CATEGORIES`, `NO_TEXT_EXT`, `LISTABLE_ARCHIVE_EXT`,
`SKIP_DIRS`, `DENY_GLOBS`, `PROJECT_ROOT`.
`access.SECRET_DIRS` is the short list of directory names a recursive walk must
exclude. Noise (`skip_dirs`) and secrecy (`SECRET_DIRS`) are different concerns and
are not conflated: requiring 30 skip-dirs in a model-written command made `du` and
`grep -r` unusable.

`llm.py` error taxonomy, and the distinction is load-bearing:
  LLMConfigError       permanent. Do NOT retry, do NOT degrade over it silently.
  LLMUnavailable       transient. Back off, retry, then fall back to the next model.
  LLMReasoningExhausted  a subclass of the above; "ask for more tokens", not
                       "try again later". `complete` handles it specially.
  EgressDenied          a prompt was refused before it left the machine.

## access.py  -- THE JAIL. One module, used by tools, command operands, and the index.
```python
AccessPolicy.from_settings(settings: Settings, cwd: Path|None=None) -> AccessPolicy
  .resolve(value, *, for_write=False, base=None, check_scope=True,
           check_deny=True) -> Path        # raises AccessDenied; THE entry point
  .try_resolve(value, **kw) -> Path|None   # filtering variant
  .check(value, **kw) -> (ok: bool, why: str)
  .walkable(path: Path) -> bool           # may a walk DESCEND here? (deny + skip)
  .in_scope(path) / .contains(path, roots) / .scope_roots() / .describe()
  fields: read_roots, write_roots, deny_globs, skip_dirs, cwd
is_denied(path, patterns) -> (denied: bool, matched_pattern: str)
class AccessDenied(PermissionError): .path, .reason  # reason: "denied"|"out_of_scope"|"empty"
```

## trace.py
```python
class Verdict(str, Enum):   pass_ = "pass"; fail = "fail"; unsure = "unsure"
class FailureType(str, Enum): none empty wrong error timeout contradiction
                            repeated inconclusive
@dataclass Event: n, kind, action, status, detail, duration_ms, ts, extra
@dataclass Attempt: attempt_id, plan_version, step_id, agent, instruction, status,
                   result_summary, verdict, failure_reason, failure_type,
                   evidence_checked, duration_ms
                   .compact() -> str;  .signature() -> (agent, normalised_instruction)

class Trace
  .event(kind, action, *, detail="", status="ok", t0=None, **extra) -> Event
  .attempt(*, plan_version, step_id, agent, instruction, status, result_summary,
           verdict, failure_reason="", failure_type=FailureType.none,
           evidence_checked="", duration_ms=None) -> Attempt
  .dead_ends() -> list[str]      # FAILURES only, compact form
  .signatures() -> set[tuple[str,str]]
  .verified_summary() -> str
  .render(last=None) -> str ; .render_attempts(last=None) -> str
```

## context.py  -- replaces the notebook's module globals
```python
new_task(query: str, *, settings=None, root=None, task_id=None, ...) -> TaskContext

@dataclass TaskContext
  query, settings, task_id, root: Path, budget: Budget, trace: Trace,
  counters: Counters, approvals: set[str]
  last_exit: str; last_stdout: str; last_stderr: str      # for the CURRENT command only
  started, finished_at, notes: list[str]
  .check() -> None                 # raises BudgetExhausted if spent
  .charge_llm(tokens=0); .charge_tool(name, n=1); .charge_iteration(); .charge_replan()
  .exhausted() -> str|None ; .elapsed_s() ; .out_of_time()
  .fingerprint(argv: list[str]) -> str        # \x00-joined, collision-free
  .approve(argv) ; .is_approved(argv) -> bool
  .log(kind, action, **kw) ; .attempt(**kw) ; .note(text)
  .close(status="ok") ; .summary() -> dict
  context manager: __exit__ swallows BudgetExhausted
class BudgetExhausted(RuntimeError)
venv_python() -> str|None ; running_in_venv() -> bool
```

## redaction.py
```python
redact(text: str, limit: int = 20_000) -> (text, n_redactions)   # never raises
scrubbed_env(workspace=None) -> dict[str, str]                   # minimal, no keys
contains_secret(text) -> bool
REDACT_PATTERNS
```

## Rules for new modules
1. **No module-level mutable state.** No `LOGS`, `PATH_LOG`, counters, caches, or
   `gateway_reset()`. Everything mutable is a parameter or a TaskContext field.
   The ONLY exception is an explicitly immutable cache (e.g. `functools.lru_cache`
   over pure functions with no arguments that change).
2. **Every filesystem path goes through `AccessPolicy.resolve`.** No `Path(...)`
   followed by an ad-hoc containment check. The index counts too.
3. **Every subprocess** uses `runner.run_process` (below), never bare
   `subprocess.run`. Never `shell=True`.
4. **Every LLM call** goes through `llm.call` and calls `ctx.charge_llm(tokens)`
   before returning.
5. **Never raise across an agent boundary.** Return a status field instead. A tool
   crash must degrade to a `blocked`/`error` observation, not an exception that
   kills the loop.
6. Type-annotate. `from __future__ import annotations` first. Docstring says WHY, not
   what -- especially where the notebook had a bug, say what it was.

## runner.py  -- the ONLY place a child process starts
```python
@dataclass ProcessResult: status, exit_code, duration_ms, stdout, stderr,
                          timed_out, command, truncated, redacted, reason
run_process(argv: list[str], *, cwd: Path, timeout_s: int, max_output_bytes: int,
            limits: Limits, expect_exit: int = 0) -> ProcessResult
```
`status` is one of: "ok" | "nonzero_exit" | "timeout" | "error".
Handles: scrubbed env, `killpg` on the child's own session, rlimits from `Limits`,
bounded ring keeping the tail, redaction. Never raises for a command that ran badly --
an exception across the gateway boundary kills the loop.

## gateway.py  -- the choke point
```python
authorize(spec, manifest, *, ctx, enabled_tier=None, settings=None)
    -> CommandResult | None        # None means ALLOWED; a refusal is a value
execute(spec, manifest, *, ctx, approve=False, dry_run=None, timeout_s=None)
    -> CommandResult               # never raises
run_postconditions(spec, cwd, *, proc, policy) -> list[CheckResult]
_unpruned(argv) -> list[str]       # secret dirs a WALK cannot exclude
```
Order: classify -> tool gate -> per-agent budget -> tier ceiling -> approval ->
dry-run -> cwd -> prune injection -> run. Returns `None` for "allowed" deliberately:
it makes `execute` unable to proceed without an explicit success from `authorize`, so
adding an early-return here cannot accidentally mean "allowed".

## router.py  -- ONE model call, three strategies
```python
route(query, *, ctx=None) -> Route      # never raises
Route: strategy (direct|loop|plan), reason, source ("llm"|"fallback"), cost
Route.escalate_to -> str | None
```
Falls back to `loop` when the model is unreachable, because loop is the only rung that
verifies its own work. A config error is reported AS a config error. A strategy the
model invents is not coerced.

## react.py  -- the loop
Actions the loop can dispatch to:
    run_command, find, grep, read, survey, index_query
`index_query` is a read-only SELECT against the metadata index, with an enforced
LIMIT and the index AGE in the observation -- a stale index produces confident
answers about files that moved, and the model cannot reason about staleness it is
not told about.
```python
run_react(goal, *, ctx, max_turns=None, use_llm_verify=True) -> ReactResult
ReactResult: answer, status, steps, answer_source, verified
             # status: answered | inconclusive | budget | stuck | refused
             # `showable` is `answer and verified`. There is deliberately no status
             # string beginning with "answered" that is not verified.
```

## pipeline.py  -- one pipeline, structured stages
```python
plan_path(query, *, ctx, strategy="auto", use_llm_verify=True, replans_left=2,
          max_turns=None, on_event=None) -> StageResult
StageResult: outcome ("found"|"near_miss"|"not_found"|"error"), text, evidence,
             strategy, reason, tried, verified, cost
synthesize_verified(query, candidate, evidence, *, ctx, use_llm_verify=True)
    -> (str | None, str)          # (None, why) when it does not hold up
run_task(...) -> str | None       # thin delegate
stage_result(...) -> StageResult  # what the CLI and eval use
```
Only `found` stops the ladder. `not_found` and `near_miss` escalate;
`error` does not -- a dead endpoint fails identically three times.

## verify_paths.py / verify_disk.py
```python
unverified_paths(answer, evidence) -> list[str]   # absolute paths not in evidence
recheck_disk_claims(answer, *, ctx, tolerance=0.02) -> (problems, checks)
```

## prune.py
```python
is_recursive(binary, argv) -> bool      # handles short-flag bundling
inject(argv, *, access, secrets=None) -> list[str]   # only ever ADDS arguments
```

## frozen/
```python
enabled(settings) -> bool          # UNKNOWN settings return True -- fail safe
approvals_enabled(settings) -> bool
interpreters_enabled(settings) -> bool   # constant False: no sandbox exists
tiers.apply(settings, binary, tier, reasons)
```

## Rules for new modules
1. **No module-level mutable state.** No counters, no caches, no `gateway_reset()`.
   Everything mutable is a parameter or a `TaskContext` field. The only exception is an
   explicitly immutable cache.
2. **Every filesystem path goes through `AccessPolicy.resolve`.** No `Path(...)`
   followed by an ad-hoc containment check. The index and the verification disk-recheck
   count too.
3. **Every subprocess** uses `runner.run_process`, never bare `subprocess.run`. Never
   `shell=True`.
4. **Every LLM call** goes through `llm.call`, which charges `ctx` BEFORE the request.
5. **Never raise across an agent boundary.** Return a status field. A tool crash
   degrades to a `blocked`/`error` observation, not an exception that kills the loop.
6. **Unknown means fail safe.** Where a check cannot evaluate -- `frozen.enabled(None)`
   -- it must assume the more restrictive answer. Assuming safety when information is
   missing is how a read-only agent ends up with a writer.
7. Type-annotate. `from __future__ import annotations` first. Docstring says WHY, not
   what -- and where there was a bug, say what it was and how it was found.
