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
  planner_cot: bool   verifier_cot: bool   verifier_rubric: bool   use_rg: bool|None
  .replaced(**kw) -> Settings

Limits fields: find_max_results, dir_hint_max, fuzzy_pool_cap, fuzzy_cutoff,
  fuzzy_min_len, near_miss_min, near_miss_max, grep_max_results, grep_max_per_file,
  read_max_bytes, read_max_lines, read_context, binary_sniff,
  max_archive_members, max_results, timeout_s, exec_timeout_s,
  exec_max_output_bytes, cpu_rlimit_s, address_space_bytes, file_size_bytes

Budget fields: max_llm_calls, max_tool_calls, max_iterations, max_tokens,
  wall_clock_s, max_replans, max_parallel_plans

LLMConfig fields: base_url, model, dspy_model, api_key, timeout_s, temperature,
  max_tokens, fallback_models
```
Also module constants: `CATEGORIES: dict[str, frozenset[str]]`,
`NO_TEXT_EXT`, `LISTABLE_ARCHIVE_EXT`, `SKIP_DIRS`, `PROJECT_ROOT`.

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

## runner.py  (being written in parallel -- assume this signature)
```python
@dataclass ProcessResult: status, exit_code, duration_ms, stdout, stderr,
                          timed_out, command
run_process(argv: list[str], *, cwd: Path, timeout_s: int, max_output_bytes: int,
            limits: Limits, expect_exit: int = 0) -> ProcessResult
```
`status` is one of: "ok" | "nonzero_exit" | "timeout" | "error".
It handles: scrubbed env, process-group kill on timeout, rlimits, no shell.
If your module needs a different signature, say so in your summary instead of
silently changing it.