#!/usr/bin/env bash
# Entry point. Uses the project venv because dspy/pydantic are NOT in system python.
#
#   ./run.sh "where is safe_resolve defined?"     ask a question
#   ./run.sh probe                                run the policy probe
#   ./run.sh test                                 run the test suite
#   ./run.sh check                                verify configuration
#   ./run.sh exec -- ls -la                       run one command through the gateway
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-$PWD/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "no venv at $PY. create one with: python3 -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi

# No model defaults here on purpose. `.env` is the single source of truth, and an
# exported fallback silently overrides it -- `jarvis.config._env` prefers
# os.environ over the file, so `export LLM_MODEL=gpt-oss-120b` here would have hidden
# a working gpt-4o-mini config behind a 404 on every run. Set the variables in your
# shell to override, or edit .env.

cmd="${1:-}"
shift || true

case "$cmd" in
  probe)  exec "$PY" -m jarvis.probe "$@" ;;
  test)   exec "$PY" -m pytest tests/ -q "$@" ;;
  check)  exec "$PY" - <<'PYEOF'
from jarvis.config import load_settings
from jarvis.llm import preflight
from jarvis.access import AccessPolicy

s = load_settings()
print(f"read roots : {[str(r) for r in s.read_roots]}")
print(f"write root : {s.write_root}")
print(f"deny       : {len(s.deny_globs)} globs")
print(f"executor   : enabled_tier={s.executor_enabled_tier} dry_run={s.executor_dry_run}")
print(f"api key    : {'set' if s.llm.api_key else 'NOT SET'}")
pf = preflight(s, verbose=False)
print(f"llm         : {'PASS' if pf['ok'] else 'FAIL'} ({pf['kind']}) {pf['detail'][:80]}")
pol = AccessPolicy.from_settings(s)
print(f"jail       : {pol.describe()[:100]}")
PYEOF
  ;;
  exec)
    # One command through the gateway. Refuses anything policy denies and reports the
    # tier it classified at, so `exec` is a way to ask the policy a question.
    # `--tier` classifies without running; without it the command actually executes,
    # which for tier 1+ needs `--approve` and still respects dry_run.
    exec "$PY" -m jarvis.gateway "$@"
    ;;
  "")
    exec "$PY" -m jarvis
    ;;
  *)
    exec "$PY" -m jarvis "$cmd" "$@"
    ;;
esac