#!/usr/bin/env bash
# Entry point. Uses the project venv because dspy/pydantic are NOT in system python.
set -euo pipefail
cd "$(dirname "$0")"
export DSPY_MODEL="${DSPY_MODEL:-qwen/qwen3.8-27b}"
# The notebook's own answer-composition client uses LLM_MODEL (default gpt-oss-120b).
# Groq's daily token limit is PER MODEL, so leaving the two on different models means
# the plan executes fine and then the final answer fails on 429. Keep them aligned.
export LLM_MODEL="${LLM_MODEL:-$DSPY_MODEL}"
PY=".venv/bin/python"

[ -x "$PY" ] || { echo "venv missing. Create it with:"; echo "  python3 -m venv .venv && .venv/bin/pip install dspy ipykernel"; exit 1; }

case "${1:-lab}" in
  check) exec "$PY" run.py check ;;
  plan|react) exec "$PY" run.py "$1" ;;
  lab)   "$PY" run.py check || true
         echo "starting Jupyter Lab -- select the 'Agent (venv: dspy)' kernel"
         exec jupyter lab ;;
  *)     exec "$PY" run.py "$1" ;;
esac
