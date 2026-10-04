"""jarvis -- a read-only file agent with a router, a discovery loop, and a verifier.

Layered so each import is cheap and the boundaries stay honest:

    config      constants and settings            (no project imports)
    trace       append-only per-task record
    context     TaskContext: budgets, counters, the trace
    access      the ONE jail: read scope + deny-list
    redaction   secret masking, env scrubbing
    runner      process hygiene: scrubbed env, process-group kill, rlimits
    policy      CommandSpec + per-binary risk classification
    gateway     the choke point every tool call passes through
    tools       find / grep / read / survey / run_command
    llm         provider interface + OpenAI-compatible client
    router      cheap classifier: direct | discovery | plan
    planner     multi-part decomposition (optional)
    verifier    code checks evidence, LLM rubric is optional
    engine      run_task, TaskResult, the escalation ladder

The notebook is now a thin driver over this package.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]