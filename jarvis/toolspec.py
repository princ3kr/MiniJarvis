"""Native tool-calling for the loop, with JSON-in-text as the fallback.

Why this exists
---------------
The configured model (`openai/gpt-oss-20b`) is a tool-calling model. Asked, in plain
text, to emit `{"command": [...]}` it emitted a real function call instead, and the
endpoint refused the request:

    HTTP 400: Tool choice is none, but model called a tool

Measured, not guessed. The trigger is not one word: `argv`, `args`, `command`,
"run shell programs", "actions available" and "you have tools" ALL trip it, while
`{"answer": "..."}` does not. So no amount of prompt rewording fixes it -- the model
wants a `tools` array and we were not giving it one.

Fighting that would mean either a different model (a config choice the user should
make, not one made for them) or a prompt so vague the loop cannot work. So the loop
speaks the model's actual dialect: it sends a `tools` array, and if the model calls
one, that call is executed.

Both paths are supported, because "works with one model" is not a property worth
having:

  * **native** -- a `tool_calls` entry on the response. Preferred. The model chose a
    structured call, so there is nothing to parse and nothing to mis-parse.
  * **text** -- a JSON object in the content. For a model with no tool-calling
    support, which is most of them.

`llm.complete` returns whichever the endpoint sent, and this module converts both into
one `Decision`. Everything downstream sees the same thing, so the loop has a single
code path for interpreting a turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The tool the model may call. Deliberately ONE, and deliberately named `run_shell`
# rather than `run_command`: the model already treats the word "command" as a tool
# name and we would be fighting our own schema.
SHELL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": (
            "Run one read-only shell program and return its output. argv is a list of "
            "strings with no shell metacharacters: no pipes, no &&, no redirects, no "
            "globs. One program per call. The policy adds deny-list exclusions to any "
            "program that walks directories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The program and its arguments, e.g. [\"ls\",\"-la\"].",
                },
                "why": {
                    "type": "string",
                    "description": "What this will tell you. Under 15 words.",
                },
            },
            "required": ["argv"],
        },
    },
}

# A second "tool" for finishing. Expressing finish as a tool rather than a JSON shape
# is what makes the native path work at all: the model must have a way to say "done",
# and a tool is the only way a tool-calling model will.
FINISH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Give the final answer, supported only by output you have actually seen. "
            "If the evidence does not answer the question, say what is missing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The answer, in prose."},
            },
            "required": ["answer"],
        },
    },
}

TOOLS: list[dict[str, Any]] = [SHELL_TOOL, FINISH_TOOL]


@dataclass
class Decision:
    """One turn's intent, whichever dialect it arrived in."""

    kind: str                      # "run" | "finish" | "none"
    argv: list[str] = None
    answer: str = ""
    why: str = ""
    thought: str = ""
    source: str = "text"           # "tool_call" | "text" | "none"


def from_tool_calls(choices: list[dict] | None) -> Decision:
    """Read a native tool call out of a chat completion."""
    if not choices:
        return Decision("none", source="none")
    msg = choices[0].get("message") or {}
    calls = msg.get("tool_calls") or []
    if not calls:
        return Decision("none", source="none")
    call = calls[0]
    fn = (call.get("function") or {})
    name = str(fn.get("name") or "")
    try:
        payload = json.loads(fn.get("arguments") or "{}")
    except Exception:
        payload = {}
    if name == FINISH_TOOL["function"]["name"]:
        return Decision("finish", answer=str(payload.get("answer") or ""),
                        source="tool_call")
    if name == SHELL_TOOL["function"]["name"]:
        argv = payload.get("argv")
        return Decision("run",
                        argv=[str(a) for a in argv] if isinstance(argv, list) else None,
                        why=str(payload.get("why") or ""),
                        source="tool_call")
    return Decision("none", source="none")


def from_text(text: str) -> Decision:
    """Read a JSON object out of plain content, for a model with no tool support."""
    from .llm import extract_json

    try:
        data = extract_json(text or "", expect="object")
    except Exception:
        return Decision("none", source="none")
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return Decision("none", source="none")

    thought = str(data.get("thought") or "")
    if data.get("answer") is not None:
        return Decision("finish", answer=str(data.get("answer") or ""),
                        thought=thought, source="text")
    argv = data.get("command")
    if argv is None:
        argv = (data.get("args") or {}).get("argv") if isinstance(
            data.get("args"), dict) else None
    if isinstance(argv, list) and argv:
        return Decision("run", argv=[str(a) for a in argv],
                        why=str(data.get("why") or ""), thought=thought, source="text")
    # The pre-rename vocabulary, still accepted so an older prompt is not fatal.
    action = str(data.get("action") or "")
    if action in ("finish", "answer", "done", "stop"):
        return Decision("finish", answer=str(data.get("answer") or
                                            (data.get("args") or {}).get("answer", "")),
                        thought=thought, source="text")
    if action:
        args = data.get("args") if isinstance(data.get("args"), dict) else {}
        argv = args.get("argv") or args.get("command")
        if isinstance(argv, list):
            return Decision("run", argv=[str(a) for a in argv], thought=thought,
                            source="text")
    return Decision("none", thought=thought, source="none")


import json  # noqa: E402  (used by from_tool_calls; imported late to keep the
# docstring-first reading order of this module)