"""The LLM seam: one interface, several providers, and an explicit egress policy.

Why an interface
----------------
Whole-machine reads mean paths and file snippets LEAVE the machine. That is the
single unresolved risk in the design, so it is expressed here as a decision the
caller makes (`EgressPolicy`) rather than something buried in a request builder.

Why not DSPy everywhere
-----------------------
DSPy gives typed Pydantic outputs and a nice optimiser, but it hides the HTTP call,
so you cannot: count tokens, fall back on a 429, or refuse to send a path. So DSPy
is used only where its typed signature actually pays (the planner's structured JSON
and the verifier's verdict), and this client is used everywhere else.

The distinction the notebook got wrong
--------------------------------------
`LLMConfigError` (bad key, unknown model -> the wiring is broken, report it loudly)
vs `LLMUnavailable` (timeout, 429, 5xx -> degrade quietly). Conflating them made a
404 model report as "no key", which sent the user looking in the wrong place.

Model fallback
--------------
Groq's daily token limit is per-MODEL, so a fallback to a different model is the
whole point, but a fallback to a different model can hit a different bucket that is
also exhausted. `call()` therefore tries each candidate once, in order, and reports
which ones it burned.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import LLMConfig, load_settings
from .context import TaskContext
from .trace import Trace


class LLMUnavailable(RuntimeError):
    """Transient: timeout, 429, 5xx, connection reset. Safe to degrade."""


class LLMConfigError(RuntimeError):
    """The wiring is wrong: no key, bad key, unknown model, no base_url.

    Must be reported loudly. Retrying it wastes the budget and hides the real fault.
    """


class EgressDenied(RuntimeError):
    """A prompt was refused by the egress policy before anything left the machine."""


# --------------------------------------------------------------------------- egress
@dataclass(frozen=True)
class EgressPolicy:
    """What may leave the machine. The default is the CONSERVATIVE answer.

    This is the unresolved design question, made explicit and testable:

      * `allow_paths`  -- path prefixes that may appear in a prompt at all
      * `allow_snippets` -- whether file CONTENT may be sent, or only metadata
      * `remote_base_urls` -- hosts permitted to receive anything

    `metadata_only=True` means the agent can learn "there are 3 files over 100MB"
    without their names ever leaving. It makes the agent much weaker, which is the
    point: it is the setting to run in while the decision is open.
    """

    allow_paths: tuple[str, ...] = ()
    allow_snippets: bool = True
    remote_base_urls: tuple[str, ...] = ()
    metadata_only: bool = False

    def check_url(self, base_url: str | None) -> None:
        if not self.remote_base_urls:
            return
        if not base_url or not any(base_url.startswith(u) for u in self.remote_base_urls):
            raise EgressDenied(f"base_url {base_url!r} is not in the egress allow-list")

    def check_text(self, text: str) -> None:
        if not self.remote_base_urls:
            return
        if self.metadata_only:
            # Strip anything that looks like an absolute path before it can leave.
            scrubbed = _PATH_RE.sub("<path>", text)
            if scrubbed != text:
                raise EgressDenied(
                    "egress policy is metadata_only but the prompt contains a path; "
                    "refusing to send. Use a local model, or allow paths explicitly.")

    def describe(self) -> str:
        return (f"metadata_only={self.metadata_only}  "
                f"snippets={self.allow_snippets}  "
                f"remote={list(self.remote_base_urls) or 'ANY (unset)'}")


# An absolute path, used to detect whether a metadata-only prompt would leak one.
_PATH_RE = re.compile(r"(?<![\w~])/(?:[\w.\-]+/){1,}[\w.\-]*")


# --------------------------------------------------------------------------- provider
class LLMProvider(Protocol):
    """What the engine needs from a model. Anything implementing this works."""

    name: str

    def complete(self, messages: list[dict], *, temperature: float,
                 max_tokens: int, trace: Trace | None = None) -> dict:
        """-> {"text": str, "usage": dict, "model": str}. Raises the two errors above."""
        ...


# --------------------------------------------------------------------------- client
@dataclass
class HTTPProvider:
    """OpenAI-compatible /chat/completions. Covers Groq, OpenAI, OpenRouter, Ollama."""

    config: LLMConfig
    egress: EgressPolicy = field(default_factory=EgressPolicy)
    timeout_s: int = 60
    _session: Any = None

    name = "http"

    def _post(self, base_url: str, model: str, key: str, payload: dict,
              trace: Trace | None, tag: str) -> dict:
        # `limit` is read back out of the payload only to make the empty-content
        # error actionable: "raise max_tokens" is useless without the number.
        import requests

        if self._session is None:
            self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        t0 = time.perf_counter()
        try:
            r = self._session.post(f"{base_url.rstrip('/')}/chat/completions",
                                   headers=headers, json=payload, timeout=self.timeout_s)
        except requests.Timeout:
            self._log(trace, "llm", tag, "timeout", t0)
            raise LLMUnavailable(f"timeout after {self.timeout_s}s") from None
        except requests.RequestException as e:
            self._log(trace, "llm", tag, type(e).__name__, t0, status="error")
            raise LLMUnavailable(f"network error: {type(e).__name__}: {e}") from e

        code = r.status_code
        if code in (401, 403):
            self._log(trace, "llm", tag, f"HTTP {code} auth", t0, status="config_error")
            raise LLMConfigError(f"HTTP {code}: key rejected by {base_url}.")
        if code == 404:
            self._log(trace, "llm", tag, f"HTTP 404 {model}", t0, status="config_error")
            raise LLMConfigError(
                f"HTTP 404: model {model!r} not found or not accessible at {base_url}. "
                f"Candidates: {', '.join(self.candidates()) or '(none configured)'}")
        if code == 429:
            self._log(trace, "llm", tag, "HTTP 429", t0, status="error")
            raise LLMUnavailable(f"rate limited (HTTP 429) on {model}")
        if code >= 500:
            self._log(trace, "llm", tag, f"HTTP {code}", t0, status="error")
            raise LLMUnavailable(f"server error HTTP {code}")
        if code != 200:
            self._log(trace, "llm", tag, f"HTTP {code}", t0, status="error")
            raise LLMUnavailable(f"HTTP {code}: {r.text[:200]}")

        data = r.json()
        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            self._log(trace, "llm", tag, "malformed response", t0, status="error")
            raise LLMUnavailable(f"unexpected shape: {str(data)[:200]}") from e

        usage = data.get("usage") or {}
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if not text:
            # Reasoning models can burn the entire budget on hidden reasoning and
            # return empty content. That is a token-budget fault, not a wire fault.
            self._log(trace, "llm", tag, "empty content", t0, status="error")
            raise LLMUnavailable(
                f"empty response from {model} at max_tokens={max_tokens_for(payload)}"
                + (f" (reasoning_tokens={reasoning})" if reasoning else "")
                + " -- raise max_tokens, or use a non-reasoning model.")
        self._log(trace, "llm", tag, f"{data.get('model', model)} -> {len(text)} chars",
                  t0, tokens=usage.get("total_tokens"))
        return {"text": text, "usage": usage, "model": data.get("model", model)}

    def candidates(self) -> list[str]:
        models = [self.config.model]
        for m in self.config.fallback_models:
            if m and m not in models:
                models.append(m)
        return [m for m in models if m]

    def complete(self, messages: list[dict], *, temperature: float | None = None,
                 max_tokens: int | None = None, trace: Trace | None = None) -> dict:
        base = self.config.base_url
        self.egress.check_url(base)
        self.egress.check_text("\n".join(str(m.get("content", "")) for m in messages))
        key = self.config.api_key or ""
        if not base:
            raise LLMConfigError("no base_url: set LLM_BASE_URL")
        if not str(base).startswith("http"):
            raise LLMConfigError(
                f"LLM_BASE_URL is not a URL: {base!r}. "
                f'This must be caught at startup -- otherwise it surfaces as '
                f'"Invalid URL None/chat/completions", which reads like a network fault.')
        models = self.candidates()
        if not models:
            raise LLMConfigError("no model: set LLM_MODEL")

        limit = max_tokens or self.config.max_tokens
        temp = self.config.temperature if temperature is None else temperature
        transient: list[str] = []
        for model in models:
            try:
                return self._post(base, model, key,
                                  {"model": model, "messages": messages,
                                   "temperature": temp, "max_tokens": limit},
                                  trace, model)
            except LLMConfigError:
                raise                      # a config fault is not worth retrying
            except LLMUnavailable as e:
                transient.append(f"{model}: {e}")
        raise LLMUnavailable(
            "all candidate models failed -> " + " | ".join(transient))

    @staticmethod
    def _log(trace: Trace | None, kind: str, action: str, detail: str, t0: float,
             status: str = "ok", **extra) -> None:
        if trace is not None:
            trace.event(kind, action, detail=detail[:80], status=status, t0=t0, **extra)


def max_tokens_for(payload: dict) -> int:
    return payload.get("max_tokens", "?")


# --------------------------------------------------------------------------- facade
def call(ctx: TaskContext, messages: list[dict], *, provider: LLMProvider | None = None,
         temperature: float | None = None, max_tokens: int | None = None,
         purpose: str = "answer") -> dict:
    """One LLM call, charged to the task budget.

    Charges BEFORE the request so an exhausted budget cannot start a call it cannot
    pay for. A provider that returns usage is billed; one that does not is billed at
    a rough 4 chars/token estimate, because under-counting a budget is how a loop
    runs away.
    """
    prov = provider or HTTPProvider(config=ctx.settings.llm)
    ctx.charge_llm(0)
    out = prov.complete(messages, temperature=ctx.settings.llm.temperature
                        if temperature is None else temperature,
                        max_tokens=max_tokens, trace=ctx.trace)
    usage = out.get("usage") or {}
    tokens = int(usage.get("total_tokens") or estimate_tokens(out.get("text", "")))
    ctx.counters.tokens += tokens
    ctx.log("llm", purpose, detail=f"{out.get('model')} -> {len(out.get('text',''))} chars",
            tokens=tokens)
    return out


def estimate_tokens(text: str) -> int:
    """Rough, deliberately conservative. ~3.6 chars/token plus per-message overhead."""
    if not text:
        return 0
    return int(len(text) / 3.6) + 4


# --------------------------------------------------------------------------- preflight
def preflight(settings=None, *, verbose: bool = True) -> dict:
    """Separate 'wired wrong' from 'service down', before any task runs.

    The notebook discovered these misconfigurations only after an agent had already
    burned a walk. This runs a single `PONG` probe.
    """
    s = settings or load_settings()
    cfg = s.llm
    out: dict[str, Any] = {
        "base_url": cfg.base_url, "model": cfg.model,
        "key_configured": bool(cfg.api_key), "ok": False, "kind": "",
        "detail": "", "models": [cfg.model] + list(cfg.fallback_models),
    }
    if verbose:
        print(f"  base   : {cfg.base_url}")
        print(f"  model  : {cfg.model}  fallbacks={list(cfg.fallback_models)}")
        print(f"  key    : {'SET' if cfg.api_key else 'NOT SET'}")
        print(f"  egress : {EgressPolicy().describe()}")

    if not cfg.base_url or not str(cfg.base_url).startswith("http"):
        out.update(kind="config", detail=f"base_url is not a URL: {cfg.base_url!r}")
    elif not cfg.model:
        out.update(kind="config", detail="LLM_MODEL is empty")
    elif not cfg.api_key:
        out.update(kind="config", detail="no API key: set GROQ_API_KEY or LLM_API_KEY")
    else:
        prov = HTTPProvider(config=cfg, timeout_s=cfg.timeout_s)
        try:
            r = prov.complete([{"role": "user", "content": "Reply with exactly: PONG"}],
                              max_tokens=256)
            out.update(ok=True, kind="ok", detail=f"{r['model']} replied {r['text'][:20]!r}")
        except LLMConfigError as e:
            out.update(kind="config", detail=str(e))
        except LLMUnavailable as e:
            out.update(kind="transient", detail=str(e))
        except EgressDenied as e:
            out.update(kind="egress", detail=str(e))

    if verbose:
        label = {True: "PASS", False: "FAIL"}[out["ok"]]
        print(f"  RESULT : {label} ({out['kind'] or 'unknown'}) - {out['detail']}")
    return out


# --------------------------------------------------------------------------- JSON
def extract_json(text: str, *, expect: str = "array"):
    """Pull the first JSON array or object out of prose-wrapped model output.

    Hand-rolled rather than a dependency, because the failure it repairs is the
    predictable one: a model that was told "JSON only" emits a fenced block with
    one sentence around it. Brace counting is string-aware, so a `}` inside a JSON
    string does not end the object early.
    """
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t.strip())
    opener, closer = ("[", "]") if expect == "array" else ("{", "}")
    start = t.find(opener)
    if start == -1:
        raise ValueError(f"no JSON {expect} in output: {text[:120]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError as e:
                    raise ValueError(f"malformed JSON: {e}") from None
    raise ValueError("unterminated JSON")


def dspy_available() -> bool:
    try:
        import dspy  # noqa: F401
        from dspy import Signature  # noqa: F401
        return True
    except ImportError:
        return False


def configure_dspy(settings=None) -> bool:
    """Point DSPy at the same endpoint as the plain client.

    `provider="groq"` is REQUIRED for Groq: litellm otherwise strips `openai/` from
    `openai/gpt-oss-120b` and routes to api.openai.com, which 404s.

    `num_retries=0` matters: DSPy's internal retry storm turned ONE rubric call into
    536 seconds of silent backoff when the provider returned 429. Fail fast and let
    the orchestrator see the error and degrade visibly.
    """
    s = settings or load_settings()
    if not dspy_available():
        return False
    import dspy
    if not s.llm.api_key:
        return False
    model = s.llm.dspy_model or s.llm.model
    provider = "groq" if "groq.com" in (s.llm.base_url or "") else "openai"
    dspy.configure(lm=dspy.LM(f"openai/{model}", provider=provider,
                              api_base=s.llm.base_url, api_key=s.llm.api_key,
                              temperature=0.0, max_tokens=2000, num_retries=0))
    return True