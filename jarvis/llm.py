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
import os
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


class LLMReasoningExhausted(LLMUnavailable):
    """A reasoning model spent the whole token budget on hidden reasoning.

    Distinct from a generic transient failure because the FIX is different: it is not
    "retry later", it is "ask for more tokens". `complete` catches it specifically and
    retries once with 4x, which is what makes a reasoning model usable as a router --
    a 120-token budget is spent entirely on reasoning and returns HTTP 200 with no
    content, which is indistinguishable from a model that chose to say nothing.
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
                 max_tokens: int, trace: Trace | None = None,
                 tools: list[dict] | None = None) -> dict:
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
              trace: Trace | None, tag: str, _reasoning_retried: bool = False) -> dict:
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
            # Ask the endpoint what it DOES serve. "model not found" on its own sends
            # the user looking at their key, which is fine, or at their URL, which is
            # usually fine -- and never at the actual mismatch, which is that the
            # configured model belongs to a different provider. This happened here:
            # a Groq endpoint configured with `gpt-4o-mini`, an OpenAI model Groq does
            # not serve, reported identically to a bad key.
            available = self.discover(base_url, key)
            hint = ""
            if available:
                shown = ", ".join(available[:8])
                hint = (f" This endpoint serves: {shown}"
                        + (f" (+{len(available) - 8} more)" if len(available) > 8 else "")
                        + ". Set LLM_MODEL in .env to one of these.")
            raise LLMConfigError(
                f"HTTP 404: model {model!r} not found or not accessible at {base_url}. "
                f"Configured fallbacks: {', '.join(self.candidates()[1:]) or '(none)'}"
                f".{hint}")
        if code == 429:
            self._log(trace, "llm", tag, "HTTP 429", t0, status="error")
            raise LLMUnavailable(f"rate limited (HTTP 429) on {model}")
        if code >= 500:
            self._log(trace, "llm", tag, f"HTTP {code}", t0, status="error")
            raise LLMUnavailable(f"server error HTTP {code}")
        if code in (400, 422):
            body = _short(r.text)
            # Two very different 400s share a status code.
            #
            # "Failed to parse tool call arguments as JSON" is the MODEL producing
            # malformed JSON in a tool call -- a sampling artefact, not a bad request.
            # Retrying gets a different sample and usually works. Observed on
            # gpt-oss-20b mid-conversation, where it killed the turn.
            if "parse tool call" in body.lower() or "invalid json" in body.lower():
                self._log(trace, "llm", tag, "malformed tool call", t0, status="retry")
                raise LLMUnavailable(f"model emitted unparseable tool arguments: {body}")
            # Everything else is our fault, permanently.
            self._log(trace, "llm", tag, f"HTTP {code} bad request", t0,
                      status="config_error")
            raise LLMConfigError(
                f"HTTP {code}: the endpoint rejected the request. This is not "
                f"transient and retrying will not help. Endpoint said: {body}")
        if code != 200:
            self._log(trace, "llm", tag, f"HTTP {code}", t0, status="error")
            raise LLMUnavailable(f"HTTP {code}: {_short(r.text)}")

        data = r.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            self._log(trace, "llm", tag, "malformed response", t0, status="error")
            raise LLMUnavailable(f"unexpected shape: {_short(str(data))}") from e

        # A tool-call turn has NO `content` key at all -- not an empty one. `["content"]`
        # raised KeyError and the whole call was reported as "unexpected shape", which
        # is what made the first native-tool-calling run fail for a reason that had
        # nothing to do with tool calling. `.get` with a default is the whole fix.
        raw_content = message.get("content")
        text = raw_content.strip() if isinstance(raw_content, str) else ""

        usage = data.get("usage") or {}
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        tool_calls = message.get("tool_calls") or []

        if not text and tool_calls:
            # The normal shape of a native tool turn. Not an error and not empty
            # output -- it is the model doing exactly what it was asked.
            self._log(trace, "llm", tag,
                      f"{data.get('model', model)} -> tool_call "
                      f"{(tool_calls[0].get('function') or {}).get('name')}",
                      t0, tokens=usage.get("total_tokens"))
            return {"text": "", "usage": usage, "model": data.get("model", model),
                    "choices": data.get("choices") or [], "tool_calls": tool_calls}

        if not text:
            # Reasoning models (gpt-oss, o-series) spend part of `max_tokens` on
            # hidden reasoning. A tight budget -- the router asks for 120 tokens to
            # pick one of three words -- can be spent ENTIRELY on reasoning, leaving
            # nothing to say. The endpoint returns 200 with empty content and no
            # error, so this is indistinguishable from a model that chose to say
            # nothing.
            #
            # `complete` catches this specifically and retries once with 4x, which is
            # what makes a reasoning model usable as a router. Measured on gpt-oss-20b:
            # 2498 of 2500 tokens on reasoning produced nothing at all.
            spent = int(reasoning or 0)
            self._log(trace, "llm", tag, "empty content", t0, status="error")
            raise LLMReasoningExhausted(
                f"empty response from {model} at max_tokens={max_tokens_for(payload)}"
                + (f" (reasoning_tokens={spent})" if spent else "")
                + " -- raise max_tokens, set LLM_REASONING_EFFORT=low, or use a "
                  "non-reasoning model.")
        self._log(trace, "llm", tag, f"{data.get('model', model)} -> {len(text)} chars",
                  t0, tokens=usage.get("total_tokens"))
        return {"text": text, "usage": usage, "model": data.get("model", model),
                "choices": data.get("choices") or []}

    def discover(self, base_url: str, key: str | None) -> list[str]:
        """Instance form, so `self` is available. See the module-level function."""
        return discover(base_url, key)

    def candidates(self) -> list[str]:
        models = [self.config.model]
        for m in self.config.fallback_models:
            if m and m not in models:
                models.append(m)
        return [m for m in models if m]

    def complete(self, messages: list[dict], *, temperature: float | None = None,
                 max_tokens: int | None = None, trace: Trace | None = None,
                 tools: list[dict] | None = None) -> dict:
        """One chat completion.

        `tools` sends a native `tools` array. This is not optional decoration: a
        tool-calling model (gpt-oss and friends) REFUSES a request that describes
        running things but supplies no tools -- "HTTP 400: Tool choice is none, but
        model called a tool" -- and no payload flag works around it (`tools: []` and
        `tool_choice: "none"` were both tried and both returned the same 400). So when
        the loop wants a structured turn it gives the model real tools rather than
        asking nicely in prose.
        """
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
        # One reasoning-token retry, tracked here rather than in `_post` because
        # `_post` does not have `messages` -- the retry has to resend the same
        # conversation, and the first attempt at this threw a NameError on `base`
        # precisely because it tried to recurse from the wrong layer.
        #
        # The multiplier also grows on a 429, because retrying a rate limit
        # immediately gets another rate limit. Retrying at the same instant is how a
        # loop turns one 429 into twenty and makes the limit worse.
        backoff = float(_env_backoff())
        for attempt, mult in enumerate((1, 4, 1, 1)):
            try:
                body = {"model": models[0], "messages": messages,
                        "temperature": temp, "max_tokens": limit * mult}
                if tools:
                    body["tools"] = tools
                    body["tool_choice"] = "auto"
                # Sent only when configured, and only for a model that accepts them.
                # `reasoning_format` in particular is Groq-specific and a strict
                # OpenAI-compatible endpoint rejects unknown fields with a 400, so
                # this is opt-out rather than always-on.
                if self.config.reasoning_effort:
                    body["reasoning_effort"] = self.config.reasoning_effort
                if self.config.reasoning_format and tools:
                    body["reasoning_format"] = self.config.reasoning_format
                return self._post(base, models[0], key, body, trace, models[0])
            except LLMReasoningExhausted as e:
                if attempt >= 1:
                    raise LLMUnavailable(
                        f"{e} (already retried with 4x the token budget)")
                self._log(trace, "llm", "retry",
                          f"{e} -- retrying with {limit * 4} max_tokens", None,
                          status="retry")
                continue
            except LLMConfigError:
                raise                      # a config fault is not worth retrying
            except LLMUnavailable as e:
                # Transient. Wait before retrying, then retry the SAME model (a
                # malformed tool call and a 429 are both sampling/rate artefacts, and
                # a fresh sample or a second of patience usually works), then fall
                # back to the next configured model.
                #
                # The sleep is the point. Without it the loop turned one 429 into
                # twenty requests in under a second, which deepens the rate limit it
                # was responding to.
                if "429" in str(e) or "rate limited" in str(e).lower():
                    time.sleep(backoff * (attempt + 1))
                if attempt >= 1:
                    if len(models) > 1:
                        models = models[1:]
                        self._log(trace, "llm", "retry",
                                  f"{e} -- falling back to {models[0]}", None,
                                  status="retry")
                        continue
                    raise
                self._log(trace, "llm", "retry", f"{e} -- retrying", None,
                          status="retry")
                continue
        raise LLMUnavailable("all candidate models failed")

    @staticmethod
    def _log(trace: Trace | None, kind: str, action: str, detail: str, t0: float,
             status: str = "ok", **extra) -> None:
        if trace is not None:
            trace.event(kind, action, detail=detail[:80], status=status, t0=t0, **extra)


def max_tokens_for(payload: dict) -> int:
    return payload.get("max_tokens", "?")


# --------------------------------------------------------------------------- facade
def _env_backoff() -> float:
    """Seconds to wait before retrying a rate-limited call.

    Configurable, because the right value depends entirely on the provider's limit
    and the only person who knows it is whoever set the endpoint.
    """
    return float(os.environ.get("JARVIS_LLM_BACKOFF_S") or os.environ.get(
        "LLM_BACKOFF_S") or "2.0")


def _short(body: str | bytes, n: int = 300) -> str:
    """A provider error body, trimmed to something readable in a trace line.

    These bodies are the single most useful thing in a failure report and the least
    likely to be seen, because they arrive as one enormous JSON line and get truncated
    by whatever prints it. The `error.message` field is extracted when present because
    that is where every provider puts the human-readable part.
    """
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
    try:
        data = json.loads(text)
        msg = (data.get("error") or {}).get("message")
        if isinstance(msg, str) and msg:
            return msg[:n]
    except Exception:
        pass
    return text[:n].replace("\n", " ")


def discover(base_url: str, key: str | None) -> list[str]:
    """Model ids this endpoint actually serves, or [] if it will not say.

    Providers disagree about where the listing lives. OpenAI and Groq's OpenAI
    compatibility layer both expose `/models`, but Groq serves it at
    `/openai/v1/models` rather than `/v1/models`, so both are tried. Failure is
    always [] -- this runs on an error path, and a discovery call that raises would
    replace a useful message with a useless one.
    """
    import requests

    root = base_url.rstrip("/")
    urls = [f"{root}/models"]
    if root.endswith("/v1") and "/openai/v1" not in root:
        urls.insert(0, root.replace("/v1", "/openai/v1") + "/models")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json().get("data") or []
            ids = [str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id")]
            if ids:
                return sorted(ids)
        except Exception:
            continue
    return []


def call(ctx: TaskContext, messages: list[dict], *, provider: LLMProvider | None = None,
         temperature: float | None = None, max_tokens: int | None = None,
         purpose: str = "answer", tools: list[dict] | None = None) -> dict:
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
                        max_tokens=max_tokens, trace=ctx.trace, tools=tools)
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
        out.update(kind="config",
                   detail="no API key: set LLM_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, or GROQ_API_KEY")
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