"""The ReAct loop: think -> act -> observe -> verify -> repeat, or report honestly.

This is the loop that was missing. `--strategy react` raised NotImplementedError and
nothing could route to it, which is what you noticed: every query took the same path.

Why hand-rolled rather than `dspy.ReAct`
-----------------------------------------
`dspy.ReAct` hides the three things this system has to be able to see:

  * **budget accounting.** Every LLM call must be charged to `ctx` BEFORE it happens,
    or the budget is decoration. A framework loop charges it for you, somewhere you
    cannot assert about.
  * **per-step verification.** The requirement is that a bad observation loops back.
    A framework loop terminates on the model's own say-so; here termination is gated
    on a verdict, and `unsure` is explicitly not success.
  * **refusal as an observation.** A `denied` command is data for the next turn. A
    framework that treats a tool error as an exception would kill the loop on the
    first refusal, and a loop that dies on refusal cannot be steered away from it.

So the loop is explicit. It is ~150 lines and every step is logged.

The loop's contract
-------------------
* Bounded by `max_iterations` AND by the task budget. Whichever runs out first ends
  the task, and the task ends *honestly* -- `inconclusive`, never a guess.
* Never raises across the agent boundary. Every failure becomes a trajectory entry.
* One action per turn. Composability comes from `bash`-style argv, not from a model
  inventing control flow.
* A repeated action is refused. The notebook's dead-end list, but applied to the
  trajectory rather than to a module global -- so it works per task and survives the
  bug where one task inherited another's do-not-repeat list.

Verification
------------
After each observation the step is judged by the same verifier the planner uses:

  pass   -> the trajectory satisfies the goal; synthesise
  unsure -> NOT success. Loop again with the uncertainty stated.
  fail   -> loop again with the failure reason in the trajectory

`unsure` being distinct from `fail` is load-bearing. A rate-limited rubric that was
treated as a failure would tell the agent its correct work was wrong, which is how an
agent learns to stop trying things that work.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import llm
from .context import BudgetExhausted, TaskContext
from .executor import CommandTool, executor_manifest
from .tools import find as find_tool
from .tools import grep as grep_tool
from .tools import read as read_tool
from .tools import survey as survey_tool
from .tools.observation import Observation
from .trace import Verdict

__all__ = ["ReactLoop", "ReactResult", "run_react"]

SYSTEM = """You are a careful shell agent working inside a sandboxed read-only scope.

You answer questions about a filesystem by RUNNING COMMANDS and READING THEIR OUTPUT.
You cannot see files except through the output of commands you run.

Each turn, reply with exactly one JSON object and nothing else:

  {"thought": "...", "action": "run_command", "args": {"argv": ["ls", "-la"],
   "why": "list the current directory"}}

  {"thought": "...", "action": "finish", "args": {"answer": "..."}}

Actions:
  run_command  argv is a list of strings. NO SHELL: no pipes, no &&, no redirects,
               no glob expansion. Run one command at a time.
               Common ones that work: ls, cat, head, tail, wc, du, df, stat, file,
               find, grep, git (status/log/diff/show), tree, sort, uniq, cut.
               A RECURSIVE command (du, grep -r, find, ls -R) is refused unless it
               excludes the private-key directories. Copy this form:
                 du -sh --exclude=.ssh --exclude=.aws --exclude=.gnupg
                      --exclude=.kube --exclude=keyring --exclude=.docker
                      --exclude=.password-store --exclude=.git-credentials .
  finish       when the evidence answers the question.

Rules:
  - Look before you leap. If a command returned nothing useful, try a different one
    rather than repeating it.
  - NEVER invent a file path. Only refer to paths that appeared in command output.
  - If a command is refused, read the reason and adapt. Do not retry it unchanged.
  - When you have the evidence, finish. Do not keep running commands.
  - Your final answer must be supported by output you actually saw."""

ACTION_PROMPT = """{system}

GOAL: {goal}

{trajectory}

Turn {n} of {max_turns}. Remaining: {budget}.

Reply with ONE JSON object:"""


_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([KMGTP]i?B?|bytes?)?", re.I)


def _unsupported_numbers(answer: str, evidence: str) -> list[str]:
    """Numbers in the answer that appear nowhere in the evidence.

    The specific way this agent lies. Asked how much disk a directory used, it ran
    `ls -l`, saw `total 376` -- a block count, not a size -- and answered
    "approximately 37K". No amount of "be careful" in the prompt prevents that; it is
    what reading a number in the wrong column does.

    A rubric asked to bless an answer over a wall of `ls` output says PASS, because
    the number IS in the output. It is just not the number being asked for. So this
    check is mechanical: an answer that asserts a quantity no output contained is a
    fail regardless of the rubric's opinion.

    Deliberately not exhaustive. Units are compared loosely, because `du` human
    sizes and restatements are spelled differently and the agent should not be
    punished for restating one. False positives cost a retry; a false negative is the
    bug above.
    """
    if not answer:
        return []
    ev = evidence.replace(",", "")
    bad: list[str] = []
    for m in _NUM.finditer(answer):
        value, unit = m.group(1), (m.group(2) or "").lower().rstrip("b")
        # A bare small integer is usually conversational ("the 3 files I saw"),
        # not a measurement. Checking those produces noise.
        if not unit and "." not in value and len(value) <= 2:
            continue
        if _present_in(value, ev):
            continue
        if unit and _present_in(f"{value}{unit}", ev.replace(" ", "").lower()):
            continue
        bad.append(f"{value}{unit}" if unit else value)
    return bad[:4]


def _present_in(needle: str, haystack: str) -> bool:
    """Is `needle` in `haystack` as a WHOLE TOKEN, not as a substring.

    Substring matching is the trap: `total 376` contains "37", so the check passed
    on exactly the confabulation it was written to catch. Same failure mode as the
    approval fingerprint that joined argv with spaces -- a value made of digits or
    words needs a delimiter around it, or "a b" matches "ab c".
    """
    if not needle:
        return False
    trimmed = needle.rstrip("0").rstrip(".")
    for cand in {needle, trimmed} - {""}:
        if re.search(rf"(?<![\w.]){re.escape(cand)}(?![\w])", haystack):
            return True
    return False


@dataclass
class Step:
    """One turn. Kept whole so a failure can be explained after the fact."""

    n: int
    thought: str
    action: str
    args: dict[str, Any]
    observation: Observation | None = None
    verdict: Verdict = Verdict.unsure
    reason: str = ""

    def render(self, max_obs: int = 700) -> str:
        obs = ""
        if self.observation is not None:
            body = (self.observation.text or self.observation.reason or "")
            obs = body[:max_obs] + (" ...[truncated]" if len(body) > max_obs else "")
        return (f"[turn {self.n}] {self.action}({json.dumps(self.args)[:160]})\n"
                f"  thought: {self.thought[:200]}\n"
                f"  observed: {obs}\n"
                f"  verdict: {self.verdict.value} -- {self.reason[:140]}")


@dataclass
class ReactResult:
    answer: str | None
    status: str                    # answered | inconclusive | budget | refused
    steps: list[Step] = field(default_factory=list)
    answer_source: str = ""        # which action produced the final answer

    @property
    def turns(self) -> int:
        return len(self.steps)

    def render(self) -> str:
        out = [f"REACT: {self.status} after {self.turns} turn(s)"]
        for s in self.steps:
            out.append("  " + s.render().replace("\n", "\n  "))
        if self.answer:
            out.append(f"\nANSWER (from {self.answer_source}):\n{self.answer}")
        return "\n".join(out)


class ReactLoop:
    def __init__(self, ctx: TaskContext, *, max_turns: int | None = None,
                 use_llm_verify: bool = True,
                 llm_call: Callable[..., dict] | None = None) -> None:
        self.ctx = ctx
        self.max_turns = max_turns or max(4, min(12, ctx.budget.max_iterations))
        self.use_llm_verify = use_llm_verify
        self._call = llm_call or llm.call
        # The shell tool, and the reading tools. All go through the same jail; only
        # run_command reaches a subprocess.
        self.shell = CommandTool(ctx=ctx, manifest=executor_manifest(ctx.settings))
        # Per-loop, not per-module: the notebook's dead-end list was a module global,
        # so a second task in the same kernel inherited the first task's
        # do-not-repeat list and silently rejected steps it had never tried.
        self._signatures: set[str] = set()
        self._goal = ""
        self.finish_rejections = 0

    # ------------------------------------------------------------------ dispatch
    def _dispatch(self, action: str, args: dict[str, Any]) -> Observation:
        """Run one action. Never raises -- a crash becomes an observation.

        This is the tool surface the model sees. It is deliberately SMALL and it is
        deliberately the same set the policy knows about, because a loop that can
        reach tools the gateway does not model is a loop with an unaudited surface.
        """
        if action in ("run_command", "shell", "exec", "bash"):
            argv = args.get("argv") or args.get("command") or []
            obs = self.shell(list(argv), why=str(args.get("why", "react")),
                             expected_exit=int(args.get("expected_exit", 0) or 0))
            return obs
        if action in ("find", "search"):
            return find_tool.find_files(str(args.get("pattern", "")),
                                        ctx=self.ctx,
                                        root=args.get("root") or self.ctx.root)
        if action in ("grep", "search_content"):
            return grep_tool.search_content(str(args.get("query", "")),
                                            ctx=self.ctx,
                                            root=args.get("root") or self.ctx.root)
        if action in ("read", "read_file"):
            return read_tool.read_file(str(args.get("path", "")),
                                       ctx=self.ctx)
        if action == "survey":
            return survey_tool.survey_files(ctx=self.ctx,
                                            root=args.get("root") or self.ctx.root)
        return Observation(tool=action, status="error", items=[],
                           text=f"unknown action {action!r}; use run_command or finish",
                           reason=f"unknown action {action!r}")

    # ------------------------------------------------------------------ one turn
    def _decide(self, goal: str, trajectory: str, n: int) -> tuple[dict[str, Any], str]:
        """Ask the model for the next action. Returns (parsed, raw)."""
        msgs = [{"role": "system", "content": ACTION_PROMPT.format(
            system=SYSTEM, goal=goal, trajectory=trajectory or "(nothing yet)",
            n=n, max_turns=self.max_turns, budget=self._budget_line())}]
        out = self._call(self.ctx, msgs, purpose="react_decide", temperature=0.0)
        raw = out.get("text", "")
        try:
            # `expect="object"`, and this is not a detail: the prompt asks for ONE
            # object, so the array-first extractor either raises or returns the
            # first *array* nested inside the reply. With `expect="array"` a
            # perfectly good decision came back as `{}` -- the loop then saw "no
            # action", logged a parse error, and burned a turn on a model that had
            # answered correctly.
            data = llm.extract_json(raw, expect="object")
        except Exception:
            return {}, raw
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            return {}, raw
        return data, raw

    def _budget_line(self) -> str:
        b = self.ctx.budget
        c = self.ctx.counters
        return (f"{b.max_llm_calls - c.llm_calls} llm calls, "
                f"{b.max_tool_calls - c.tool_calls} tool calls, "
                f"{b.max_iterations - c.iterations} iterations, "
                f"{b.wall_clock_s - self.ctx.elapsed_s():.0f}s left")

    def _verify(self, step: Step, goal: str) -> tuple[Verdict, str]:
        """Judge one observation. `unsure` is not success and not failure."""
        obs = step.observation
        if obs is None:
            return Verdict.fail, "no observation"
        if obs.status == "blocked":
            # A refusal is not a failed attempt at the goal, it is information the
            # model should act on. `fail` puts it in the trajectory as a mistake to
            # avoid repeating, which is exactly right.
            return Verdict.fail, f"refused: {obs.reason[:120]}"
        if obs.status == "error":
            return Verdict.fail, f"error: {obs.reason[:120]}"
        if not (obs.text or obs.items):
            return Verdict.unsure, "no output to judge"

        if not self.use_llm_verify:
            return Verdict.unsure, "verification skipped; treating output as sufficient"

        # Ask the model whether THIS observation bears on the goal. A cheap rubric
        # rather than the planner's full Signature, because the question is narrow.
        msgs = [
            {"role": "system", "content":
             "You judge whether a command's output advances a goal. Reply with one "
             "word: PASS, UNSURE, or FAIL, then a colon and a reason of under 15 words.\n"
             "PASS = the output answers the goal.\n"
             "UNSURE = related but not yet decisive.\n"
             "FAIL = irrelevant, empty, or an error."},
            {"role": "user", "content":
             f"GOAL: {goal}\n\nCOMMAND: {json.dumps(step.args.get('argv', []))[:200]}\n"
             f"OUTPUT:\n{(obs.text or '')[:1500]}"},
        ]
        try:
            out = self._call(self.ctx, msgs, purpose="react_verify", temperature=0.0)
            txt = (out.get("text") or "").strip()
            head = txt.split(":", 1)[0].strip().upper()
            rest = txt.split(":", 1)[1].strip() if ":" in txt else txt[:80]
            verdict = {"PASS": Verdict.pass_, "UNSURE": Verdict.unsure,
                       "FAIL": Verdict.fail}.get(head, Verdict.unsure)
            return verdict, rest[:140]
        except BudgetExhausted:
            raise
        except Exception as e:
            # The verifier NEVER raises. An unproven observation is `unsure`, and
            # `unsure` loops -- so a flaky rubric costs iterations, not correctness.
            return Verdict.unsure, f"verifier degraded: {type(e).__name__}"

    def _verify_finish(self, answer: str, steps: list[Step]) -> tuple[Verdict, str]:
        """Is the ANSWER supported by what the commands actually returned?

        Two structural checks first, because they catch the common failure for free
        and without a model call:

          * **no successful observation at all** -- finishing after only refusals and
            errors means the answer is invented, and no rubric should be asked to
            bless it
          * **a number in the answer that appears nowhere in the evidence** -- this is
            the `total 376` -> "37K" class of confabulation. Cheap, and it catches
            the specific way an LLM misreads command output.

        Then the rubric, which is the only check that can understand prose.
        """
        good = [(s.observation.text or s.observation.reason or "")
               for s in steps
               if s.observation is not None and s.observation.status == "ok"
               and (s.observation.text or s.observation.reason)]
        if not good:
            return Verdict.fail, ("no successful command output backs this answer; "
                                  "run something that works first")

        if not self.use_llm_verify:
            return Verdict.unsure, "verification skipped"

        unsupported = _unsupported_numbers(answer, "\n".join(good))
        try:
            out = self._call(self.ctx, [
                {"role": "system", "content":
                 "You check whether an answer is supported by command output. Reply "
                 "PASS, UNSURE or FAIL, then a colon and a reason under 15 words.\n"
                 "PASS = every claim in the answer appears in the output.\n"
                 "UNSURE = partly supported.\n"
                 "FAIL = claims something the output does not show, or misreads a "
                 "number, or the output does not answer the question at all."},
                {"role": "user", "content":
                 f"QUESTION: {self._goal}\n\nANSWER:\n{answer[:1200]}\n\n"
                 f"EVIDENCE:\n" + "\n".join(e[:900] for e in good[-6:])[:6000]}],
                purpose="react_check_answer", temperature=0.0)
            txt = (out.get("text") or "").strip()
            head = txt.split(":", 1)[0].strip().upper()
            why = txt.split(":", 1)[1].strip() if ":" in txt else txt[:80]
            verdict = {"PASS": Verdict.pass_, "UNSURE": Verdict.unsure,
                       "FAIL": Verdict.fail}.get(head, Verdict.unsure)
            # A structural miss is decisive even when the rubric is feeling kind.
            # The rubric reading a paragraph of `ls` output and saying PASS is
            # exactly the failure that produced "approximately 37K".
            if unsupported and verdict is Verdict.pass_:
                return Verdict.fail, (f"answer states {unsupported} which appears in "
                                      f"no command output")
            return verdict, why[:140]
        except BudgetExhausted:
            raise
        except Exception as e:
            return Verdict.unsure, f"verifier degraded: {type(e).__name__}"

    # ------------------------------------------------------------------ main
    def run(self, goal: str) -> ReactResult:
        self._goal = goal
        steps: list[Step] = []
        trajectory: list[Step] = []
        budget_hit = False

        for n in range(1, self.max_turns + 1):
            # Budget before the request, so a call that cannot be paid for never
            # starts. Without this the loop is unbounded by construction.
            try:
                self.ctx.charge_iteration()
            except BudgetExhausted as e:
                self.ctx.note(f"react stopped before turn {n}: {e}")
                return ReactResult(None, "budget", steps)

            traj_text = "\n\n".join(s.render() for s in trajectory[-6:])
            try:
                data, raw = self._decide(goal, traj_text, n)
            except BudgetExhausted as e:
                self.ctx.note(f"react budget exhausted at turn {n}: {e}")
                return ReactResult(None, "budget", steps)
            except Exception as e:
                self.ctx.log("react", "decide", detail=f"{type(e).__name__}",
                             status="error")
                return ReactResult(None, "inconclusive", steps,
                                   answer_source="no valid decision")

            action = str(data.get("action") or "").strip()
            args = data.get("args") if isinstance(data.get("args"), dict) else {}
            thought = str(data.get("thought") or "")[:600]

            # No action, or an unparseable reply. Do not burn a turn silently --
            # say so and retry, because a malformed reply is usually a formatting
            # slip rather than a dead end.
            if not action:
                self.ctx.log("react", "parse", detail=raw[:120], status="error")
                s = Step(n, thought, "(none)", {},
                         Observation(tool="-", status="error",
                                     reason="no action in reply"),
                         Verdict.fail, "no action in reply")
                steps.append(s)
                trajectory.append(s)
                continue

            # finish
            if action in ("finish", "answer", "done", "stop"):
                ans = str(args.get("answer") or data.get("answer") or "").strip()
                if not ans:
                    s = Step(n, thought, action, args,
                             Observation(tool="-", status="error",
                                         reason="finish with no answer"),
                             Verdict.fail, "finish with no answer")
                    steps.append(s)
                    trajectory.append(s)
                    continue
                # The finish is VERIFIED like any other step. This is the whole
                # point of the loop and the first version skipped it: `finish`
                # returned whatever the model said, unverified. Asked how much disk
                # space a directory used, the model ran `ls -l`, misread the block
                # count in `total 376` as a size, and answered "approximately 37K"
                # -- confidently, with no evidence behind it. A model that can end
                # the loop by asserting is a model with no loop.
                verdict, why = self._verify_finish(ans, steps)
                if verdict is Verdict.fail:
                    self.ctx.log("react", "finish", detail=f"rejected: {why[:80]}",
                                 status="blocked", turn=n)
                    s = Step(
                        n, thought, action, args,
                        Observation(tool=action, status="error",
                                    reason=f"answer rejected: {why}"),
                        Verdict.fail, f"finish rejected: {why}")
                    steps.append(s)
                    trajectory.append(s)
                    continue
                if verdict is Verdict.unsure and self.finish_rejections < 2:
                    # An UNPROVEN answer is not an answer. Let the model gather more
                    # evidence -- up to a point. Two rejections is the limit, because
                    # a verifier that keeps saying "unsure" would otherwise loop until
                    # the budget runs out, which costs money and returns nothing.
                    # `unsure` is deliberately not treated as failure elsewhere: an
                    # unproven *step* is fine to retry, but an unproven *conclusion*
                    # is the thing the user is being asked to trust.
                    self.finish_rejections += 1
                    self.ctx.log("react", "finish", detail=f"unproven: {why[:70]}",
                                 status="uncertain", turn=n)
                    s = Step(
                        n, thought, action, args,
                        Observation(tool=action, status="error",
                                    reason=f"answer not verified: {why}. Gather more "
                                           f"evidence, or say plainly what is missing."),
                        Verdict.unsure, f"finish unproven: {why}")
                    steps.append(s)
                    trajectory.append(s)
                    continue
                steps.append(Step(n, thought, action, args, None, verdict, why))
                self.ctx.log("react", "finish", detail=ans[:80],
                             status="ok" if verdict is Verdict.pass_ else "uncertain",
                             turn=n)
                # An answer that came back `unsure` is returned, but the status says
                # so. A caller that wants only proven answers can check `status`.
                return ReactResult(ans,
                                   "answered" if verdict is Verdict.pass_
                                   else "answered_unverified",
                                   steps, answer_source=action)

            # Repeat rejection. Per task, from the trajectory -- so it cannot leak
            # across tasks the way the notebook's module-level PATH_LOG did.
            sig = f"{action}:{json.dumps(args.get('argv', args), sort_keys=True)}"
            if sig in self._signatures:
                self.ctx.log("react", "repeat", detail=sig[:80], status="blocked")
                s = Step(
                    n, thought, action, args,
                    Observation(tool=action, status="blocked",
                                reason="already tried this exact action"),
                    Verdict.fail, "repeated action")
                # Into `steps` as well as the trajectory. `steps` is the record the
                # caller gets back and the thing `render()` prints, so a turn that
                # only reached the trajectory is a turn that did not happen as far as
                # anyone can see -- which is how a loop silently burning turns on
                # repeats becomes invisible.
                steps.append(s)
                trajectory.append(s)
                continue

            try:
                self.ctx.charge_tool(action)
                obs = self._dispatch(action, args)
            except BudgetExhausted as e:
                self.ctx.note(f"react budget exhausted at turn {n}: {e}")
                return ReactResult(None, "budget", steps)
            except Exception as e:
                # CONTRACT rule 5. A tool that crashes must not kill the loop.
                obs = Observation(tool=action, status="error", items=[],
                                  reason=f"{type(e).__name__}: {str(e)[:120]}")

            self._signatures.add(sig)
            step = Step(n, thought, action, args, obs)
            verdict, reason = self._verify(step, goal)
            step.verdict, step.reason = verdict, reason
            steps.append(step)
            trajectory.append(step)

            self.ctx.log("react", "step",
                         detail=f"{action} -> {obs.status} ({verdict.value})",
                         status=obs.status, turn=n, verdict=verdict.value,
                         duration_ms=obs.duration_ms)

            # Feed only FILE CONTENTS into the taint set, never a directory listing.
            # A filename in an `ls` listing is not an injection -- it is a name the
            # agent legitimately read from the filesystem, and tainting it made the
            # agent unable to read anything it had just listed. File contents are the
            # actual vector: text inside a file can be an instruction aimed at the
            # model. `policy.classify` then decides whether the tainted command's tier
            # makes that matter.
            if obs.text and obs.tool in ("read", "read_file"):
                self.shell.note_read_text(obs.text)

            if verdict is Verdict.pass_ and obs.status == "ok":
                # The rubric says this observation answers the goal. Ask for the
                # synthesis rather than treating raw output as the answer -- raw
                # `ls` output is evidence, not a sentence.
                return self._synthesise(goal, steps)

        return ReactResult(None, "inconclusive", steps)

    def _synthesise(self, goal: str, steps: list[Step]) -> ReactResult:
        """Turn a passing trajectory into an answer, or report inconclusive."""
        evidence = "\n\n".join(
            f"$ {' '.join(s.args.get('argv', []))}\n{(s.observation.text or '')[:1200]}"
            for s in steps if s.observation is not None and s.verdict is Verdict.pass_)
        try:
            out = self._call(self.ctx, [
                {"role": "system", "content":
                 "Answer the question using ONLY the command output below. If the "
                 "output does not contain the answer, say so plainly and name what is "
                 "missing. Never add facts that are not in the output. Be brief."},
                {"role": "user", "content":
                 f"QUESTION: {goal}\n\nEVIDENCE:\n{evidence[:8000]}"}],
                purpose="react_answer", temperature=0.0)
            return ReactResult(out.get("text", "").strip(), "answered", steps,
                               answer_source="verified observation")
        except BudgetExhausted:
            return ReactResult(None, "budget", steps)
        except Exception as e:
            self.ctx.log("react", "answer", detail=type(e).__name__, status="error")
            return ReactResult(None, "inconclusive", steps)


def run_react(goal: str, *, ctx: TaskContext, max_turns: int | None = None,
              use_llm_verify: bool = True) -> ReactResult:
    return ReactLoop(ctx, max_turns=max_turns,
                     use_llm_verify=use_llm_verify).run(goal)