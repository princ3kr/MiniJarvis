"""Frozen executor code: tiers 1 and 2, approvals, interpreters, taint escalation.

Not imported at import time, and that is the point.

The plan says to move this out of the import path until a sandbox exists, and the
reason is not tidiness. Every module in this package can be imported and exercised
without a single line of it being reachable in a running task. `policy.py` deciding
that `rm -rf` is tier 1 is a decision nobody makes, but it is a decision that LOOKS
like it could be made, and the next person to wire up a manifest at `max_tier=2` gets
writers, interpreters and network with `isolation: "none"` -- which the manifest
itself admits is not a sandbox.

So the import is the guard. `enabled()` is one comparison against
`Settings.executor_enabled_tier`, and the ceiling ships at 0, so:

  * `import jarvis.frozen.tiers` does nothing on its own -- the caller has to ask
  * `tiers.floor(...)` returns None while the ceiling is 0, so `policy.classify` skips
    the whole branch rather than computing a tier it will then refuse
  * reaching any of it requires editing `.env` (`EXECUTOR_ENABLED_TIER=2`), which is a
    deliberate, greppable, revertible act -- not a default

What is here, and why each is kept rather than deleted:

  * **tier floors.** `tar -cf out.tar /home` passes every flag check and is not a read.
    A grammar says which FLAGS are legal; it says nothing about what the binary DOES
    with them. This is the only place that answers the second question.
  * **taint escalation.** A file can say "run this": its content reaches the model and
    the model can echo it back as argv. Gated to tier >= 1 for a reason found the
    hard way -- escalating tier 0 made the loop unable to read the directory it had
    just listed.
  * **approvals.** Per-exact-argv, per-task, `\x00`-joined. The notebook used a module
    set with `" ".join`, which matched `["a b"]` against `["a", "b"]`.

None of this is reachable while the ceiling is 0, so none of it is load-bearing, and
that is why it can sit here untouched.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..policy import Tier


def enabled(settings: Any) -> bool:
    """Is ANY frozen tier reachable? One comparison against the ceiling.

    **Unknown settings mean ENABLED.** That is the whole subtlety here, and getting it
    backwards is a fail-open: with `settings=None` the first version read
    `getattr(None, "executor_enabled_tier", 0)`, got 0, concluded "frozen tiers are
    unreachable", and classified `touch` and `mkdir` as tier 0. The probe reported both
    as PERMISSIVE for exactly that reason.

    The asymmetry is deliberate and it is one-directional. Applying a tier floor can
    only ever make a command MORE restricted -- `classify` returns a higher tier, and
    the gateway refuses anything above the ceiling. So when policy lacks the settings
    it must assume the worst and the system is no less safe. The reverse -- assuming
    safety when information is missing -- is how a "read-only" agent ends up with a
    writer.

    The ceiling itself is still enforced separately by `gateway.authorize`, against
    `ctx.settings`. So `enabled(True)` here cannot cause a frozen command to run; it
    only means policy stops vouching for the command's safety.

    The ceiling is `Settings.executor_enabled_tier`, read per call, so raising it in
    `.env` is the only thing that turns this on and there is nothing to reset between
    tasks. The notebook read these from module globals at import time, which meant a
    test could not change them and a config change needed a restart.
    """
    if settings is None:
        return True                      # unknown -> assume reachable -> fail safe
    try:
        return int(getattr(settings, "executor_enabled_tier", 0)) >= 1
    except (TypeError, ValueError):
        return True


def approvals_enabled(settings: Any) -> bool:
    """Tier 2 is where approvals appear, so they appear later than tier 1.

    Same asymmetry as `enabled`: unknown means enabled. Approving on a guess is worse
    than demanding approval for a command that did not need it, because the failure
    mode is a command running that should not have.
    """
    if settings is None:
        return True
    try:
        return int(getattr(settings, "executor_enabled_tier", 0)) >= 2
    except (TypeError, ValueError):
        return True


def interpreters_enabled(settings: Any) -> bool:
    """Interpreters need a sandbox. `isolation: "none"` means there is not one, so
    they stay frozen regardless of the ceiling."""
    return False


__all__ = ["enabled", "approvals_enabled", "interpreters_enabled"]