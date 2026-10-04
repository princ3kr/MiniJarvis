"""Agent manifests: the single source of truth for what an agent may do.

A manifest is the ceiling. Policy decides a command's tier; the manifest caps it.
Both must agree for a command to run, which is why `classify` takes the manifest as
a required argument rather than reading a global.

Per the design, the discovery agent is `max_tier=0`: read-only filesystem tools, no
command execution, no network. That makes tier>=1 unreachable for an agent even if
someone flips `EXECUTOR_ENABLED_TIER` in the environment, because the manifest check
happens after the tier check and both must pass.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Limits(BaseModel):
    """Per-agent ceilings. Defaults are deliberately tighter than the global ones."""

    model_config = ConfigDict(extra="forbid")
    max_calls: int = Field(default=8, ge=1, le=50)
    timeout_s: int = Field(default=15, ge=1, le=120)
    max_output_bytes: int = Field(default=20_000, ge=256, le=500_000)


class AgentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    tools: tuple[str, ...]
    read_roots: tuple[str, ...] = ()
    write_roots: tuple[str, ...] = ()
    network: bool = False
    max_tier: int = Field(ge=0, le=3)
    isolation: Literal["none", "bwrap", "docker"] = "none"
    limits: Limits = Limits()
    description: str = ""


def build_manifests(read_roots: tuple[str, ...],
                    write_root: str) -> dict[str, AgentManifest]:
    """Build the manifest table from the configured roots.

    Built from settings rather than hard-coded, so a test that points the scope at a
    tmp dir gets a manifest that agrees with the jail. A hard-coded manifest and a
    configured jail can disagree, and the disagreement is a hole.

    Accepts `Path` or `str` roots: `Settings.read_roots` is a tuple of `Path` while
    the manifest stores strings, and the caller should not have to care which it has.
    Coercing here rather than at each call site means the manifest can never end up
    holding a `Path` that pydantic would have rejected.
    """
    roots = tuple(str(r) for r in read_roots)
    write = str(write_root)
    return {
        "discovery": AgentManifest(
            name="discovery", tools=("find", "grep", "read", "survey", "index_query"),
            read_roots=roots, write_roots=(), network=False,
            max_tier=0, limits=Limits(max_calls=8, timeout_s=15),
            description="read-only filesystem discovery"),
        # KNOWN BUG (removal scheduled, Phase 5): webSearch is declared so plans can
        # target it, but nothing implements it. A plan step naming it fails at run
        # time instead of at planning time. Phase 5 restricts the planner vocabulary
        # to implemented agents, or implements webSearch behind an importlib check.
        "webSearch": AgentManifest(
            name="webSearch", tools=("search", "fetch"), read_roots=(),
            network=True, max_tier=0),
        "planner": AgentManifest(
            name="planner", tools=(), read_roots=(), max_tier=0,
            description="no tools, structurally"),
        "answerer": AgentManifest(
            name="answerer", tools=(), read_roots=(), max_tier=0),
        # Frozen. Present so the machinery is exercised, but max_tier=0 for the
        # discovery agent means it cannot be reached from a normal task.
        "executor": AgentManifest(
            name="executor", tools=("run_command",),
            read_roots=roots, write_roots=(write,),
            network=False, max_tier=2, isolation="none",
            limits=Limits(max_calls=4, timeout_s=60),
            description="runs shell commands under policy; NO SANDBOX"),
    }


class Registry:
    """Manifest lookup with a helpful error."""

    def __init__(self, manifests: dict[str, AgentManifest]) -> None:
        self.manifests = manifests

    def __getitem__(self, name: str) -> AgentManifest:
        m = self.manifests.get(name)
        if m is None:
            raise KeyError(f"unknown agent {name!r}; known: {sorted(self.manifests)}")
        return m

    def get(self, name: str) -> AgentManifest | None:
        return self.manifests.get(name)

    def implemented(self) -> set[str]:
        """Agents with a runner. The planner's vocabulary is this set in Phase 5."""
        return {"discovery", "planner", "answerer", "executor"}