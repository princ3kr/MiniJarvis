from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Observation:
    tool: str
    status: Literal["ok", "empty", "blocked", "not_found", "error", "skipped", "binary"]
    path: str | None = None
    text: str = ""  # CAPPED excerpt
    total_lines: int = 0  # lines in the WHOLE source, not the window
    n_lines: int = 0  # lines in this window
    start_line: int = 1
    truncated: bool = False
    handle: str | None = None  # opaque paging handle, see below
    items: list[dict] = field(default_factory=list)  # structured rows (find hits, grep hits)
    meta: dict = field(default_factory=dict)
    reason: str = ""
    duration_ms: float | None = None