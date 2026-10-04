from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from .observation import Observation


@dataclass
class ToolSpec:
    name: str
    description: str
    params: dict[str, Any]
    handler: Callable[..., Any]


TOOL_MANIFEST: dict[str, ToolSpec] = {}
__all__ = ["Observation", "ToolSpec", "TOOL_MANIFEST"]


from . import find, grep, read, survey
from .observation import Observation
from .base import exclusion_flags, run_tool

TOOL_MANIFEST.update({
    "find": ToolSpec(
        name="find",
        description="Find files matching pattern with ranking",
        params={
            "pattern": {"type": "string"},
            "root": {"type": "string", "nullable": True},
            "max_results": {"type": "integer", "nullable": True},
            "fuzzy": {"type": "boolean"},
            "include_dirs": {"type": "boolean"},
        },
        handler=find.find_files,
    ),
    "grep": ToolSpec(
        name="grep",
        description="Search file contents",
        params={
            "query": {"type": "string"},
            "root": {"type": "string", "nullable": True},
            "max_results": {"type": "integer", "nullable": True},
            "regex": {"type": "boolean"},
            "max_per_file": {"type": "integer", "nullable": True},
            "case_sensitive": {"type": "boolean"},
        },
        handler=grep.search_content,
    ),
    "read": ToolSpec(
        name="read",
        description="Read file contents",
        params={
            "path": {"type": "string", "nullable": True},
            "handle": {"type": "string", "nullable": True},
            "start_line": {"type": "integer"},
            "max_lines": {"type": "integer", "nullable": True},
            "max_bytes": {"type": "integer", "nullable": True},
        },
        handler=read.read_file,
    ),
    "survey": ToolSpec(
        name="survey",
        description="Survey files by categories/extensions",
        params={
            "root": {"type": "string", "nullable": True},
            "categories": {"type": "array", "nullable": True},
            "extensions": {"type": "array", "nullable": True},
        },
        handler=survey.survey_files,
    ),
})
