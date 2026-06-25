"""Session-memory MCP tool specs."""

from __future__ import annotations

from typing import List

from ..tool_adapter import ToolSpec
from .common import factory

_SESSION_MEMORY = factory("cs_copilot.tools.io.session_memory:SessionMemoryToolkit")

_METHODS = [
    ("list_session_objects", "List structured objects stored in the active session.", True),
    ("list_loadable_session_data", "List session-resident datasets that can be reloaded.", True),
    ("get_session_object", "Return a session object by id.", True),
    ("select_session_object", "Mark a session object as the current one for its role.", False),
    ("resolve_session_reference", "Resolve a free-form reference to a session object id.", True),
    ("resolve_candidate_set", "Resolve a candidate-set reference to a stored object.", True),
    ("load_candidate_set_artifact", "Materialise a candidate-set artifact path.", True),
    (
        "materialize_candidate_set_dataset",
        "Materialise the dataset rows of a candidate set.",
        False,
    ),
    ("summarize_session_memory", "Return a compact textual summary of session memory.", False),
]

SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=f"session_{name}",
        toolkit_factory=_SESSION_MEMORY,
        method=name,
        summary=summary,
        read_only=read_only,
    )
    for name, summary, read_only in _METHODS
]
