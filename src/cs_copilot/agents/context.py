"""Explicit context budgets for bounded, auditable agent handoffs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .contracts import HandoffEnvelope

_TRUNCATION_MARKER = "\n[truncated by context budget]"


@dataclass(frozen=True)
class ContextBudget:
    """Per-component token allocation for a specialist invocation."""

    system_policy: int = 1_500
    procedure: int = 3_000
    run_summary: int = 2_000
    tool_schemas: int = 3_000
    recent_messages: int = 1_500

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if value < 0:
                raise ValueError(f"context budget {name} cannot be negative")
        if self.total <= 0:
            raise ValueError("context budget total must be positive")

    @property
    def total(self) -> int:
        return sum(self.as_dict().values())

    def as_dict(self) -> dict[str, int]:
        return {
            "system_policy": self.system_policy,
            "procedure": self.procedure,
            "run_summary": self.run_summary,
            "tool_schemas": self.tool_schemas,
            "recent_messages": self.recent_messages,
        }


DEFAULT_CONTEXT_BUDGET = ContextBudget()


@dataclass(frozen=True)
class BoundedContext:
    """Prompt sections and their measured size after explicit truncation."""

    sections: Mapping[str, str]
    token_counts: Mapping[str, int]
    truncated: tuple[str, ...]

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts.values())

    def render(self) -> str:
        return "\n\n".join(
            f"## {name.replace('_', ' ').title()}\n{text}"
            for name, text in self.sections.items()
            if text
        )


class ContextBuilder:
    """Build bounded receiver context without forwarding a full conversation."""

    def __init__(
        self,
        budget: ContextBudget = DEFAULT_CONTEXT_BUDGET,
        *,
        count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        self.budget = budget
        self._count_tokens = count_tokens or _estimate_tokens

    def build(
        self,
        handoff: HandoffEnvelope,
        *,
        system_policy: str,
        procedure: str,
        run_summary: str,
        tool_schemas: str,
        recent_messages: Iterable[str] = (),
    ) -> BoundedContext:
        """Fit each component independently and report every truncation."""
        handoff_text = _render_handoff(handoff)
        recent_text = _bounded_recent_messages(recent_messages)
        raw_sections = {
            "system_policy": system_policy,
            "procedure": f"{handoff_text}\n\nProcedure:\n{procedure}".strip(),
            "run_summary": run_summary,
            "tool_schemas": tool_schemas,
            "recent_messages": recent_text,
        }
        allocations = self.budget.as_dict()
        sections: dict[str, str] = {}
        token_counts: dict[str, int] = {}
        truncated: list[str] = []
        for name, text in raw_sections.items():
            fitted, was_truncated = _fit_text(text or "", allocations[name], self._count_tokens)
            sections[name] = fitted
            token_counts[name] = self._count_tokens(fitted)
            if was_truncated:
                truncated.append(name)
        return BoundedContext(sections, token_counts, tuple(truncated))


def _render_handoff(envelope: HandoffEnvelope) -> str:
    payload = envelope.to_dict()
    lines = ["Handoff contract:"]
    for key in (
        "run_id",
        "workflow_slug",
        "task_id",
        "sender_role",
        "receiver_role",
        "objective",
        "constraints",
        "required_capabilities",
        "input_artifact_ids",
        "expected_output_artifacts",
        "expected_output_schema",
        "acceptance_criteria",
        "context_summary",
        "budget",
        "trace_id",
        "span_id",
        "parent_span_id",
    ):
        value = payload.get(key)
        if value not in (None, "", []):
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _bounded_recent_messages(messages: Iterable[str], *, maximum: int = 8) -> str:
    # Only a small explicit tail is accepted. Callers cannot pass an unbounded
    # full history and have it silently included.
    tail = [str(message).strip() for message in messages if str(message).strip()][-maximum:]
    # Newest-first ordering ensures a later size cap preserves the most recent
    # facts rather than silently retaining older conversation content.
    return "\n".join(f"- {message}" for message in reversed(tail))


def _fit_text(
    text: str,
    budget: int,
    count_tokens: Callable[[str], int],
) -> tuple[str, bool]:
    if count_tokens(text) <= budget:
        return text, False
    if budget <= 0:
        return "", bool(text)
    marker_tokens = count_tokens(_TRUNCATION_MARKER)
    if marker_tokens > budget:
        compact_marker = "[truncated]"
        while compact_marker and count_tokens(compact_marker) > budget:
            compact_marker = compact_marker[:-1]
        return compact_marker, True
    content_budget = max(0, budget - marker_tokens)
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(text[:middle]) <= content_budget:
            low = middle
        else:
            high = middle - 1
    fitted = text[:low].rstrip() + _TRUNCATION_MARKER
    return fitted, True


def _estimate_tokens(text: str) -> int:
    """Conservative provider-neutral fallback used when no tokenizer is available."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
