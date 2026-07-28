"""Typed contracts for specialist roles and multi-agent handoffs.

These contracts deliberately carry task facts and artifact references only.
Conversation histories, scratchpads, private reasoning, and chain-of-thought are
not valid handoff fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from cs_copilot.workflows.runtime import SCHEMA_VERSION, HandoffEnvelope

HANDOFF_SCHEMA_VERSION = SCHEMA_VERSION


@dataclass(frozen=True)
class ExecutionBudget:
    """Bounded resources a receiving role may spend on one task."""

    max_tokens: int | None = None
    max_tool_calls: int | None = None
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_tokens", self.max_tokens),
            ("max_tool_calls", self.max_tool_calls),
            ("timeout_seconds", self.timeout_seconds),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "max_tokens": self.max_tokens,
            "max_tool_calls": self.max_tool_calls,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class RolePolicy:
    """Allowlist for one in-process Agno specialist role."""

    role: str
    profile: str
    allowed_toolkits: frozenset[str]
    allowed_functions: frozenset[str] = frozenset()

    def allows(self, tool: Any) -> bool:
        if callable(tool) and hasattr(tool, "__name__") and not hasattr(tool, "__dict__"):
            return tool.__name__ in self.allowed_functions
        if callable(tool) and getattr(tool, "__name__", None) in self.allowed_functions:
            return True
        return tool.__class__.__name__ in self.allowed_toolkits

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "profile": self.profile,
            "allowed_toolkits": sorted(self.allowed_toolkits),
            "allowed_functions": sorted(self.allowed_functions),
        }


def _policy(
    role: str,
    profile: str,
    toolkits: Iterable[str],
    functions: Iterable[str] = (),
) -> RolePolicy:
    return RolePolicy(role, profile, frozenset(toolkits), frozenset(functions))


ROLE_POLICIES: Mapping[str, RolePolicy] = MappingProxyType(
    {
        "coordinator": _policy(
            "coordinator",
            "standard",
            ("SessionMemoryToolkit", "SkillToolkit"),
        ),
        "chembl_downloader": _policy(
            "chembl_downloader",
            "chembl-retrieval",
            ("ChemblToolkit", "PointerPandasTools", "SkillToolkit"),
        ),
        "gtm_agent": _policy(
            "gtm_agent",
            "gtm-analysis",
            ("GTMToolkit", "PointerPandasTools", "SessionMemoryToolkit", "SkillToolkit"),
            ("save_gtm_landscape_plot", "save_gtm_plot"),
        ),
        "chemoinformatician": _policy(
            "chemoinformatician",
            "chemoinformatics",
            ("ChemicalSimilarityToolkit", "PointerPandasTools", "GTMToolkit", "SkillToolkit"),
        ),
        "report_generator": _policy(
            "report_generator",
            "reporting",
            ("PointerPandasTools", "SkillToolkit"),
            (
                "save_gtm_landscape_plot",
                "save_gtm_plot",
                "save_rich_report",
                "save_markdown_report",
            ),
        ),
        "molecular_designer": _policy(
            "molecular_designer",
            "molecular-design",
            (
                "MolecularDesignerToolkit",
                "AutoencoderToolkit",
                "GTMToolkit",
                "ChemicalSimilarityToolkit",
                "PointerPandasTools",
                "SkillToolkit",
            ),
        ),
        "peptide_designer": _policy(
            "peptide_designer",
            "peptide-design",
            ("PeptideDesignerToolkit", "GTMToolkit", "PointerPandasTools", "SkillToolkit"),
            ("save_gtm_landscape_plot", "save_gtm_plot"),
        ),
        "synplanner": _policy(
            "synplanner",
            "retrosynthesis",
            ("SynPlannerToolkit", "SkillToolkit"),
        ),
        "robustness_evaluation": _policy(
            "robustness_evaluation",
            "robustness",
            ("PointerPandasTools", "RobustnessAnalysisToolkit", "SkillToolkit"),
        ),
        "single_agent": _policy(
            "single_agent",
            "standard",
            (
                "ChemblToolkit",
                "GTMToolkit",
                "ChemicalSimilarityToolkit",
                "MolecularDesignerToolkit",
                "AutoencoderToolkit",
                "PeptideDesignerToolkit",
                "SynPlannerToolkit",
                "SessionMemoryToolkit",
                "PointerPandasTools",
                "SkillToolkit",
            ),
            (
                "save_gtm_landscape_plot",
                "save_gtm_plot",
                "save_rich_report",
                "save_markdown_report",
            ),
        ),
    }
)


def get_role_policy(role: str) -> RolePolicy:
    try:
        return ROLE_POLICIES[role]
    except KeyError as exc:
        raise KeyError(f"No tool allowlist declared for agent role '{role}'") from exc


def validate_role_tools(policy: RolePolicy, tools: Iterable[Any]) -> None:
    """Reject a factory configuration that grants tools outside its role policy."""
    denied = [_tool_identity(tool) for tool in tools if not policy.allows(tool)]
    if denied:
        raise ValueError(
            f"Role '{policy.role}' is not allowed to use configured tools: "
            + ", ".join(sorted(denied))
        )


def record_handoff(runtime: Any, envelope: HandoffEnvelope) -> bool:
    """Record a handoff through a v2 ``RunContext`` when one is available."""
    if runtime is None:
        return False
    recorder = getattr(runtime, "record_handoff", None)
    if not callable(recorder):
        raise TypeError("workflow runtime does not expose record_handoff")
    recorder(envelope)
    return True


def _tool_identity(tool: Any) -> str:
    return getattr(tool, "__name__", None) or tool.__class__.__name__
