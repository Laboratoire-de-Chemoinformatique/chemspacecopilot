"""Deterministic chemical-space analysis preflight policy.

Like the ChEMBL gate, this validates structured fields the external MCP reasoner
supplies — it does not parse the user's free text. The caller (the connected LLM)
classifies the requested analysis intents and the dataset source; this module
enforces that both are present and maps the chosen intents to the recommended
execution tools.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Recognized analysis intents (the caller selects from these).
_KNOWN_INTENTS = (
    "chembl_retrieval",
    "gtm_build",
    "gtm_analysis",
    "density_landscape",
    "activity_landscape",
    "gtm_projection",
    "chemotype_sar_analysis",
    "report_generation",
)

# Recognized dataset sources.
_KNOWN_DATASET_SOURCES = (
    "session_clean_dataset",
    "explicit_path",
    "uploaded_dataset",
    "chembl_retrieval",
)


@dataclass(frozen=True)
class ChemicalSpaceDecision:
    """Structured planning result for chemical-space MCP workflows."""

    can_proceed: bool
    needs_clarification: bool
    missing_requirements: list[str] = field(default_factory=list)
    clarifying_questions: list[str] = field(default_factory=list)
    analysis_intents: list[str] = field(default_factory=list)
    dataset_source: str | None = None
    recommended_next_tools: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_chemical_space_analysis(
    analysis_intents: list[str] | str | None = None,
    dataset_source: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Validate a chemical-space analysis plan from caller-supplied fields.

    Parameters
    ----------
    analysis_intents:
        One or more of: ``chembl_retrieval``, ``gtm_build``, ``gtm_analysis``,
        ``density_landscape``, ``activity_landscape``, ``gtm_projection``,
        ``chemotype_sar_analysis``, ``report_generation``. Leave empty if the user
        has not narrowed a broad "analyze chemical space" request; do not infer.
    dataset_source:
        One of ``session_clean_dataset``, ``explicit_path``, ``uploaded_dataset``,
        or ``chembl_retrieval``.
    notes:
        Optional free-text context; recorded, never parsed for the decision.
    """

    intents = _normalize_intents(analysis_intents)
    source = _clean(dataset_source) or None
    missing: list[str] = []
    questions: list[str] = []
    plan_notes: list[str] = []

    if not intents:
        missing.append("analysis_intent")
        questions.append(
            "Which analysis do you want: ChEMBL retrieval, GTM density map, activity "
            "landscape, projection, chemotype/SAR analysis, or report generation?"
        )

    if "chembl_retrieval" in intents:
        plan_notes.append("Use chembl_prepare_retrieval before calling ChEMBL retrieval tools.")
        source = source or "chembl_retrieval"

    if not source:
        missing.append("data_source")
        questions.append(
            "Which data source should be used: an existing session clean dataset, an explicit "
            "CSV/Parquet path, uploaded data, or a new ChEMBL retrieval?"
        )

    can_proceed = not missing
    return ChemicalSpaceDecision(
        can_proceed=can_proceed,
        needs_clarification=not can_proceed,
        missing_requirements=missing,
        clarifying_questions=questions,
        analysis_intents=intents,
        dataset_source=source,
        recommended_next_tools=_recommended_tools(intents, can_proceed),
        notes=plan_notes,
    ).as_dict()


def _clean(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _normalize_intents(value: list[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [part.strip() for part in value.replace(";", ",").split(",")]
    else:
        items = [str(part).strip() for part in value]
    return list(dict.fromkeys(item.lower() for item in items if item))


def _recommended_tools(intents: list[str], can_proceed: bool) -> list[str]:
    if not can_proceed:
        if "chembl_retrieval" in intents:
            return ["chembl_prepare_retrieval"]
        return []

    tools: list[str] = []
    if "chembl_retrieval" in intents:
        tools.append("chembl_prepare_retrieval")
    if any(intent in intents for intent in ("gtm_build", "density_landscape")):
        tools.extend(["gtm_optimization", "gtm_save_model_and_data"])
    if any(intent in intents for intent in ("gtm_analysis", "activity_landscape")):
        tools.extend(["gtm_load_model_only", "gtm_load_and_prep_data"])
    if "activity_landscape" in intents:
        tools.append("gtm_create_activity_landscapes")
    if "density_landscape" in intents:
        tools.extend(["gtm_save_density_plot", "gtm_get_density_summary"])
    if "gtm_projection" in intents:
        tools.append("gtm_project_data")
    if "chemotype_sar_analysis" in intents:
        tools.append("gtm_sample_dense_nodes")
    if "report_generation" in intents:
        tools.append("report_save_rich")
    return list(dict.fromkeys(tools))
