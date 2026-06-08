"""Deterministic chemical-space analysis preflight policy."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


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
    user_request: str,
    session_summary: str | None = None,
) -> dict[str, Any]:
    """Return a structured preflight plan for chemical-space analysis."""

    request = _clean(user_request)
    context = _clean("\n".join(value for value in (user_request, session_summary) if value))
    if not request:
        return ChemicalSpaceDecision(
            can_proceed=False,
            needs_clarification=True,
            missing_requirements=["user_request"],
            clarifying_questions=["What chemical-space analysis should be planned?"],
        ).as_dict()

    intents = _detect_intents(request)
    dataset_source = _detect_dataset_source(context)
    missing: list[str] = []
    questions: list[str] = []
    notes: list[str] = []

    if not intents or _is_broad_chemical_space_request(request, intents):
        missing.append("analysis_intent")
        questions.append(
            "Which analysis do you want: ChEMBL retrieval, GTM density map, activity "
            "landscape, projection, chemotype/SAR analysis, or report generation?"
        )

    if "chembl_retrieval" in intents:
        notes.append("Use chembl_prepare_retrieval before calling ChEMBL retrieval tools.")
        dataset_source = dataset_source or "chembl_retrieval"

    if not dataset_source:
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
        dataset_source=dataset_source,
        recommended_next_tools=_recommended_tools(intents, can_proceed),
        notes=notes,
    ).as_dict()


def _clean(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _detect_intents(request: str) -> list[str]:
    lower = request.lower()
    intents: list[str] = []

    if any(term in lower for term in ("chembl", "fetch", "retrieve", "download")):
        intents.append("chembl_retrieval")
    if "activity landscape" in lower or "active region" in lower or "actives" in lower:
        intents.append("activity_landscape")
    if "density" in lower or "distribution" in lower:
        intents.append("density_landscape")
    if "project" in lower or "projection" in lower or "map new data" in lower:
        intents.append("gtm_projection")
    if "gtm" in lower or "map" in lower:
        if any(term in lower for term in ("build", "train", "optimize", "create")):
            intents.append("gtm_build")
        else:
            intents.append("gtm_analysis")
    if any(term in lower for term in ("chemotype", "scaffold", "sar")):
        intents.append("chemotype_sar_analysis")
    if "report" in lower or "summary" in lower:
        intents.append("report_generation")

    return list(dict.fromkeys(intents))


def _detect_dataset_source(context: str) -> str | None:
    lower = context.lower()
    if re.search(r"\b(?:s3://\S+|\S+\.(?:csv|parquet|tsv))\b", context):
        return "explicit_path"
    if any(
        marker in lower
        for marker in (
            "clean_dataset_path",
            "dataset_path",
            "current clean dataset",
            "existing clean dataset",
            "session clean dataset",
        )
    ):
        return "session_clean_dataset"
    if "uploaded dataset" in lower or "uploaded file" in lower:
        return "uploaded_dataset"
    return None


def _is_broad_chemical_space_request(request: str, intents: list[str]) -> bool:
    lower = request.lower().strip()
    broad_phrases = {
        "analyze chemical space",
        "analyse chemical space",
        "chemical space analysis",
        "analyze compounds",
        "help me analyze compounds",
    }
    return lower in broad_phrases or (lower == "chemical space" and not intents)


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
        tools.append("gtm_load_density_matrix")
    if "gtm_projection" in intents:
        tools.append("gtm_project_data")
    if "chemotype_sar_analysis" in intents:
        tools.append("gtm_sample_dense_nodes")
    if "report_generation" in intents:
        tools.append("report_save_rich")
    return list(dict.fromkeys(tools))
