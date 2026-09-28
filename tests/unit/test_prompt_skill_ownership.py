"""Guards for prompt vs skill/workflow ownership.

Deliberate divergences between the slim Agno-team prompts and the MCP-framed
catalog (do NOT "fix" these to literally match):
- Tool vocabulary: catalogs use MCP names (``chembl_fetch_compounds``) while the
  Agno toolkits expose native names (``fetch_compounds``); the MCP facades bridge
  the two.
- LLM work is *delegated, not disabled*: under MCP ``llm_policy="external"`` the
  tool returns a ``needs_external_llm`` task for the outer agent to complete; the
  Agno team uses its own model inline. Skills must describe delegation, never
  claim the LLM engine is "unavailable".
- Session/cache mechanics (e.g. ``gtm_cache``) are Agno-team internals and are
  intentionally absent from the execution-context-agnostic catalog.
"""

from __future__ import annotations

import re

from cs_copilot.agents import instructions
from cs_copilot.skills import get_skill

PROMPT_CONSTANTS = (
    instructions.CHEMBL_INSTRUCTIONS,
    instructions.CHEMOINFORMATICIAN_INSTRUCTIONS,
    instructions.MOLECULAR_DESIGNER_INSTRUCTIONS,
    instructions.GTM_AGENT_INSTRUCTIONS,
    instructions.REPORT_GENERATOR_INSTRUCTIONS,
    instructions.AGENT_TEAM_INSTRUCTIONS,
    instructions.SYNPLANNER_INSTRUCTIONS,
    instructions.PEPTIDE_DESIGNER_INSTRUCTIONS,
    instructions.ROBUSTNESS_EVALUATION_INSTRUCTIONS,
)


def _joined_prompt_text() -> str:
    return "\n".join(line for constant in PROMPT_CONSTANTS for line in constant)


def test_agent_prompts_do_not_reintroduce_step_by_step_tool_procedures():
    text = _joined_prompt_text()

    assert not re.search(r"\bStep\s+\d+\s*:", text, flags=re.IGNORECASE)
    assert "source of truth" in text
    assert "fetch_skill" in text


def test_high_risk_procedures_live_in_skills():
    expectations = {
        "chembl-target-retrieval": [
            "chembl_prepare_retrieval",
            "chembl_fetch_compounds",
            "target specificity",
            "standardization report",
            "B/F/A",
            "case-insensitive substring",
            "achiral",
        ],
        "gtm-density-landscape": [
            "chemspace_plan_analysis",
            "gtm_save_density_plot",
            "gtm_get_density_summary",
            "density analysis",
            "filtered_density",
            "neighborhood score",
        ],
        "gtm-activity-landscape": [
            "chemspace_plan_analysis",
            "gtm_create_activity_landscapes",
            "gtm-density-landscape",
            "Default",
            "filtered_reg_density",
            "3-bullet SAR",
        ],
        "report-generation": [
            "report_save_rich",
            "report_save_markdown",
            "synthesis reports",
            "<file>...</file>",
            "scaffold-frequency",
            "compass-annotated",
        ],
        "molecular-design": [
            "mol_design_molecules",
            "mol_register_design_candidates",
            "session_materialize_candidate_set_dataset",
            "noise_scale",
            "registered_candidate_set_id",
            "experimentally verified",
        ],
        "peptide-design": [
            "peptide_design_peptides",
            "peptide_validate_design_candidates",
            "gtm_create_peptide_activity_landscapes",
            "wae_peptides",
            "V, W, Y, Z",
        ],
        "retrosynthesis-planning": [
            "synplanner_plan_synthesis",
            "synplanner_get_route_visualizations",
            "not SynPlanner-validated",
            "![Route",
        ],
        "chemoinformatics-analysis": [
            "Murcko",
            "silhouette",
            "activity cliff",
            "chemotype_analysis",
            "sar_analysis",
        ],
        "robustness-report": [
            "data_similarity",
            "process_consistency",
            "0.70",
        ],
        "dataset-normalization": [
            "activity_mapping",
            "final_activity_mapping",
        ],
    }

    for slug, snippets in expectations.items():
        skill_text = get_skill(slug).skill_md
        for snippet in snippets:
            assert snippet in skill_text, f"{slug} is missing {snippet!r}"


def test_llm_work_described_as_delegation_not_unavailable():
    """The MCP server delegates LLM work to the outer agent; skills must say so."""

    for slug in ("molecular-design", "peptide-design", "candidate-design-to-gtm"):
        text = get_skill(slug).skill_md
        assert "needs_external_llm" in text, f"{slug} should describe external LLM delegation"
        assert (
            "is unavailable because" not in text
        ), f"{slug} still claims the LLM engine is unavailable"

    # The ChEMBL LLM-as-judge is delegated, not silently disabled.
    chembl = get_skill("chembl-target-retrieval").skill_md
    assert "delegated, not skipped" in chembl
