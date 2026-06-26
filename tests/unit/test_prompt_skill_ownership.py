"""Guards for prompt vs skill/workflow ownership."""

from __future__ import annotations

import re

from cs_copilot.agents import prompts
from cs_copilot.skills import get_skill

PROMPT_CONSTANTS = (
    prompts.CHEMBL_INSTRUCTIONS,
    prompts.CHEMOINFORMATICIAN_INSTRUCTIONS,
    prompts.MOLECULAR_DESIGNER_INSTRUCTIONS,
    prompts.GTM_AGENT_INSTRUCTIONS,
    prompts.REPORT_GENERATOR_INSTRUCTIONS,
    prompts.AGENT_TEAM_INSTRUCTIONS,
    prompts.SYNPLANNER_INSTRUCTIONS,
    prompts.PEPTIDE_DESIGNER_INSTRUCTIONS,
    prompts.ROBUSTNESS_EVALUATION_INSTRUCTIONS,
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
            "standardization report paths",
        ],
        "gtm-activity-landscape": [
            "chemspace_plan_analysis",
            "gtm_create_activity_landscapes",
            "gtm_save_density_plot",
            "Default",
        ],
        "report-generation": [
            "report_save_rich",
            "report_save_markdown",
            "synthesis reports",
            "<file>...</file>",
        ],
        "molecular-design": [
            "mol_design_molecules",
            "mol_register_design_candidates",
            "session_materialize_candidate_set_dataset",
        ],
        "peptide-design": [
            "peptide_design_peptides",
            "peptide_validate_design_candidates",
            "gtm_create_peptide_activity_landscapes",
        ],
        "retrosynthesis-planning": [
            "synplanner_plan_synthesis",
            "synplanner_get_route_visualizations",
            "not SynPlanner-validated",
        ],
    }

    for slug, snippets in expectations.items():
        skill_text = get_skill(slug).skill_md
        for snippet in snippets:
            assert snippet in skill_text, f"{slug} is missing {snippet!r}"
