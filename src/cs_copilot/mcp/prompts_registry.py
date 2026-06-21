"""Registry mapping cs_copilot prompt constants to MCP prompts.

Each entry exposes one of the curated agent / team instruction lists in
``src/cs_copilot/agents/prompts.py`` (pure string content, no Agno team
imports) plus the two ChEMBL LLM-as-judge prompt templates so that an
external reasoner can recreate the in-process filtering itself.

Importing this module never instantiates an Agno agent or team.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class PromptSpec:
    """Declarative description of one MCP prompt."""

    mcp_name: str
    summary: str
    render: Callable[..., str]
    arguments: Sequence[Mapping[str, Any]] = ()


def _join_instructions(instructions: Sequence[str]) -> str:
    return "\n".join(str(line) for line in instructions)


def _agent_prompt(constant_name: str, mcp_name: str, summary: str) -> PromptSpec:
    def _render() -> str:
        from cs_copilot.agents import prompts as _prompts

        value = getattr(_prompts, constant_name)
        return _join_instructions(value)

    return PromptSpec(mcp_name=mcp_name, summary=summary, render=_render)


def _render_mcp_workflow_prompt() -> str:
    return _join_instructions(
        (
            "You are the external MCP reasoner for cs_copilot.",
            "Do not delegate to the Agno team unless the private agno_team_run tool is "
            "explicitly enabled and the user asks for trusted delegation.",
            "For each new user request, call mcp_bootstrap first with the user request "
            "and optional workflow_slug if one is known.",
            "Treat mcp_bootstrap as an organization step: fetch its recommended prompt, "
            "workflow, and skill documents, then run its listed read-only preflight tools.",
            "Use chembl_prepare_retrieval before ChEMBL retrieval and "
            "chemspace_plan_analysis before broad chemical-space or GTM work.",
            "If bootstrap itself returns bootstrap_questions, ask them before continuing.",
            "If a preflight tool returns needs_clarification=true, ask the returned "
            "questions before calling write tools.",
            "Do not invent missing target, organism, assay type, mechanism, analysis "
            "intent, dataset source, or workflow details just to avoid asking the user.",
            "Call MCP tools directly by name; the agent-style prompts are role guidance, "
            "not separate workers in default MCP mode.",
            "Use llm_* task tools when a tool returns status needs_external_llm.",
            "Treat cscopilot://session resources and session_* tools as the source of "
            "truth for prior artifacts, datasets, candidates, GTM maps, reports, and "
            "synthesis plans.",
            "Review write actions before running them and report saved artifact paths back "
            "to the user.",
        )
    )


# Curated agent / workflow prompts. The names are stable; descriptions describe
# the role the prompt asks the external reasoner to adopt.
_AGENT_PROMPTS: List[PromptSpec] = [
    PromptSpec(
        mcp_name="cs_copilot_mcp_workflow",
        summary=(
            "MCP-native top-level orchestration prompt. Use this first when "
            "driving cs_copilot as an external MCP reasoner."
        ),
        render=_render_mcp_workflow_prompt,
    ),
    PromptSpec(
        mcp_name="cs_copilot_workflow",
        summary=(
            "Top-level cs_copilot orchestration prompt. Adopt this when driving "
            "the cs_copilot toolkits as an external reasoner — it covers agent "
            "selection, molecule-vs-peptide routing, and workflow composition."
        ),
        render=lambda: _join_instructions(_load_prompts().AGENT_TEAM_INSTRUCTIONS),
    ),
    _agent_prompt(
        "CHEMBL_INSTRUCTIONS",
        "chembl_agent",
        "Act as the ChEMBL data-retrieval agent: target validation, organism, "
        "assay-type and mechanism workflow.",
    ),
    _agent_prompt(
        "GTM_AGENT_INSTRUCTIONS",
        "gtm_agent",
        "Act as the GTM agent: build, load, project, density and activity "
        "landscape workflows on Generative Topographic Maps.",
    ),
    _agent_prompt(
        "CHEMOINFORMATICIAN_INSTRUCTIONS",
        "chemoinformatician_agent",
        "Act as the chemoinformatician agent: scaffold, clustering, SAR and "
        "similarity analyses on prepared datasets.",
    ),
    _agent_prompt(
        "MOLECULAR_DESIGNER_INSTRUCTIONS",
        "molecular_designer_agent",
        "Act as the molecular designer agent: autoencoder + LLM design and "
        "candidate validation for small molecules.",
    ),
    _agent_prompt(
        "PEPTIDE_DESIGNER_INSTRUCTIONS",
        "peptide_designer_agent",
        "Act as the peptide designer agent: WAE + LLM design and antimicrobial "
        "landscape analysis for peptides.",
    ),
    _agent_prompt(
        "SYNPLANNER_INSTRUCTIONS",
        "synplanner_agent",
        "Act as the SynPlanner agent: retrosynthetic planning and route " "visualisation.",
    ),
    _agent_prompt(
        "REPORT_GENERATOR_INSTRUCTIONS",
        "report_generator_agent",
        "Act as the report generator agent: produce markdown / rich (HTML/PDF) "
        "reports and persist them to session storage.",
    ),
    _agent_prompt(
        "ROBUSTNESS_EVALUATION_INSTRUCTIONS",
        "robustness_evaluation",
        "Act as the robustness evaluation agent: review LLM-judge robustness " "test results.",
    ),
    _agent_prompt(
        "HANDLING_NEW_FILES_INSTRUCTIONS",
        "handling_new_files",
        "Convention for sharing produced files between agents using " "<file>...</file> tags.",
    ),
]


def _load_prompts():
    from cs_copilot.agents import prompts as _prompts

    return _prompts


# ChEMBL judge prompts — these are parameterised, so the MCP client supplies
# the arguments and gets a fully-rendered prompt back.
def _render_chembl_retrieval_judge(
    target_query: str,
    keywords: str,
    organism_filter: Optional[str] = None,
    items: str = "[]",
) -> str:
    from cs_copilot.tools.databases.chembl import CHEMBL_RETRIEVAL_JUDGE_TEMPLATE

    return CHEMBL_RETRIEVAL_JUDGE_TEMPLATE.format(
        target_query=target_query,
        keywords_csv=keywords,
        organism_filter=organism_filter or "none",
        items_json=items,
    )


def _render_chembl_metadata_judge(
    target_query: str,
    keywords: str,
    organism_filter: Optional[str] = None,
    items: str = "[]",
) -> str:
    from cs_copilot.tools.databases.chembl import CHEMBL_METADATA_JUDGE_TEMPLATE

    return CHEMBL_METADATA_JUDGE_TEMPLATE.format(
        target_query=target_query,
        keywords_csv=keywords,
        organism_filter=organism_filter or "none",
        items_json=items,
    )


_JUDGE_ARGUMENTS = (
    {"name": "target_query", "description": "Original target / query text.", "required": True},
    {
        "name": "keywords",
        "description": 'Comma-separated search keywords (e.g. "CDK,2").',
        "required": True,
    },
    {
        "name": "organism_filter",
        "description": "Organism filter applied to the fetch, or null/none.",
        "required": False,
    },
    {
        "name": "items",
        "description": "JSON array of candidate rows produced by the toolkit.",
        "required": False,
    },
)


_JUDGE_PROMPTS: List[PromptSpec] = [
    PromptSpec(
        mcp_name="chembl_retrieval_judge",
        summary=(
            "ChEMBL short-keyword retrieval LLM-as-judge prompt — use this to "
            "replicate the in-process retrieval-judge filtering with this "
            "client's reasoning engine."
        ),
        render=_render_chembl_retrieval_judge,
        arguments=_JUDGE_ARGUMENTS,
    ),
    PromptSpec(
        mcp_name="chembl_metadata_judge",
        summary=(
            "ChEMBL populated-target metadata LLM-as-judge prompt — use this "
            "to replicate the in-process metadata-judge filtering with this "
            "client's reasoning engine."
        ),
        render=_render_chembl_metadata_judge,
        arguments=_JUDGE_ARGUMENTS,
    ),
]


def iter_specs() -> Iterable[PromptSpec]:
    """Yield every :class:`PromptSpec` exposed by the MCP server."""

    yield from _AGENT_PROMPTS
    yield from _JUDGE_PROMPTS


def all_specs() -> List[PromptSpec]:
    """Return every :class:`PromptSpec` as a list (handy for tests)."""

    return list(iter_specs())
