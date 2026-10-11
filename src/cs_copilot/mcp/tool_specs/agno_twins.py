"""In-process (Agno) implementations of MCP operations served through facades.

MCP specs whose factory builds a real toolkit are bound to
``<toolkit>.<method>`` automatically. Operations exposed through an MCP facade
(which adapts arguments, lazy loading, or LLM policy) name the toolkit method or
bare function that implements the same operation in the Agno runtime here, so
both runtimes record the same tool identity and execution contract.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

_CHEMBL = "cs_copilot.tools.databases.chembl:ChemblToolkit"
_GTM = "cs_copilot.tools.chemography.gtm:GTMToolkit"
_GTM_OPERATIONS = "cs_copilot.tools.chemography.gtm_operations"
_PANDAS = "cs_copilot.tools.io.pointer_pandas_tools:PointerPandasTools"
_REPORTS = "cs_copilot.tools.io.report_export"
_SKILLS = "cs_copilot.tools.io.skill_toolkit:SkillToolkit"

AGNO_TWINS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "chembl_prepare_retrieval": (f"{_CHEMBL}.prepare_retrieval",),
        "chembl_fetch_compounds": (f"{_CHEMBL}.fetch_compounds",),
        "chemspace_plan_analysis": (f"{_GTM}.plan_analysis",),
        "gtm_save_density_plot": (f"{_GTM_OPERATIONS}:save_gtm_plot",),
        "gtm_save_landscape_plot": (f"{_GTM_OPERATIONS}:save_gtm_landscape_plot",),
        "pandas_create_dataframe": (f"{_PANDAS}.create_pandas_dataframe",),
        "pandas_run_operation": (f"{_PANDAS}.run_dataframe_operation",),
        "pandas_load_dataframe_from_session": (f"{_PANDAS}.load_dataframe_from_session",),
        "report_save_markdown": (f"{_REPORTS}:save_markdown_report",),
        "report_save_rich": (f"{_REPORTS}:save_rich_report",),
        "report_get_evidence": ("cs_copilot.tools.io.reporting_evidence:get_report_evidence",),
        "skill_list": (f"{_SKILLS}.list_skills",),
        "skill_search": (f"{_SKILLS}.search_skills",),
        "skill_fetch": (f"{_SKILLS}.fetch_skill",),
    }
)

# Groups whose facade methods keep the toolkit's method names.
FACADE_GROUP_TOOLKITS: Mapping[str, str] = MappingProxyType(
    {
        "molecular_design": (
            "cs_copilot.tools.chemistry.molecular_designer_toolkit:MolecularDesignerToolkit"
        ),
        "peptide_design": "cs_copilot.tools.chemistry.peptide_designer_toolkit:PeptideDesignerToolkit",
    }
)
