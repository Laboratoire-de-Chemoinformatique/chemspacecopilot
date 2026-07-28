"""Behavior tests for direct MCP parity facades."""

from __future__ import annotations

import asyncio
import inspect

from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.llm import LLMBroker
from cs_copilot.mcp.tool_adapter import build_tool
from cs_copilot.mcp.tools_registry import all_specs
from cs_copilot.storage import S3
from cs_copilot.workflows import RunContext


def _spec(name: str):
    return next(spec for spec in all_specs() if spec.mcp_name == name)


def test_skill_tools_work_directly():
    skills = _spec("skill_list").toolkit_factory()

    listed = skills.list()
    searched = skills.search("molecular design")
    fetched = skills.fetch("molecular-design")

    assert any(skill["slug"] == "molecular-design" for skill in listed)
    assert searched[0]["slug"] == "molecular-design"
    assert fetched["slug"] == "molecular-design"
    assert "#" in fetched["skill_md"]


def test_workflow_tools_work_directly():
    workflows = _spec("workflow_list").toolkit_factory()

    listed = workflows.list()
    searched = workflows.search("chembl gtm report")
    fetched = workflows.fetch("chembl-to-gtm-report")

    assert any(workflow["slug"] == "chembl-to-gtm-report" for workflow in listed)
    assert searched[0]["slug"] == "chembl-to-gtm-report"
    assert fetched["slug"] == "chembl-to-gtm-report"
    assert "report_save_rich" in fetched["workflow_md"]


def test_molecular_validate_and_rank_work_without_model_access():
    tools = _spec("mol_validate_design_candidates").toolkit_factory()

    validated = tools.validate_design_candidates(["CCO", "c1ccccc1", "bad"])
    ranked = tools.rank_design_candidates(validated, seed_smiles="CCO")

    assert ranked[0]["smiles"] == "CCO"
    assert ranked[0]["properties"]["seed_tanimoto"] == 1.0
    assert ranked[-1]["valid"] is False


def test_peptide_validate_and_rank_work_without_model_access():
    tools = _spec("peptide_validate_design_candidates").toolkit_factory()

    validated = tools.validate_design_candidates(["ACD", "A C E", "B"])
    ranked = tools.rank_design_candidates(validated, seed_sequence="A C D")

    assert ranked[0]["sequence"] == "A C D"
    assert ranked[0]["properties"]["seed_sequence_similarity"] == 1.0
    assert ranked[-1]["valid"] is False


def test_llm_design_engines_create_external_tasks_in_default_mcp():
    ctx = MCPAgentContext()
    ctx.llm = LLMBroker(ctx)
    mol_spec = _spec("mol_design_molecules")
    pep_spec = _spec("peptide_design_peptides")

    mol_tool = build_tool(mol_spec, mol_spec.toolkit_factory(), ctx)
    pep_tool = build_tool(pep_spec, pep_spec.toolkit_factory(), ctx)

    mol_result = asyncio.run(mol_tool(goal="design molecules", engine="llm"))
    pep_result = asyncio.run(pep_tool(goal="design peptides", engine="llm"))

    assert mol_result["status"] == "success"
    assert mol_result["data"]["status"] == "needs_external_llm"
    assert mol_result["data"]["task_type"] == "molecular.design"
    assert pep_result["status"] == "success"
    assert pep_result["data"]["status"] == "needs_external_llm"
    assert pep_result["data"]["task_type"] == "peptide.design"
    pending = ctx.llm.list_tasks()
    assert [task["task_type"] for task in pending] == [
        "molecular.design",
        "peptide.design",
    ]


def test_llm_design_engines_fail_when_llm_policy_is_disabled():
    ctx = MCPAgentContext(llm_policy="disabled")
    ctx.llm = LLMBroker(ctx)
    mol_spec = _spec("mol_design_molecules")
    mol_tool = build_tool(mol_spec, mol_spec.toolkit_factory(), ctx)

    result = asyncio.run(mol_tool(goal="design molecules", engine="llm"))
    assert result["status"] == "error"
    assert "llm_policy='disabled'" in result["error"]["message"]


def test_llm_lifecycle_tools_work_directly():
    ctx = MCPAgentContext()
    ctx.llm = LLMBroker(ctx)

    create_spec = _spec("llm_create_task")
    create_tool = build_tool(create_spec, create_spec.toolkit_factory(), ctx)
    list_spec = _spec("llm_list_pending_tasks")
    list_tool = build_tool(list_spec, list_spec.toolkit_factory(), ctx)
    submit_spec = _spec("llm_submit_task_result")
    submit_tool = build_tool(submit_spec, submit_spec.toolkit_factory(), ctx)

    created = asyncio.run(
        create_tool(task_type="unit.test", prompt_text="Return JSON.", consumer_tool="unit")
    )
    pending = asyncio.run(list_tool())
    completed = asyncio.run(
        submit_tool(task_id=created["data"]["task_id"], result={"answer": "ok"})
    )

    assert pending["data"][0]["task_id"] == created["data"]["task_id"]
    assert completed["data"]["status"] == "completed"
    assert completed["data"]["result"] == {"answer": "ok"}


def test_chembl_external_judge_task_and_submit_work_without_backend_probe():
    from cs_copilot.mcp.facades.chembl import ChemblMCPFacade
    from cs_copilot.tools.databases.chembl import ChemblToolkit

    ctx = MCPAgentContext()
    ctx.llm = LLMBroker(ctx)
    facade = ChemblMCPFacade()
    facade._inner = ChemblToolkit.__new__(ChemblToolkit)

    task = facade.create_external_judge_task(
        judge_type="retrieval",
        target_query="CDK2",
        keywords="CDK",
        organism_filter="Homo sapiens",
        items=[
            {
                "item_id": "item_1",
                "judge_scope": "protein",
                "judge_basis": "target_pref_name",
                "value": "Cyclin-dependent kinase 2",
                "row_count": 1,
                "assay_chembl_ids": ["CHEMBL1"],
                "sample_descriptions": ["CDK2 inhibition"],
            }
        ],
        agent=ctx,
    )
    submitted = facade.submit_external_judge_result(
        task_id=task["task_id"],
        result={"decisions": [{"item_id": "item_1", "keep": True, "explanation": "match"}]},
        expected_item_ids=["item_1"],
        agent=ctx,
    )

    assert task["prompt_name"] == "chembl_retrieval_judge"
    assert "Cyclin-dependent kinase 2" in task["prompt_text"]
    assert submitted["status"] == "completed"


def test_chembl_external_judge_tasks_are_created_from_current_dataset(tmp_path, monkeypatch):
    from cs_copilot.mcp.facades.chembl import ChemblMCPFacade
    from cs_copilot.tools.databases.chembl import ChemblToolkit

    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/chembl-external-judge")
    ctx = MCPAgentContext()
    ctx.llm = LLMBroker(ctx)
    raw_path = "chembl_raw.csv"
    with S3.open(raw_path, "w") as handle:
        handle.write(
            "query_keywords,target_pref_name,target_organism,target_type,"
            "assay_chembl_id,description\n"
            "CDK,Cyclin-dependent kinase 2,Homo sapiens,SINGLE PROTEIN,"
            "CHEMBL1,CDK2 inhibition\n"
        )
    ctx.session_state["session_objects"] = {
        "current": {"dataset": "ds_001"},
        "datasets": {
            "ds_001": {
                "raw_dataset_path": raw_path,
                "query_keywords": ["CDK"],
                "standardization_summary": {
                    "chembl_retrieval_filtering": {
                        "judge_status": "disabled",
                        "suspicious_row_count": 1,
                        "metadata_judge_status": "disabled",
                        "metadata_judge_row_count": 1,
                    }
                },
            }
        },
    }
    facade = ChemblMCPFacade()
    facade._inner = ChemblToolkit.__new__(ChemblToolkit)

    tasks = facade._create_external_judge_tasks_from_current_dataset(
        target_query="CDK2",
        organism_filter="Homo sapiens",
        agent=ctx,
        session_state=ctx.session_state,
    )

    assert [task["task_type"] for task in tasks] == [
        "chembl.retrieval_judge",
        "chembl.metadata_judge",
    ]
    assert len(ctx.llm.list_tasks()) == 2


def test_pandas_facade_schema_hides_injected_parameters():
    ctx = MCPAgentContext()
    spec = _spec("pandas_normalize_for_analysis")
    tool = build_tool(spec, spec.toolkit_factory(), ctx)
    signature = inspect.signature(tool)

    assert "agent" not in signature.parameters
    assert "session_state" not in signature.parameters
    assert "df_path" in signature.parameters


def test_synplanner_identify_input_handles_smiles_and_fallback_names():
    tools = _spec("synplanner_identify_input").toolkit_factory()

    smiles = tools.identify_input("CCO")
    aspirin = tools.identify_input("aspirin")

    assert smiles["source"] == "smiles"
    assert smiles["smiles"] == "CCO"
    assert aspirin["source"] == "name"
    assert aspirin["smiles"]


def test_new_direct_tools_write_mcp_events(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "direct-tool-manifest"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    ctx.run_context = RunContext.create(
        "mcp-session",
        session_state=ctx.session_state,
        run_id=session_name,
    )
    spec = _spec("skill_fetch")
    tool = build_tool(spec, spec.toolkit_factory(), ctx)

    result = asyncio.run(tool(slug="molecular-design", include_content=False))

    assert result["data"]["slug"] == "molecular-design"
    payloads = [
        event.payload
        for event in ctx.run_context.events
        if event.event_type == "tool_call_recorded"
    ]
    assert len(payloads) == 1
    assert payloads[0]["tool_name"] == "skill_fetch"
    assert payloads[0]["status"] == "success"


def test_synplanner_backend_failure_surfaces_as_error_envelope(monkeypatch):
    from cs_copilot.tools.chemistry.synplanner_toolkit import SynPlannerError

    spec = _spec("synplanner_plan_synthesis")
    instance = spec.toolkit_factory()

    def fail_backend():
        raise SynPlannerError("SynPlanner backend missing")

    monkeypatch.setattr(instance, "_load_synplanner_components", fail_backend)
    tool = build_tool(spec, instance, MCPAgentContext())

    result = asyncio.run(tool(query="CCO"))
    assert result["status"] == "error"
    assert "SynPlanner backend missing" in result["error"]["message"]
