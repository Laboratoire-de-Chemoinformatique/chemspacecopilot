"""MCP-aware ChEMBL facade."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Sequence

from ..errors import MCPToolError
from ..llm import normalize_llm_policy

logger = logging.getLogger(__name__)

_CHEMBL_JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "keep": {"type": "boolean"},
                    "explanation": {"type": "string"},
                },
                "required": ["item_id", "keep"],
            },
        }
    },
    "required": ["decisions"],
}


def _as_keywords(keywords: str | Sequence[str]) -> list[str]:
    if isinstance(keywords, str):
        return [item.strip() for item in keywords.split(",") if item.strip()]
    return [str(item).strip() for item in keywords if str(item).strip()]


def _as_items(items: str | Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(items, str):
        try:
            parsed = json.loads(items)
        except json.JSONDecodeError as exc:
            raise MCPToolError(f"items must be JSON or a list of objects: {exc}") from exc
    else:
        parsed = list(items)
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise MCPToolError("items must be a JSON array/list of objects.")
    return [dict(item) for item in parsed]


class ChemblMCPFacade:
    """Policy-aware wrapper around ``ChemblToolkit``."""

    def __init__(self) -> None:
        self._inner: Any | None = None

    def _toolkit(self) -> Any:
        if self._inner is None:
            from cs_copilot.tools.databases.chembl import ChemblToolkit

            self._inner = ChemblToolkit()
        return self._inner

    def fetch_compounds(
        self,
        query: str = "bioactivity data",
        organism: Optional[str] = None,
        assay_types: Optional[Sequence[str]] = None,
        mechanism: Optional[str] = None,
        agent: Any | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> str:
        """Fetch ChEMBL compounds with judge behavior controlled by MCP LLM policy."""

        policy = normalize_llm_policy(getattr(agent, "llm_policy", "external"))
        use_internal_judge = policy == "agno-model" and getattr(agent, "model", None) is not None
        result = self._toolkit().fetch_compounds(
            query=query,
            organism=organism,
            assay_types=assay_types,
            mechanism=mechanism,
            enable_retrieval_judge=use_internal_judge,
            enable_metadata_judge=use_internal_judge,
            agent=agent,
            session_state=session_state,
        )
        if use_internal_judge or policy == "disabled":
            return result
        try:
            tasks = self._create_external_judge_tasks_from_current_dataset(
                target_query=query,
                organism_filter=organism,
                agent=agent,
                session_state=session_state,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to create external ChEMBL judge tasks: %s", exc)
            tasks = []
            task_note = (
                "Could not create follow-up external ChEMBL judge tasks from the "
                f"saved dataset: {exc}"
            )
        else:
            task_note = (
                "Created external ChEMBL judge task(s): "
                + ", ".join(task["task_id"] for task in tasks)
                if tasks
                else "No external ChEMBL judge task was needed for this result."
            )
        return (
            f"{result}\n\n"
            "MCP LLM policy: external. In-process ChEMBL judges were not run. "
            f"{task_note} Submit decisions with chembl_submit_external_judge_result "
            "or llm_submit_task_result."
        )

    def create_external_judge_task(
        self,
        judge_type: str,
        target_query: str,
        keywords: str | Sequence[str],
        items: str | Sequence[dict[str, Any]],
        organism_filter: Optional[str] = None,
        agent: Any | None = None,
    ) -> dict[str, Any]:
        """Create a pending external ChEMBL LLM-as-judge task."""

        broker = getattr(agent, "llm", None)
        if broker is None:
            raise MCPToolError("MCP LLM broker is not configured for this session.")

        normalized_type = str(judge_type or "").strip().lower().replace("-", "_")
        keywords_list = _as_keywords(keywords)
        judge_items = _as_items(items)
        if normalized_type in {"retrieval", "short_keyword", "short_keyword_retrieval"}:
            prompt_name = "chembl_retrieval_judge"
            task_type = "chembl.retrieval_judge"
            prompt = self._toolkit()._build_retrieval_judge_prompt(
                judge_items,
                target_query=target_query,
                organism_filter=organism_filter,
                keywords=keywords_list,
            )
        elif normalized_type in {"metadata", "target_metadata", "metadata_judge"}:
            prompt_name = "chembl_metadata_judge"
            task_type = "chembl.metadata_judge"
            prompt = self._toolkit()._build_metadata_judge_prompt(
                judge_items,
                target_query=target_query,
                organism_filter=organism_filter,
                keywords=keywords_list,
            )
        else:
            raise MCPToolError("judge_type must be one of: retrieval, short_keyword, metadata.")

        return broker.create_task(
            task_type=task_type,
            prompt_name=prompt_name,
            prompt_text=prompt,
            input_payload={
                "judge_type": normalized_type,
                "target_query": target_query,
                "keywords": keywords_list,
                "organism_filter": organism_filter,
                "items": judge_items,
            },
            output_schema=_CHEMBL_JUDGE_OUTPUT_SCHEMA,
            consumer_tool="chembl_fetch_compounds",
            metadata={
                "next_tools": [
                    "llm_get_task",
                    "chembl_submit_external_judge_result",
                    "llm_submit_task_result",
                ],
            },
        )

    def submit_external_judge_result(
        self,
        task_id: str,
        result: Any,
        expected_item_ids: Sequence[str] | None = None,
        agent: Any | None = None,
    ) -> dict[str, Any]:
        """Validate and submit ChEMBL external judge decisions for an LLM task."""

        broker = getattr(agent, "llm", None)
        if broker is None:
            raise MCPToolError("MCP LLM broker is not configured for this session.")
        parsed = self._toolkit()._parse_retrieval_judge_response(result)
        decisions = [decision.model_dump() for decision in parsed.decisions]
        if expected_item_ids is not None:
            expected = {str(item_id) for item_id in expected_item_ids}
            received = {str(decision["item_id"]) for decision in decisions}
            missing = sorted(expected - received)
            if missing:
                raise MCPToolError(f"Judge result omitted item ids: {missing}")
        return broker.submit_task_result(
            task_id=task_id,
            result={"decisions": decisions},
        )

    def _create_external_judge_tasks_from_current_dataset(
        self,
        *,
        target_query: str,
        organism_filter: Optional[str],
        agent: Any | None,
        session_state: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        state = session_state or getattr(agent, "session_state", None)
        if not isinstance(state, dict):
            return []
        record = self._current_dataset_record(state)
        if not record:
            return []
        summary = (
            record.get("standardization_summary", {}).get("chembl_retrieval_filtering", {})
            if isinstance(record.get("standardization_summary"), dict)
            else {}
        )
        if not isinstance(summary, dict):
            return []
        raw_dataset_path = record.get("raw_dataset_path")
        if not raw_dataset_path:
            return []

        needs_retrieval = (
            summary.get("judge_status") == "disabled"
            and int(summary.get("suspicious_row_count") or 0) > 0
        )
        needs_metadata = (
            summary.get("metadata_judge_status") == "disabled"
            and int(summary.get("metadata_judge_row_count") or 0) > 0
        )
        if not needs_retrieval and not needs_metadata:
            return []

        import pandas as pd

        from cs_copilot.storage import S3

        with S3.open(str(raw_dataset_path), "r") as handle:
            df = pd.read_csv(handle)

        keywords = record.get("query_keywords") or summary.get("query_keywords") or []
        tasks: list[dict[str, Any]] = []
        if needs_retrieval and "query_keywords" in df.columns:
            suspicious_df = df[df["query_keywords"].apply(self._toolkit()._short_keyword_only)]
            retrieval_items, _row_items = self._toolkit()._build_judge_items(
                suspicious_df,
                organism_filter,
            )
            if retrieval_items:
                tasks.append(
                    self.create_external_judge_task(
                        judge_type="retrieval",
                        target_query=target_query,
                        keywords=keywords,
                        organism_filter=organism_filter,
                        items=retrieval_items,
                        agent=agent,
                    )
                )

        if needs_metadata:
            metadata_items, _metadata_row_items = self._toolkit()._build_metadata_judge_items(df)
            if metadata_items:
                tasks.append(
                    self.create_external_judge_task(
                        judge_type="metadata",
                        target_query=target_query,
                        keywords=keywords,
                        organism_filter=organism_filter,
                        items=metadata_items,
                        agent=agent,
                    )
                )
        return tasks

    @staticmethod
    def _current_dataset_record(session_state: dict[str, Any]) -> dict[str, Any] | None:
        memory = session_state.get("session_objects")
        if not isinstance(memory, dict):
            return None
        current = memory.get("current")
        datasets = memory.get("datasets")
        if not isinstance(current, dict) or not isinstance(datasets, dict):
            return None
        dataset_id = current.get("dataset")
        record = datasets.get(dataset_id)
        return record if isinstance(record, dict) else None


def chembl_mcp_facade() -> ChemblMCPFacade:
    return ChemblMCPFacade()
