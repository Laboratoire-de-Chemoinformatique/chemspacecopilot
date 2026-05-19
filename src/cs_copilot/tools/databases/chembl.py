#!/usr/bin/env python
# coding: utf-8
"""
ChEMBL-specific database toolkit implementation.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
from agno.agent import Agent
from pydantic import BaseModel, Field

from cs_copilot.storage import S3, OutputOperation, scoped_artifact_path
from cs_copilot.tools.chemistry.activity_schema import build_compound_memory_preview
from cs_copilot.tools.chemistry.clean_dataset import prepare_clean_dataset
from cs_copilot.tools.chemistry.standardize import standardize_smiles_column
from cs_copilot.tools.io.session_memory import (
    register_session_object,
    update_state_targets,
)

from .base import BaseDatabaseToolkit, DatabaseError, NotFound, RateLimited, ValidationError
from .chembl_fetcher import ChemblDataFetcher, RestChemblFetcher, SqlChemblFetcher
from .types import DBConfig, PaginationMode, QueryParams, ResultPage

# Any run of hyphens or whitespace counts as a single token separator.
_HYPHEN_OR_SPACE_RUN = re.compile(r"[-\s]+")

# Letter→digit boundary (e.g. "CDK2" → "CDK|2", "PDE4A" → "PDE|4A").
# Only splits where a letter is immediately followed by a digit; the
# digit and any trailing letters stay together as one unit.
_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=[a-zA-Z])(?=\d)")


def _build_punctuation_regex(keyword: str) -> str | None:
    """Build a regex that matches all hyphen/space variants of *keyword*.

    Returns ``None`` when the keyword cannot be split into multiple
    parts, signalling the caller to use plain substring matching.

    **Two splitting passes**:

    1. Split on explicit hyphens/spaces (``_HYPHEN_OR_SPACE_RUN``).
    2. Sub-split each resulting token at letter→digit boundaries
       (``_LETTER_DIGIT_BOUNDARY``), so ``"CDK2"`` → ``["CDK", "2"]``
       and ``"PDE4A"`` → ``["PDE", "4A"]``.

    This means single-token abbreviations like ``CDK2`` now produce a
    regex (``CDK[- ]?2``) that matches ``CDK2``, ``CDK-2``, and
    ``CDK 2``.

    **Symmetry guarantee**: inputs that tokenize identically across both
    passes produce the same regex.

    For *n* ≤ 3 final tokens the separator is ``[- ]?`` (optional),
    covering the concat form (e.g. ``"CDK2"`` from ``"CDK-2"``).
    For *n* ≥ 4 tokens the separator is ``[- ]`` (required).

    Uses ``[- ]`` (literal hyphen + space) rather than ``[-\\s]`` for
    maximum SQL dialect compatibility.

    Each token is ``re.escape``'d to handle metacharacters.
    """
    if not keyword:
        return None
    base = keyword.strip()
    if not base:
        return None

    # Pass 1: split on explicit hyphens/spaces.
    raw_tokens = [t for t in _HYPHEN_OR_SPACE_RUN.split(base) if t]
    if not raw_tokens:
        return None

    # Pass 2: sub-split each token at letter→digit boundaries.
    tokens: list[str] = []
    for t in raw_tokens:
        parts = _LETTER_DIGIT_BOUNDARY.split(t)
        tokens.extend(p for p in parts if p)

    if len(tokens) <= 1:
        return None
    escaped = [re.escape(t) for t in tokens]
    sep = "[- ]?" if len(tokens) <= 3 else "[- ]"
    return sep.join(escaped)


logger = logging.getLogger(__name__)


class _ChemblJudgeDecision(BaseModel):
    item_id: str = Field(..., description="Exact item_id from the validation request.")
    keep: bool = Field(..., description="Whether rows represented by this item should be kept.")
    explanation: Optional[str] = Field(
        default=None,
        description="Brief reason for the keep/filter decision.",
    )


class _ChemblJudgeResponse(BaseModel):
    decisions: list[_ChemblJudgeDecision] = Field(
        default_factory=list,
        description="Validation decisions for suspicious short-keyword retrieval items.",
    )


@dataclass
class _ChemblRetrievalFilterResult:
    retained_df: pd.DataFrame
    filtered_df: pd.DataFrame
    filtered_rows_path: Optional[str]
    summary: Dict[str, Any]


class ChemblToolkit(BaseDatabaseToolkit):
    """
    ChEMBL-specific database toolkit implementation.

    Supports multiple SQL backends (SQLite, PostgreSQL, MySQL) and the ChEMBL
    REST API.  The backend is selected automatically based on environment
    configuration.
    """

    # ChEMBL-specific constants
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

    # Default field mappings for different ChEMBL resources
    ACTIVITY_FIELDS = [
        "activity_id",
        "assay_chembl_id",
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "activity_comment",
        "data_validity_comment",
        "potential_duplicate",
    ]

    MOLECULE_FIELDS = ["molecule_chembl_id", "canonical_smiles", "molecule_structures"]

    ASSAY_FIELDS = [
        "assay_chembl_id",
        "description",
        "assay_type",
        "assay_organism",
        "target_chembl_id",
        "target_pref_name",
        "target_type",
        "target_organism",
        "target_tax_id",
        "target_species_group_flag",
    ]

    TARGET_METADATA_COLUMNS = [
        "target_chembl_id",
        "target_pref_name",
        "target_type",
        "target_organism",
        "target_tax_id",
        "target_species_group_flag",
    ]

    def __init__(
        self,
        config: Optional[DBConfig] = None,
        backend: str = "auto",
        **toolkit_kwargs,
    ):
        """
        Initialize ChEMBL toolkit.

        Args:
            config: Database configuration (optional, uses defaults if not provided)
            backend: Data source backend. One of:
                - "auto": Auto-detect from env vars (SQLite > PostgreSQL > MySQL > REST).
                  If an optional SQL driver is missing, fall back to the next candidate.
                - "rest": Force REST API (chembl_webresource_client)
                - "mysql": Force MySQL database
                - "postgresql": Force PostgreSQL database
                - "sqlite": Force SQLite database
            **toolkit_kwargs: Additional arguments for Agno Toolkit
        """
        if backend == "auto":
            resolved, fetcher = self._create_auto_fetcher()
        else:
            resolved = self._resolve_backend(backend)
            fetcher = self._create_fetcher(resolved)

        self._active_backend = resolved
        is_sql = resolved in ("mysql", "postgresql", "sqlite")

        if config is None:
            config = DBConfig(
                uri=self.BASE_URL,
                timeout_s=30.0,
                retries=3,
                page_size=100,
                rate_limit=10.0,
                supports_sql=is_sql,
                supports_http_api=True,
                pagination_mode=PaginationMode.OFFSET_LIMIT,
            )

        _BACKEND_LABELS = {
            "mysql": "Connected to a LOCAL MySQL ChEMBL database",
            "postgresql": "Connected to a LOCAL PostgreSQL ChEMBL database",
            "sqlite": "Connected to a LOCAL SQLite ChEMBL database",
        }
        if "instructions" not in toolkit_kwargs:
            if resolved in _BACKEND_LABELS:
                backend_label = (
                    f"Active backend: {_BACKEND_LABELS[resolved]}. "
                    "The ChEMBL REST API is also available as a fallback."
                )
            else:
                backend_label = "Active backend: Using the ChEMBL REST API."
            toolkit_kwargs["instructions"] = (
                "ChEMBL database toolkit for fetching compound bioactivity data, "
                "generating EDA reports, and analyzing chemical datasets. "
                f"{backend_label}"
            )

        super().__init__(
            config,
            name="chembl_toolkit",
            **toolkit_kwargs,
        )
        self._client = None  # Will be initialized lazily
        self._client_init_error = None  # Store initialization error if any
        self._fetcher = fetcher
        self.register(self.fetch_compounds)
        self.register(self.describe_dataset)
        self.register(self.convert_to_chembl_query)

    def _ensure_client(self):
        """Lazy initialization of ChEMBL client."""
        if self._client is None and self._client_init_error is None:
            try:
                from chembl_webresource_client.new_client import new_client

                self._client = new_client
                logger.info("ChEMBL client initialized successfully")
            except Exception as e:
                self._client_init_error = e
                error_msg = f"Failed to initialize ChEMBL client: {e}. The ChEMBL API may be temporarily unavailable."
                logger.error(error_msg)
                raise DatabaseError(error_msg) from e
        elif self._client_init_error is not None:
            raise DatabaseError(f"ChEMBL client unavailable: {self._client_init_error}")
        return self._client

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        """Resolve 'auto' to a concrete backend name."""
        import os

        if backend == "auto":
            if os.getenv("CHEMBL_SQLITE_PATH"):
                return "sqlite"
            if os.getenv("CHEMBL_PG_HOST"):
                return "postgresql"
            if os.getenv("CHEMBL_MYSQL_HOST"):
                return "mysql"
            return "rest"
        valid = ("rest", "mysql", "postgresql", "sqlite")
        if backend in valid:
            return backend
        raise ValueError(
            f"Unknown ChEMBL backend: {backend!r}. Use one of: {', '.join(valid)}, or 'auto'."
        )

    @staticmethod
    def _auto_backend_candidates() -> list[str]:
        """Return backend candidates for auto-detection in priority order."""
        import os

        candidates: list[str] = []
        if os.getenv("CHEMBL_SQLITE_PATH"):
            candidates.append("sqlite")
        if os.getenv("CHEMBL_PG_HOST"):
            candidates.append("postgresql")
        if os.getenv("CHEMBL_MYSQL_HOST"):
            candidates.append("mysql")
        candidates.append("rest")
        return candidates

    def _create_auto_fetcher(self) -> tuple[str, ChemblDataFetcher]:
        """Create the first usable backend for auto mode.

        Optional SQL backends are tried in priority order. Missing database
        driver packages do not abort startup; they trigger a fallback to the
        next candidate, ending with the REST API.
        """
        for candidate in self._auto_backend_candidates():
            try:
                return candidate, self._create_fetcher(candidate)
            except ImportError as exc:
                logger.warning(
                    "ChEMBL backend %s unavailable during auto-detection (%s). "
                    "Trying the next backend.",
                    candidate,
                    exc,
                )

        # The REST fetcher is expected to be always constructible, so reaching
        # this point would indicate an internal bug rather than configuration.
        raise RuntimeError("Failed to create any ChEMBL backend during auto-detection.")

    def _create_fetcher(self, backend: str) -> ChemblDataFetcher:
        """Create the data fetcher for an already-resolved backend."""
        if backend == "mysql":
            return SqlChemblFetcher.from_mysql_env()
        if backend == "postgresql":
            return SqlChemblFetcher.from_postgres_env()
        if backend == "sqlite":
            return SqlChemblFetcher.from_sqlite()
        return RestChemblFetcher(self._ensure_client)

    def _get_resource_client(self, resource: str):
        client = self._ensure_client()
        try:
            return getattr(client, resource)
        except AttributeError as exc:
            raise ValidationError(f"Unsupported ChEMBL resource: {resource}") from exc

    def connect(self) -> None:
        """Connect to ChEMBL data source (validate access)."""
        try:
            self._fetcher.connect()
            self._connected = True
            logger.info("Connected to ChEMBL successfully")
        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect to ChEMBL: {e}")
            raise DatabaseError(f"ChEMBL connection failed: {e}") from e

    def ping(self) -> bool:
        """Test ChEMBL data source connectivity."""
        try:
            return self._fetcher.ping()
        except Exception as e:
            logger.warning(f"ChEMBL ping failed: {e}")
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Get information about database capabilities including the active backend."""
        caps = super().get_capabilities()
        caps["active_backend"] = self._active_backend
        return caps

    def query(self, params: QueryParams) -> ResultPage:
        """
        Execute a single-page query using the ChEMBL webresource client.

        The implementation mirrors the readable style from the official
        `demo_wrc.ipynb` notebook: pick a resource, apply filters, optionally
        restrict fields, and slice results with Pythonic indexing.
        """

        start_time = time.perf_counter()
        resource_name = params.extra_params.get("resource", "activity")
        limit = params.limit or self.config.page_size
        offset = params.offset or 0

        try:
            resource = self._get_resource_client(resource_name)

            # 1) Build the query using provided filters
            query = resource.filter(**(params.filters or {}))

            # 2) Keep only requested fields when provided
            if params.fields:
                query = query.only(list(params.fields))

            # 3) Apply offset via ``skip`` then pull a single page just like the
            #    notebook's ``activities[:20]`` pattern
            if offset:
                query = query.skip(offset)

            raw_records = list(query[:limit])
            mapped_records = [self.map_fields(record) for record in raw_records]

            has_more = len(raw_records) == limit
            next_offset = offset + limit if has_more else None

            query_time_ms = (time.perf_counter() - start_time) * 1000
            return ResultPage(
                records=mapped_records,
                total=None,
                next_offset=next_offset,
                has_more=has_more,
                query_time_ms=query_time_ms,
            )

        except Exception as exc:
            logger.error(f"ChEMBL query failed: {exc}")
            raise self.handle_error(exc) from exc

    def map_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map ChEMBL-specific field names to standard names.

        Args:
            record: Raw ChEMBL record

        Returns:
            Record with mapped field names
        """
        # ChEMBL records are generally well-structured, but we might want
        # to flatten some nested structures
        mapped = record.copy()

        # Handle nested molecule structures
        if "molecule_structures" in mapped and mapped["molecule_structures"]:
            structures = mapped["molecule_structures"]
            if isinstance(structures, dict) and "canonical_smiles" in structures:
                mapped["canonical_smiles"] = structures["canonical_smiles"]

        return mapped

    def handle_error(self, error: Exception) -> DatabaseError:
        """Map ChEMBL-specific errors to standard error types."""
        error_str = str(error).lower()

        if "timeout" in error_str or "timed out" in error_str:
            return RateLimited(f"ChEMBL API timeout: {error}")
        elif "not found" in error_str or "404" in error_str:
            return NotFound(f"ChEMBL resource not found: {error}")
        elif "rate limit" in error_str or "429" in error_str:
            return RateLimited(f"ChEMBL rate limit exceeded: {error}")
        elif "invalid" in error_str or "400" in error_str:
            return ValidationError(f"Invalid ChEMBL query: {error}")
        else:
            return DatabaseError(f"ChEMBL API error: {error}")

    def _normalize_assay_types(self, assay_types: Optional[Sequence[str]]) -> list:
        """Normalize human-readable assay types to ChEMBL codes."""

        if assay_types is None:
            return []

        if isinstance(assay_types, str):
            values = [assay_types]
        else:
            values = list(assay_types)

        mapping = {
            "binding": "B",
            "functional": "F",
            "admet": "A",
        }

        normalized = []
        for raw in values:
            if raw is None:
                continue
            item = str(raw).strip()
            if not item:
                continue

            lowered = item.lower()
            if lowered in mapping:
                normalized.append(mapping[lowered])
            elif item.upper() in mapping.values():
                normalized.append(item.upper())
            else:
                valid = ", ".join(sorted({*mapping.keys(), *mapping.values()}))
                raise ValueError(f"Invalid assay type '{item}'. Expected one of: {valid}.")

        # Remove duplicates while preserving order
        seen = set()
        ordered = []
        for code in normalized:
            if code not in seen:
                ordered.append(code)
                seen.add(code)

        return ordered

    def fetch_compounds(
        self,
        query: str = "bioactivity data",
        organism: Optional[str] = None,
        assay_types: Optional[Sequence[str]] = None,
        mechanism: Optional[str] = None,
        agent: Optional[Agent] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Fetch compound bioactivity data from ChEMBL database using multiple keywords.

        Args:
            query: Search term(s) for assay descriptions. Can be:
                - A single string (e.g., "kinase")
                - A comma-separated string (e.g., "kinase, BRAF, PPAR-alpha")
                - Multiple queries will be processed separately and merged
            organism: Optional organism filter (e.g., "Homo sapiens", "Influenza A")
            assay_types: Optional list of assay types to keep. MUST be a list, not a dict.
                Examples:
                - ["B", "F"] for binding and functional assays (default)
                - ["binding", "functional"] (human-friendly names)
                - [] to allow all assay types (empty list, not empty dict)
                - None to use default (binding + functional)
                Common mistake: Do NOT use {} (empty dict), use [] (empty list) instead.
            mechanism: Optional mechanism of action filter. When provided, only assays
                whose description contains this term (case-insensitive) are kept.
                Examples: "agonist", "antagonist", "modulator", "inverse agonist", None.
            session_state: Shared session state auto-injected by Agno.

        Returns:
            Status message with information about fetched data

        Raises:
            ValueError: If query is empty or invalid
            DatabaseError: If ChEMBL API request fails
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        if mechanism is not None:
            if not isinstance(mechanism, str):
                raise ValueError("mechanism must be a string or None")
            mechanism = mechanism.strip() or None

        # Handle LLM confusion: convert empty dict {} to empty list []
        if isinstance(assay_types, dict):
            if len(assay_types) == 0:
                assay_types = []
            else:
                raise ValueError(
                    "assay_types must be a list/sequence, not a dict. "
                    "Use a list like ['B', 'F'] or [] for all types."
                )

        default_assay_types = ["B", "F"] if assay_types is None else assay_types
        assay_type_codes = self._normalize_assay_types(default_assay_types)

        # Parse multiple keywords from query string
        raw_keywords = [kw.strip() for kw in query.split(",") if kw.strip()]
        if not raw_keywords:
            raise ValueError("query must contain at least one non-empty keyword")

        keywords: list[str] = raw_keywords

        logger.info(f"Fetching ChEMBL compounds for {len(keywords)} keyword(s): {keywords}")

        all_dataframes = []
        all_assay_ids = set()  # Track unique assays across all keywords

        try:
            # Process each keyword separately
            for keyword in keywords:
                logger.info(f"Processing keyword: '{keyword}'")

                # Step 1: Fetch assays for this keyword
                regex = _build_punctuation_regex(keyword)
                if regex:
                    logger.debug(f"Using regex for '{keyword}': {regex}")

                assays = self._fetcher.fetch_assays(keyword, organism, regex_pattern=regex)

                if mechanism:
                    mechanism_lower = mechanism.lower()
                    before = len(assays)
                    assays = [
                        a for a in assays if mechanism_lower in (a.get("description") or "").lower()
                    ]
                    logger.info(
                        f"Mechanism filter '{mechanism}' reduced assays from {before} "
                        f"to {len(assays)} for keyword '{keyword}'"
                    )

                assay_ids = [assay["assay_chembl_id"] for assay in assays]

                if not assay_ids:
                    logger.warning(f"No assays found for keyword: {keyword}")
                    continue

                logger.info(f"Found {len(assay_ids)} assays for keyword: '{keyword}'")
                all_assay_ids.update(assay_ids)

                # Step 2: Fetch activities for these assays
                logger.debug(f"Fetching activities from ChEMBL for keyword: '{keyword}'")

                activities = self._fetcher.fetch_activities(
                    assay_ids, assay_type_codes, self.ACTIVITY_FIELDS
                )

                if not activities:
                    logger.warning(f"No activity data found for keyword: {keyword}")
                    continue

                logger.info(
                    f"Retrieved {len(activities)} activity records for keyword: '{keyword}'"
                )

                # Step 3: Convert to DataFrame and add SMILES
                df = self.to_dataframe(activities)
                df = self._merge_assay_data(df, assays)
                df["smi"] = df["canonical_smiles"]  # Use standard 'smi' column name
                df["query_keywords"] = keyword
                all_dataframes.append(df)

            # Step 4: Merge all DataFrames and remove duplicates
            if not all_dataframes:
                return f"No data found for any of the keywords: {keywords}"

            logger.info(f"Merging {len(all_dataframes)} datasets and removing duplicates")
            merged_df = pd.concat(all_dataframes, ignore_index=True)

            # Determine dedup key
            if "activity_id" in merged_df.columns:
                dedup_key = ["activity_id"]
            else:
                dedup_key = ["molecule_chembl_id", "assay_chembl_id"]
                if "standard_value" in merged_df.columns:
                    dedup_key.append("standard_value")

            # Aggregate query_keywords for duplicate rows (multi-keyword only)
            if len(keywords) > 1:
                keywords_agg = (
                    merged_df.groupby(dedup_key, sort=False)["query_keywords"]
                    .apply(lambda s: "|".join(sorted(s.unique())))
                    .reset_index()
                )
                merged_df = merged_df.drop(columns=["query_keywords"]).merge(
                    keywords_agg, on=dedup_key, how="left"
                )

            # Remove duplicates
            initial_count = len(merged_df)
            merged_df = merged_df.drop_duplicates(subset=dedup_key, keep="first")

            duplicates_removed = initial_count - len(merged_df)
            logger.info(
                f"Removed {duplicates_removed} duplicate records. Final dataset: {len(merged_df)} records"
            )

            query_slug = "_".join(
                [kw.replace(" ", "_") for kw in keywords[:3]]
            )  # Limit to first 3 keywords for filename
            if len(keywords) > 3:
                query_slug += "_and_more"
            if agent is not None and not isinstance(getattr(agent, "session_state", None), dict):
                agent.session_state = {}
            session_for_artifacts = session_state or getattr(agent, "session_state", None)

            filtering = self._filter_suspicious_short_keyword_rows(
                merged_df,
                keywords=keywords,
                target_query=query,
                organism_filter=organism,
                query_slug=query_slug,
                agent=agent,
                session_state=session_for_artifacts,
            )
            merged_df = filtering.retained_df

            if merged_df.empty:
                filtering_report_path = self._write_retrieval_filtering_only_report(
                    query_slug,
                    filtering.summary,
                    session_for_artifacts,
                )
                for state in update_state_targets(agent, session_state):
                    state.setdefault("data_file_paths", {})
                    state["data_file_paths"]["filtered_dataset_path"] = filtering.filtered_rows_path
                    state["data_file_paths"]["standardization_report_path"] = filtering_report_path
                return self._format_all_rows_filtered_message(
                    keywords,
                    filtering.summary,
                    filtering.filtered_rows_path,
                    filtering_report_path,
                )

            # Step 5: Prepare raw, clean, descriptor, and report artifacts.
            prepared = prepare_clean_dataset(
                merged_df,
                source_name=f"chembl_{query_slug}",
                smiles_column="smi",
                raw_filename=f"chembl_{query_slug}_raw.csv",
                clean_filename=f"chembl_{query_slug}_clean.csv",
                descriptor_filename=f"chembl_{query_slug}_descriptors.parquet",
                report_filename=f"chembl_{query_slug}_standardization_report.md",
                session_state=session_for_artifacts,
            )
            prepared.standardization_summary["chembl_retrieval_filtering"] = filtering.summary
            self._append_retrieval_filtering_report(
                prepared.standardization_report_path,
                filtering.summary,
            )
            total_assays = len(all_assay_ids)
            clean_df = prepared.clean_df
            activity_mapping = prepared.activity_mapping.to_dict()
            final_activity_mapping = prepared.final_activity_mapping.to_dict()

            # Store dataset path and compact memory objects for cross-agent access.
            for state in update_state_targets(agent, session_state):
                if "data_file_paths" not in state:
                    state["data_file_paths"] = {}
                state["data_file_paths"]["raw_dataset_path"] = prepared.raw_dataset_path
                state["data_file_paths"]["clean_dataset_path"] = prepared.clean_dataset_path
                state["data_file_paths"][
                    "descriptor_parquet_path"
                ] = prepared.descriptor_parquet_path
                state["data_file_paths"][
                    "standardization_report_path"
                ] = prepared.standardization_report_path
                if filtering.filtered_rows_path:
                    state["data_file_paths"]["filtered_dataset_path"] = filtering.filtered_rows_path
                state["data_file_paths"]["dataset_path"] = prepared.clean_dataset_path
                dataset_id = register_session_object(
                    state,
                    "dataset",
                    {
                        "dataset_path": prepared.clean_dataset_path,
                        "raw_dataset_path": prepared.raw_dataset_path,
                        "clean_dataset_path": prepared.clean_dataset_path,
                        "descriptor_parquet_path": prepared.descriptor_parquet_path,
                        "standardization_report_path": prepared.standardization_report_path,
                        "filtered_dataset_path": filtering.filtered_rows_path,
                        "query_keywords": keywords,
                        "row_count": int(len(clean_df)),
                        "raw_row_count": int(len(merged_df)),
                        "retrieved_raw_row_count": int(filtering.summary["retrieved_row_count"]),
                        "retrieval_filtered_row_count": int(
                            filtering.summary["filtered_row_count"]
                        ),
                        "unique_compounds": int(clean_df["smiles"].nunique()),
                        "assay_count": total_assays,
                        "organism_filter": organism,
                        "assay_type_codes": assay_type_codes,
                        "mechanism_filter": mechanism,
                        "activity_mapping": activity_mapping,
                        "final_activity_mapping": final_activity_mapping,
                        "activity_merge_policy": prepared.standardization_summary[
                            "activity_merge_policy"
                        ],
                        "stereochemistry_removed": prepared.standardization_summary[
                            "stereochemistry_removed"
                        ],
                        "standardization_summary": prepared.standardization_summary,
                    },
                    label=f"ChEMBL dataset: {', '.join(keywords[:3])}",
                    source_agent=getattr(agent, "name", None),
                    source_tool="fetch_compounds",
                    set_current=True,
                )
                for idx, compound in enumerate(
                    build_compound_memory_preview(clean_df, prepared.final_activity_mapping),
                    start=1,
                ):
                    register_session_object(
                        state,
                        "compound",
                        {**compound, "related": {"dataset_id": dataset_id}},
                        label=f"ChEMBL compound {idx}",
                        source_agent=getattr(agent, "name", None),
                        source_tool="fetch_compounds",
                        set_current=False,
                    )
                logger.info(
                    "Stored clean dataset_path in session_state: %s",
                    prepared.clean_dataset_path,
                )

            return self._format_success_message(
                clean_df,
                keywords,
                prepared.clean_dataset_path,
                total_assays,
                duplicates_removed,
                organism_filter=organism,
                assay_type_codes=assay_type_codes,
                mechanism_filter=mechanism,
                raw_dataset_path=prepared.raw_dataset_path,
                descriptor_parquet_path=prepared.descriptor_parquet_path,
                standardization_report_path=prepared.standardization_report_path,
                standardization_summary=prepared.standardization_summary,
                retrieval_filtering_summary=filtering.summary,
            )

        except Exception as e:
            logger.error(f"Error fetching ChEMBL compounds: {e}")
            raise

    def _filter_suspicious_short_keyword_rows(
        self,
        df: pd.DataFrame,
        *,
        keywords: Sequence[str],
        target_query: str,
        organism_filter: Optional[str],
        query_slug: str,
        agent: Optional[Agent],
        session_state: Optional[Dict[str, Any]],
    ) -> _ChemblRetrievalFilterResult:
        """Filter suspicious short-keyword hits and incorrect populated target metadata."""
        base_summary: Dict[str, Any] = {
            "enabled": True,
            "short_keyword_threshold": 5,
            "fallback_policy": "filter_rows",
            "metadata_fallback_policy": "keep_rows",
            "retrieved_row_count": int(len(df)),
            "suspicious_row_count": 0,
            "metadata_judge_row_count": 0,
            "metadata_missing_decision_count": 0,
            "filtered_row_count": 0,
            "metadata_filtered_row_count": 0,
            "retained_row_count": int(len(df)),
            "filtered_rows_path": None,
            "judge_status": "not_needed",
            "metadata_judge_status": "not_needed",
            "query_keywords": list(keywords),
            "organism_filter": organism_filter,
        }
        if df.empty:
            return _ChemblRetrievalFilterResult(
                retained_df=df.copy(),
                filtered_df=df.iloc[0:0].copy(),
                filtered_rows_path=None,
                summary=base_summary,
            )

        if "query_keywords" in df.columns:
            suspicious_mask = df["query_keywords"].apply(self._short_keyword_only)
            suspicious_df = df[suspicious_mask].copy()
        else:
            suspicious_df = df.iloc[0:0].copy()
        base_summary["suspicious_row_count"] = int(len(suspicious_df))
        filtered_metadata: Dict[Any, Dict[str, Any]] = {}

        if not suspicious_df.empty:
            judge_items, row_items = self._build_judge_items(suspicious_df, organism_filter)

            missing_basis_indices = set(suspicious_df.index) - set(row_items)
            for row_index in missing_basis_indices:
                filtered_metadata[row_index] = {
                    "filter_reason": "missing_judge_basis",
                    "judge_basis": None,
                    "judge_value": None,
                    "judge_decision": "filter",
                    "judge_explanation": (
                        "No target preferred name, organism, or assay description was available "
                        "to validate a short-keyword hit."
                    ),
                }

            decisions: Dict[str, _ChemblJudgeDecision] = {}
            if judge_items:
                try:
                    decisions = self._run_chembl_retrieval_judge(
                        judge_items,
                        target_query=target_query,
                        organism_filter=organism_filter,
                        keywords=keywords,
                        agent=agent,
                    )
                    base_summary["judge_status"] = "completed"
                except Exception as exc:
                    logger.warning("ChEMBL short-keyword judge unavailable: %s", exc)
                    base_summary["judge_status"] = "unavailable"
                    for row_index, item in row_items.items():
                        filtered_metadata[row_index] = {
                            "filter_reason": "judge_unavailable",
                            "judge_basis": item["judge_basis"],
                            "judge_value": item["value"],
                            "judge_decision": "filter",
                            "judge_explanation": (
                                "Short-keyword hit could not be validated because the judge "
                                "was unavailable or returned invalid output."
                            ),
                        }

            for row_index, item in row_items.items():
                if row_index in filtered_metadata:
                    continue
                decision = decisions.get(item["item_id"])
                if decision is None:
                    filtered_metadata[row_index] = {
                        "filter_reason": "judge_unavailable",
                        "judge_basis": item["judge_basis"],
                        "judge_value": item["value"],
                        "judge_decision": "filter",
                        "judge_explanation": (
                            "Short-keyword hit did not receive a valid judge decision."
                        ),
                    }
                    continue
                if not decision.keep:
                    filtered_metadata[row_index] = {
                        "filter_reason": "llm_judge_rejected",
                        "judge_basis": item["judge_basis"],
                        "judge_value": item["value"],
                        "judge_decision": "filter",
                        "judge_explanation": decision.explanation,
                    }

        metadata_candidates = df.drop(index=list(filtered_metadata), errors="ignore")
        metadata_items, metadata_row_items = self._build_metadata_judge_items(metadata_candidates)
        base_summary["metadata_judge_row_count"] = int(len(metadata_row_items))

        metadata_decisions: Dict[str, _ChemblJudgeDecision] = {}
        if metadata_items:
            expected_metadata_item_ids = {item["item_id"] for item in metadata_items}
            try:
                metadata_decisions = self._run_chembl_metadata_judge(
                    metadata_items,
                    target_query=target_query,
                    organism_filter=organism_filter,
                    keywords=keywords,
                    agent=agent,
                )
                missing_metadata_item_ids = expected_metadata_item_ids - set(metadata_decisions)
                base_summary["metadata_missing_decision_count"] = len(missing_metadata_item_ids)
                base_summary["metadata_judge_status"] = (
                    "partial" if missing_metadata_item_ids else "completed"
                )
            except Exception as exc:
                logger.warning("ChEMBL target metadata judge unavailable: %s", exc)
                base_summary["metadata_judge_status"] = "unavailable"

        metadata_filtered_count = 0
        for row_index, item in metadata_row_items.items():
            if row_index in filtered_metadata:
                continue
            decision = metadata_decisions.get(item["item_id"])
            if decision is None:
                continue
            if not decision.keep:
                filtered_metadata[row_index] = {
                    "filter_reason": "metadata_llm_judge_rejected",
                    "judge_basis": item["judge_basis"],
                    "judge_value": item["value"],
                    "judge_decision": "filter",
                    "judge_explanation": decision.explanation,
                }
                metadata_filtered_count += 1

        filtered_indices = list(filtered_metadata)
        if filtered_indices:
            filtered_df = df.loc[filtered_indices].copy()
            for column in (
                "filter_reason",
                "judge_basis",
                "judge_value",
                "judge_decision",
                "judge_explanation",
            ):
                filtered_df[column] = [
                    filtered_metadata[row_index].get(column) for row_index in filtered_indices
                ]
            retained_df = df.drop(index=filtered_indices).copy()
        else:
            filtered_df = df.iloc[0:0].copy()
            retained_df = df.copy()

        filtered_rows_path = self._save_filtered_rows(
            filtered_df,
            query_slug=query_slug,
            session_state=session_state,
        )
        base_summary.update(
            {
                "filtered_row_count": int(len(filtered_df)),
                "metadata_filtered_row_count": metadata_filtered_count,
                "retained_row_count": int(len(retained_df)),
                "filtered_rows_path": filtered_rows_path,
                "decision_counts": (
                    filtered_df["filter_reason"].value_counts().to_dict()
                    if not filtered_df.empty and "filter_reason" in filtered_df.columns
                    else {}
                ),
            }
        )
        return _ChemblRetrievalFilterResult(
            retained_df=retained_df,
            filtered_df=filtered_df,
            filtered_rows_path=filtered_rows_path,
            summary=base_summary,
        )

    @staticmethod
    def _is_short_keyword(keyword: Any) -> bool:
        return len(str(keyword or "").strip()) < 5

    def _short_keyword_only(self, query_keywords: Any) -> bool:
        keywords = [kw.strip() for kw in str(query_keywords or "").split("|") if kw.strip()]
        return bool(keywords) and all(self._is_short_keyword(keyword) for keyword in keywords)

    def _build_judge_items(
        self,
        suspicious_df: pd.DataFrame,
        organism_filter: Optional[str],
    ) -> tuple[list[Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
        items_by_key: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        row_items: Dict[Any, Dict[str, Any]] = {}

        for row_index, row in suspicious_df.iterrows():
            scope = self._infer_judge_scope(row, organism_filter)
            basis_column = self._judge_basis_column(row, scope)
            basis_value = self._clean_cell_value(row.get(basis_column))
            if not basis_column or not basis_value:
                continue

            key = (scope, basis_column, basis_value)
            item = items_by_key.get(key)
            if item is None:
                item = {
                    "item_id": f"item_{len(items_by_key) + 1}",
                    "judge_scope": scope,
                    "judge_basis": basis_column,
                    "value": basis_value,
                    "row_count": 0,
                    "assay_chembl_ids": [],
                    "sample_descriptions": [],
                }
                items_by_key[key] = item

            item["row_count"] += 1
            assay_id = self._clean_cell_value(row.get("assay_chembl_id"))
            if assay_id and assay_id not in item["assay_chembl_ids"]:
                item["assay_chembl_ids"].append(assay_id)
            description = self._clean_cell_value(row.get("description"))
            if description and description not in item["sample_descriptions"]:
                item["sample_descriptions"].append(description)
            item["assay_chembl_ids"] = item["assay_chembl_ids"][:10]
            item["sample_descriptions"] = item["sample_descriptions"][:3]
            row_items[row_index] = item

        return list(items_by_key.values()), row_items

    def _build_metadata_judge_items(
        self,
        df: pd.DataFrame,
    ) -> tuple[list[Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
        items_by_key: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
        row_items: Dict[Any, Dict[str, Any]] = {}

        for row_index, row in df.iterrows():
            target_type = self._clean_cell_value(row.get("target_type"))
            target_pref_name = self._clean_cell_value(row.get("target_pref_name"))
            target_organism = self._clean_cell_value(row.get("target_organism"))

            fields_to_validate = []
            if target_pref_name and self._is_protein_target_row(row):
                fields_to_validate.append("target_pref_name")
            if target_organism:
                fields_to_validate.append("target_organism")
            if not fields_to_validate:
                continue

            query_keywords = self._clean_cell_value(row.get("query_keywords"))
            key = (target_type, target_pref_name, target_organism, query_keywords)
            item = items_by_key.get(key)
            if item is None:
                item = {
                    "item_id": f"metadata_item_{len(items_by_key) + 1}",
                    "judge_scope": "target_metadata",
                    "judge_basis": "|".join(fields_to_validate),
                    "fields_to_validate": fields_to_validate,
                    "target_type": target_type or None,
                    "target_pref_name": target_pref_name or None,
                    "target_organism": target_organism or None,
                    "query_keywords": query_keywords or None,
                    "value": self._format_metadata_judge_value(
                        fields_to_validate,
                        target_pref_name,
                        target_organism,
                    ),
                    "row_count": 0,
                    "assay_chembl_ids": [],
                    "sample_descriptions": [],
                }
                items_by_key[key] = item

            item["row_count"] += 1
            assay_id = self._clean_cell_value(row.get("assay_chembl_id"))
            if assay_id and assay_id not in item["assay_chembl_ids"]:
                item["assay_chembl_ids"].append(assay_id)
            description = self._clean_cell_value(row.get("description"))
            if description and description not in item["sample_descriptions"]:
                item["sample_descriptions"].append(description)
            item["assay_chembl_ids"] = item["assay_chembl_ids"][:10]
            item["sample_descriptions"] = item["sample_descriptions"][:3]
            row_items[row_index] = item

        return list(items_by_key.values()), row_items

    def _infer_judge_scope(self, row: pd.Series, organism_filter: Optional[str]) -> str:
        target_type = self._clean_cell_value(row.get("target_type")).lower()
        if any(token in target_type for token in ("organism", "cell-line", "cell line", "tissue")):
            return "organism"
        if "protein" in target_type or self._clean_cell_value(row.get("target_pref_name")):
            return "protein"
        if organism_filter and any(
            token in str(organism_filter).lower()
            for token in ("virus", "viral", "hiv", "influenza", "sars", "coronavirus")
        ):
            return "organism"
        return "protein"

    def _judge_basis_column(self, row: pd.Series, scope: str) -> Optional[str]:
        if scope == "organism":
            for column in ("target_organism", "assay_organism", "description"):
                if self._clean_cell_value(row.get(column)):
                    return column
            return None

        if self._clean_cell_value(row.get("target_pref_name")):
            return "target_pref_name"
        if self._clean_cell_value(row.get("description")):
            return "description"
        return None

    def _is_protein_target_row(self, row: pd.Series) -> bool:
        target_type = self._clean_cell_value(row.get("target_type")).lower()
        if "protein" in target_type:
            return True
        if target_type:
            return False
        return bool(self._clean_cell_value(row.get("target_pref_name")))

    @staticmethod
    def _format_metadata_judge_value(
        fields_to_validate: Sequence[str],
        target_pref_name: str,
        target_organism: str,
    ) -> str:
        values = []
        if "target_pref_name" in fields_to_validate:
            values.append(f"target_pref_name={target_pref_name}")
        if "target_organism" in fields_to_validate:
            values.append(f"target_organism={target_organism}")
        return "; ".join(values)

    @staticmethod
    def _clean_cell_value(value: Any) -> str:
        if value is None:
            return ""
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "null"}:
            return ""
        return text

    def _run_chembl_retrieval_judge(
        self,
        judge_items: list[Dict[str, Any]],
        *,
        target_query: str,
        organism_filter: Optional[str],
        keywords: Sequence[str],
        agent: Optional[Agent],
    ) -> Dict[str, _ChemblJudgeDecision]:
        model = getattr(agent, "model", None)
        if model is None:
            raise RuntimeError("No agent model is available for ChEMBL short-keyword judging.")

        prompt = self._build_retrieval_judge_prompt(
            judge_items,
            target_query=target_query,
            organism_filter=organism_filter,
            keywords=keywords,
        )
        judge = Agent(
            model=model,
            name="chembl_short_keyword_retrieval_judge",
            description="Validate suspicious ChEMBL rows retrieved by short search keywords.",
            instructions=[
                "Decide whether each item is consistent with the requested ChEMBL target.",
                "Return one decision for every provided item_id.",
                "Reject unrelated proteins, organisms, assay systems, and generic short-string hits.",
                "Keep only direct matches, accepted synonyms, orthologs within the requested context, "
                "or organism strain/parent-name matches.",
            ],
            output_schema=_ChemblJudgeResponse,
            structured_outputs=True,
            use_json_mode=True,
            markdown=False,
            telemetry=False,
        )
        response = judge.run(prompt, stream=False)
        parsed = self._parse_retrieval_judge_response(response.content)
        decisions = {decision.item_id: decision for decision in parsed.decisions}
        missing = {item["item_id"] for item in judge_items} - set(decisions)
        if missing:
            raise ValueError(f"Judge omitted decisions for item ids: {sorted(missing)}")
        return decisions

    def _run_chembl_metadata_judge(
        self,
        judge_items: list[Dict[str, Any]],
        *,
        target_query: str,
        organism_filter: Optional[str],
        keywords: Sequence[str],
        agent: Optional[Agent],
    ) -> Dict[str, _ChemblJudgeDecision]:
        model = getattr(agent, "model", None)
        if model is None:
            raise RuntimeError("No agent model is available for ChEMBL metadata judging.")

        prompt = self._build_metadata_judge_prompt(
            judge_items,
            target_query=target_query,
            organism_filter=organism_filter,
            keywords=keywords,
        )
        judge = Agent(
            model=model,
            name="chembl_target_metadata_judge",
            description="Validate populated ChEMBL target metadata against the requested target.",
            instructions=[
                "Decide whether each ChEMBL target metadata item is consistent with the request.",
                "Return one decision for every provided item_id.",
                "Reject rows with populated target metadata that clearly refers to the wrong "
                "protein target or organism.",
                "Keep rows when the populated metadata is consistent, broadly compatible, "
                "or the request does not specify enough context to identify a conflict.",
            ],
            output_schema=_ChemblJudgeResponse,
            structured_outputs=True,
            use_json_mode=True,
            markdown=False,
            telemetry=False,
        )
        response = judge.run(prompt, stream=False)
        parsed = self._parse_retrieval_judge_response(response.content)
        decisions = {decision.item_id: decision for decision in parsed.decisions}
        missing = {item["item_id"] for item in judge_items} - set(decisions)
        if missing:
            logger.warning(
                "ChEMBL metadata judge omitted decisions for item ids: %s; "
                "keeping rows represented by omitted items.",
                sorted(missing),
            )
        return decisions

    @staticmethod
    def _build_retrieval_judge_prompt(
        judge_items: list[Dict[str, Any]],
        *,
        target_query: str,
        organism_filter: Optional[str],
        keywords: Sequence[str],
    ) -> str:
        return (
            "Validate ChEMBL activity rows that were retrieved only through short "
            "search keywords, which are prone to false substring matches.\n"
            f"Original target/query context: {target_query}\n"
            f"Search keywords: {', '.join(keywords)}\n"
            f"Organism filter: {organism_filter or 'none'}\n\n"
            "For protein scope, keep target preferred names only when they are the "
            "requested protein, a direct synonym, or a clear ortholog in context. "
            "If the basis is an assay description, keep it only when the description "
            "clearly names the requested protein/target.\n"
            "For organism scope, keep organism names only when they match the requested "
            "organism, virus, strain, or accepted parent/synonym. If the basis is an "
            "assay description, keep it only when the description clearly refers to the "
            "requested organism.\n\n"
            "Return structured decisions with item_id, keep, and explanation.\n"
            f"Items:\n{json.dumps(judge_items, indent=2, sort_keys=True)}"
        )

    @staticmethod
    def _build_metadata_judge_prompt(
        judge_items: list[Dict[str, Any]],
        *,
        target_query: str,
        organism_filter: Optional[str],
        keywords: Sequence[str],
    ) -> str:
        return (
            "Validate populated ChEMBL target metadata before molecular standardization.\n"
            f"Original target/query context: {target_query}\n"
            f"Search keywords: {', '.join(keywords)}\n"
            f"Organism filter: {organism_filter or 'none'}\n\n"
            "Judge only the fields listed in fields_to_validate for each item. Rows where "
            "target_pref_name or target_organism is empty are intentionally not included.\n"
            "For target_pref_name, validate protein targets: keep direct target matches, "
            "accepted synonyms, clear orthologs in context, and broad family members when "
            "the request asks for a family. Reject clearly different proteins or targets.\n"
            "For target_organism, reject only when the populated organism conflicts with "
            "an organism, virus, strain, species, or host explicitly requested in the query "
            "or organism filter. If no organism scope is requested or inferable, keep it.\n"
            "If any populated field being validated is clearly incorrect, set keep=false. "
            "Otherwise set keep=true.\n\n"
            "Return structured decisions with item_id, keep, and explanation.\n"
            f"Items:\n{json.dumps(judge_items, indent=2, sort_keys=True)}"
        )

    @staticmethod
    def _parse_retrieval_judge_response(content: Any) -> _ChemblJudgeResponse:
        if isinstance(content, _ChemblJudgeResponse):
            return content
        if isinstance(content, dict):
            return _ChemblJudgeResponse.model_validate(content)
        if hasattr(content, "model_dump"):
            return _ChemblJudgeResponse.model_validate(content.model_dump())
        if isinstance(content, str):
            text = content.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text).strip()
            try:
                return _ChemblJudgeResponse.model_validate_json(text)
            except Exception:
                return _ChemblJudgeResponse.model_validate(json.loads(text))
        raise ValueError(f"Unsupported judge response type: {type(content)!r}")

    def _save_filtered_rows(
        self,
        filtered_df: pd.DataFrame,
        *,
        query_slug: str,
        session_state: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if filtered_df.empty:
            return None
        filename = f"chembl_{query_slug}_suspicious_filtered.csv"
        rel_path = scoped_artifact_path(
            filename,
            OutputOperation.CHEMICAL_SPACE,
            "datasets",
            "filtered",
            session_state=session_state,
            workflow_slug="chemical_space",
        )
        with S3.open(rel_path, "w") as handle:
            filtered_df.to_csv(handle, index=False)
        return S3.path(rel_path)

    def _append_retrieval_filtering_report(
        self,
        report_path: str,
        filtering_summary: Dict[str, Any],
    ) -> None:
        if not filtering_summary:
            return
        if (
            filtering_summary.get("suspicious_row_count", 0) == 0
            and filtering_summary.get("metadata_judge_row_count", 0) == 0
            and filtering_summary.get("filtered_row_count", 0) == 0
        ):
            return
        section = self._format_retrieval_filtering_report_section(filtering_summary)
        try:
            if self._is_remote_or_explicit_path(report_path):
                try:
                    with S3.open(report_path, "r") as handle:
                        existing = handle.read()
                except Exception:
                    existing = ""
                if not isinstance(existing, str):
                    existing = ""
                with S3.open(report_path, "w") as handle:
                    handle.write(existing.rstrip() + "\n\n" + section)
            else:
                path = Path(report_path)
                existing = path.read_text() if path.exists() else ""
                path.write_text(existing.rstrip() + "\n\n" + section)
        except Exception as exc:
            logger.warning("Could not append ChEMBL retrieval filtering report: %s", exc)

    def _write_retrieval_filtering_only_report(
        self,
        query_slug: str,
        filtering_summary: Dict[str, Any],
        session_state: Optional[Dict[str, Any]],
    ) -> str:
        filename = f"chembl_{query_slug}_standardization_report.md"
        rel_path = scoped_artifact_path(
            filename,
            OutputOperation.CHEMICAL_SPACE,
            "standardization",
            session_state=session_state,
            workflow_slug="chemical_space",
        )
        report = (
            "# Dataset Standardization Report\n\n"
            "No molecular standardization was run because all retrieved ChEMBL rows "
            "were removed during retrieval validation.\n\n"
            f"{self._format_retrieval_filtering_report_section(filtering_summary)}"
        )
        with S3.open(rel_path, "w") as handle:
            handle.write(report)
        return S3.path(rel_path)

    @staticmethod
    def _is_remote_or_explicit_path(path: str) -> bool:
        return isinstance(path, str) and (
            path.startswith("s3://") or path.startswith("/") or path.startswith("file://")
        )

    @staticmethod
    def _format_retrieval_filtering_report_section(summary: Dict[str, Any]) -> str:
        filtered_path = summary.get("filtered_rows_path") or "None"
        decision_counts = summary.get("decision_counts") or {}
        metadata_filtered_count = summary.get("metadata_filtered_row_count", 0)
        short_filtered_count = max(
            summary.get("filtered_row_count", 0) - metadata_filtered_count,
            0,
        )
        return (
            "## ChEMBL Retrieval Filtering\n"
            "- Applied before molecular standardization to short-keyword retrieval hits "
            "and populated target metadata.\n"
            f"- Retrieved rows before filtering: {summary.get('retrieved_row_count', 0)}\n"
            f"- Suspicious short-keyword rows evaluated: "
            f"{summary.get('suspicious_row_count', 0)}\n"
            f"- Target metadata rows evaluated: "
            f"{summary.get('metadata_judge_row_count', 0)}\n"
            f"- Rows filtered out: {summary.get('filtered_row_count', 0)}\n"
            f"- Rows filtered by short-keyword judge: {short_filtered_count}\n"
            f"- Rows filtered by target metadata judge: "
            f"{metadata_filtered_count}\n"
            f"- Rows retained for standardization: {summary.get('retained_row_count', 0)}\n"
            f"- Judge status: {summary.get('judge_status', 'unknown')}\n"
            f"- Short-keyword judge status: {summary.get('judge_status', 'unknown')}\n"
            f"- Target metadata judge status: "
            f"{summary.get('metadata_judge_status', 'unknown')}\n"
            f"- Target metadata omitted decisions: "
            f"{summary.get('metadata_missing_decision_count', 0)}\n"
            f"- Fallback policy: {summary.get('fallback_policy', 'filter_rows')}\n"
            f"- Short-keyword fallback policy: "
            f"{summary.get('fallback_policy', 'filter_rows')}\n"
            f"- Target metadata fallback policy: "
            f"{summary.get('metadata_fallback_policy', 'keep_rows')}\n"
            f"- Filtered rows artifact: `{filtered_path}`\n"
            f"- Filter reasons: "
            f"`{json.dumps(decision_counts, sort_keys=True, default=str)}`\n"
        )

    @staticmethod
    def _format_all_rows_filtered_message(
        keywords: Sequence[str],
        filtering_summary: Dict[str, Any],
        filtered_rows_path: Optional[str],
        filtering_report_path: Optional[str],
    ) -> str:
        keywords_str = ", ".join([f"'{kw}'" for kw in keywords])
        return (
            "No clean ChEMBL dataset was created because all retrieved rows were "
            "filtered during retrieval validation.\n"
            f"Keywords: {keywords_str}\n"
            f"Suspicious rows evaluated: {filtering_summary.get('suspicious_row_count', 0)}\n"
            f"Target metadata rows evaluated: "
            f"{filtering_summary.get('metadata_judge_row_count', 0)}\n"
            f"Rows filtered out: {filtering_summary.get('filtered_row_count', 0)}\n"
            f"Filtered rows artifact: `{filtered_rows_path}`\n"
            f"Retrieval filtering report: `{filtering_report_path}`"
        )

    @staticmethod
    def _memory_compound_preview(df: pd.DataFrame, limit: int = 50) -> list[Dict[str, Any]]:
        """Return compact activity-bearing compound records for session memory."""
        return build_compound_memory_preview(df, limit=limit)

    def _add_smiles_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMILES structures to activity DataFrame."""
        logger.debug("Fetching SMILES structures")

        # Get unique molecule ChEMBL IDs
        unique_molecule_ids = df["molecule_chembl_id"].unique().tolist()
        logger.info(f"Fetching SMILES for {len(unique_molecule_ids)} unique molecules")

        # Fetch molecules with SMILES data using the ChEMBL client
        molecule_query = (
            self._ensure_client()
            .molecule.filter(molecule_chembl_id__in=unique_molecule_ids)
            .only(self.MOLECULE_FIELDS)
        )
        molecules = list(molecule_query[: len(unique_molecule_ids)])

        # Build SMILES lookup dictionary
        smiles_dict = {}
        for mol in molecules:
            cid = mol.get("molecule_chembl_id")
            smi = mol.get("canonical_smiles")

            # Try nested structure if top-level SMILES not available
            if not smi:
                structs = mol.get("molecule_structures") or {}
                smi = structs.get("canonical_smiles")

            smiles_dict[cid] = smi

        # Add SMILES column to DataFrame
        df["smi"] = df["molecule_chembl_id"].map(smiles_dict)
        df = standardize_smiles_column(df, "smi")

        smiles_count = df["smi"].notna().sum()
        logger.info(f"Successfully mapped SMILES for {smiles_count}/{len(df)} records")

        return df

    def _merge_assay_data(
        self, activities_df: pd.DataFrame, assays: Sequence[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Attach assay metadata to activity data using assay_chembl_id."""
        if activities_df.empty or not assays:
            return activities_df

        assay_df = self.to_dataframe(assays)
        if assay_df.empty or "assay_chembl_id" not in assay_df.columns:
            logger.warning("Assay data missing assay_chembl_id; skipping assay merge")
            return activities_df
        for column in self.TARGET_METADATA_COLUMNS:
            if column not in assay_df.columns:
                assay_df[column] = None

        return activities_df.merge(
            assay_df,
            on="assay_chembl_id",
            how="left",
            suffixes=("", "_assay"),
        )

    def _save_chembl_data(self, df: pd.DataFrame, query: str) -> str:
        """Save ChEMBL data and return the resolved storage path."""
        filename = f"chembl_{query.replace(' ', '_')}.csv"
        saved_path = S3.path(filename)

        try:
            with S3.open(filename, "w") as f:
                df.to_csv(f, index=False)
            logger.info(f"Saved ChEMBL data to {saved_path}")
            return saved_path
        except Exception as e:
            logger.error(f"Error saving ChEMBL data: {e}")
            raise

    def _format_success_message(
        self,
        df: pd.DataFrame,
        keywords: list,
        filename: str,
        total_assays: int = 0,
        duplicates_removed: int = 0,
        organism_filter: Optional[str] = None,
        assay_type_codes: Optional[Sequence[str]] = None,
        mechanism_filter: Optional[str] = None,
        raw_dataset_path: Optional[str] = None,
        descriptor_parquet_path: Optional[str] = None,
        standardization_report_path: Optional[str] = None,
        standardization_summary: Optional[Dict[str, Any]] = None,
        retrieval_filtering_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format success message for ChEMBL data fetch."""
        sample_row = df.head(1).to_string(index=False) if not df.empty else "No data"
        columns_preview = ", ".join(df.columns[:6].tolist()) + (
            "..." if len(df.columns) > 6 else ""
        )

        keywords_str = ", ".join([f"'{kw}'" for kw in keywords])
        save_label = "Saved to S3" if filename.startswith("s3://") else "Saved locally"
        message = (
            f"✅ Fetched and cleaned {len(df)} compound records from "
            f"{len(keywords)} keyword(s): {keywords_str}\n"
            f"📄 Clean dataset ({save_label}): `{filename}`\n"
        )
        if raw_dataset_path:
            message += f"📄 Raw dataset: `{raw_dataset_path}`\n"
        if descriptor_parquet_path:
            message += f"🧮 Descriptor Parquet: `{descriptor_parquet_path}`\n"
        if standardization_report_path:
            message += f"🧾 Standardization report: `{standardization_report_path}`\n"

        if total_assays > 0:
            message += f"🔬 Found {total_assays} unique assays across all keywords\n"

        assay_type_labels = {"B": "Binding", "F": "Functional", "A": "ADMET"}
        if assay_type_codes:
            readable_types = [assay_type_labels.get(code, code) for code in assay_type_codes]
            message += f"🧪 Assay types filtered: {', '.join(readable_types)}\n"

        if mechanism_filter:
            message += f"⚙️ Mechanism of action filter: {mechanism_filter}\n"

        if organism_filter:
            message += f"🧬 Target organism filter: {organism_filter}\n"

        if duplicates_removed > 0:
            message += f"🔄 Removed {duplicates_removed} duplicate raw activity records\n"

        if retrieval_filtering_summary and retrieval_filtering_summary.get("suspicious_row_count"):
            filtered_count = retrieval_filtering_summary.get("filtered_row_count", 0)
            metadata_filtered_count = retrieval_filtering_summary.get(
                "metadata_filtered_row_count",
                0,
            )
            short_filtered_count = max(filtered_count - metadata_filtered_count, 0)
            filtered_path = retrieval_filtering_summary.get("filtered_rows_path")
            message += (
                "🧯 Short-keyword retrieval filtering: "
                f"evaluated {retrieval_filtering_summary.get('suspicious_row_count', 0)} "
                f"suspicious rows and filtered {short_filtered_count}\n"
            )
            if filtered_path:
                message += f"📄 Filtered rows: `{filtered_path}`\n"

        if retrieval_filtering_summary and retrieval_filtering_summary.get(
            "metadata_judge_row_count"
        ):
            message += (
                "🧯 Target metadata filtering: "
                f"evaluated {retrieval_filtering_summary.get('metadata_judge_row_count', 0)} "
                "rows with populated target metadata and filtered "
                f"{retrieval_filtering_summary.get('metadata_filtered_row_count', 0)}\n"
            )
            missing_count = retrieval_filtering_summary.get("metadata_missing_decision_count", 0)
            if missing_count:
                message += f"🧯 Target metadata judge omitted {missing_count} item(s); kept them\n"
            filtered_path = retrieval_filtering_summary.get("filtered_rows_path")
            if filtered_path and not retrieval_filtering_summary.get("suspicious_row_count"):
                message += f"📄 Filtered rows: `{filtered_path}`\n"

        if standardization_summary:
            message += (
                "🧹 Standardization: "
                f"{standardization_summary.get('raw_rows')} raw rows → "
                f"{standardization_summary.get('rows_after_standardization')} valid rows → "
                f"{standardization_summary.get('clean_rows')} clean compounds; "
                f"{standardization_summary.get('invalid_smiles_rows')} invalid rows; "
                f"{standardization_summary.get('duplicate_rows_after_standardization')} "
                "post-standardization duplicate rows\n"
            )

        message += (
            f"📊 Columns: {columns_preview}\n"
            f"🔍 Sample row:\n{sample_row}\n\n"
            f"💡 Downstream agents will use the clean dataset path: `{filename}`"
        )

        return message

    def describe_dataset(self, path_to_dataset: str) -> str:
        """
        Compute and return descriptive statistics for a tabular dataset.

        Args:
            path_to_dataset: S3 or local path to the CSV file containing the dataset

        Returns:
            String representation of the pandas DataFrame's descriptive statistics

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If path is empty or file is empty
            pandas.errors.ParserError: If the CSV file cannot be parsed
        """
        if not path_to_dataset:
            raise ValueError("path_to_dataset cannot be empty")

        logger.info(f"Describing dataset: {path_to_dataset}")

        try:
            with S3.open(path_to_dataset, "r") as f:
                df = pd.read_csv(f)

            if df.empty:
                raise ValueError(f"Dataset at '{path_to_dataset}' is empty")

            logger.info(f"Loaded dataset with shape {df.shape} for description")
            description = df.describe(include="all")
            return str(description)

        except Exception as e:
            logger.error(f"Error describing dataset {path_to_dataset}: {e}")
            raise

    def convert_to_chembl_query(self, natural_prompt: str) -> str:
        """
        Generate multiple semantic keyword variations from a target description
        for ChEMBL assay description searches.

        The returned instruction directs the LLM to focus on **semantic**
        variants (abbreviations, synonyms, Greek letter replacement).
        Punctuation and hyphenation variants (``"epidermal growth factor
        receptor"`` vs ``"epidermal-growth-factor receptor"`` vs
        ``"epidermal growth factor-receptor"``) are automatically matched
        downstream by ``fetch_compounds`` via a regex built by
        :func:`_build_punctuation_regex` — the LLM should NOT spend
        its keyword budget on them.

        Args:
            natural_prompt: Target description (should already have generic
                terms removed in previous steps)

        Returns:
            Formatted instruction for generating ChEMBL search keywords
        """
        if not natural_prompt or not isinstance(natural_prompt, str):
            raise ValueError("natural_prompt must be a non-empty string")

        return (
            f"You are preparing queries for ChEMBL's `assay_description__icontains` filter.\n"
            f"Given this target description: '{natural_prompt.strip()}', generate multiple "
            f"SEMANTIC keyword variations (typically 2-4). Focus on:\n"
            f"  - Gene symbols / abbreviations (e.g., 'epidermal growth factor receptor' → 'egfr', "
            f"'phosphodiesterase 4A' → 'pde4a', 'B-Raf proto-oncogene' → 'braf', "
            f"'Janus kinase 2' → 'jak2').\n"
            f"  - Common full-name variants (e.g., 'phosphodiesterase 4A', "
            f"'epidermal growth factor receptor').\n"
            f"  - Literature synonyms (e.g., 'ERBB1' for EGFR; 'PRKCA' for "
            f"'protein kinase C alpha').\n"
            f"  - Greek character replacement (e.g., 'α' → 'alpha', 'β' → 'beta').\n"
            f"DO NOT generate punctuation / spacing variants yourself — the downstream "
            f"`fetch_compounds` tool automatically matches all hyphen/space combinations via "
            f"regex (e.g., 'epidermal growth factor receptor' automatically matches "
            f"'epidermal-growth factor receptor', 'epidermal growth-factor-receptor', etc.).\n"
            f"Output a comma-separated list of keyword phrases suitable for assay description "
            f"searches.\n"
            f"Example: For 'phosphodiesterase 4A', generate: 'pde4a, phosphodiesterase 4A'.\n"
            f"Example: For 'epidermal growth factor receptor', generate: "
            f"'egfr, epidermal growth factor receptor, erbb1'."
        )
