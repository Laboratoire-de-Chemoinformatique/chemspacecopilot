"""Explicit registry of cs_copilot toolkit methods exposed as MCP tools.

The registry is intentionally curated rather than introspected: it lets us
choose names, hide internal helpers, and document overrides like the ChEMBL
LLM-as-judge gating without surfacing those knobs to the MCP client.

Toolkit instances are created lazily by their factories on first use so that
heavy module imports (torch, RDKit caches, ChEMBL DB drivers) are paid only
when the corresponding tool is actually called.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Any, Callable, Iterable, List

from .errors import MCPToolError
from .tool_adapter import ToolSpec


def _factory(import_path: str) -> Callable[[], Any]:
    """Return a memoised factory that lazily imports and instantiates a toolkit."""

    @functools.lru_cache(maxsize=1)
    def _build() -> Any:
        module_name, _, class_name = import_path.rpartition(":")
        if not module_name or not class_name:
            raise ValueError(f"Invalid factory path: {import_path!r}")
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls()

    return _build


_CHEMBL = _factory("cs_copilot.tools.databases.chembl:ChemblToolkit")
_GTM = _factory("cs_copilot.tools.chemography.gtm:GTMToolkit")
_SIMILARITY = _factory("cs_copilot.tools.chemistry.similarity_toolkit:ChemicalSimilarityToolkit")
_SESSION_MEMORY = _factory("cs_copilot.tools.io.session_memory:SessionMemoryToolkit")
_ROBUSTNESS = _factory("cs_copilot.tools.analysis.robustness_toolkit:RobustnessAnalysisToolkit")


# Report export is a pair of module-level functions, not a toolkit instance.
# The facade below adapts them so the adapter can treat them uniformly.
class _ReportExportFacade:
    """Tiny adapter that exposes ``report_export`` module functions as methods."""

    def __init__(self) -> None:
        from cs_copilot.tools.io.report_export import save_markdown_report, save_rich_report

        self.save_markdown = save_markdown_report
        self.save_rich = save_rich_report


@functools.lru_cache(maxsize=1)
def _report_facade() -> _ReportExportFacade:
    return _ReportExportFacade()


def _ensure_llm_engine_available(
    engine: str,
    agent: Any | None,
    *,
    domain: str,
    fallback_engine: str,
) -> None:
    if str(engine or "").strip().lower() != "llm":
        return
    if getattr(agent, "model", None) is not None:
        return
    raise MCPToolError(
        f"LLM-backed {domain} design is unavailable in default MCP because "
        "MCPAgentContext.model is None. Use agno_team_run or the Agno team "
        f"runtime for internal-model design, or choose engine='{fallback_engine}'."
    )


def _backend_unavailable(name: str, exc: Exception) -> MCPToolError:
    return MCPToolError(f"{name} backend is unavailable for this tool call: {exc}")


class _SkillFacade:
    """Direct MCP access to the pure-Python cs_copilot skill catalog."""

    def list(self, include_content: bool = False) -> List[dict[str, Any]]:
        """List reusable cs_copilot workflow skills."""
        from cs_copilot.skills import list_skills

        return [spec.as_dict(include_content=include_content) for spec in list_skills()]

    def search(
        self,
        query: str,
        limit: int = 10,
        include_content: bool = False,
    ) -> List[dict[str, Any]]:
        """Search reusable cs_copilot workflow skills by metadata or tool names."""
        from cs_copilot.skills import search_skills

        return [
            spec.as_dict(include_content=include_content)
            for spec in search_skills(query, limit=limit)
        ]

    def fetch(self, slug: str, include_content: bool = True) -> dict[str, Any]:
        """Fetch one reusable cs_copilot workflow skill by slug."""
        from cs_copilot.skills import get_skill

        return get_skill(slug).as_dict(include_content=include_content)


class _WorkflowPolicyFacade:
    """Read-only workflow preflight helpers for external MCP reasoners."""

    def prepare_chembl_retrieval(
        self,
        user_request: str,
        session_summary: str | None = None,
    ) -> dict[str, Any]:
        """Preflight a ChEMBL request before calling retrieval tools."""
        from cs_copilot.workflows import prepare_chembl_retrieval

        return prepare_chembl_retrieval(
            user_request=user_request,
            session_summary=session_summary,
        )

    def plan_chemical_space_analysis(
        self,
        user_request: str,
        session_summary: str | None = None,
    ) -> dict[str, Any]:
        """Preflight a chemical-space analysis request before mutating tools."""
        from cs_copilot.workflows import plan_chemical_space_analysis

        return plan_chemical_space_analysis(
            user_request=user_request,
            session_summary=session_summary,
        )


class _MolecularDesignerFacade:
    """MCP-safe facade that loads the autoencoder only for generation calls."""

    def __init__(self) -> None:
        self._inner: Any | None = None

    def _toolkit(self) -> Any:
        if self._inner is None:
            try:
                from cs_copilot.tools.chemistry.autoencoder_toolkit import AutoencoderToolkit
                from cs_copilot.tools.chemistry.molecular_designer_toolkit import (
                    MolecularDesignerToolkit,
                )

                self._inner = MolecularDesignerToolkit(autoencoder_toolkit=AutoencoderToolkit())
            except Exception as exc:  # noqa: BLE001
                raise _backend_unavailable("Molecular designer", exc) from exc
        return self._inner

    def list_design_engines(self) -> dict[str, Any]:
        """List available small-molecule design engines."""
        return {
            "engines": [
                {
                    "name": "autoencoder",
                    "description": "LSTM autoencoder latent-space generation for SMILES.",
                    "supported_modes": ["analog", "interpolate", "neighborhood", "sample"],
                },
                {
                    "name": "llm",
                    "description": "LLM SMILES proposal followed by RDKit validation.",
                    "supported_modes": ["analog", "design", "neighborhood", "sample"],
                },
            ],
            "default_engine": "autoencoder",
        }

    def design_molecules(
        self,
        goal: str,
        engine: str = "autoencoder",
        n_candidates: int = 20,
        seed_smiles: str | None = None,
        constraints: dict[str, Any] | None = None,
        generation_mode: str = "sample",
        temperature: float = 1.0,
        decode_mode: str = "sample",
        noise_scale: float = 0.1,
        include_invalid: bool = False,
        return_format: str = "summary",
        session_key: str = "designed_molecules",
        agent: Any | None = None,
        session_state: dict[str, Any] | None = None,
        _source_tool: str = "design_molecules",
    ) -> Any:
        """Design small-molecule candidates with a selected design engine."""
        _ensure_llm_engine_available(
            engine,
            agent,
            domain="molecular",
            fallback_engine="autoencoder",
        )
        return self._toolkit().design_molecules(
            goal=goal,
            engine=engine,
            n_candidates=n_candidates,
            seed_smiles=seed_smiles,
            constraints=constraints,
            generation_mode=generation_mode,
            temperature=temperature,
            decode_mode=decode_mode,
            noise_scale=noise_scale,
            include_invalid=include_invalid,
            return_format=return_format,
            session_key=session_key,
            agent=agent,
            session_state=session_state,
            _source_tool=_source_tool,
        )

    def generate_analogs(
        self,
        seed_smiles: str,
        goal: str = "Generate close small-molecule analogs of the seed structure.",
        engine: str = "autoencoder",
        n_analogs: int = 10,
        noise_scale: float = 0.1,
        temperature: float = 0.5,
        include_invalid: bool = False,
        return_format: str = "summary",
        session_key: str = "designed_analogs",
        agent: Any | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> Any:
        """Generate small-molecule analogs around a seed SMILES."""
        _ensure_llm_engine_available(
            engine,
            agent,
            domain="molecular",
            fallback_engine="autoencoder",
        )
        return self._toolkit().generate_analogs(
            seed_smiles=seed_smiles,
            goal=goal,
            engine=engine,
            n_analogs=n_analogs,
            noise_scale=noise_scale,
            temperature=temperature,
            include_invalid=include_invalid,
            return_format=return_format,
            session_key=session_key,
            agent=agent,
            session_state=session_state,
        )

    def interpolate_molecules(
        self,
        smiles1: str,
        smiles2: str,
        n_steps: int = 10,
        temperature: float = 0.1,
        return_format: str = "summary",
        session_key: str = "designed_interpolation",
        agent: Any | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> Any:
        """Interpolate between two molecules using the autoencoder engine."""
        return self._toolkit().interpolate_molecules(
            smiles1=smiles1,
            smiles2=smiles2,
            n_steps=n_steps,
            temperature=temperature,
            return_format=return_format,
            session_key=session_key,
            agent=agent,
            session_state=session_state,
        )

    def validate_design_candidates(
        self,
        smiles_list: str | List[str],
        engine: str = "manual",
    ) -> List[dict[str, Any]]:
        """Validate, standardize, and annotate molecular design candidates."""
        from cs_copilot.tools.chemistry.molecular_designer_toolkit import (
            _dedupe_candidates,
            _validate_candidate,
        )

        values = [smiles_list] if isinstance(smiles_list, str) else smiles_list
        candidates = [_validate_candidate(smiles, engine=engine) for smiles in values]
        return [candidate.to_dict() for candidate in _dedupe_candidates(candidates)]

    def rank_design_candidates(
        self,
        candidates: List[dict[str, Any]],
        seed_smiles: str | None = None,
        prefer_qed: bool = True,
    ) -> List[dict[str, Any]]:
        """Rank validated molecular design candidates."""
        from cs_copilot.tools.chemistry.molecular_designer_toolkit import _similarity_to_seed

        ranked: List[dict[str, Any]] = []
        for candidate in candidates:
            item = dict(candidate)
            smiles = item.get("smiles")
            if smiles:
                similarity = _similarity_to_seed(smiles, seed_smiles)
                if similarity is not None:
                    item.setdefault("properties", {})["seed_tanimoto"] = similarity
                if prefer_qed and "qed" in item.get("properties", {}):
                    item["ranking_score"] = item["properties"]["qed"]
                if similarity is not None:
                    item["ranking_score"] = similarity
            ranked.append(item)
        return sorted(
            ranked,
            key=lambda item: (
                bool(item.get("valid")),
                item.get("ranking_score") if item.get("ranking_score") is not None else -1,
            ),
            reverse=True,
        )

    def register_design_candidates(
        self,
        candidates: Any,
        engine: str = "autoencoder",
        generation_mode: str = "manual",
        seed_smiles: str | None = None,
        goal: str = "Register generated molecular design candidates.",
        session_key: str = "registered_design_candidates",
        include_invalid: bool = False,
        agent: Any | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist final molecular design candidates as a generated candidate set."""
        return self._toolkit().register_design_candidates(
            candidates=candidates,
            engine=engine,
            generation_mode=generation_mode,
            seed_smiles=seed_smiles,
            goal=goal,
            session_key=session_key,
            include_invalid=include_invalid,
            agent=agent,
            session_state=session_state,
        )


class _PeptideDesignerFacade:
    """MCP-safe peptide facade that defers WAE model loading to WAE calls."""

    def __init__(self) -> None:
        self._inner: Any | None = None

    def _toolkit(self) -> Any:
        if self._inner is None:
            try:
                from cs_copilot.tools.chemistry.peptide_designer_toolkit import (
                    PeptideDesignerToolkit,
                )

                self._inner = PeptideDesignerToolkit()
            except Exception as exc:  # noqa: BLE001
                raise _backend_unavailable("Peptide WAE", exc) from exc
        return self._inner

    def list_design_engines(self) -> dict[str, Any]:
        """List available peptide design engines."""
        return {
            "engines": [
                {
                    "name": "wae",
                    "description": "Peptide WAE latent-space generation for sequences.",
                    "supported_modes": ["analog", "interpolate", "neighborhood", "sample"],
                },
                {
                    "name": "llm",
                    "description": "LLM peptide proposal followed by sequence validation.",
                    "supported_modes": ["analog", "design", "neighborhood", "sample"],
                },
            ],
            "default_engine": "wae",
        }

    def design_peptides(
        self,
        goal: str,
        engine: str = "wae",
        n_candidates: int = 20,
        seed_sequence: str | None = None,
        constraints: dict[str, Any] | None = None,
        generation_mode: str = "sample",
        temperature: float = 1.0,
        decode_mode: str = "categorical",
        noise_scale: float = 0.1,
        latent_std: float = 1.0,
        include_invalid: bool = False,
        return_format: str = "summary",
        session_key: str = "designed_peptides",
        agent: Any | None = None,
        session_state: dict[str, Any] | None = None,
        _source_tool: str = "design_peptides",
    ) -> Any:
        """Design peptide candidates with a selected design engine."""
        _ensure_llm_engine_available(
            engine,
            agent,
            domain="peptide",
            fallback_engine="wae",
        )
        return self._toolkit().design_peptides(
            goal=goal,
            engine=engine,
            n_candidates=n_candidates,
            seed_sequence=seed_sequence,
            constraints=constraints,
            generation_mode=generation_mode,
            temperature=temperature,
            decode_mode=decode_mode,
            noise_scale=noise_scale,
            latent_std=latent_std,
            include_invalid=include_invalid,
            return_format=return_format,
            session_key=session_key,
            agent=agent,
            session_state=session_state,
            _source_tool=_source_tool,
        )

    def generate_peptide_analogs(
        self,
        seed_sequence: str,
        goal: str = "Generate close peptide analogs of the seed sequence.",
        engine: str = "wae",
        n_analogs: int = 10,
        noise_scale: float = 0.1,
        temperature: float = 1.0,
        include_invalid: bool = False,
        return_format: str = "summary",
        session_key: str = "designed_peptide_analogs",
        agent: Any | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> Any:
        """Generate peptide analogs around a seed sequence."""
        _ensure_llm_engine_available(
            engine,
            agent,
            domain="peptide",
            fallback_engine="wae",
        )
        return self._toolkit().generate_peptide_analogs(
            seed_sequence=seed_sequence,
            goal=goal,
            engine=engine,
            n_analogs=n_analogs,
            noise_scale=noise_scale,
            temperature=temperature,
            include_invalid=include_invalid,
            return_format=return_format,
            session_key=session_key,
            agent=agent,
            session_state=session_state,
        )

    def design_peptide_interpolation(
        self,
        sequence1: str,
        sequence2: str,
        n_steps: int = 10,
        temperature: float = 1.0,
        method: str = "linear",
        return_format: str = "summary",
        session_key: str = "designed_peptide_interpolation",
        agent: Any | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> Any:
        """Interpolate between two peptides using the WAE engine."""
        return self._toolkit().design_peptide_interpolation(
            sequence1=sequence1,
            sequence2=sequence2,
            n_steps=n_steps,
            temperature=temperature,
            method=method,
            return_format=return_format,
            session_key=session_key,
            agent=agent,
            session_state=session_state,
        )

    def validate_design_candidates(
        self,
        sequences: str | List[str],
        engine: str = "manual",
    ) -> List[dict[str, Any]]:
        """Validate, normalize, and annotate peptide design candidates."""
        from cs_copilot.tools.chemistry.peptide_designer_toolkit import (
            _dedupe_peptide_candidates,
            _validate_peptide_candidate,
        )

        values = [sequences] if isinstance(sequences, str) else sequences
        candidates = [_validate_peptide_candidate(sequence, engine=engine) for sequence in values]
        return [candidate.to_dict() for candidate in _dedupe_peptide_candidates(candidates)]

    def rank_design_candidates(
        self,
        candidates: List[dict[str, Any]],
        seed_sequence: str | None = None,
        prefer_shorter: bool = False,
    ) -> List[dict[str, Any]]:
        """Rank validated peptide design candidates."""
        from cs_copilot.tools.chemistry.peptide_designer_toolkit import _sequence_similarity

        ranked: List[dict[str, Any]] = []
        for candidate in candidates:
            item = dict(candidate)
            sequence = item.get("sequence")
            if sequence:
                similarity = _sequence_similarity(sequence, seed_sequence)
                if similarity is not None:
                    item.setdefault("properties", {})["seed_sequence_similarity"] = similarity
                    item["ranking_score"] = similarity
                elif item.get("score") is not None:
                    item["ranking_score"] = item["score"]
                elif prefer_shorter:
                    length = item.get("properties", {}).get("length")
                    if length:
                        item["ranking_score"] = 1 / length
            ranked.append(item)
        return sorted(
            ranked,
            key=lambda item: (
                bool(item.get("valid")),
                item.get("ranking_score") if item.get("ranking_score") is not None else -1,
            ),
            reverse=True,
        )

    def load_peptide_design_candidates(
        self,
        reference: str = "designed_peptides",
        include_candidates: bool = True,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load peptide design candidates from a session pointer or artifact path."""
        import json

        from cs_copilot.storage import S3
        from cs_copilot.tools.chemistry.peptide_designer_toolkit import (
            _compact_peptide_preview,
        )

        artifact_path = reference
        pointer = None
        if isinstance(session_state, dict):
            raw_pointer = session_state.get(reference)
            if isinstance(raw_pointer, dict):
                pointer = raw_pointer
                artifact_path = (
                    pointer.get("artifact_rel_path")
                    or pointer.get("artifact_path")
                    or artifact_path
                )

        with S3.open(str(artifact_path), "r") as handle:
            payload = json.load(handle)

        candidates = list(payload.get("candidates") or [])
        result: dict[str, Any] = {
            "status": "loaded",
            "reference": reference,
            "peptide_candidate_set_id": payload.get("peptide_candidate_set_id"),
            "metadata": payload.get("metadata") or {},
            "count": len(candidates),
            "preview": _compact_peptide_preview(candidates),
        }
        if pointer:
            result["session_pointer"] = pointer
        if include_candidates:
            result["candidates"] = candidates
        return result

    def validate_model_loaded(self) -> bool:
        """Return whether the Peptide WAE model is loaded and usable."""
        return self._toolkit().validate_model_loaded()

    def get_latent_dimension(self) -> int:
        """Get the peptide WAE latent dimension."""
        return self._toolkit().get_latent_dimension()

    def encode_peptides(self, sequences: str | List[str], batch_size: int = 32) -> Any:
        """Encode peptide sequences to latent vectors."""
        return self._toolkit().encode_peptides(sequences=sequences, batch_size=batch_size)

    def decode_latent(
        self,
        latent_vectors: List[float] | List[List[float]],
        temperature: float = 1.0,
        decode_mode: str = "categorical",
        max_length: int = 25,
    ) -> List[str]:
        """Decode latent vectors to peptide sequences."""
        return self._toolkit().decode_latent(
            latent_vectors=latent_vectors,
            temperature=temperature,
            decode_mode=decode_mode,
            max_length=max_length,
        )

    def sample_peptides(
        self,
        n_samples: int = 5000,
        latent_std: float = 1.0,
        temperature: float = 1.0,
        decode_mode: str = "categorical",
        max_length: int = 25,
        filter_valid_unique: bool = True,
        return_format: str = "summary",
        session_key: str = "sampled_peptides",
        agent: Any | None = None,
    ) -> Any:
        """Sample new peptides from the WAE latent space."""
        return self._toolkit().sample_peptides(
            n_samples=n_samples,
            latent_std=latent_std,
            temperature=temperature,
            decode_mode=decode_mode,
            max_length=max_length,
            filter_valid_unique=filter_valid_unique,
            return_format=return_format,
            session_key=session_key,
            agent=agent,
        )

    def interpolate_peptides(
        self,
        seq1: str,
        seq2: str,
        n_steps: int = 10,
        temperature: float = 1.0,
        decode_mode: str = "categorical",
        method: str = "linear",
    ) -> List[str]:
        """Interpolate between two peptides in WAE latent space."""
        return self._toolkit().interpolate_peptides(
            seq1=seq1,
            seq2=seq2,
            n_steps=n_steps,
            temperature=temperature,
            decode_mode=decode_mode,
            method=method,
        )

    def reconstruct_sequence(
        self,
        sequence: str,
        temperature: float = 0.1,
        decode_mode: str = "greedy",
    ) -> str:
        """Reconstruct a peptide sequence by encoding and decoding it."""
        return self._toolkit().reconstruct_sequence(
            sequence=sequence,
            temperature=temperature,
            decode_mode=decode_mode,
        )

    def explore_latent_neighborhood(
        self,
        base_sequence: str,
        noise_scale: float = 0.1,
        n_neighbors: int = 5,
        temperature: float = 1.0,
        decode_mode: str = "categorical",
    ) -> List[str]:
        """Explore the WAE latent neighborhood around a peptide sequence."""
        return self._toolkit().explore_latent_neighborhood(
            base_sequence=base_sequence,
            noise_scale=noise_scale,
            n_neighbors=n_neighbors,
            temperature=temperature,
            decode_mode=decode_mode,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded Peptide WAE model."""
        return self._toolkit().get_model_info()


class _PointerPandasFacade:
    """MCP-safe wrapper around PointerPandasTools with JSON-friendly schemas."""

    def __init__(self) -> None:
        from cs_copilot.tools.io.pointer_pandas_tools import PointerPandasTools

        self._toolkit = PointerPandasTools()

    def load_dataframe_from_session(
        self,
        dataframe_name: str,
        session_key: str,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load a session DataFrame or CSV path into the pandas registry."""
        return dict(
            self._toolkit.load_dataframe_from_session(
                dataframe_name=dataframe_name,
                session_key=session_key,
                session_state=session_state,
            )
        )

    def create_dataframe(
        self,
        dataframe_name: str,
        create_using_function: str,
        function_parameters: Any | None = None,
    ) -> dict[str, Any]:
        """Create a DataFrame and store it in the pandas registry."""
        return dict(
            self._toolkit.create_pandas_dataframe(
                dataframe_name=dataframe_name,
                create_using_function=create_using_function,
                function_parameters=function_parameters,
            )
        )

    def run_operation(
        self,
        dataframe_name: str,
        operation: str,
        operation_parameters: Any | None = None,
        function_parameters: Any | None = None,
    ) -> Any:
        """Run a pandas operation against a registered DataFrame."""
        return self._toolkit.run_dataframe_operation(
            dataframe_name=dataframe_name,
            operation=operation,
            operation_parameters=operation_parameters,
            function_parameters=function_parameters,
        )

    def normalize_for_analysis(
        self,
        df_path: str,
        cluster_col: str | None = None,
        smiles_col: str | None = None,
        activity_col: str | None = None,
        agent: Any | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Normalize a DataFrame to the standard analysis format."""
        return dict(
            self._toolkit.normalize_for_analysis(
                df_path=df_path,
                cluster_col=cluster_col,
                smiles_col=smiles_col,
                activity_col=activity_col,
                agent=agent,
                session_state=session_state,
            )
        )


@functools.lru_cache(maxsize=1)
def _skill_facade() -> _SkillFacade:
    return _SkillFacade()


@functools.lru_cache(maxsize=1)
def _molecular_designer_facade() -> _MolecularDesignerFacade:
    return _MolecularDesignerFacade()


@functools.lru_cache(maxsize=1)
def _peptide_designer_facade() -> _PeptideDesignerFacade:
    return _PeptideDesignerFacade()


@functools.lru_cache(maxsize=1)
def _pointer_pandas_facade() -> _PointerPandasFacade:
    return _PointerPandasFacade()


@functools.lru_cache(maxsize=1)
def _workflow_policy_facade() -> _WorkflowPolicyFacade:
    return _WorkflowPolicyFacade()


_MOLECULAR_DESIGN = _molecular_designer_facade
_PEPTIDE_DESIGN = _peptide_designer_facade
_SYNPLANNER = _factory("cs_copilot.tools.chemistry.synplanner_toolkit:SynPlannerToolkit")
_PANDAS = _pointer_pandas_facade
_SKILLS = _skill_facade
_WORKFLOW_POLICY = _workflow_policy_facade


# ChEMBL ---------------------------------------------------------------------

_CHEMBL_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=_WORKFLOW_POLICY,
        method="prepare_chembl_retrieval",
        summary=(
            "Preflight a ChEMBL retrieval request. Use this read-only workflow gate "
            "before chembl_fetch_compounds to identify missing target, organism, "
            "assay-type, or mechanism clarification."
        ),
        read_only=True,
    ),
    ToolSpec(
        mcp_name="chembl_fetch_compounds",
        toolkit_factory=_CHEMBL,
        method="fetch_compounds",
        summary=(
            "Low-level execution tool that fetches ChEMBL bioactivity data for "
            "one or more keyword targets. For vague user requests, call "
            "chembl_prepare_retrieval first and fetch only after can_proceed=true. "
            "In MCP mode the in-process LLM-as-judge filtering is disabled; "
            "use the chembl_retrieval_judge / chembl_metadata_judge prompts "
            "if you want to perform equivalent filtering with this client."
        ),
        forces={"enable_retrieval_judge": False, "enable_metadata_judge": False},
    ),
    ToolSpec(
        mcp_name="chembl_describe_dataset",
        toolkit_factory=_CHEMBL,
        method="describe_dataset",
        summary="Return a structural summary of a previously fetched ChEMBL dataset by path.",
        read_only=True,
    ),
    ToolSpec(
        mcp_name="chembl_convert_to_chembl_query",
        toolkit_factory=_CHEMBL,
        method="convert_to_chembl_query",
        summary=(
            "Rewrite a free-form natural language query into the canonical "
            "ChEMBL keyword form accepted by chembl_fetch_compounds."
        ),
        read_only=True,
    ),
]


# GTM ------------------------------------------------------------------------

# (mcp_name, method_name, summary, read_only) - explicit names avoid both
# `gtm_gtm_*` doubles and bare names like `save_gtm_and_data`.
_GTM_METHODS = [
    (
        "gtm_optimization",
        "gtm_optimization",
        "Build a GTM model from a dataset (optimisation pass).",
        False,
    ),
    (
        "gtm_save_model_and_data",
        "save_gtm_and_data",
        "Persist a fitted GTM model and the projected source dataset.",
        False,
    ),
    (
        "gtm_load_model_only",
        "load_gtm_model_only",
        "Load a previously saved GTM model into the session.",
        False,
    ),
    (
        "gtm_load_density_matrix",
        "load_gtm_get_density_matrix",
        "Load a GTM model and return its node density / responsibility matrix.",
        False,
    ),
    (
        "gtm_load_and_prep_data",
        "load_and_prep_data",
        "Project a dataset onto a loaded GTM model and prepare lookup tables.",
        False,
    ),
    (
        "gtm_analyze_scaffolds_in_nodes",
        "analyze_scaffolds_in_nodes",
        "Summarise scaffolds residing in the given GTM node ids.",
        True,
    ),
    (
        "gtm_check_source_datasets_in_nodes",
        "check_source_datasets_in_nodes",
        "Report which source datasets contribute to the given GTM node ids.",
        True,
    ),
    (
        "gtm_node_id_from_coords",
        "node_id_from_coords",
        "Return the GTM node id closest to a (x, y) latent coordinate.",
        True,
    ),
    (
        "gtm_get_density_summary",
        "get_density_summary",
        "Return the top-N densest GTM nodes.",
        True,
    ),
    (
        "gtm_get_activity_landscape_summary",
        "get_activity_landscape_summary",
        "Summarise an activity landscape view built from the loaded GTM map.",
        True,
    ),
    (
        "gtm_get_node_lookup_summary",
        "get_node_lookup_summary",
        "Return a compact lookup table for the loaded GTM map.",
        True,
    ),
    (
        "gtm_sample_nodes",
        "sample_nodes",
        "Sample molecules located inside the given GTM nodes.",
        False,
    ),
    (
        "gtm_sample_dense_nodes",
        "sample_dense_nodes",
        "Sample molecules from the densest GTM nodes.",
        False,
    ),
    (
        "gtm_sample_activity_landscape_nodes",
        "sample_activity_landscape_nodes",
        "Sample molecules from activity-landscape regions of interest.",
        False,
    ),
    (
        "gtm_sample_top_activity_molecules",
        "sample_top_activity_molecules",
        "Sample top-activity molecules on the loaded GTM map.",
        False,
    ),
    (
        "gtm_sample_by_coordinates",
        "sample_by_coordinates",
        "Sample molecules near the supplied (x, y) latent coordinates.",
        False,
    ),
    (
        "gtm_create_activity_landscapes",
        "create_activity_landscapes",
        "Build activity landscape views from the loaded GTM map and dataset.",
        False,
    ),
    (
        "gtm_load_activity_landscape_csv",
        "load_activity_landscape_csv",
        "Load a previously saved activity-landscape CSV back into the session.",
        False,
    ),
    (
        "gtm_save_landscape_plot",
        "save_gtm_landscape_plot",
        "Save a static GTM activity / density landscape plot.",
        False,
    ),
    (
        "gtm_project_data",
        "project_data_on_gtm",
        "Project a new dataset onto a loaded GTM model.",
        False,
    ),
    (
        "gtm_train_on_latent_space",
        "train_gtm_on_latent_space",
        "Train a GTM model on autoencoder latent vectors.",
        False,
    ),
    (
        "gtm_load_latent_data",
        "load_latent_data_on_gtm",
        "Load latent-space data and project it onto a loaded GTM model.",
        False,
    ),
    (
        "gtm_create_peptide_activity_landscapes",
        "create_peptide_activity_landscapes",
        "Build peptide-specific activity landscape views.",
        False,
    ),
]


_GTM_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=mcp_name,
        toolkit_factory=_GTM,
        method=method,
        summary=summary,
        read_only=read_only,
    )
    for mcp_name, method, summary, read_only in _GTM_METHODS
]


# Chemical similarity --------------------------------------------------------

_SIMILARITY_METHODS = [
    ("calculate_tanimoto_similarity", "Tanimoto similarity for one or more SMILES pairs."),
    ("calculate_dice_similarity", "Dice similarity for one or more SMILES pairs."),
    ("calculate_tversky_similarity", "Tversky similarity for one or more SMILES pairs."),
    ("calculate_cosine_similarity", "Cosine similarity for one or more SMILES pairs."),
    (
        "calculate_euclidean_distance",
        "Euclidean distance between fingerprint vectors of SMILES pairs.",
    ),
    (
        "calculate_all_similarities",
        "Compute Tanimoto / Dice / Tversky / cosine in a single call.",
    ),
    (
        "find_most_similar",
        "Find the most similar molecules to a query SMILES from a candidate list.",
    ),
]


_SIMILARITY_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=f"chem_{name}",
        toolkit_factory=_SIMILARITY,
        method=name,
        summary=summary,
        read_only=True,
    )
    for name, summary in _SIMILARITY_METHODS
]


# Session memory ------------------------------------------------------------

_SESSION_METHODS = [
    ("list_session_objects", "List structured objects stored in the active session.", True),
    ("list_loadable_session_data", "List session-resident datasets that can be reloaded.", True),
    ("get_session_object", "Return a session object by id.", True),
    ("select_session_object", "Mark a session object as the current one for its role.", False),
    ("resolve_session_reference", "Resolve a free-form reference to a session object id.", True),
    ("resolve_candidate_set", "Resolve a candidate-set reference to a stored object.", True),
    ("load_candidate_set_artifact", "Materialise a candidate-set artifact path.", True),
    (
        "materialize_candidate_set_dataset",
        "Materialise the dataset rows of a candidate set.",
        False,
    ),
    ("summarize_session_memory", "Return a compact textual summary of session memory.", False),
]


_SESSION_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=f"session_{name}",
        toolkit_factory=_SESSION_MEMORY,
        method=name,
        summary=summary,
        read_only=read_only,
    )
    for name, summary, read_only in _SESSION_METHODS
]


# Reports --------------------------------------------------------------------

_REPORT_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name="report_save_markdown",
        toolkit_factory=_report_facade,
        method="save_markdown",
        summary="Save a markdown report into the session-scoped storage layout.",
    ),
    ToolSpec(
        mcp_name="report_save_rich",
        toolkit_factory=_report_facade,
        method="save_rich",
        summary="Save an image-rich (HTML/PDF/Markdown) report into the session layout.",
    ),
]


# Workflow preflight ---------------------------------------------------------

_WORKFLOW_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name="chemspace_plan_analysis",
        toolkit_factory=_WORKFLOW_POLICY,
        method="plan_chemical_space_analysis",
        summary=(
            "Preflight a broad chemical-space analysis request before ChEMBL, "
            "GTM, chemotype, or report-generation tools are called."
        ),
        read_only=True,
    ),
]


# Robustness analysis -------------------------------------------------------

_ROBUSTNESS_METHODS = [
    ("load_test_results", "Load the raw results of a robustness test run.", True),
    ("load_test_summary_csv", "Load the per-prompt summary CSV of a robustness test run.", True),
    ("list_available_test_runs", "List robustness test runs available under the data root.", True),
    ("analyze_score_distribution", "Summarise score distribution for a robustness run.", True),
    ("identify_failing_prompts", "List failing prompts above a score threshold.", True),
    ("compare_test_runs", "Compare two robustness test runs side by side.", True),
    ("analyze_temporal_trends", "Summarise robustness score trends across runs.", True),
    ("generate_insights", "Generate textual insights about a robustness run.", True),
    ("export_analysis_report", "Persist a robustness analysis report to storage.", False),
]


_ROBUSTNESS_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=f"robustness_{name}",
        toolkit_factory=_ROBUSTNESS,
        method=name,
        summary=summary,
        read_only=read_only,
    )
    for name, summary, read_only in _ROBUSTNESS_METHODS
]


# Skills ---------------------------------------------------------------------

_SKILL_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name="skill_list",
        toolkit_factory=_SKILLS,
        method="list",
        summary="List reusable cs_copilot workflow skills from the local skill catalog.",
        read_only=True,
    ),
    ToolSpec(
        mcp_name="skill_search",
        toolkit_factory=_SKILLS,
        method="search",
        summary="Search reusable cs_copilot workflow skills by metadata and tool names.",
        read_only=True,
    ),
    ToolSpec(
        mcp_name="skill_fetch",
        toolkit_factory=_SKILLS,
        method="fetch",
        summary="Fetch one reusable cs_copilot workflow skill, including SKILL.md content.",
        read_only=True,
    ),
]


# PointerPandas facade --------------------------------------------------------

_PANDAS_METHODS = [
    (
        "pandas_load_dataframe_from_session",
        "load_dataframe_from_session",
        "Load a session DataFrame or CSV artifact into the MCP pandas registry.",
        False,
    ),
    (
        "pandas_create_dataframe",
        "create_dataframe",
        "Create a DataFrame and store it in the MCP pandas registry.",
        False,
    ),
    (
        "pandas_run_operation",
        "run_operation",
        "Run a pandas operation against a registered DataFrame.",
        False,
    ),
    (
        "pandas_normalize_for_analysis",
        "normalize_for_analysis",
        "Normalize a DataFrame for downstream cs_copilot analysis workflows.",
        False,
    ),
]


_PANDAS_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=mcp_name,
        toolkit_factory=_PANDAS,
        method=method,
        summary=summary,
        read_only=read_only,
    )
    for mcp_name, method, summary, read_only in _PANDAS_METHODS
]


# Molecular design ------------------------------------------------------------

_MOLECULAR_DESIGN_METHODS = [
    (
        "mol_list_design_engines",
        "list_design_engines",
        "List available molecular design engines and supported generation modes.",
        True,
        {},
    ),
    (
        "mol_design_molecules",
        "design_molecules",
        "Design small-molecule candidates with a selected molecular design engine.",
        False,
        {"_source_tool": "design_molecules"},
    ),
    (
        "mol_generate_analogs",
        "generate_analogs",
        "Generate small-molecule analogs around a seed SMILES.",
        False,
        {},
    ),
    (
        "mol_interpolate_molecules",
        "interpolate_molecules",
        "Interpolate between two molecules using the molecular autoencoder engine.",
        False,
        {},
    ),
    (
        "mol_validate_design_candidates",
        "validate_design_candidates",
        "Validate, standardize, and annotate proposed molecular design candidates.",
        True,
        {},
    ),
    (
        "mol_rank_design_candidates",
        "rank_design_candidates",
        "Rank validated molecular design candidates by seed similarity and quality.",
        True,
        {},
    ),
    (
        "mol_register_design_candidates",
        "register_design_candidates",
        "Persist final molecular design candidates as a generated candidate set.",
        False,
        {},
    ),
]


_MOLECULAR_DESIGN_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=mcp_name,
        toolkit_factory=_MOLECULAR_DESIGN,
        method=method,
        summary=summary,
        forces=forces,
        read_only=read_only,
    )
    for mcp_name, method, summary, read_only, forces in _MOLECULAR_DESIGN_METHODS
]


# Peptide design --------------------------------------------------------------

_PEPTIDE_DESIGN_METHODS = [
    (
        "peptide_list_design_engines",
        "list_design_engines",
        "List available peptide design engines and supported generation modes.",
        True,
        {},
    ),
    (
        "peptide_design_peptides",
        "design_peptides",
        "Design peptide candidates with a selected peptide design engine.",
        False,
        {"_source_tool": "design_peptides"},
    ),
    (
        "peptide_generate_analogs",
        "generate_peptide_analogs",
        "Generate peptide analogs around a seed sequence.",
        False,
        {},
    ),
    (
        "peptide_design_interpolation",
        "design_peptide_interpolation",
        "Interpolate between two peptide sequences using the WAE engine.",
        False,
        {},
    ),
    (
        "peptide_validate_design_candidates",
        "validate_design_candidates",
        "Validate, normalize, and annotate proposed peptide design candidates.",
        True,
        {},
    ),
    (
        "peptide_rank_design_candidates",
        "rank_design_candidates",
        "Rank validated peptide design candidates by seed similarity and quality.",
        True,
        {},
    ),
    (
        "peptide_load_design_candidates",
        "load_peptide_design_candidates",
        "Load peptide design candidates from a session pointer or artifact path.",
        True,
        {},
    ),
    (
        "peptide_validate_model_loaded",
        "validate_model_loaded",
        "Check whether the Peptide WAE model is loaded and usable.",
        True,
        {},
    ),
    (
        "peptide_get_latent_dimension",
        "get_latent_dimension",
        "Return the Peptide WAE latent dimension.",
        True,
        {},
    ),
    (
        "peptide_encode_peptides",
        "encode_peptides",
        "Encode peptide sequences to latent vectors.",
        True,
        {},
    ),
    (
        "peptide_decode_latent",
        "decode_latent",
        "Decode latent vectors to peptide sequences.",
        True,
        {},
    ),
    (
        "peptide_sample_peptides",
        "sample_peptides",
        "Sample new peptides from the WAE latent space.",
        False,
        {},
    ),
    (
        "peptide_interpolate_peptides",
        "interpolate_peptides",
        "Interpolate between two peptides in WAE latent space.",
        True,
        {},
    ),
    (
        "peptide_reconstruct_sequence",
        "reconstruct_sequence",
        "Reconstruct a peptide sequence by encoding and decoding it.",
        True,
        {},
    ),
    (
        "peptide_explore_latent_neighborhood",
        "explore_latent_neighborhood",
        "Explore the WAE latent neighborhood around a peptide sequence.",
        True,
        {},
    ),
    (
        "peptide_get_model_info",
        "get_model_info",
        "Return metadata about the loaded Peptide WAE model.",
        True,
        {},
    ),
]


_PEPTIDE_DESIGN_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=mcp_name,
        toolkit_factory=_PEPTIDE_DESIGN,
        method=method,
        summary=summary,
        forces=forces,
        read_only=read_only,
    )
    for mcp_name, method, summary, read_only, forces in _PEPTIDE_DESIGN_METHODS
]


# SynPlanner ------------------------------------------------------------------

_SYNPLANNER_METHODS = [
    (
        "synplanner_identify_input",
        "identify_input",
        "Identify whether a retrosynthesis query is a SMILES string or molecule name.",
        True,
    ),
    (
        "synplanner_convert_name_to_smiles",
        "convert_name_to_smiles",
        "Convert a molecule name to canonical SMILES for SynPlanner input.",
        True,
    ),
    (
        "synplanner_plan_synthesis",
        "plan_synthesis",
        "Run SynPlanner retrosynthesis planning for a SMILES string or molecule name.",
        False,
    ),
    (
        "synplanner_describe_plan",
        "describe_plan",
        "Return a human-readable description of the latest SynPlanner plan.",
        True,
    ),
    (
        "synplanner_get_route_visualizations",
        "get_route_visualizations",
        "Generate or fetch route visualization artifacts for a SynPlanner plan.",
        False,
    ),
]


_SYNPLANNER_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=mcp_name,
        toolkit_factory=_SYNPLANNER,
        method=method,
        summary=summary,
        read_only=read_only,
    )
    for mcp_name, method, summary, read_only in _SYNPLANNER_METHODS
]


def _with_group(specs: Iterable[ToolSpec], group: str) -> Iterable[ToolSpec]:
    for spec in specs:
        yield replace(spec, group=spec.group or group)


def iter_specs() -> Iterable[ToolSpec]:
    """Yield every :class:`ToolSpec` exposed by the MCP server."""

    yield from _with_group(_CHEMBL_SPECS, "chembl")
    yield from _with_group(_GTM_SPECS, "gtm")
    yield from _with_group(_SIMILARITY_SPECS, "chem")
    yield from _with_group(_SESSION_SPECS, "session")
    yield from _with_group(_REPORT_SPECS, "report")
    yield from _with_group(_WORKFLOW_SPECS, "workflow")
    yield from _with_group(_ROBUSTNESS_SPECS, "robustness")
    yield from _with_group(_SKILL_SPECS, "skills")
    yield from _with_group(_PANDAS_SPECS, "pandas")
    yield from _with_group(_MOLECULAR_DESIGN_SPECS, "molecular_design")
    yield from _with_group(_PEPTIDE_DESIGN_SPECS, "peptide_design")
    yield from _with_group(_SYNPLANNER_SPECS, "synplanner")


def all_specs() -> List[ToolSpec]:
    """Return every :class:`ToolSpec` as a list (handy for tests)."""

    return list(iter_specs())
