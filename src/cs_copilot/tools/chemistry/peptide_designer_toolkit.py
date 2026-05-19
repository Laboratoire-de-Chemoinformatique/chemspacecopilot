#!/usr/bin/env python
# coding: utf-8
"""
Peptide Designer toolkit and engine facade for peptide sequence generation.

The public agent should reason in terms of peptide design engines. The
deepchemography Peptide WAE remains one engine implementation, while LLM-based
sequence proposal is exposed as another engine with the same validation and
artifact-backed output contract.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Union

import numpy as np
import torch
from agno.agent import Agent
from agno.models.base import Model
from agno.tools.toolkit import Toolkit
from pydantic import BaseModel, Field

from cs_copilot.storage import S3, OutputOperation, operation_rel_path
from cs_copilot.tools.constants import (
    DEFAULT_PEPTIDE_DESIGNER_MODEL_PATH,
    HUGGINGFACE_PEPTIDE_WAE_REPO,
)
from cs_copilot.tools.io.session_memory import register_session_object, update_state_targets

logger = logging.getLogger(__name__)

SampleReturnFormat = Literal["summary", "list"]
SUPPORTED_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTUVYWZ")
DEFAULT_MAX_SEQUENCE_LENGTH = 25
PEPTIDE_DESIGN_ARTIFACT_DIR = "peptide_candidate_sets"
PEPTIDE_DESIGN_ARTIFACT_FORMAT = "json"


def _filter_valid_unique_peptides(raw: List[Any]) -> List[str]:
    """Drop non-string/empty entries and deduplicate peptide sequences.

    Vocab-constrained decoding already limits characters to the model's
    amino-acid alphabet, so this filter mainly removes empty/whitespace
    outputs and exact duplicates.
    """
    seen: set = set()
    out: List[str] = []
    for s in raw:
        if not isinstance(s, str):
            continue
        s_norm = s.strip()
        if not s_norm or s_norm in seen:
            continue
        seen.add(s_norm)
        out.append(s_norm)
    return out


class PeptideDesignerError(Exception):
    """Exception raised for peptide designer-related errors."""

    pass


@dataclass
class PeptideDesignRequest:
    """Engine-independent request for peptide design."""

    goal: str
    n_candidates: int = 20
    seed_sequence: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    generation_mode: str = "sample"
    temperature: float = 1.0
    decode_mode: str = "categorical"
    noise_scale: float = 0.1
    latent_std: float = 1.0


@dataclass
class PeptideCandidate:
    """Validated peptide candidate returned by any design engine."""

    sequence: Optional[str]
    original_sequence: Optional[str]
    engine: str
    valid: bool
    rationale: Optional[str] = None
    score: Optional[float] = None
    error: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "sequence": self.sequence,
            "original_sequence": self.original_sequence,
            "engine": self.engine,
            "valid": self.valid,
            "rationale": self.rationale,
            "score": self.score,
            "error": self.error,
            "properties": self.properties,
        }


@dataclass
class PeptideDesignResult:
    """Engine-independent peptide design result."""

    engine: str
    candidates: List[PeptideCandidate]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def valid_candidates(self) -> List[PeptideCandidate]:
        """Return valid candidates only."""
        return [candidate for candidate in self.candidates if candidate.valid]


class PeptideDesignEngine(Protocol):
    """Protocol implemented by peptide generative engines."""

    engine_name: str

    def supports(self, generation_mode: str) -> bool:
        """Return whether this engine supports a generation mode."""
        ...

    def design(self, request: PeptideDesignRequest) -> PeptideDesignResult:
        """Generate peptide candidates for a request."""
        ...


class _LLMPeptideCandidate(BaseModel):
    sequence: str = Field(
        ...,
        description=(
            "Candidate peptide sequence as single-letter amino acids, either compact "
            "or space-separated."
        ),
    )
    rationale: Optional[str] = Field(
        default=None, description="Short reason this peptide matches the design goal."
    )
    score: Optional[float] = Field(
        default=None, description="Optional model confidence or desirability score from 0 to 1."
    )


class _LLMPeptideDesignResponse(BaseModel):
    candidates: List[_LLMPeptideCandidate] = Field(
        default_factory=list, description="Candidate peptide sequences proposed by the LLM."
    )


def _normalize_peptide_sequence(raw_sequence: Any) -> Optional[str]:
    if not isinstance(raw_sequence, str):
        return None

    text = raw_sequence.strip()
    if not text:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith(">")]
    text = " ".join(lines).replace(",", " ").replace("-", " ")
    tokens = [token.upper() for token in text.split() if token.strip()]

    if len(tokens) == 1 and len(tokens[0]) > 1:
        tokens = list(tokens[0])

    if not tokens:
        return None
    return " ".join(tokens)


def _peptide_properties(sequence: str) -> Dict[str, Any]:
    tokens = sequence.split()
    composition: Dict[str, int] = {}
    for token in tokens:
        composition[token] = composition.get(token, 0) + 1
    return {
        "length": len(tokens),
        "amino_acid_composition": composition,
    }


def _validate_peptide_candidate(
    raw_sequence: Any,
    *,
    engine: str,
    rationale: Optional[str] = None,
    score: Optional[float] = None,
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> PeptideCandidate:
    normalized = _normalize_peptide_sequence(raw_sequence)
    if normalized is None:
        return PeptideCandidate(
            sequence=None,
            original_sequence=str(raw_sequence) if raw_sequence is not None else None,
            engine=engine,
            valid=False,
            rationale=rationale,
            score=score,
            error="Peptide sequence must be a non-empty string.",
        )

    tokens = normalized.split()
    invalid = sorted({token for token in tokens if token not in SUPPORTED_AMINO_ACIDS})
    if invalid:
        return PeptideCandidate(
            sequence=None,
            original_sequence=raw_sequence,
            engine=engine,
            valid=False,
            rationale=rationale,
            score=score,
            error=f"Unsupported amino acid token(s): {', '.join(invalid)}.",
        )

    if len(tokens) > max_length:
        return PeptideCandidate(
            sequence=None,
            original_sequence=raw_sequence,
            engine=engine,
            valid=False,
            rationale=rationale,
            score=score,
            error=f"Peptide sequence length {len(tokens)} exceeds max length {max_length}.",
        )

    return PeptideCandidate(
        sequence=normalized,
        original_sequence=raw_sequence,
        engine=engine,
        valid=True,
        rationale=rationale,
        score=score,
        properties=_peptide_properties(normalized),
    )


def _dedupe_peptide_candidates(
    candidates: Sequence[PeptideCandidate],
) -> List[PeptideCandidate]:
    seen: set[str] = set()
    out: List[PeptideCandidate] = []
    for candidate in candidates:
        if candidate.valid and candidate.sequence:
            if candidate.sequence in seen:
                continue
            seen.add(candidate.sequence)
        out.append(candidate)
    return out


def _sequence_similarity(sequence: str, seed_sequence: Optional[str]) -> Optional[float]:
    seed = _normalize_peptide_sequence(seed_sequence)
    target = _normalize_peptide_sequence(sequence)
    if not seed or not target:
        return None

    seed_tokens = seed.split()
    target_tokens = target.split()
    max_len = max(len(seed_tokens), len(target_tokens))
    if max_len == 0:
        return None

    matches = sum(
        1 for left, right in zip(seed_tokens, target_tokens, strict=False) if left == right
    )
    return round(matches / max_len, 4)


def _compact_peptide_preview(
    candidates: Sequence[Any],
    *,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    preview: List[Dict[str, Any]] = []
    for candidate in list(candidates or [])[:limit]:
        item: Dict[str, Any] = {}
        if isinstance(candidate, str):
            item["sequence"] = candidate
        elif isinstance(candidate, dict):
            sequence = candidate.get("sequence") or candidate.get("peptide") or candidate.get("seq")
            if sequence:
                item["sequence"] = str(sequence)
            if candidate.get("valid") is not None:
                item["valid"] = bool(candidate["valid"])
            if candidate.get("score") is not None:
                item["score"] = candidate["score"]
            if candidate.get("error"):
                item["error"] = str(candidate["error"])
        if item:
            preview.append(item)
    return preview


def _peptide_design_artifact_rel_path(
    session_key: str,
    run_id: str,
    *,
    session_state: Optional[Dict[str, Any]] = None,
) -> str:
    safe_key = str(session_key or "peptide_design").strip() or "peptide_design"
    return operation_rel_path(
        OutputOperation.ANALOG_GENERATION,
        PEPTIDE_DESIGN_ARTIFACT_DIR,
        safe_key,
        f"{run_id}.{PEPTIDE_DESIGN_ARTIFACT_FORMAT}",
        session_state=session_state,
        workflow_slug="peptide_design",
    )


def _save_peptide_design_artifact(
    session_state: Dict[str, Any],
    *,
    session_key: str,
    candidates: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    counter = int(session_state.get("_peptide_design_run_counter", 0)) + 1
    session_state["_peptide_design_run_counter"] = counter
    run_id = f"pep_cset_{counter:03d}"
    rel_path = _peptide_design_artifact_rel_path(
        session_key,
        run_id,
        session_state=session_state,
    )
    payload = {
        "peptide_candidate_set_id": run_id,
        "candidates": list(candidates),
        "metadata": metadata,
    }
    with S3.open(rel_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return {
        "peptide_candidate_set_id": run_id,
        "artifact_path": S3.path(rel_path),
        "artifact_rel_path": rel_path,
        "artifact_format": PEPTIDE_DESIGN_ARTIFACT_FORMAT,
    }


class WAEPeptideDesignEngine:
    """Peptide design engine backed by the existing Peptide WAE model."""

    engine_name = "wae"
    _SUPPORTED_MODES = {"sample", "analog", "neighborhood", "interpolate"}

    def __init__(self, toolkit: "PeptideDesignerToolkit"):
        self.toolkit = toolkit

    def supports(self, generation_mode: str) -> bool:
        return generation_mode in self._SUPPORTED_MODES

    def design(self, request: PeptideDesignRequest) -> PeptideDesignResult:
        mode = request.generation_mode
        if not self.supports(mode):
            raise PeptideDesignerError(f"WAE engine does not support mode: {mode}")

        if mode == "interpolate":
            sequence2 = request.constraints.get("sequence2") or request.constraints.get("seq2")
            if not request.seed_sequence or not sequence2:
                raise PeptideDesignerError(
                    "seed_sequence and constraints['sequence2'] are required for interpolation."
                )
            raw = self.toolkit.interpolate_peptides(
                seq1=request.seed_sequence,
                seq2=sequence2,
                n_steps=request.n_candidates,
                temperature=request.temperature,
                decode_mode=request.decode_mode,
                method=str(request.constraints.get("method") or "linear"),
            )
        elif mode in {"analog", "neighborhood"} or request.seed_sequence:
            if not request.seed_sequence:
                raise PeptideDesignerError("seed_sequence is required for analog generation.")
            raw = self.toolkit.explore_latent_neighborhood(
                base_sequence=request.seed_sequence,
                noise_scale=request.noise_scale,
                n_neighbors=request.n_candidates,
                temperature=request.temperature,
                decode_mode=request.decode_mode,
            )
        else:
            raw = self.toolkit.sample_peptides(
                n_samples=request.n_candidates,
                latent_std=request.latent_std,
                temperature=request.temperature,
                decode_mode=request.decode_mode,
                filter_valid_unique=False,
                return_format="list",
            )

        candidates = [
            _validate_peptide_candidate(sequence, engine=self.engine_name, rationale=request.goal)
            for sequence in raw
        ]
        return PeptideDesignResult(
            engine=self.engine_name,
            candidates=_dedupe_peptide_candidates(candidates),
            metadata={
                "generation_mode": mode,
                "n_requested": request.n_candidates,
                "seed_sequence": request.seed_sequence,
            },
        )


class LLMPeptideDesignEngine:
    """Peptide design engine backed by an LLM sequence proposal step."""

    engine_name = "llm"
    _SUPPORTED_MODES = {"sample", "design", "analog", "neighborhood"}

    def __init__(self, model: Model):
        self.model = model

    def supports(self, generation_mode: str) -> bool:
        return generation_mode in self._SUPPORTED_MODES

    def design(self, request: PeptideDesignRequest) -> PeptideDesignResult:
        if not self.supports(request.generation_mode):
            raise PeptideDesignerError(
                f"LLM engine does not support mode: {request.generation_mode}"
            )

        prompt = self._build_prompt(request)
        proposer = Agent(
            model=self.model,
            name="llm_peptide_design_engine",
            description="Propose chemically plausible peptide sequence candidates.",
            instructions=[
                "Return only peptide sequences using single-letter amino acid codes.",
                "Do not return SMILES, protein FASTA headers, explanatory markdown, or names only.",
                "Respect the requested candidate count and constraints as much as possible.",
                "Keep sequences at or below 25 amino acids unless the user explicitly asks otherwise.",
                "Prefer space-separated single-letter amino acid sequences in the output.",
            ],
            output_schema=_LLMPeptideDesignResponse,
            structured_outputs=True,
            use_json_mode=True,
            markdown=False,
            telemetry=False,
        )
        response = proposer.run(prompt, stream=False)
        llm_response = self._parse_response(response.content)

        candidates = [
            _validate_peptide_candidate(
                item.sequence,
                engine=self.engine_name,
                rationale=item.rationale,
                score=item.score,
            )
            for item in llm_response.candidates
        ]
        return PeptideDesignResult(
            engine=self.engine_name,
            candidates=_dedupe_peptide_candidates(candidates),
            metadata={
                "generation_mode": request.generation_mode,
                "n_requested": request.n_candidates,
                "seed_sequence": request.seed_sequence,
                "constraints": request.constraints,
            },
        )

    def _build_prompt(self, request: PeptideDesignRequest) -> str:
        return (
            "Design peptide sequence candidates.\n"
            f"Goal: {request.goal}\n"
            f"Generation mode: {request.generation_mode}\n"
            f"Requested candidates: {request.n_candidates}\n"
            f"Seed peptide sequence: {request.seed_sequence or 'none'}\n"
            f"Constraints: {json.dumps(request.constraints or {}, sort_keys=True)}\n"
            "Return candidates as structured data with sequence, rationale, and optional score."
        )

    def _parse_response(self, content: Any) -> _LLMPeptideDesignResponse:
        if isinstance(content, _LLMPeptideDesignResponse):
            return content
        if isinstance(content, dict):
            return _LLMPeptideDesignResponse.model_validate(content)
        if isinstance(content, str):
            try:
                return _LLMPeptideDesignResponse.model_validate_json(content)
            except Exception:
                return _LLMPeptideDesignResponse.model_validate(json.loads(content))
        if hasattr(content, "model_dump"):
            return _LLMPeptideDesignResponse.model_validate(content.model_dump())
        raise PeptideDesignerError(f"Unsupported LLM design response type: {type(content)!r}")


class PeptideDesignerToolkit(Toolkit):
    """
    Facade toolkit for peptide sequence design.

    This class provides a public Peptide Designer interface over a deepchemography
    Peptide WAE model for encoding amino acid sequences to latent representations
    and sampling new peptides from the latent space.

    Input format: Space-separated single-letter amino acid codes
    Example: "M L L L L L A L A L L A L L L A L L L"
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the PeptideDesignerToolkit.

        Args:
            model_path: Path to the trained WAE model directory.
                       If None, uses default path or downloads from HuggingFace.
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__("peptide_designer")

        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set up model path
        if model_path is None:
            model_path = os.getenv(
                "PEPTIDE_DESIGNER_MODEL_PATH", DEFAULT_PEPTIDE_DESIGNER_MODEL_PATH
            )
            if model_path == DEFAULT_PEPTIDE_DESIGNER_MODEL_PATH:
                logger.info(f"Using default Peptide Designer model path: {model_path}")
            else:
                logger.info(f"Using PEPTIDE_DESIGNER_MODEL_PATH from environment: {model_path}")

        self.model_path = model_path

        # Check if model exists, download from HuggingFace if not
        self._ensure_model_exists()

        # Initialize model components
        self.model = None
        self.vocab = None
        self.config = None
        self._load_model()

        self.wae_engine = WAEPeptideDesignEngine(self)

        # Register peptide design facade tools
        self.register(self.list_design_engines)
        self.register(self.design_peptides)
        self.register(self.generate_peptide_analogs)
        self.register(self.design_peptide_interpolation)
        self.register(self.validate_design_candidates)
        self.register(self.rank_design_candidates)
        self.register(self.load_peptide_design_candidates)

        # Register low-level peptide design tools
        self.register(self.encode_peptides)
        self.register(self.decode_latent)
        self.register(self.sample_peptides)
        self.register(self.interpolate_peptides)
        self.register(self.reconstruct_sequence)
        self.register(self.get_latent_dimension)
        self.register(self.validate_model_loaded)
        self.register(self.explore_latent_neighborhood)
        self.register(self.get_model_info)

    def list_design_engines(self) -> Dict[str, Any]:
        """
        List available peptide design engines and their supported modes.

        Returns:
            Dictionary describing available peptide design engines.
        """
        return {
            "engines": [
                {
                    "name": "wae",
                    "description": "Peptide WAE latent-space generation for amino acid sequences.",
                    "supported_modes": sorted(WAEPeptideDesignEngine._SUPPORTED_MODES),
                },
                {
                    "name": "llm",
                    "description": "LLM peptide sequence proposal followed by sequence validation.",
                    "supported_modes": sorted(LLMPeptideDesignEngine._SUPPORTED_MODES),
                },
            ],
            "default_engine": "wae",
        }

    def design_peptides(
        self,
        goal: str,
        engine: str = "wae",
        n_candidates: int = 20,
        seed_sequence: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        generation_mode: str = "sample",
        temperature: float = 1.0,
        decode_mode: str = "categorical",
        noise_scale: float = 0.1,
        latent_std: float = 1.0,
        include_invalid: bool = False,
        return_format: SampleReturnFormat = "summary",
        session_key: str = "designed_peptides",
        agent: Optional[Agent] = None,
        session_state: Optional[Dict[str, Any]] = None,
        _source_tool: str = "design_peptides",
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Design peptide candidates using a selected generative engine.

        Args:
            goal: Natural-language design objective or rationale.
            engine: Design engine name: "wae" or "llm".
            n_candidates: Number of candidates to attempt.
            seed_sequence: Optional seed peptide for analog/neighborhood design.
            constraints: Optional structured constraints, such as desired motifs.
            generation_mode: "sample", "design", "analog", "neighborhood", or "interpolate".
            temperature: Sampling temperature for compatible engines.
            decode_mode: Decode mode for compatible engines.
            noise_scale: Latent perturbation scale for WAE analog generation.
            latent_std: Standard deviation for WAE prior sampling.
            include_invalid: Whether to keep invalid candidates in returned/stored results.
            return_format: "summary" saves full results as an artifact and stores a compact
                session-state pointer; "list" returns all inline.
            session_key: Session-state key for the artifact pointer in summary mode.
            agent: Agent instance auto-injected by Agno.
            session_state: Shared session state auto-injected by Agno.

        Returns:
            Compact summary or a list of candidate dictionaries.
        """
        if n_candidates <= 0:
            raise PeptideDesignerError("n_candidates must be positive.")

        request = PeptideDesignRequest(
            goal=goal,
            n_candidates=n_candidates,
            seed_sequence=seed_sequence,
            constraints=constraints or {},
            generation_mode=generation_mode,
            temperature=temperature,
            decode_mode=decode_mode,
            noise_scale=noise_scale,
            latent_std=latent_std,
        )

        result = self._get_engine(engine, agent).design(request)
        candidates = result.candidates if include_invalid else result.valid_candidates()
        candidates = self._rank_candidate_objects(candidates, seed_sequence=seed_sequence)
        candidate_dicts = [candidate.to_dict() for candidate in candidates]

        state_targets = update_state_targets(agent, session_state)
        selected_artifact: Dict[str, Any] = {}
        selected_analysis_id: Optional[str] = None
        for state in state_targets:
            use_state_for_summary = state is session_state or (
                session_state is None and not selected_artifact
            )
            metadata = {
                "origin_agent": "peptide_designer",
                "generation_engine": result.engine,
                "generation_mode": generation_mode,
                "source_tool": _source_tool,
                "session_key": session_key,
                "goal": goal,
                "seed_sequence": seed_sequence,
                "count_attempted": len(result.candidates),
                "count_returned": len(candidate_dicts),
                "engine_metadata": result.metadata,
            }
            artifact = _save_peptide_design_artifact(
                state,
                session_key=session_key,
                candidates=candidate_dicts,
                metadata=metadata,
            )
            state[session_key] = {
                **artifact,
                "origin_agent": "peptide_designer",
                "generation_engine": result.engine,
                "generation_mode": generation_mode,
                "count_attempted": len(result.candidates),
                "count_returned": len(candidate_dicts),
                "preview": _compact_peptide_preview(candidate_dicts),
            }
            analysis_id = register_session_object(
                state,
                "analysis",
                {
                    "analysis_type": "peptide_design",
                    "engine": result.engine,
                    "generation_mode": generation_mode,
                    "goal": goal,
                    "session_key": session_key,
                    "count_attempted": len(result.candidates),
                    "count_returned": len(candidate_dicts),
                    "artifact_path": artifact["artifact_path"],
                    "artifact_rel_path": artifact["artifact_rel_path"],
                    "peptide_candidate_set_id": artifact["peptide_candidate_set_id"],
                },
                label=f"Peptide design run ({result.engine})",
                source_agent=getattr(agent, "name", None),
                source_tool=_source_tool,
                set_current=True,
                current_role="analysis",
            )
            if use_state_for_summary:
                selected_artifact = artifact
                selected_analysis_id = analysis_id

        if return_format == "list" or not state_targets:
            if return_format == "summary" and not state_targets:
                logger.info(
                    "design_peptides called with return_format='summary' but no session "
                    "state was available; falling back to list."
                )
            return candidate_dicts

        return {
            "engine": result.engine,
            "generation_mode": generation_mode,
            "count_attempted": len(result.candidates),
            "count_returned": len(candidate_dicts),
            "include_invalid": include_invalid,
            "preview": _compact_peptide_preview(candidate_dicts),
            "session_key": session_key,
            "registered_analysis_id": selected_analysis_id,
            "peptide_candidate_set_id": selected_artifact.get("peptide_candidate_set_id"),
            "artifact_path": selected_artifact.get("artifact_path"),
            "artifact_format": selected_artifact.get("artifact_format"),
            "metadata": result.metadata,
            "note": (
                f"Full {len(candidate_dicts)}-item peptide design result saved as an "
                "artifact. Use session_key, peptide_candidate_set_id, or artifact_path "
                "for downstream analysis."
            ),
        }

    def generate_peptide_analogs(
        self,
        seed_sequence: str,
        goal: str = "Generate close peptide analogs of the seed sequence.",
        engine: str = "wae",
        n_analogs: int = 10,
        noise_scale: float = 0.1,
        temperature: float = 1.0,
        include_invalid: bool = False,
        return_format: SampleReturnFormat = "summary",
        session_key: str = "designed_peptide_analogs",
        agent: Optional[Agent] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate peptide analogs around a seed sequence.

        Args:
            seed_sequence: Seed peptide sequence.
            goal: Design objective.
            engine: "wae" or "llm".
            n_analogs: Number of analogs to generate.
            noise_scale: Latent perturbation scale for the WAE engine.
            temperature: Sampling temperature.
            include_invalid: Whether to keep invalid candidates.
            return_format: "summary" or "list".
            session_key: Session-state key for summary mode.
            agent: Agent instance auto-injected by Agno.
            session_state: Shared session state auto-injected by Agno.

        Returns:
            Compact summary or list of analog candidate dictionaries.
        """
        return self.design_peptides(
            goal=goal,
            engine=engine,
            n_candidates=n_analogs,
            seed_sequence=seed_sequence,
            generation_mode="analog",
            temperature=temperature,
            noise_scale=noise_scale,
            include_invalid=include_invalid,
            return_format=return_format,
            session_key=session_key,
            agent=agent,
            session_state=session_state,
            _source_tool="generate_peptide_analogs",
        )

    def design_peptide_interpolation(
        self,
        sequence1: str,
        sequence2: str,
        n_steps: int = 10,
        temperature: float = 1.0,
        method: str = "linear",
        return_format: SampleReturnFormat = "summary",
        session_key: str = "designed_peptide_interpolation",
        agent: Optional[Agent] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Interpolate between two peptides using the WAE engine.

        Args:
            sequence1: First endpoint peptide sequence.
            sequence2: Second endpoint peptide sequence.
            n_steps: Number of interpolation steps.
            temperature: Decoding temperature.
            method: Interpolation method: "linear", "slerp", or "tanh".
            return_format: "summary" or "list".
            session_key: Session-state key for summary mode.
            agent: Agent instance auto-injected by Agno.
            session_state: Shared session state auto-injected by Agno.

        Returns:
            Compact summary or list of interpolation candidate dictionaries.
        """
        return self.design_peptides(
            goal="Interpolate between two peptide sequences in latent space.",
            engine="wae",
            n_candidates=n_steps,
            seed_sequence=sequence1,
            constraints={"sequence2": sequence2, "method": method},
            generation_mode="interpolate",
            temperature=temperature,
            decode_mode="greedy",
            return_format=return_format,
            session_key=session_key,
            agent=agent,
            session_state=session_state,
            _source_tool="design_peptide_interpolation",
        )

    def validate_design_candidates(
        self, sequences: Union[str, List[str]], engine: str = "manual"
    ) -> List[Dict[str, Any]]:
        """
        Validate, normalize, and annotate proposed peptide design candidates.

        Args:
            sequences: Single peptide sequence or list of peptide sequence candidates.
            engine: Provenance label to attach to the validation results.

        Returns:
            Candidate dictionaries including validity, normalized sequence, and properties.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        candidates = [
            _validate_peptide_candidate(sequence, engine=engine) for sequence in sequences
        ]
        return [candidate.to_dict() for candidate in _dedupe_peptide_candidates(candidates)]

    def rank_design_candidates(
        self,
        candidates: List[Dict[str, Any]],
        seed_sequence: Optional[str] = None,
        prefer_shorter: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Rank validated peptide design candidates.

        Args:
            candidates: Candidate dictionaries from design or validation tools.
            seed_sequence: Optional seed sequence for position-wise similarity scoring.
            prefer_shorter: If True, use shorter valid peptides as a secondary score.

        Returns:
            Ranked candidate dictionaries.
        """
        ranked: List[Dict[str, Any]] = []
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
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Load peptide design candidates from a session pointer or artifact path.

        Args:
            reference: Session key, artifact path, or artifact-relative path.
            include_candidates: Whether to include the full candidate list.
            session_state: Shared session state auto-injected by Agno.

        Returns:
            Artifact metadata and optionally full peptide candidate dictionaries.
        """
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
        result = {
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

    def _get_hf_token_safe(self):
        """Get HuggingFace token safely from environment or local login."""
        try:
            from huggingface_hub import get_token

            return get_token()
        except Exception as e:
            logger.warning(f"Failed to get HuggingFace token: {e}")
            return None

    def _ensure_model_exists(self):
        """
        Ensure required files exist locally; download from HuggingFace if missing.
        """
        import os
        import shutil
        from pathlib import Path

        base_path = Path(self.model_path)
        files = ["model.pt", "vocab.dict"]

        if all((base_path / f).exists() for f in files):
            logger.info(f"Peptide Designer model files found at {self.model_path}")
            return

        logger.warning(
            f"Peptide Designer model files not found at {self.model_path}. "
            "Attempting to download from HuggingFace..."
        )
        base_path.mkdir(parents=True, exist_ok=True)

        # Resolve token
        hf_token = (
            os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN") or self._get_hf_token_safe()
        )

        try:
            from huggingface_hub import snapshot_download

            cache_dir = os.path.expanduser(
                os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HOME") or str(base_path)
            )

            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

            snapshot_download(
                repo_id=HUGGINGFACE_PEPTIDE_WAE_REPO,
                cache_dir=cache_dir,
                local_dir=str(base_path),
                resume_download=True,
                token=hf_token,
            )

            # Verify files
            missing = [f for f in files if not (base_path / f).exists()]
            if missing:
                raise PeptideDesignerError(
                    f"Downloaded files incomplete. Missing: {', '.join(missing)} at {self.model_path}"
                )
            logger.info(f"Successfully fetched Peptide Designer model files into {self.model_path}")

            # Cleanup cache
            try:
                cache_dir_path = base_path / ".cache"
                if cache_dir_path.exists() and cache_dir_path.is_dir():
                    shutil.rmtree(cache_dir_path, ignore_errors=True)
            except Exception:
                pass

        except ImportError as e:
            raise PeptideDesignerError(
                "huggingface_hub not installed. Install it with: pip install huggingface_hub"
            ) from e
        except Exception as e:
            raise PeptideDesignerError(
                f"Failed to download Peptide Designer model from HuggingFace "
                f"({HUGGINGFACE_PEPTIDE_WAE_REPO}): {repr(e)}. "
                f"Original model path: {self.model_path}"
            ) from e

    def _load_model(self):
        """Load the trained Peptide Designer WAE model and vocabulary."""
        try:
            from deepchemography.peptides import PeptideVocab, PeptideWAE, get_default_config

            base_path = Path(self.model_path)
            model_file = str(base_path / "model.pt")
            vocab_file = str(base_path / "vocab.dict")

            # Load config
            self.config = get_default_config()

            # Load vocabulary
            self.vocab = PeptideVocab(vocab_file, max_seq_len=self.config["max_seq_len"])

            # Create model
            self.model = PeptideWAE(
                n_vocab=self.vocab.size(),
                max_seq_len=self.config["max_seq_len"],
                z_dim=self.config["z_dim"],
                c_dim=self.config["c_dim"],
                emb_dim=self.config["emb_dim"],
                encoder_config=self.config["encoder"],
                decoder_config=self.config["decoder"],
            )

            # Load weights
            state_dict = torch.load(model_file, map_location=self.device)
            # Filter out classifier weights if present
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
            self.model.load_state_dict(state_dict, strict=False)

            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Peptide Designer model loaded successfully from {self.model_path}")
            logger.info(f"  Vocabulary size: {self.vocab.size()}")
            logger.info(f"  Latent dimension: {self.config['z_dim']}")
            logger.info(f"  Device: {self.device}")

        except ImportError as e:
            raise PeptideDesignerError(f"Failed to import deepchemography.peptides: {e}") from e
        except Exception as e:
            raise PeptideDesignerError(f"Failed to load Peptide Designer model: {e}") from e

    def validate_model_loaded(self) -> bool:
        """
        Check if the Peptide Designer model is properly loaded.

        Returns:
            True if model is loaded and ready to use
        """
        return self.model is not None and self.vocab is not None and self.config is not None

    def get_latent_dimension(self) -> int:
        """
        Get the dimension of the latent space.

        Returns:
            Latent dimension size (100)
        """
        if not self.validate_model_loaded():
            raise PeptideDesignerError("Model not loaded")
        return self.config["z_dim"]

    def encode_peptides_array(
        self, sequences: Union[str, List[str]], batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode peptide sequences and return latent vectors as a numpy array.

        This is the public interface for obtaining raw numpy arrays of latent vectors,
        suitable for downstream operations like GTM training or projection.

        Args:
            sequences: Single sequence or list of sequences.
                      Format: space-separated amino acids, e.g., "M L L L A L A"
            batch_size: Batch size for encoding (currently processes one at a time)

        Returns:
            numpy array of shape (n_sequences, latent_dim)
        """
        return self._encode_peptides_ndarray(sequences, batch_size=batch_size)

    def _encode_peptides_ndarray(
        self, sequences: Union[str, List[str]], batch_size: int = 32
    ) -> np.ndarray:
        """
        Internal helper: encode peptide sequences and return numpy array.
        """
        if not self.validate_model_loaded():
            raise PeptideDesignerError("Model not loaded")

        # Handle single sequence
        if isinstance(sequences, str):
            sequences = [sequences]
            return_single = True
        else:
            return_single = False

        if not sequences:
            raise PeptideDesignerError("No peptide sequences provided")

        self.model.eval()
        latent_vectors = []

        with torch.no_grad():
            for seq in sequences:
                try:
                    enc_inputs = self.vocab.to_ix(seq)
                    enc_inputs = enc_inputs.to(self.device)
                    mu, _ = self.model.forward_encoder(enc_inputs)
                    latent_vectors.append(mu.cpu().numpy())
                except Exception as e:
                    logger.warning(f"Error encoding sequence '{seq}': {e}")
                    # Return zero vector for failed encodings
                    latent_vectors.append(np.zeros((1, self.config["z_dim"])))

        result = np.vstack(latent_vectors)

        if return_single:
            return result[0:1]
        return result

    def encode_peptides(
        self, sequences: Union[str, List[str]], batch_size: int = 32
    ) -> Union[List[float], List[List[float]]]:
        """
        Encode peptide sequences to latent vectors.

        Args:
            sequences: Single sequence or list of sequences.
                      Format: space-separated amino acids, e.g., "M L L L A L A"
            batch_size: Batch size for encoding (currently processes one at a time)

        Returns:
            Latent vector(s) as JSON-serializable list(s)
        """
        arr = self._encode_peptides_ndarray(sequences, batch_size=batch_size)
        if isinstance(sequences, str) or (isinstance(sequences, list) and len(sequences) == 1):
            return arr[0].tolist()
        return arr.tolist()

    def decode_latent(
        self,
        latent_vectors: Union[List[float], List[List[float]]],
        temperature: float = 1.0,
        decode_mode: str = "categorical",
        max_length: int = 25,
    ) -> List[str]:
        """
        Decode latent vectors to peptide sequences.

        Args:
            latent_vectors: Latent vector(s) to decode
            temperature: Sampling temperature (higher = more random)
            decode_mode: 'categorical' for stochastic, 'greedy' for deterministic
            max_length: Maximum sequence length (default 25)

        Returns:
            List of peptide sequences (space-separated amino acids)
        """
        if not self.validate_model_loaded():
            raise PeptideDesignerError("Model not loaded")

        # Convert to numpy array
        z = np.array(latent_vectors)
        if z.ndim == 1:
            z = z.reshape(1, -1)

        z_tensor = torch.tensor(z, dtype=torch.float32).to(self.device)
        n_samples = z_tensor.size(0)

        with torch.no_grad():
            c = self.model.sample_c_prior(n_samples)
            samples, _, _ = self.model.generate_sentences(
                n_samples,
                z=z_tensor,
                c=c,
                sample_mode=decode_mode,
                temp=temperature,
            )

        # Convert to strings
        predictions = []
        for sample in samples:
            seq_str = self.vocab.to_string(sample, print_special_tokens=False)
            predictions.append(seq_str)

        return predictions

    def sample_peptides(
        self,
        n_samples: int = 5000,
        latent_std: float = 1.0,
        temperature: float = 1.0,
        decode_mode: str = "categorical",
        max_length: int = 25,
        filter_valid_unique: bool = True,
        return_format: SampleReturnFormat = "summary",
        session_key: str = "sampled_peptides",
        agent: Optional[Agent] = None,
    ) -> Union[List[str], Dict[str, Any]]:
        """
        Sample new peptides from the latent space using Gaussian prior.

        Args:
            n_samples: Number of peptides to generate. Defaults to 5000 for
                meaningful peptide-space exploration; pass a smaller value
                explicitly for quick demos.
            latent_std: Standard deviation for Gaussian sampling
            temperature: Sampling temperature (higher = more random)
            decode_mode: 'categorical' for stochastic, 'greedy' for deterministic
            max_length: Maximum sequence length
            filter_valid_unique: If True (default), drop empty sequences and
                deduplicate before returning.
            return_format: "summary" (default) persists the full list into
                agent.session_state[session_key] and returns a compact dict with
                count, preview (first 20), and the session key. "list" returns
                the raw List[str] directly (may inflate LLM context at large N).
            session_key: Key under which the full list is stored in
                agent.session_state when return_format="summary".
            agent: Agent instance (auto-injected by agno). Required for
                "summary" format; if None, gracefully falls back to "list".

        Returns:
            Dict summary (default) or List[str] (when return_format="list" or
            no agent available).
        """
        if not self.validate_model_loaded():
            raise PeptideDesignerError("Model not loaded")

        with torch.no_grad():
            z = torch.randn(n_samples, self.config["z_dim"]).to(self.device) * latent_std
            c = self.model.sample_c_prior(n_samples)

            samples, _, _ = self.model.generate_sentences(
                n_samples,
                z=z,
                c=c,
                sample_mode=decode_mode,
                temp=temperature,
            )

        raw: List[str] = []
        for sample in samples:
            seq_str = self.vocab.to_string(sample, print_special_tokens=False)
            raw.append(seq_str)

        sampled = _filter_valid_unique_peptides(raw) if filter_valid_unique else list(raw)

        if return_format == "list" or agent is None:
            if return_format == "summary" and agent is None:
                logger.info(
                    "sample_peptides called with return_format='summary' but no agent "
                    "was provided; falling back to raw list."
                )
            return sampled

        if agent.session_state is None:
            agent.session_state = {}
        agent.session_state[session_key] = sampled

        return {
            "count_attempted": n_samples,
            "count_returned": len(sampled),
            "filter_valid_unique": filter_valid_unique,
            "preview": sampled[:20],
            "session_key": session_key,
            "note": (
                f"Full {len(sampled)}-item peptide list persisted to "
                f"agent.session_state['{session_key}']. Retrieve it from session state "
                f"for downstream analysis (encoding, clustering, activity prediction, etc.) "
                f"instead of asking for the whole list inline."
            ),
        }

    def interpolate_peptides(
        self,
        seq1: str,
        seq2: str,
        n_steps: int = 10,
        temperature: float = 1.0,
        decode_mode: str = "categorical",
        method: str = "linear",
    ) -> List[str]:
        """
        Interpolate between two peptides in latent space.

        Args:
            seq1: First peptide sequence (space-separated amino acids)
            seq2: Second peptide sequence (space-separated amino acids)
            n_steps: Number of interpolation steps (excluding endpoints)
            temperature: Sampling temperature for decoding
            decode_mode: 'categorical' for stochastic, 'greedy' for deterministic
            method: Interpolation method ('linear', 'slerp', or 'tanh')

        Returns:
            List of interpolated peptide sequences (including endpoints)
        """
        if not self.validate_model_loaded():
            raise PeptideDesignerError("Model not loaded")

        # Encode both sequences
        z1 = self._encode_peptides_ndarray([seq1])
        z2 = self._encode_peptides_ndarray([seq2])

        # Compute interpolation weights
        weights = [0.0] + [1.0 / (n_steps + 1) * i for i in range(1, n_steps + 1)] + [1.0]

        # Interpolate
        z_list = [z1]
        for w in weights[1:-1]:
            if method == "linear":
                z_interp = (1 - w) * z1 + w * z2
            elif method == "slerp":
                z1_norm = z1 / np.linalg.norm(z1)
                z2_norm = z2 / np.linalg.norm(z2)
                omega = np.arccos(np.clip(np.dot(z1_norm.flatten(), z2_norm.flatten()), -1, 1))
                if np.abs(omega) < 1e-6:
                    z_interp = (1 - w) * z1 + w * z2
                else:
                    z_interp = (np.sin((1 - w) * omega) * z1 + np.sin(w * omega) * z2) / np.sin(
                        omega
                    )
            elif method == "tanh":
                w_tanh = (np.tanh(w * 4 - 2) + 1) / 2
                z_interp = (1 - w_tanh) * z1 + w_tanh * z2
            else:
                raise PeptideDesignerError(f"Unknown interpolation method: {method}")
            z_list.append(z_interp)
        z_list.append(z2)

        # Decode
        z_array = np.vstack(z_list)
        return self.decode_latent(
            z_array.tolist(), temperature=temperature, decode_mode=decode_mode
        )

    def reconstruct_sequence(
        self, sequence: str, temperature: float = 0.1, decode_mode: str = "greedy"
    ) -> str:
        """
        Reconstruct a peptide sequence by encoding and decoding it.

        Args:
            sequence: Input peptide sequence (space-separated amino acids)
            temperature: Sampling temperature for decoding
            decode_mode: 'greedy' for deterministic, 'categorical' for stochastic

        Returns:
            Reconstructed peptide sequence
        """
        if not self.validate_model_loaded():
            raise PeptideDesignerError("Model not loaded")

        latent = self._encode_peptides_ndarray([sequence])
        reconstructed = self.decode_latent(
            latent.tolist(), temperature=temperature, decode_mode=decode_mode
        )
        return reconstructed[0]

    def explore_latent_neighborhood(
        self,
        base_sequence: str,
        noise_scale: float = 0.1,
        n_neighbors: int = 5,
        temperature: float = 1.0,
        decode_mode: str = "categorical",
    ) -> List[str]:
        """
        Explore the neighborhood of a peptide in latent space.

        Args:
            base_sequence: Base peptide sequence (space-separated amino acids)
            noise_scale: Standard deviation of noise to add
                        (0.05-0.15 = close analogs, 0.2-0.4 = moderate, 0.5+ = diverse)
            n_neighbors: Number of neighbors to generate
            temperature: Sampling temperature for decoding
            decode_mode: 'categorical' for stochastic, 'greedy' for deterministic

        Returns:
            List of generated neighbor peptide sequences
        """
        if not self.validate_model_loaded():
            raise PeptideDesignerError("Model not loaded")

        # Encode base sequence
        z_base = self._encode_peptides_ndarray([base_sequence])

        # Generate neighbors by adding noise
        z_neighbors = z_base + np.random.randn(n_neighbors, z_base.shape[1]) * noise_scale

        # Decode neighbors
        return self.decode_latent(
            z_neighbors.tolist(), temperature=temperature, decode_mode=decode_mode
        )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        if not self.validate_model_loaded():
            raise PeptideDesignerError("Model not loaded")

        return {
            "model_path": str(self.model_path),
            "vocabulary_size": self.vocab.size(),
            "latent_dimension": self.config["z_dim"],
            "condition_dimension": self.config["c_dim"],
            "embedding_dimension": self.config["emb_dim"],
            "max_sequence_length": self.config["max_seq_len"],
            "device": str(self.device),
            "encoder_type": "bidirectional_gru",
            "decoder_type": "gru",
            "supported_amino_acids": "A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, U, V, W, Y, Z",
        }

    def _get_engine(self, engine: str, agent: Optional[Agent]) -> PeptideDesignEngine:
        engine_key = engine.lower().strip()
        if engine_key == "wae":
            return self.wae_engine
        if engine_key == "llm":
            model = getattr(agent, "model", None) if agent is not None else None
            if model is None:
                raise PeptideDesignerError(
                    "LLM peptide design requires an agent with a model. Use this tool "
                    "through the Peptide Designer agent or choose engine='wae'."
                )
            return LLMPeptideDesignEngine(model)
        raise PeptideDesignerError(
            f"Unknown peptide design engine: {engine}. Available engines: wae, llm."
        )

    def _rank_candidate_objects(
        self,
        candidates: Sequence[PeptideCandidate],
        seed_sequence: Optional[str],
    ) -> List[PeptideCandidate]:
        ranked = []
        for candidate in candidates:
            if candidate.valid and candidate.sequence:
                similarity = _sequence_similarity(candidate.sequence, seed_sequence)
                if similarity is not None:
                    candidate.properties["seed_sequence_similarity"] = similarity
                    candidate.score = similarity
                elif candidate.score is None:
                    length = candidate.properties.get("length")
                    candidate.score = round(1 / float(length), 4) if length else None
            ranked.append(candidate)
        return sorted(
            ranked,
            key=lambda candidate: (
                candidate.valid,
                candidate.score if candidate.score is not None else -1,
            ),
            reverse=True,
        )
