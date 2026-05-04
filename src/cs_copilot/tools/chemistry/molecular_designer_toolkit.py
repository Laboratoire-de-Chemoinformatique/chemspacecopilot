#!/usr/bin/env python
# coding: utf-8
"""
Molecular designer toolkit and engine facade for small-molecule generation.

The public agent should reason in terms of molecular design engines. The
autoencoder remains one engine implementation, while LLM-based SMILES proposal
is exposed as another engine with the same validation and session-state output
contract.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Union

from agno.agent import Agent
from agno.models.base import Model
from agno.tools.toolkit import Toolkit
from pydantic import BaseModel, Field
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Descriptors, rdFingerprintGenerator

from cs_copilot.tools.io.session_memory import (
    register_compounds_from_candidates,
    register_generated_candidate_set,
    register_session_object,
    update_state_targets,
)

from .autoencoder_toolkit import AutoencoderToolkit
from .base_chemistry import ChemistryError
from .standardize import standardize_smiles

logger = logging.getLogger(__name__)

SampleReturnFormat = Literal["summary", "list"]


class MolecularDesignerError(ChemistryError):
    """Exception raised for molecular-designer operations."""

    pass


@dataclass
class MolecularDesignRequest:
    """Engine-independent request for small-molecule design."""

    goal: str
    n_candidates: int = 20
    seed_smiles: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    generation_mode: str = "sample"
    temperature: float = 1.0
    decode_mode: str = "sample"
    noise_scale: float = 0.1


@dataclass
class MolecularCandidate:
    """Validated molecular candidate returned by any design engine."""

    smiles: Optional[str]
    original_smiles: Optional[str]
    engine: str
    valid: bool
    rationale: Optional[str] = None
    score: Optional[float] = None
    error: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "smiles": self.smiles,
            "original_smiles": self.original_smiles,
            "engine": self.engine,
            "valid": self.valid,
            "rationale": self.rationale,
            "score": self.score,
            "error": self.error,
            "properties": self.properties,
        }


@dataclass
class MolecularDesignResult:
    """Engine-independent design result."""

    engine: str
    candidates: List[MolecularCandidate]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def valid_candidates(self) -> List[MolecularCandidate]:
        """Return valid candidates only."""
        return [candidate for candidate in self.candidates if candidate.valid]


class MolecularDesignEngine(Protocol):
    """Protocol implemented by small-molecule generative engines."""

    engine_name: str

    def supports(self, generation_mode: str) -> bool:
        """Return whether this engine supports a generation mode."""
        ...

    def design(self, request: MolecularDesignRequest) -> MolecularDesignResult:
        """Generate candidate molecules for a request."""
        ...


class _LLMCandidate(BaseModel):
    smiles: str = Field(..., description="Candidate molecule as a SMILES string.")
    rationale: Optional[str] = Field(
        default=None, description="Short reason this molecule matches the design goal."
    )
    score: Optional[float] = Field(
        default=None, description="Optional model confidence or desirability score from 0 to 1."
    )


class _LLMDesignResponse(BaseModel):
    candidates: List[_LLMCandidate] = Field(
        default_factory=list, description="Candidate molecules proposed by the LLM."
    )


def _candidate_properties(smiles: str) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 4),
        "logp": round(Descriptors.MolLogP(mol), 4),
        "tpsa": round(Descriptors.TPSA(mol), 4),
        "hbd": int(Descriptors.NumHDonors(mol)),
        "hba": int(Descriptors.NumHAcceptors(mol)),
        "rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
        "qed": round(float(QED.qed(mol)), 4),
    }


def _validate_candidate(
    raw_smiles: Any,
    *,
    engine: str,
    rationale: Optional[str] = None,
    score: Optional[float] = None,
) -> MolecularCandidate:
    if not isinstance(raw_smiles, str) or not raw_smiles.strip():
        return MolecularCandidate(
            smiles=None,
            original_smiles=str(raw_smiles) if raw_smiles is not None else None,
            engine=engine,
            valid=False,
            rationale=rationale,
            score=score,
            error="SMILES must be a non-empty string.",
        )

    standardized = standardize_smiles(raw_smiles)
    if standardized is None:
        return MolecularCandidate(
            smiles=None,
            original_smiles=raw_smiles,
            engine=engine,
            valid=False,
            rationale=rationale,
            score=score,
            error="SMILES could not be standardized.",
        )

    mol = Chem.MolFromSmiles(standardized)
    if mol is None:
        return MolecularCandidate(
            smiles=None,
            original_smiles=raw_smiles,
            engine=engine,
            valid=False,
            rationale=rationale,
            score=score,
            error="SMILES is invalid after standardization.",
        )

    canonical = Chem.MolToSmiles(mol)
    return MolecularCandidate(
        smiles=canonical,
        original_smiles=raw_smiles,
        engine=engine,
        valid=True,
        rationale=rationale,
        score=score,
        properties=_candidate_properties(canonical),
    )


def _dedupe_candidates(candidates: Sequence[MolecularCandidate]) -> List[MolecularCandidate]:
    seen: set[str] = set()
    out: List[MolecularCandidate] = []
    for candidate in candidates:
        if candidate.valid and candidate.smiles:
            if candidate.smiles in seen:
                continue
            seen.add(candidate.smiles)
        out.append(candidate)
    return out


def _similarity_to_seed(smiles: str, seed_smiles: Optional[str]) -> Optional[float]:
    if not seed_smiles:
        return None
    seed_std = standardize_smiles(seed_smiles)
    if seed_std is None:
        return None
    seed_mol = Chem.MolFromSmiles(seed_std)
    mol = Chem.MolFromSmiles(smiles)
    if seed_mol is None or mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return round(
        float(
            DataStructs.TanimotoSimilarity(
                generator.GetFingerprint(seed_mol), generator.GetFingerprint(mol)
            )
        ),
        4,
    )


class AutoencoderDesignEngine:
    """Small-molecule design engine backed by the existing LSTM autoencoder."""

    engine_name = "autoencoder"
    _SUPPORTED_MODES = {"sample", "analog", "neighborhood", "interpolate"}

    def __init__(self, toolkit: AutoencoderToolkit):
        self.toolkit = toolkit

    def supports(self, generation_mode: str) -> bool:
        return generation_mode in self._SUPPORTED_MODES

    def design(self, request: MolecularDesignRequest) -> MolecularDesignResult:
        mode = request.generation_mode
        if not self.supports(mode):
            raise MolecularDesignerError(f"Autoencoder engine does not support mode: {mode}")

        if mode == "interpolate":
            smiles2 = request.constraints.get("smiles2")
            if not request.seed_smiles or not smiles2:
                raise MolecularDesignerError(
                    "seed_smiles and constraints['smiles2'] are required for interpolation."
                )
            raw = self.toolkit.interpolate_molecules(
                smiles1=request.seed_smiles,
                smiles2=smiles2,
                n_steps=request.n_candidates,
                temperature=request.temperature,
                decode_mode=request.decode_mode,
            )
        elif mode in {"analog", "neighborhood"} or request.seed_smiles:
            if not request.seed_smiles:
                raise MolecularDesignerError("seed_smiles is required for analog generation.")
            raw = self.toolkit.explore_latent_neighborhood(
                base_smiles=request.seed_smiles,
                noise_scale=request.noise_scale,
                n_neighbors=request.n_candidates,
                temperature=request.temperature,
                decode_mode=request.decode_mode,
            )
        else:
            raw = self.toolkit.sample_molecules(
                n_samples=request.n_candidates,
                temperature=request.temperature,
                decode_mode=request.decode_mode,
                filter_valid_unique=False,
                return_format="list",
            )

        candidates = [
            _validate_candidate(smiles, engine=self.engine_name, rationale=request.goal)
            for smiles in raw
        ]
        return MolecularDesignResult(
            engine=self.engine_name,
            candidates=_dedupe_candidates(candidates),
            metadata={
                "generation_mode": mode,
                "n_requested": request.n_candidates,
                "seed_smiles": request.seed_smiles,
            },
        )


class LLMDesignEngine:
    """Small-molecule design engine backed by an LLM SMILES proposal step."""

    engine_name = "llm"
    _SUPPORTED_MODES = {"sample", "design", "analog", "neighborhood"}

    def __init__(self, model: Model):
        self.model = model

    def supports(self, generation_mode: str) -> bool:
        return generation_mode in self._SUPPORTED_MODES

    def design(self, request: MolecularDesignRequest) -> MolecularDesignResult:
        if not self.supports(request.generation_mode):
            raise MolecularDesignerError(
                f"LLM engine does not support mode: {request.generation_mode}"
            )

        prompt = self._build_prompt(request)
        proposer = Agent(
            model=self.model,
            name="llm_molecular_design_engine",
            description="Propose chemically plausible small-molecule SMILES candidates.",
            instructions=[
                "Return only small-molecule SMILES candidates.",
                "Do not return peptide sequences, reaction SMILES, explanatory markdown, or names only.",
                "Respect the requested candidate count and constraints as much as possible.",
                "Prefer syntactically valid, neutral, drug-like organic molecules unless asked otherwise.",
            ],
            output_schema=_LLMDesignResponse,
            structured_outputs=True,
            use_json_mode=True,
            markdown=False,
            telemetry=False,
        )
        response = proposer.run(prompt, stream=False)
        llm_response = self._parse_response(response.content)

        candidates = [
            _validate_candidate(
                item.smiles,
                engine=self.engine_name,
                rationale=item.rationale,
                score=item.score,
            )
            for item in llm_response.candidates
        ]
        return MolecularDesignResult(
            engine=self.engine_name,
            candidates=_dedupe_candidates(candidates),
            metadata={
                "generation_mode": request.generation_mode,
                "n_requested": request.n_candidates,
                "seed_smiles": request.seed_smiles,
                "constraints": request.constraints,
            },
        )

    def _build_prompt(self, request: MolecularDesignRequest) -> str:
        return (
            "Design small-molecule SMILES candidates.\n"
            f"Goal: {request.goal}\n"
            f"Generation mode: {request.generation_mode}\n"
            f"Requested candidates: {request.n_candidates}\n"
            f"Seed SMILES: {request.seed_smiles or 'none'}\n"
            f"Constraints: {json.dumps(request.constraints or {}, sort_keys=True)}\n"
            "Return candidates as structured data with smiles, rationale, and optional score."
        )

    def _parse_response(self, content: Any) -> _LLMDesignResponse:
        if isinstance(content, _LLMDesignResponse):
            return content
        if isinstance(content, dict):
            return _LLMDesignResponse.model_validate(content)
        if isinstance(content, str):
            try:
                return _LLMDesignResponse.model_validate_json(content)
            except Exception:
                return _LLMDesignResponse.model_validate(json.loads(content))
        if hasattr(content, "model_dump"):
            return _LLMDesignResponse.model_validate(content.model_dump())
        raise MolecularDesignerError(f"Unsupported LLM design response type: {type(content)!r}")


class MolecularDesignerToolkit(Toolkit):
    """Facade toolkit for small-molecule design engines."""

    def __init__(self, autoencoder_toolkit: Optional[AutoencoderToolkit] = None):
        super().__init__("molecular_designer")
        self.autoencoder_toolkit = autoencoder_toolkit or AutoencoderToolkit()
        self.autoencoder_engine = AutoencoderDesignEngine(self.autoencoder_toolkit)

        self.register(self.list_design_engines)
        self.register(self.design_molecules)
        self.register(self.generate_analogs)
        self.register(self.interpolate_molecules)
        self.register(self.validate_design_candidates)
        self.register(self.rank_design_candidates)

    def list_design_engines(self) -> Dict[str, Any]:
        """
        List available molecular design engines and their supported modes.

        Returns:
            Dictionary describing available small-molecule design engines.
        """
        return {
            "engines": [
                {
                    "name": "autoencoder",
                    "description": "LSTM autoencoder latent-space generation for SMILES.",
                    "supported_modes": sorted(AutoencoderDesignEngine._SUPPORTED_MODES),
                },
                {
                    "name": "llm",
                    "description": "LLM SMILES proposal followed by RDKit validation.",
                    "supported_modes": sorted(LLMDesignEngine._SUPPORTED_MODES),
                },
            ],
            "default_engine": "autoencoder",
        }

    def design_molecules(
        self,
        goal: str,
        engine: str = "autoencoder",
        n_candidates: int = 20,
        seed_smiles: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        generation_mode: str = "sample",
        temperature: float = 1.0,
        decode_mode: str = "sample",
        noise_scale: float = 0.1,
        include_invalid: bool = False,
        return_format: SampleReturnFormat = "summary",
        session_key: str = "designed_molecules",
        agent: Optional[Agent] = None,
        session_state: Optional[Dict[str, Any]] = None,
        _source_tool: str = "design_molecules",
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Design small-molecule candidates using a selected generative engine.

        Args:
            goal: Natural-language design objective or rationale.
            engine: Design engine name: "autoencoder" or "llm".
            n_candidates: Number of candidates to attempt.
            seed_smiles: Optional seed SMILES for analog/neighborhood design.
            constraints: Optional structured constraints, such as desired property ranges.
            generation_mode: "sample", "analog", "neighborhood", or "interpolate".
            temperature: Sampling temperature for compatible engines.
            decode_mode: Decode mode for compatible engines.
            noise_scale: Latent perturbation scale for analog generation.
            include_invalid: Whether to keep invalid candidates in returned/stored results.
            return_format: "summary" persists full results in session state; "list" returns all inline.
            session_key: Session-state key for full results when return_format="summary".
            agent: Agent instance auto-injected by Agno.
            session_state: Shared session state auto-injected by Agno.

        Returns:
            Compact summary or a list of candidate dictionaries.
        """
        if n_candidates <= 0:
            raise MolecularDesignerError("n_candidates must be positive.")

        request = MolecularDesignRequest(
            goal=goal,
            n_candidates=n_candidates,
            seed_smiles=seed_smiles,
            constraints=constraints or {},
            generation_mode=generation_mode,
            temperature=temperature,
            decode_mode=decode_mode,
            noise_scale=noise_scale,
        )

        result = self._get_engine(engine, agent).design(request)
        candidates = result.candidates if include_invalid else result.valid_candidates()
        candidates = self._rank_candidate_objects(candidates, seed_smiles=seed_smiles)
        candidate_dicts = [candidate.to_dict() for candidate in candidates]

        state_targets = update_state_targets(agent, session_state)
        registered_compound_ids: List[str] = []
        registered_candidate_set_id: Optional[str] = None
        for state in state_targets:
            state[session_key] = candidate_dicts
            candidate_ids = register_compounds_from_candidates(
                state,
                candidate_dicts,
                source_agent=getattr(agent, "name", None),
                source_tool=_source_tool,
                label_prefix=f"{result.engine} design candidate",
                related={
                    "session_key": session_key,
                    "goal": goal,
                    "generation_mode": generation_mode,
                    "seed_smiles": seed_smiles,
                },
                provenance={
                    "origin_type": "generated",
                    "origin_agent": "molecular_designer",
                    "generation_engine": result.engine,
                },
                set_current_first=bool(candidate_dicts),
            )
            if state is session_state:
                registered_compound_ids = candidate_ids
            candidate_set_id = register_generated_candidate_set(
                state,
                candidate_ids,
                source_agent=getattr(agent, "name", None),
                source_tool=_source_tool,
                origin_agent="molecular_designer",
                generation_engine=result.engine,
                generation_mode=generation_mode,
                session_key=session_key,
                label=f"Molecular Designer candidates ({result.engine})",
                seed_smiles=seed_smiles,
                goal=goal,
                count_attempted=len(result.candidates),
                metadata=result.metadata,
            )
            if state is session_state:
                registered_candidate_set_id = candidate_set_id
            register_session_object(
                state,
                "analysis",
                {
                    "analysis_type": "molecular_design",
                    "engine": result.engine,
                    "generation_mode": generation_mode,
                    "goal": goal,
                    "session_key": session_key,
                    "count_attempted": len(result.candidates),
                    "count_returned": len(candidate_dicts),
                    "compound_ids": candidate_ids,
                    "candidate_set_id": candidate_set_id,
                },
                label=f"Molecular design run ({result.engine})",
                source_agent=getattr(agent, "name", None),
                source_tool=_source_tool,
                set_current=True,
                current_role="analysis",
            )

        if return_format == "list" or not state_targets:
            if return_format == "summary" and not state_targets:
                logger.info(
                    "design_molecules called with return_format='summary' but no session "
                    "state was available; falling back to list."
                )
            return candidate_dicts

        return {
            "engine": result.engine,
            "generation_mode": generation_mode,
            "count_attempted": len(result.candidates),
            "count_returned": len(candidate_dicts),
            "include_invalid": include_invalid,
            "preview": candidate_dicts[:20],
            "session_key": session_key,
            "registered_compound_ids": registered_compound_ids,
            "registered_candidate_set_id": registered_candidate_set_id,
            "metadata": result.metadata,
            "note": (
                f"Full {len(candidate_dicts)}-item molecular design result persisted to "
                f"session_state['{session_key}']; use that key for downstream "
                "ranking, filtering, GTM projection, or reporting."
            ),
        }

    def generate_analogs(
        self,
        seed_smiles: str,
        goal: str = "Generate close small-molecule analogs of the seed structure.",
        engine: str = "autoencoder",
        n_analogs: int = 10,
        noise_scale: float = 0.1,
        temperature: float = 0.5,
        include_invalid: bool = False,
        return_format: SampleReturnFormat = "summary",
        session_key: str = "designed_analogs",
        agent: Optional[Agent] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate small-molecule analogs around a seed SMILES.

        Args:
            seed_smiles: Seed molecule as SMILES.
            goal: Design objective.
            engine: "autoencoder" or "llm".
            n_analogs: Number of analogs to generate.
            noise_scale: Latent perturbation scale for the autoencoder engine.
            temperature: Sampling temperature.
            include_invalid: Whether to keep invalid candidates.
            return_format: "summary" or "list".
            session_key: Session-state key for summary mode.
            agent: Agent instance auto-injected by Agno.
            session_state: Shared session state auto-injected by Agno.

        Returns:
            Compact summary or list of analog candidate dictionaries.
        """
        return self.design_molecules(
            goal=goal,
            engine=engine,
            n_candidates=n_analogs,
            seed_smiles=seed_smiles,
            generation_mode="analog",
            temperature=temperature,
            noise_scale=noise_scale,
            include_invalid=include_invalid,
            return_format=return_format,
            session_key=session_key,
            agent=agent,
            session_state=session_state,
            _source_tool="generate_analogs",
        )

    def interpolate_molecules(
        self,
        smiles1: str,
        smiles2: str,
        n_steps: int = 10,
        temperature: float = 0.1,
        return_format: SampleReturnFormat = "summary",
        session_key: str = "designed_interpolation",
        agent: Optional[Agent] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Interpolate between two molecules using the autoencoder engine.

        Args:
            smiles1: First endpoint SMILES.
            smiles2: Second endpoint SMILES.
            n_steps: Number of interpolation steps.
            temperature: Decoding temperature.
            return_format: "summary" or "list".
            session_key: Session-state key for summary mode.
            agent: Agent instance auto-injected by Agno.
            session_state: Shared session state auto-injected by Agno.

        Returns:
            Compact summary or list of interpolation candidate dictionaries.
        """
        return self.design_molecules(
            goal="Interpolate between two small molecules in latent space.",
            engine="autoencoder",
            n_candidates=n_steps,
            seed_smiles=smiles1,
            constraints={"smiles2": smiles2},
            generation_mode="interpolate",
            temperature=temperature,
            decode_mode="greedy",
            return_format=return_format,
            session_key=session_key,
            agent=agent,
            session_state=session_state,
            _source_tool="interpolate_molecules",
        )

    def validate_design_candidates(
        self, smiles_list: Union[str, List[str]], engine: str = "manual"
    ) -> List[Dict[str, Any]]:
        """
        Validate, standardize, and annotate proposed molecular design candidates.

        Args:
            smiles_list: Single SMILES or list of SMILES candidates.
            engine: Provenance label to attach to the validation results.

        Returns:
            Candidate dictionaries including validity, standardized SMILES, and properties.
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        candidates = [_validate_candidate(smiles, engine=engine) for smiles in smiles_list]
        return [candidate.to_dict() for candidate in _dedupe_candidates(candidates)]

    def rank_design_candidates(
        self,
        candidates: List[Dict[str, Any]],
        seed_smiles: Optional[str] = None,
        prefer_qed: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Rank validated molecular design candidates.

        Args:
            candidates: Candidate dictionaries from design or validation tools.
            seed_smiles: Optional seed for Tanimoto similarity scoring.
            prefer_qed: If True, use QED as a secondary quality score.

        Returns:
            Ranked candidate dictionaries.
        """
        ranked: List[Dict[str, Any]] = []
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

    def _get_engine(self, engine: str, agent: Optional[Agent]) -> MolecularDesignEngine:
        engine_key = engine.lower().strip()
        if engine_key == "autoencoder":
            return self.autoencoder_engine
        if engine_key == "llm":
            model = getattr(agent, "model", None) if agent is not None else None
            if model is None:
                raise MolecularDesignerError(
                    "LLM design requires an agent with a model. Use this tool through "
                    "the Molecular Designer agent or choose engine='autoencoder'."
                )
            return LLMDesignEngine(model)
        raise MolecularDesignerError(
            f"Unknown molecular design engine: {engine}. Available engines: autoencoder, llm."
        )

    def _rank_candidate_objects(
        self, candidates: Sequence[MolecularCandidate], seed_smiles: Optional[str]
    ) -> List[MolecularCandidate]:
        ranked = []
        for candidate in candidates:
            if candidate.valid and candidate.smiles:
                similarity = _similarity_to_seed(candidate.smiles, seed_smiles)
                if similarity is not None:
                    candidate.properties["seed_tanimoto"] = similarity
                    candidate.score = similarity
                elif candidate.score is None:
                    qed = candidate.properties.get("qed")
                    candidate.score = float(qed) if qed is not None else None
            ranked.append(candidate)
        return sorted(
            ranked,
            key=lambda candidate: (
                candidate.valid,
                candidate.score if candidate.score is not None else -1,
            ),
            reverse=True,
        )
