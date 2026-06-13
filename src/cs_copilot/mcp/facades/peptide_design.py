"""MCP-safe peptide design facade."""

from __future__ import annotations

import functools
from typing import Any, List

from .common import backend_unavailable, ensure_llm_engine_available


class PeptideDesignerFacade:
    """Defer WAE model loading until peptide generation or latent calls."""

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
                raise backend_unavailable("Peptide WAE", exc) from exc
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
        ensure_llm_engine_available(
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
        ensure_llm_engine_available(
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


@functools.lru_cache(maxsize=1)
def peptide_designer_facade() -> PeptideDesignerFacade:
    return PeptideDesignerFacade()
