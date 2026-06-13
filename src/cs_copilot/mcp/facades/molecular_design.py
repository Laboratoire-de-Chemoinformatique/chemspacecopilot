"""MCP-safe small-molecule design facade."""

from __future__ import annotations

import functools
from typing import Any, List

from .common import backend_unavailable, ensure_llm_engine_available


class MolecularDesignerFacade:
    """Load the molecular autoencoder only for generation calls."""

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
                raise backend_unavailable("Molecular designer", exc) from exc
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
        ensure_llm_engine_available(
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
        ensure_llm_engine_available(
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


@functools.lru_cache(maxsize=1)
def molecular_designer_facade() -> MolecularDesignerFacade:
    return MolecularDesignerFacade()
