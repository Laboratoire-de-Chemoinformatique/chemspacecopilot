#!/usr/bin/env python
# coding: utf-8
"""
Autoencoder toolkit for SMILES encoding and molecular generation.

This module encodes SMILES strings to latent representations and samples new
molecules from the latent space. The latent engine is supplied as an
``AutoencoderEngine`` object and defaults to chembed.
"""

import logging
import os
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np
import torch
from agno.agent import Agent
from rdkit import Chem

from cs_copilot.generation_audit import (
    capture_generation_audit,
    generation_audit_summary,
    save_generation_audit,
)
from cs_copilot.tools.io.session_memory import (
    compact_candidate_preview,
    register_compounds_from_candidates,
    register_generated_candidate_set,
    register_session_object,
    update_state_targets,
)

from .base_chemistry import BaseChemistryToolkit, ChemistryError
from .standardize import standardize_smiles

logger = logging.getLogger(__name__)

SampleReturnFormat = Literal["summary", "list"]

#: Prior-sampling default when the caller names no count. 5000 is the number the
#: sampling docstring argues for: enough to survive validity/uniqueness filtering
#: and still describe the chemical space. chembed decodes autoregressively, which
#: costs roughly 35 s on a GPU but around 17 minutes on CPU, so a CPU host gets a
#: smaller default rather than an interactive call that appears to hang.
DEFAULT_SAMPLE_COUNT_GPU = 5000
DEFAULT_SAMPLE_COUNT_CPU = 500


def _filter_valid_unique_smiles(raw: List[Any]) -> List[str]:
    """Drop non-string entries, invalid SMILES, and duplicates (by canonical form).

    Gaussian prior sampling produces many invalid/duplicate SMILES; this filter
    yields the chemically meaningful subset used by downstream analysis.
    """
    seen: set = set()
    out: List[str] = []
    for s in raw:
        if not isinstance(s, str) or not s:
            continue
        std = standardize_smiles(s)
        if std is None:
            continue
        mol = Chem.MolFromSmiles(std)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


class AutoencoderError(ChemistryError):
    """Exception raised for autoencoder-related errors."""

    pass


class AutoencoderEngine(Protocol):
    """The latent model behind :class:`AutoencoderToolkit`.

    Implementations are supplied to the toolkit by composition, never by
    subclassing it: ``execution_contracts._default_contract`` builds each Agno
    tool name from ``type(owner).__name__`` and memoises it under the method's
    ``(module, qualname)``, which subclasses share. A toolkit subclass would
    therefore rename every registered tool and poison the shared cache.
    """

    name: str

    def load(self, *, model_path: Optional[str], device: Any) -> Tuple[Any, Any, Any]:
        """Return ``(model, vocab, config)``; the toolkit assigns them onto itself."""
        ...

    def encode(self, smiles_list: List[str], *, batch_size: int) -> np.ndarray:
        """Return one latent row per input SMILES, shape ``(n, latent_dim)`` float32."""
        ...

    def decode(self, z: Any, *, max_len: int, temp: float, decode: str) -> List[str]:
        """Decode latent vectors to SMILES strings, unfiltered."""
        ...

    def latent_dim(self) -> int:
        """Return the width of one latent row."""
        ...

    def info(self) -> Dict[str, Any]:
        """Return model metadata for :meth:`AutoencoderToolkit.get_model_info`."""
        ...

    def decode_settings(self, *, temp: float, decode: str, max_len: int) -> Dict[str, Any]:
        """Return the decode settings actually applied, for the generation audit."""
        ...

    def default_noise_scale(self) -> float:
        """Return the latent perturbation scale suited to this engine's latent."""
        ...


class AutoencoderToolkit(BaseChemistryToolkit):
    """
    Autoencoder toolkit for SMILES encoding and molecular generation.

    Encodes SMILES strings to latent representations and samples new molecules
    from the latent space, using whichever :class:`AutoencoderEngine` it is
    given, chembed by default.
    """

    # Class-level default: tests build this toolkit with ``__new__`` and never
    # run ``__init__``, so every hook below must still find the attribute.
    engine: Optional[AutoencoderEngine] = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        engine: Optional[AutoencoderEngine] = None,
    ):
        """
        Initialize the AutoencoderToolkit.

        Args:
            model_path: A local chembed ``.ckpt`` file. If None,
                ``AUTOENCODER_MODEL_PATH`` is used when it names a file; otherwise
                the engine loads its pinned checkpoint from Hugging Face.
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
            engine: Latent engine to use. When None, chembed.
        """
        super().__init__("autoencoder")

        if engine is None:
            # Imported here because chembed_engine imports AutoencoderError from
            # this module. The check refuses a stale AUTOENCODER_BACKEND that
            # still names the retired LSTM.
            from .chembed_engine import ChembedEngine

            resolve_autoencoder_backend()
            engine = ChembedEngine()
        self.engine = engine

        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_path = model_path or os.getenv("AUTOENCODER_MODEL_PATH")

        # Initialize model components
        self.model = None
        self.vocab = None
        self.config = None
        self._load_model()

        # Register all autoencoder tools
        self.register(self.encode_smiles)
        self.register(self.decode_latent)
        self.register(self.sample_molecules)
        self.register(self.interpolate_molecules)
        self.register(self.reconstruct_smiles)
        self.register(self.get_latent_dimension)
        self.register(self.validate_model_loaded)
        self.register(self.explore_latent_neighborhood)
        self.register(self.get_model_info)

    def _load_model(self):
        """Load the engine's model, vocabulary and config."""
        self.model, self.vocab, self.config = self.engine.load(
            model_path=self.model_path, device=self.device
        )
        # Record what was actually loaded: a model_path that names no file is
        # ignored, and would otherwise end up in every generation audit.
        checkpoint_path = getattr(self.engine, "checkpoint_path", None)
        if checkpoint_path:
            self.model_path = checkpoint_path

    # ------------------------------------------------------------------
    # Engine hooks. None is registered as an Agno tool, so overriding them
    # moves no contract name.
    # ------------------------------------------------------------------
    def _latent_dim(self) -> int:
        """Return the width of one latent vector."""
        return self.engine.latent_dim()

    def _decode_z(self, z: Any, *, max_len: int, temp: float, decode: str) -> List[str]:
        """Decode latent vectors to raw, unfiltered SMILES strings."""
        return self.engine.decode(z, max_len=max_len, temp=temp, decode=decode)

    def _resolve_sample_count(self, n_samples: Optional[int]) -> int:
        """Return the sample count to use, and log when the device decided it.

        Args:
            n_samples: Caller-supplied count, or None to pick one for this device.

        Returns:
            The caller's count when given, else a device-dependent default.
        """
        if n_samples is not None:
            return int(n_samples)
        on_gpu = self.device.type == "cuda"
        resolved = DEFAULT_SAMPLE_COUNT_GPU if on_gpu else DEFAULT_SAMPLE_COUNT_CPU
        logger.info(
            "n_samples not given; sampling %d molecules on %s. Pass n_samples "
            "explicitly to make this independent of the host.",
            resolved,
            self.device.type,
        )
        return resolved

    def _default_noise_scale(self) -> float:
        """Return the latent perturbation this engine needs to leave the parent."""
        return self.engine.default_noise_scale()

    @property
    def backend_name(self) -> str:
        """Name the latent engine behind this toolkit.

        Callers recording provenance use this rather than the design-engine
        name, which stays ``"autoencoder"`` whatever the engine.
        """
        return self.engine.name

    def _engine_info(self) -> Dict[str, Any]:
        """Return engine-specific model metadata."""
        return self.engine.info()

    def _decode_settings(self, *, temp: float, decode: str, max_len: int) -> Dict[str, Any]:
        """Return the decode settings actually applied, for the generation audit."""
        return self.engine.decode_settings(temp=temp, decode=decode, max_len=max_len)

    def validate_model_loaded(self) -> bool:
        """
        Check if the autoencoder model is properly loaded.

        Returns:
            True if model is loaded and ready to use
        """
        return self.model is not None and self.vocab is not None and self.config is not None

    def get_latent_dimension(self) -> int:
        """
        Get the dimension of the latent space.

        Returns:
            Latent dimension size
        """
        if not self.validate_model_loaded():
            raise AutoencoderError("Model not loaded")
        return self._latent_dim()

    def _encode_smiles_ndarray(
        self, smiles_list: Union[str, List[str]], batch_size: int = 32
    ) -> np.ndarray:
        """
        Internal helper: encode SMILES and return a numpy.ndarray.
        """
        if not self.validate_model_loaded():
            raise AutoencoderError("Model not loaded")

        # Handle single SMILES string
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            return_single = True
        else:
            return_single = False

        if not smiles_list:
            raise AutoencoderError("No SMILES strings provided")

        result = self.engine.encode(list(smiles_list), batch_size=batch_size)
        return result[0:1] if return_single else result

    def encode_smiles_array(
        self, smiles_list: Union[str, List[str]], batch_size: int = 32
    ) -> np.ndarray:
        """
        Public API for internal callers that need a numpy.ndarray result.
        """
        return self._encode_smiles_ndarray(smiles_list, batch_size=batch_size)

    def encode_smiles(
        self, smiles_list: Union[str, List[str]], batch_size: int = 32
    ) -> Union[List[float], List[List[float]]]:
        """
        Tool-exposed API: returns JSON-serializable Python lists to avoid
        downstream truthiness issues with numpy arrays in the agent runtime.
        """
        arr = self._encode_smiles_ndarray(smiles_list, batch_size=batch_size)
        # If a single vector (shape (1, d)) was returned, squeeze appropriately
        if isinstance(smiles_list, str) or (
            isinstance(smiles_list, list) and len(smiles_list) == 1
        ):
            return arr[0].tolist()
        return arr.tolist()

    def sample_from_latent(
        self,
        z: Optional[np.ndarray] = None,
        n_samples: int = 5000,
        latent_std: float = 1.0,
        max_len: int = 100,
        temp: float = 1.0,
        decode: str = "greedy",
        audit_path: Optional[str] = None,
    ) -> List[str]:
        """
        Sample SMILES from the autoencoder latent space.

        Based on the sample_from_latent function from the notebook.

        Args:
            z: Optional latent vectors (numpy array). If None, samples from Gaussian
            n_samples: Number of samples to generate when z is None. Default 5000 —
                the recommended minimum for meaningful chemical-space exploration
                after validity/uniqueness filtering. Pass a smaller value
                explicitly for quick demos.
            latent_std: Standard deviation for Gaussian sampling (only used if z is None)
            max_len: Maximum length of generated SMILES
            temp: Temperature for sampling (higher = more random, lower = more deterministic)
            decode: 'greedy' for deterministic, 'sample' for stochastic
            audit_path: Optional local JSON path for unfiltered backend outputs.

        Returns:
            List of generated SMILES strings
        """
        if not self.validate_model_loaded():
            raise AutoencoderError("Model not loaded")

        # If no latent vectors provided, sample from Gaussian prior
        sampled_from_prior = z is None
        if z is None:
            latent_dim = self._latent_dim()
            z = torch.randn(n_samples, latent_dim, device=self.device) * latent_std
        elif isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)

        samples = self._decode_z(z, max_len=max_len, temp=temp, decode=decode)

        # Preserve invalid outputs and duplicates before any toolkit standardization.
        samples = list(samples)
        self.last_generation_audit = capture_generation_audit(
            samples,
            source="autoencoder_backend_outputs_before_standardization",
            requested_count=int(z.shape[0]),
            raw_outputs_available=True,
        )
        self.last_generation_audit["settings"] = {
            **self._decode_settings(temp=temp, decode=decode, max_len=max_len),
            "latent_std": latent_std if sampled_from_prior else None,
            "latent_source": "gaussian_prior" if sampled_from_prior else "provided_vectors",
            "model_path": str(self.model_path),
        }
        save_generation_audit(self.last_generation_audit, audit_path=audit_path)
        standardized_smiles = []
        for smi_raw in samples:
            smiles_std = standardize_smiles(smi_raw) if isinstance(smi_raw, str) else None
            standardized_smiles.append(smiles_std if smiles_std is not None else smi_raw)
        return standardized_smiles

    def decode_latent(
        self,
        latent_vectors: Union[np.ndarray, List[np.ndarray]],
        temperature: float = 0.1,
        decode_mode: str = "greedy",
        max_length: int = 100,
    ) -> List[str]:
        """
        Decode latent vectors back to SMILES strings.

        Args:
            latent_vectors: Latent vectors to decode
            temperature: Temperature for sampling (higher = more random)
            decode_mode: 'greedy' for deterministic, 'sample' for stochastic
            max_length: Maximum length of generated SMILES

        Returns:
            List of generated SMILES strings
        """
        return self.sample_from_latent(
            z=latent_vectors, temp=temperature, decode=decode_mode, max_len=max_length
        )

    def sample_molecules(
        self,
        n_samples: Optional[int] = None,
        latent_std: float = 1.0,
        temperature: float = 1.0,
        decode_mode: str = "sample",
        max_length: int = 100,
        filter_valid_unique: bool = True,
        return_format: SampleReturnFormat = "summary",
        session_key: str = "sampled_molecules",
        agent: Optional[Agent] = None,
        session_state: Optional[Dict[str, Any]] = None,
        audit_path: Optional[str] = None,
    ) -> Union[List[str], Dict[str, Any]]:
        """
        Sample new molecules from the latent space using Gaussian prior.

        Args:
            n_samples: Number of molecules to generate. When omitted, defaults to
                5000 on a GPU and 500 on CPU, because the transformer decoder is
                roughly 30x slower without one. Pass a value explicitly for a
                count that does not depend on the host.
            latent_std: Standard deviation for Gaussian sampling
            temperature: Temperature for sampling (higher = more random)
            decode_mode: 'greedy' for deterministic, 'sample' for stochastic
            max_length: Maximum length of generated SMILES
            filter_valid_unique: If True (default), drop non-parseable SMILES
                and deduplicate by canonical form before returning.
            return_format: "summary" (default) saves the full list as a
                candidate-set artifact and returns a compact dict with count,
                preview, session key, and artifact path. "list" returns
                the raw List[str] directly (legacy behavior — may inflate LLM
                context with large n_samples).
            session_key: Key under which the artifact pointer is stored in
                session state when return_format="summary".
            agent: Agent instance (auto-injected by agno). Required for
                "summary" format; if None, gracefully falls back to "list".
            session_state: Shared session state auto-injected by Agno.
            audit_path: Optional local JSON path; otherwise save an audit with session artifacts.

        Returns:
            Dict summary (default) or List[str] (when return_format="list" or
            no session state available).
        """
        self.last_generation_audit = None
        requested_n_samples = n_samples
        n_samples = self._resolve_sample_count(n_samples)
        raw = self.sample_from_latent(
            z=None,
            n_samples=n_samples,
            latent_std=latent_std,
            temp=temperature,
            decode=decode_mode,
            max_len=max_length,
        )

        audit = self.last_generation_audit or capture_generation_audit(
            raw,
            source="sampling_outputs_backend_capture_unavailable",
            requested_count=n_samples,
            raw_outputs_available=False,
        )
        self.last_generation_audit = audit
        sampled = _filter_valid_unique_smiles(raw) if filter_valid_unique else list(raw)

        state_targets = update_state_targets(agent, session_state)
        audit_artifact_path = save_generation_audit(
            audit, session_state=state_targets[0] if state_targets else None, audit_path=audit_path
        )
        audit_metadata = generation_audit_summary(audit)
        audit_metadata["artifact_path"] = audit_artifact_path
        # A host-dependent default would otherwise make two runs of the same call
        # differ with no record of why.
        audit_metadata["n_samples_requested"] = requested_n_samples
        audit_metadata["n_samples_resolved"] = n_samples
        audit_metadata["n_samples_source"] = (
            "caller" if requested_n_samples is not None else f"default_{self.device.type}"
        )
        registered_compound_ids: List[str] = []
        registered_candidate_set_id: Optional[str] = None
        registered_artifact_path: Optional[str] = None
        for state in state_targets:
            use_state_for_summary = state is session_state or (
                session_state is None and not registered_compound_ids
            )
            candidate_ids = register_compounds_from_candidates(
                state,
                sampled,
                source_agent=getattr(agent, "name", None),
                source_tool="sample_molecules",
                label_prefix="Autoencoder sample",
                related={"session_key": session_key},
                provenance={
                    "origin_type": "generated",
                    "origin_agent": "autoencoder_toolkit",
                    "generation_engine": "autoencoder",
                    "latent_engine": self.backend_name,
                },
                set_current_first=bool(sampled),
            )
            if use_state_for_summary:
                registered_compound_ids = candidate_ids
            candidate_set_id = register_generated_candidate_set(
                state,
                candidate_ids,
                source_agent=getattr(agent, "name", None),
                source_tool="sample_molecules",
                origin_agent="autoencoder_toolkit",
                generation_engine="autoencoder",
                generation_mode="sample",
                session_key=session_key,
                label="Autoencoder sampled candidates",
                count_attempted=n_samples,
                metadata={
                    "generation_audit": audit_metadata,
                    "count_attempted_scope": "requested_samples_not_backend_attempts",
                    "latent_engine": self.backend_name,
                    "latent_std": latent_std,
                    "temperature": temperature,
                    "decode_mode": decode_mode,
                    "filter_valid_unique": filter_valid_unique,
                },
                candidates=sampled,
            )
            pointer = state.get(session_key)
            artifact_path = pointer.get("artifact_path") if isinstance(pointer, dict) else None
            if use_state_for_summary:
                registered_candidate_set_id = candidate_set_id
                registered_artifact_path = artifact_path
            register_session_object(
                state,
                "analysis",
                {
                    "analysis_type": "autoencoder_sampling",
                    "session_key": session_key,
                    "count_attempted": n_samples,
                    "count_returned": len(sampled),
                    "compound_ids": candidate_ids,
                    "candidate_set_id": candidate_set_id,
                    "artifact_path": artifact_path,
                },
                label="Autoencoder sampling run",
                source_agent=getattr(agent, "name", None),
                source_tool="sample_molecules",
                set_current=True,
                current_role="analysis",
            )

        if return_format == "list" or not state_targets:
            if return_format == "summary" and not state_targets:
                logger.info(
                    "sample_molecules called with return_format='summary' but no session "
                    "state was available; falling back to raw list."
                )
            return sampled

        return {
            "generation_audit": audit_metadata,
            "count_attempted_scope": "requested_samples_not_backend_attempts",
            "count_attempted": n_samples,
            "count_returned": len(sampled),
            "filter_valid_unique": filter_valid_unique,
            "preview": compact_candidate_preview(sampled),
            "session_key": session_key,
            "registered_compound_ids": registered_compound_ids,
            "registered_candidate_set_id": registered_candidate_set_id,
            "artifact_path": registered_artifact_path,
            "artifact_format": "json" if registered_artifact_path else None,
            "note": (
                f"Full {len(sampled)}-item SMILES list saved as an artifact. Use "
                "registered_candidate_set_id or artifact_path for downstream analysis "
                "(property calculation, clustering, GTM projection, etc.)."
            ),
        }

    def interpolate_molecules(
        self,
        smiles1: str,
        smiles2: str,
        n_steps: int = 10,
        temperature: float = 0.1,
        decode_mode: str = "greedy",
    ) -> List[str]:
        """
        Interpolate between two molecules in latent space.

        Based on the interpolation example from the notebook.

        Args:
            smiles1: First molecule SMILES string
            smiles2: Second molecule SMILES string
            n_steps: Number of interpolation steps
            temperature: Temperature for decoding
            decode_mode: 'greedy' for deterministic, 'sample' for stochastic

        Returns:
            List of interpolated SMILES strings
        """
        if not self.validate_model_loaded():
            raise AutoencoderError("Model not loaded")

        # Encode both molecules
        z1 = self.encode_smiles_array([smiles1])
        z2 = self.encode_smiles_array([smiles2])

        # Linear interpolation in latent space
        alphas = np.linspace(0, 1, n_steps)
        z_interp = []

        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            z_interp.append(z)

        z_interp = np.vstack(z_interp)

        # Decode interpolated latent vectors
        interpolated_smiles = self.decode_latent(
            z_interp, temperature=temperature, decode_mode=decode_mode
        )

        return interpolated_smiles

    def reconstruct_smiles(
        self, smiles: str, temperature: float = 0.1, decode_mode: str = "greedy"
    ) -> str:
        """
        Reconstruct a SMILES string by encoding and decoding it.

        Args:
            smiles: Input SMILES string
            temperature: Temperature for decoding
            decode_mode: 'greedy' for deterministic, 'sample' for stochastic

        Returns:
            Reconstructed SMILES string
        """
        if not self.validate_model_loaded():
            raise AutoencoderError("Model not loaded")

        # Encode and decode
        latent = self.encode_smiles_array([smiles])
        reconstructed = self.decode_latent(latent, temperature=temperature, decode_mode=decode_mode)

        return reconstructed[0]

    def explore_latent_neighborhood(
        self,
        base_smiles: str,
        noise_scale: Optional[float] = None,
        n_neighbors: int = 5,
        temperature: float = 0.5,
        decode_mode: str = "sample",
    ) -> List[str]:
        """
        Explore the neighborhood of a molecule in latent space.

        Based on the neighborhood exploration example from the notebook.

        Args:
            base_smiles: Base molecule SMILES string
            noise_scale: Standard deviation of the latent noise. When omitted, the
                active engine's own default is used, because a value tuned for
                one latent space produces either nothing or noise in another.
            n_neighbors: Number of neighbors to generate
            temperature: Temperature for sampling
            decode_mode: 'greedy' for deterministic, 'sample' for stochastic

        Returns:
            List of generated neighbor SMILES strings
        """
        if not self.validate_model_loaded():
            raise AutoencoderError("Model not loaded")

        if noise_scale is None:
            noise_scale = self._default_noise_scale()

        # Encode base molecule
        z_base = self.encode_smiles_array([base_smiles])

        # Generate neighbors by adding noise
        z_neighbors = z_base + np.random.randn(n_neighbors, z_base.shape[1]) * noise_scale

        # Decode neighbors
        neighbors = self.decode_latent(
            z_neighbors, temperature=temperature, decode_mode=decode_mode
        )

        return neighbors

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        if not self.validate_model_loaded():
            raise AutoencoderError("Model not loaded")

        return {
            "model_path": str(self.model_path),
            "device": str(self.device),
            **self._engine_info(),
        }


#: Values of ``AUTOENCODER_BACKEND`` and the robustness ``latent_engine`` key that
#: name chembed, or no engine in particular.
_CHEMBED_BACKEND_NAMES = {"chembed", "autoencoder", "default", ""}


def resolve_autoencoder_backend(backend: Optional[str] = None) -> str:
    """Return ``"chembed"``, the only small-molecule latent engine.

    The deepchemography LSTM was retired after v0.4.0. Asking for it, by argument
    or through ``AUTOENCODER_BACKEND``, raises instead of generating on chembed,
    so no result gets attributed to a model that did not produce it.

    Args:
        backend: Caller-supplied engine name, or None to check the environment.

    Returns:
        ``"chembed"``.

    Raises:
        AutoencoderError: If the name is anything else.
    """
    name = backend if backend is not None else os.getenv("AUTOENCODER_BACKEND", "")
    name = name.strip().lower()
    if name not in _CHEMBED_BACKEND_NAMES:
        raise AutoencoderError(
            f"Latent engine {name!r} is not available: chembed is the only small-molecule "
            "engine. The deepchemography LSTM was retired after v0.4.0; check out that tag "
            "to reproduce results recorded on it."
        )
    return "chembed"


def make_autoencoder_toolkit(
    *,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> AutoencoderToolkit:
    """Build an :class:`AutoencoderToolkit` on chembed.

    Args:
        model_path: A local chembed ``.ckpt`` file, or None for the pinned checkpoint.
        device: Device to run the model on.

    Returns:
        A toolkit whose class is always ``AutoencoderToolkit``, so the Agno tool
        names it registers do not depend on the engine.
    """
    return AutoencoderToolkit(model_path=model_path, device=device)
