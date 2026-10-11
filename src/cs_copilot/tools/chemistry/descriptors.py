#!/usr/bin/env python
# coding: utf-8
"""Utilities for computing molecular descriptors and embeddings."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .autoencoder_toolkit import AutoencoderToolkit, make_autoencoder_toolkit
from .base_chemistry import calc_morgan_fp_batch

logger = logging.getLogger(__name__)

# DEFAULT_DESCRIPTOR_TYPE = "autoencoder"
DEFAULT_DESCRIPTOR_TYPE = "morgan"
DEFAULT_DESCRIPTOR_COLUMN = "descriptor_vector"


class MolecularDescriptorEncoder:
    """Factory-style encoder for generating molecular descriptor vectors.

    The encoder provides a simple interface that supports multiple descriptor
    backends (autoencoder embeddings, Morgan fingerprints, ...). Alternative
    descriptors can be requested via the ``descriptor_type`` argument when
    calling :meth:`encode`.
    """

    _ALIASES: Dict[str, str] = {
        "autoencoder": "autoencoder",
        "ae": "autoencoder",
        "latent": "autoencoder",
        "embedding": "autoencoder",
        "autoencoder_embedding": "autoencoder",
        "chembed": "chembed",
        "selfies_vae": "chembed",
        "transformer_vae": "chembed",
        "chembed_embedding": "chembed",
        "morgan": "morgan",
        "morgan_fp": "morgan",
        "fingerprint": "morgan",
        "mfp": "morgan",
    }

    # chembed and the deepchemography autoencoder both produce 256-wide latents,
    # so a shared column name would let vectors from one be silently reused as
    # the other: _compute_descriptors treats an existing column as a cache and
    # never checks which model wrote it. Distinct names are what prevents that.
    _COLUMN_NAMES: Dict[str, str] = {
        "autoencoder": "autoencoder_embedding",
        "chembed": "chembed_embedding",
        "morgan": "morgan_fingerprint",
    }

    # Each descriptor family pins its latent engine. Resolving these through
    # AUTOENCODER_BACKEND instead would make "autoencoder" mean whichever engine
    # happened to be configured, writing one model's vectors into the other's
    # cache column. The environment variable selects the engine used to
    # *generate* molecules; it must not silently redefine a descriptor.
    _DESCRIPTOR_ENGINES: Dict[str, str] = {
        "autoencoder": "deepchemography",
        "chembed": "chembed",
    }

    def __init__(self, default_descriptor: str = DEFAULT_DESCRIPTOR_TYPE):
        self.default_descriptor = self._normalise_descriptor(default_descriptor)
        self._autoencoder_toolkits: Dict[str, AutoencoderToolkit] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(
        self,
        smiles_list: Sequence[str],
        descriptor_type: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode SMILES strings into descriptor vectors.

        Args:
            smiles_list: Iterable of SMILES strings to encode.
            descriptor_type: Name of the descriptor backend to use. When ``None``
                the encoder's default descriptor is applied.
            **kwargs: Additional backend-specific keyword arguments. Supported
                options include ``batch_size``/``model_path``/``device`` for the
                autoencoder and ``nbits`` for Morgan fingerprints.

        Returns:
            ``numpy.ndarray`` containing one descriptor vector per SMILES string.
        """

        if not smiles_list:
            raise ValueError("Cannot encode an empty SMILES collection")

        descriptor_key = self._normalise_descriptor(descriptor_type or self.default_descriptor)
        smiles = list(smiles_list)

        if descriptor_key in {"autoencoder", "chembed"}:
            return self._encode_autoencoder(
                smiles,
                backend=descriptor_key,
                batch_size=kwargs.pop("batch_size", 32),
                model_path=kwargs.pop("model_path", None),
                device=kwargs.pop("device", None),
            )

        if descriptor_key == "morgan":
            return self._encode_morgan(
                smiles,
                nbits=int(kwargs.pop("nbits", 1024)),
            )

        raise ValueError(f"Unsupported descriptor_type '{descriptor_type}'")

    def column_name(self, descriptor_type: Optional[str] = None) -> str:
        """Return the recommended DataFrame column name for *descriptor_type*."""

        descriptor_key = self._normalise_descriptor(descriptor_type or self.default_descriptor)
        return self._COLUMN_NAMES.get(descriptor_key, DEFAULT_DESCRIPTOR_COLUMN)

    def supported_descriptors(self) -> List[str]:
        """List the canonical names of supported descriptor backends."""

        return sorted(set(self._ALIASES.values()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalise_descriptor(self, descriptor_type: str) -> str:
        key = descriptor_type.lower()
        if key not in self._ALIASES:
            raise ValueError(f"Unknown descriptor type '{descriptor_type}'")
        return self._ALIASES[key]

    def _encode_autoencoder(
        self,
        smiles: Sequence[str],
        *,
        backend: str,
        batch_size: int,
        model_path: Optional[str],
        device: Optional[str],
    ) -> np.ndarray:
        toolkit = self._get_autoencoder_toolkit(
            backend=self._DESCRIPTOR_ENGINES[backend], model_path=model_path, device=device
        )
        embeddings = toolkit.encode_smiles_array(list(smiles), batch_size=batch_size)
        return np.asarray(embeddings, dtype=np.float32)

    def _encode_morgan(self, smiles: Sequence[str], *, nbits: int) -> np.ndarray:
        smiles_values = list(smiles)
        results = calc_morgan_fp_batch(smiles_values, nbits)

        fingerprints: List[np.ndarray] = []
        for smi, fp in zip(smiles_values, results, strict=True):
            if fp is None:
                logger.warning(
                    "Invalid SMILES '%s' encountered during Morgan encoding;"
                    " substituting a zero vector",
                    smi,
                )
                fp = np.zeros(nbits, dtype=np.float64)
            fingerprints.append(fp.astype(np.float64))
        return np.vstack(fingerprints)

    def _get_autoencoder_toolkit(
        self, *, backend: str, model_path: Optional[str], device: Optional[str]
    ) -> AutoencoderToolkit:
        # Allow callers to override the cached toolkit when explicit arguments
        # are provided. Otherwise reuse one lazily initialised toolkit per
        # backend -- caching a single one would hand chembed latents to callers
        # asking for the deepchemography model, and the reverse.
        if model_path or device:
            return make_autoencoder_toolkit(backend, model_path=model_path, device=device)

        if backend not in self._autoencoder_toolkits:
            self._autoencoder_toolkits[backend] = make_autoencoder_toolkit(backend)
        return self._autoencoder_toolkits[backend]


__all__ = [
    "DEFAULT_DESCRIPTOR_TYPE",
    "DEFAULT_DESCRIPTOR_COLUMN",
    "MolecularDescriptorEncoder",
]
