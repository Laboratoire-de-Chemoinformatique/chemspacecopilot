#!/usr/bin/env python
# coding: utf-8
"""chembed latent engine for :class:`~.autoencoder_toolkit.AutoencoderToolkit`.

chembed is a SELFIES transformer VAE (Talibart & Gilis, *J. Cheminform.* 2026).
It replaces the deepchemography LSTM autoencoder for small molecules, which
cannot encode SMILES longer than roughly 50 characters and so returns almost
nothing for the larger, more potent chemotypes.

Two differences from the LSTM matter to callers:

* Decoding is greedy and deterministic. ``temperature`` and ``decode_mode``
  have no effect, and ``max_length`` is fixed by the checkpoint. The engine
  accepts all three, ignores them, and records in the generation audit both
  what was requested and what was applied, so no artifact claims a setting
  that was never used.
* All diversity therefore comes from latent noise, and the noise scales tuned
  for the LSTM do not transfer: a single encoded molecule has a latent
  standard deviation near 0.66.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem

from cs_copilot.tools.constants import (
    CHEMBED_CHECKPOINT_FILENAME,
    HUGGINGFACE_CHEMBED_REPO,
)

from .autoencoder_toolkit import AutoencoderError

logger = logging.getLogger(__name__)

#: chembed batches at this width internally; the autoregressive decode is slow
#: enough that the toolkit's LSTM-era default of 32 wastes most of the batch.
DEFAULT_BATCH_SIZE = 256

#: Stand-in for a SMILES RDKit cannot parse. Substituting keeps one latent row
#: per input, which GTM relies on, rather than silently shortening the matrix.
UNPARSEABLE_PLACEHOLDER = "C"


class ChembedEngine:
    """Latent engine backed by the chembed SELFIES transformer VAE."""

    name = "chembed"

    def __init__(
        self,
        repo_id: str = HUGGINGFACE_CHEMBED_REPO,
        filename: str = CHEMBED_CHECKPOINT_FILENAME,
        checkpath: Optional[str] = None,
    ):
        """Configure where the checkpoint comes from.

        Args:
            repo_id: Hugging Face repository holding the pre-trained checkpoint.
            filename: Checkpoint file within *repo_id*.
            checkpath: Local checkpoint to load instead of the Hugging Face one.
        """
        self.repo_id = repo_id
        self.filename = filename
        self.checkpath = checkpath
        self.checkpoint_path = checkpath or f"{repo_id}/{filename}"
        self.vae: Any = None
        self.unknown_token_substitutions = 0
        self._warned_ignored_parameters = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, *, model_path: Optional[str], device: Any) -> Tuple[Any, Any, Any]:
        """Load the VAE and return ``(model, vocab, config)``.

        A *model_path* naming an existing file is treated as a local checkpoint.
        The toolkit's default path points at the deepchemography model directory
        and is ignored.

        Args:
            model_path: Toolkit-resolved path; used only when it is a real file.
            device: Torch device to load onto.

        Returns:
            ``(vae, vocab, model_config)``.

        Raises:
            AutoencoderError: If chembed is missing, the checkpoint cannot be
                loaded, or the checkpoint has a bottleneck this engine cannot
                reshape for.
        """
        try:
            from chembed import checkpoint_utils
        except ImportError as exc:
            raise AutoencoderError(
                "chembed is not installed. Install it with: uv sync --frozen"
            ) from exc

        checkpath = self.checkpath
        if checkpath is None and model_path and Path(model_path).is_file():
            checkpath = model_path

        try:
            if checkpath is not None:
                self.checkpoint_path = str(checkpath)
                vae = checkpoint_utils.load_vae_from_checkpoint(Path(checkpath), device=device)
            else:
                self.checkpoint_path = f"{self.repo_id}/{self.filename}"
                vae = checkpoint_utils.load_vae_from_hub(
                    device=device, repo_id=self.repo_id, filename=self.filename
                )
        except Exception as exc:
            raise AutoencoderError(
                f"Failed to load chembed checkpoint from {self.checkpoint_path}: {exc!r}"
            ) from exc

        if int(vae.d_bottleneck) != 1:
            raise AutoencoderError(
                f"chembed checkpoint has d_bottleneck={vae.d_bottleneck}; this engine flattens "
                "the latent to one row per molecule and only supports a bottleneck of 1."
            )

        vae.eval()
        self.vae = vae
        logger.info(
            "chembed loaded from %s (d_model=%s, max_len=%s, vocab=%s)",
            self.checkpoint_path,
            vae.d_model,
            vae.max_len,
            len(vae.vocab),
        )
        return vae, vae.vocab, vae.model_config

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------
    def encode(self, smiles_list: List[str], *, batch_size: int) -> np.ndarray:
        """Encode SMILES to one latent row each.

        The input is not standardized here: ``encode_multiple_smiles`` applies
        chembed's own ChEMBL structure pipeline, and canonicalizing first with a
        different kekulization risks SELFIES tokens the vocabulary lacks.

        Args:
            smiles_list: SMILES strings to encode.
            batch_size: Encoder batch size.

        Returns:
            ``(len(smiles_list), latent_dim)`` float32 array.
        """
        from chembed.encode import encode_multiple_smiles

        safe = [
            smi if Chem.MolFromSmiles(smi) is not None else UNPARSEABLE_PLACEHOLDER
            for smi in smiles_list
        ]
        unparseable = sum(1 for smi, kept in zip(smiles_list, safe, strict=True) if smi != kept)
        if unparseable:
            logger.warning(
                "%d/%d SMILES could not be parsed and were encoded as '%s'",
                unparseable,
                len(smiles_list),
                UNPARSEABLE_PLACEHOLDER,
            )

        # replace_if_not_in_vocab keeps one unseen token from aborting the whole
        # batch. The substitutions change the molecule, so they are counted and
        # reported rather than left silent.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            zs = encode_multiple_smiles(
                safe,
                self.vae,
                batch_size=batch_size or DEFAULT_BATCH_SIZE,
                replace_if_not_in_vocab=True,
                use_sigma=False,
            )
        replaced = sum(1 for w in caught if type(w.message).__name__ == "TokenReplacementWarning")
        self.unknown_token_substitutions = replaced
        if replaced:
            logger.warning(
                "chembed replaced %d SELFIES token(s) outside its vocabulary while encoding "
                "%d molecules",
                replaced,
                len(safe),
            )

        # (N, d_bottleneck, d_model) -> (N, d_bottleneck * d_model). reshape
        # rather than squeeze so a wider bottleneck would flatten consistently.
        return zs.reshape(zs.shape[0], -1).cpu().numpy().astype(np.float32)

    def decode(self, z: Any, *, max_len: int, temp: float, decode: str) -> List[str]:
        """Decode latent vectors to raw SMILES.

        Args:
            z: ``(n, latent_dim)`` or ``(n, 1, latent_dim)`` latents.
            max_len: Ignored; the checkpoint fixes the decode length.
            temp: Ignored; decoding is greedy.
            decode: Ignored; decoding is greedy.

        Returns:
            One SMILES string per latent row, unfiltered.
        """
        import torch
        from chembed.decode import decode_zs_to_smiles

        self._warn_ignored_parameters(temp=temp, decode=decode, max_len=max_len)
        tensor = torch.as_tensor(np.asarray(z), dtype=torch.float32)
        tensor = tensor.reshape(tensor.shape[0], 1, -1)
        return decode_zs_to_smiles(tensor, self.vae, batch_size=DEFAULT_BATCH_SIZE)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def latent_dim(self) -> int:
        """Return the flattened latent width, ``d_bottleneck * d_model``."""
        return int(self.vae.d_bottleneck) * int(self.vae.d_model)

    def info(self) -> Dict[str, Any]:
        """Return model metadata, including what this engine cannot do."""
        config = self.vae.model_config
        return {
            "backend": self.name,
            "architecture": "selfies_transformer_vae",
            "vocabulary_size": len(self.vae.vocab),
            "latent_dimension": self.latent_dim(),
            "d_model": int(self.vae.d_model),
            "d_bottleneck": int(self.vae.d_bottleneck),
            "max_len": int(self.vae.max_len),
            "repo_id": self.repo_id,
            "checkpoint_path": self.checkpoint_path,
            "encoder_type": "transformer",
            "encoder_layers": config.get("nb_layers_encoder"),
            "encoder_hidden_size": config.get("dim_feedforward_encoder"),
            "decoder_type": "transformer",
            "decoder_layers": config.get("nb_layers_decoder"),
            "decoder_hidden_size": config.get("dim_feedforward_decoder"),
            "nhead": config.get("nhead"),
            "batch_normalization": False,
            "bidirectional_encoder": False,
            # Agents read this; without it they keep offering temperature sweeps
            # that silently do nothing.
            "supports_temperature": False,
            "supports_stochastic_decode": False,
        }

    def decode_settings(self, *, temp: float, decode: str, max_len: int) -> Dict[str, Any]:
        """Return the decode settings applied, alongside those requested."""
        return {
            "temperature": None,
            "decode_mode": "greedy",
            "max_length": int(self.vae.max_len),
            "requested_temperature": temp,
            "requested_decode_mode": decode,
            "requested_max_length": max_len,
            "ignored_parameters": ["temperature", "decode_mode", "max_length"],
            "backend": self.name,
            "repo_id": self.repo_id,
            "checkpoint_path": self.checkpoint_path,
            "unknown_token_substitutions": self.unknown_token_substitutions,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _warn_ignored_parameters(self, *, temp: float, decode: str, max_len: int) -> None:
        """Warn once per engine when a caller asks for decoding chembed cannot do."""
        if self._warned_ignored_parameters:
            return
        if decode != "greedy" or temp != 1.0:
            logger.warning(
                "chembed decodes greedily: temperature=%s and decode_mode=%r are ignored. "
                "Vary noise_scale or latent_std instead. The generation audit records both "
                "the requested and the applied settings.",
                temp,
                decode,
            )
            self._warned_ignored_parameters = True
