#!/usr/bin/env python
# coding: utf-8
"""
Peptide WAE toolkit for peptide sequence encoding and generation.

This module provides integration with the deepchemography Peptide WAE
for encoding amino acid sequences to latent representations and sampling
new peptides from the latent space.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from agno.tools.toolkit import Toolkit

from cs_copilot.tools.constants import (
    DEFAULT_PEPTIDE_WAE_MODEL_PATH,
    HUGGINGFACE_PEPTIDE_WAE_REPO,
)

logger = logging.getLogger(__name__)


class PeptideWAEError(Exception):
    """Exception raised for peptide WAE-related errors."""

    pass


class PeptideWAEToolkit(Toolkit):
    """
    Peptide WAE toolkit for amino acid sequence encoding and generation.

    This class provides integration with the deepchemography Peptide WAE
    for encoding amino acid sequences to latent representations and sampling
    new peptides from the latent space.

    Input format: Space-separated single-letter amino acid codes
    Example: "M L L L L L A L A L L A L L L A L L L"
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the PeptideWAEToolkit.

        Args:
            model_path: Path to the trained WAE model directory.
                       If None, uses default path or downloads from HuggingFace.
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__("peptide_wae")

        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set up model path
        if model_path is None:
            model_path = os.getenv("PEPTIDE_WAE_MODEL_PATH", DEFAULT_PEPTIDE_WAE_MODEL_PATH)
            if model_path == DEFAULT_PEPTIDE_WAE_MODEL_PATH:
                logger.info(f"Using default peptide WAE model path: {model_path}")
            else:
                logger.info(f"Using PEPTIDE_WAE_MODEL_PATH from environment: {model_path}")

        self.model_path = model_path

        # Check if model exists, download from HuggingFace if not
        self._ensure_model_exists()

        # Initialize model components
        self.model = None
        self.vocab = None
        self.config = None
        self._load_model()

        # Register all peptide WAE tools
        self.register(self.encode_peptides)
        self.register(self.decode_latent)
        self.register(self.sample_peptides)
        self.register(self.interpolate_peptides)
        self.register(self.reconstruct_sequence)
        self.register(self.get_latent_dimension)
        self.register(self.validate_model_loaded)
        self.register(self.explore_latent_neighborhood)
        self.register(self.get_model_info)

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
            logger.info(f"Peptide WAE model files found at {self.model_path}")
            return

        logger.warning(
            f"Peptide WAE model files not found at {self.model_path}. "
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
                raise PeptideWAEError(
                    f"Downloaded files incomplete. Missing: {', '.join(missing)} at {self.model_path}"
                )
            logger.info(f"Successfully fetched peptide WAE model files into {self.model_path}")

            # Cleanup cache
            try:
                cache_dir_path = base_path / ".cache"
                if cache_dir_path.exists() and cache_dir_path.is_dir():
                    shutil.rmtree(cache_dir_path, ignore_errors=True)
            except Exception:
                pass

        except ImportError as e:
            raise PeptideWAEError(
                "huggingface_hub not installed. Install it with: pip install huggingface_hub"
            ) from e
        except Exception as e:
            raise PeptideWAEError(
                f"Failed to download peptide WAE model from HuggingFace "
                f"({HUGGINGFACE_PEPTIDE_WAE_REPO}): {repr(e)}. "
                f"Original model path: {self.model_path}"
            ) from e

    def _load_model(self):
        """Load the trained peptide WAE model and vocabulary."""
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

            logger.info(f"Peptide WAE model loaded successfully from {self.model_path}")
            logger.info(f"  Vocabulary size: {self.vocab.size()}")
            logger.info(f"  Latent dimension: {self.config['z_dim']}")
            logger.info(f"  Device: {self.device}")

        except ImportError as e:
            raise PeptideWAEError(f"Failed to import deepchemography.peptides: {e}") from e
        except Exception as e:
            raise PeptideWAEError(f"Failed to load peptide WAE model: {e}") from e

    def validate_model_loaded(self) -> bool:
        """
        Check if the peptide WAE model is properly loaded.

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
            raise PeptideWAEError("Model not loaded")
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
            raise PeptideWAEError("Model not loaded")

        # Handle single sequence
        if isinstance(sequences, str):
            sequences = [sequences]
            return_single = True
        else:
            return_single = False

        if not sequences:
            raise PeptideWAEError("No peptide sequences provided")

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
            raise PeptideWAEError("Model not loaded")

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
        n_samples: int = 10,
        latent_std: float = 1.0,
        temperature: float = 1.0,
        decode_mode: str = "categorical",
        max_length: int = 25,
    ) -> List[str]:
        """
        Sample new peptides from the latent space using Gaussian prior.

        Args:
            n_samples: Number of peptides to generate
            latent_std: Standard deviation for Gaussian sampling
            temperature: Sampling temperature (higher = more random)
            decode_mode: 'categorical' for stochastic, 'greedy' for deterministic
            max_length: Maximum sequence length

        Returns:
            List of generated peptide sequences
        """
        if not self.validate_model_loaded():
            raise PeptideWAEError("Model not loaded")

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

        predictions = []
        for sample in samples:
            seq_str = self.vocab.to_string(sample, print_special_tokens=False)
            predictions.append(seq_str)

        return predictions

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
            raise PeptideWAEError("Model not loaded")

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
                raise PeptideWAEError(f"Unknown interpolation method: {method}")
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
            raise PeptideWAEError("Model not loaded")

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
            raise PeptideWAEError("Model not loaded")

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
            raise PeptideWAEError("Model not loaded")

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
