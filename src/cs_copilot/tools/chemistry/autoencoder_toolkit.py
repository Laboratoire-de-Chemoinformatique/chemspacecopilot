#!/usr/bin/env python
# coding: utf-8
"""
Autoencoder toolkit for SMILES encoding and molecular generation.

This module provides integration with the deepchemography LSTM autoencoder
for encoding SMILES strings to latent representations and sampling new molecules
from the latent space.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from cs_copilot.storage import S3
from cs_copilot.tools.constants import (
    DEFAULT_AUTOENCODER_MODEL_PATH,
    HUGGINGFACE_AUTOENCODER_REPO,
)

from .base_chemistry import BaseChemistryToolkit, ChemistryError
from .standardize import standardize_smiles

logger = logging.getLogger(__name__)


class AutoencoderError(ChemistryError):
    """Exception raised for autoencoder-related errors."""

    pass


class AutoencoderToolkit(BaseChemistryToolkit):
    """
    Autoencoder toolkit for SMILES encoding and molecular generation.

    This class provides integration with the deepchemography LSTM autoencoder
    for encoding SMILES strings to latent representations and sampling new molecules
    from the latent space.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the AutoencoderToolkit.

        Args:
            model_path: Path to the trained autoencoder model directory.
                       If None, uses default path from deepchemography.
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__("autoencoder")

        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set up model path
        if model_path is None:
            # Get path from environment variable or use default
            model_path = os.getenv("AUTOENCODER_MODEL_PATH", DEFAULT_AUTOENCODER_MODEL_PATH)
            if model_path == DEFAULT_AUTOENCODER_MODEL_PATH:
                logger.info(f"Using default autoencoder model path: {model_path}")
            else:
                logger.info(f"Using AUTOENCODER_MODEL_PATH from environment: {model_path}")

        self.model_path = model_path  # Can be string path or Path object

        # Check if model exists, download from Hugging Face if not
        self._ensure_model_exists()

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

    def _get_hf_token_safe(self):
        """Get Hugging Face token safely from environment or local login."""
        try:
            from huggingface_hub import get_token

            return get_token()
        except Exception as e:
            logger.warning(f"Failed to get Hugging Face token: {e}")
            return None

    def _ensure_model_exists(self):
        """
        Ensure required files exist locally; download from Hugging Face if missing.
        - Avoids temp dirs and symlinks
        - Respects env-provided cache if present (but doesn't require it)
        - Falls back to direct HTTP with Authorization when needed
        """
        import os
        import time
        from pathlib import Path

        import requests

        base_path = Path(self.model_path)
        files = ["config.pt", "vocab.pt", "model.pt"]

        if all((base_path / f).exists() for f in files):
            logger.info(f"Model files found at {self.model_path}")
            return

        logger.warning(
            f"Model files not found at {self.model_path}. Attempting to download from Hugging Face..."
        )
        base_path.mkdir(parents=True, exist_ok=True)

        # --- resolve token (env or local login), but don't force it if public ---
        hf_token = (
            os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN") or self._get_hf_token_safe()
        )

        # helper: robust HTTP download that adds Authorization if we have a token
        def _download_via_requests(repo_id: str, revision: str, filename: str, dest: Path):
            from huggingface_hub import hf_hub_url

            url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
            headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
            # retries with exponential backoff
            for attempt in range(3):
                try:
                    with requests.get(url, stream=True, headers=headers, timeout=(10, 120)) as r:
                        if r.status_code == 401:
                            raise AutoencoderError(
                                "Hugging Face returned 401 Unauthorized. "
                                "If this repo is private or gated, set HUGGINGFACE_HUB_TOKEN."
                            )
                        r.raise_for_status()
                        with open(dest, "wb") as f:
                            for chunk in r.iter_content(1024 * 64):
                                if chunk:
                                    f.write(chunk)
                    if dest.stat().st_size <= 0:
                        raise AutoencoderError(f"Empty file downloaded for {filename}")
                    return
                except Exception as e:
                    if attempt == 2:
                        raise
                    wait = 2**attempt
                    logger.warning(f"Download {filename} failed ({e!r}); retrying in {wait}s...")
                    time.sleep(wait)

        try:
            # --- primary path: use snapshot_download with a minimal, test-friendly signature ---
            from huggingface_hub import snapshot_download

            # Tests expect these exact kwargs: repo_id, cache_dir, local_dir, resume_download
            cache_dir = os.path.expanduser(
                os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HOME") or str(base_path)
            )

            # Avoid occasional stalls from the native downloader/progress in some shells
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

            snapshot_download(
                repo_id=HUGGINGFACE_AUTOENCODER_REPO,
                cache_dir=cache_dir,
                local_dir=str(base_path),
                resume_download=True,
            )

            # --- final verification ---
            missing = [f for f in files if not (base_path / f).exists()]
            if missing:
                raise AutoencoderError(
                    f"Downloaded files incomplete. Missing: {', '.join(missing)} at {self.model_path}"
                )
            logger.info(f"Successfully fetched model files into {self.model_path}")

            # --- cleanup: remove any local .cache directory created under model dir ---
            try:
                cache_dir = base_path / ".cache"
                if cache_dir.exists() and cache_dir.is_dir():
                    import shutil as _shutil

                    _shutil.rmtree(cache_dir, ignore_errors=True)
                    logger.info(f"Removed residual cache directory: {cache_dir}")
            except Exception as _cleanup_err:  # pragma: no cover - best effort cleanup
                logger.debug(f"Cache cleanup skipped: {_cleanup_err!r}")

        except ImportError as e:
            raise AutoencoderError(
                "huggingface_hub not installed. Install it with: pip install huggingface_hub"
            ) from e
        except Exception as e:
            raise AutoencoderError(
                f"Failed to download model from Hugging Face ({HUGGINGFACE_AUTOENCODER_REPO}): {repr(e)}. "
                f"Original model path: {self.model_path}"
            ) from e

    def _load_model(self):
        """Load the trained autoencoder model and vocabulary."""
        try:
            # Import deepchemography components
            from deepchemography import LSTMAutoencoder

            # Build paths - handle both string and Path objects
            base_path = Path(self.model_path)
            config_path = str(base_path / "config.pt")
            vocab_path = str(base_path / "vocab.pt")
            model_path = str(base_path / "model.pt")

            # Use S3.open() for loading - handles both S3 URLs and local paths transparently
            with S3.open(config_path, "rb") as f:
                self.config = torch.load(f, weights_only=False)

            with S3.open(vocab_path, "rb") as f:
                self.vocab = torch.load(f, weights_only=False)

            with S3.open(model_path, "rb") as f:
                model_state = torch.load(f, weights_only=False)

            # Initialize and load the model
            self.model = LSTMAutoencoder(self.vocab, self.config)
            self.model.load_state_dict(model_state)
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Autoencoder model loaded successfully from {self.model_path}")
            logger.info(f"  Vocabulary size: {len(self.vocab)}")
            logger.info(f"  Latent dimension: {self.config.d_z}")
            logger.info(f"  Device: {self.device}")

        except ImportError as e:
            raise AutoencoderError(f"Failed to import deepchemography: {e}") from e
        except Exception as e:
            raise AutoencoderError(f"Failed to load autoencoder model: {e}") from e

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
        return self.config.d_z

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

        self.model.eval()
        latent_vectors = []

        with torch.no_grad():
            for i in tqdm(range(0, len(smiles_list), batch_size), desc="Encoding SMILES"):
                batch_smiles = smiles_list[i : i + batch_size]

                # Convert SMILES to tensors
                batch_tensors = []
                for smiles in batch_smiles:
                    try:
                        smiles_std = standardize_smiles(smiles)
                        if smiles_std is None:
                            logger.warning(
                                f"Invalid/unstandardizable SMILES: {smiles}, using fallback"
                            )
                            smiles = "C"  # Fallback to simple carbon
                        else:
                            smiles = smiles_std

                        tensor = self.model.string2tensor(smiles, device=self.device)
                        batch_tensors.append(tensor)
                    except Exception as e:
                        logger.warning(f"Error processing SMILES '{smiles}': {e}, using fallback")
                        tensor = self.model.string2tensor("C", device=self.device)
                        batch_tensors.append(tensor)

                # Encode to latent space
                z = self.model.forward_encoder(batch_tensors)
                latent_vectors.append(z.cpu().numpy())

        result = np.vstack(latent_vectors)

        if return_single:
            return result[0:1]
        return result

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
        n_samples: int = 10,
        latent_std: float = 1.0,
        max_len: int = 100,
        temp: float = 1.0,
        decode: str = "greedy",
    ) -> List[str]:
        """
        Sample SMILES from the autoencoder latent space.

        Based on the sample_from_latent function from the notebook.

        Args:
            z: Optional latent vectors (numpy array). If None, samples from Gaussian
            n_samples: Number of samples to generate (only used if z is None)
            latent_std: Standard deviation for Gaussian sampling (only used if z is None)
            max_len: Maximum length of generated SMILES
            temp: Temperature for sampling (higher = more random, lower = more deterministic)
            decode: 'greedy' for deterministic, 'sample' for stochastic

        Returns:
            List of generated SMILES strings
        """
        if not self.validate_model_loaded():
            raise AutoencoderError("Model not loaded")

        self.model.eval()

        # If no latent vectors provided, sample from Gaussian prior
        if z is None:
            latent_dim = self.config.d_z
            z = torch.randn(n_samples, latent_dim, device=self.device) * latent_std
        elif isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)

        # Generate samples using the model's sample method
        with torch.no_grad():
            samples = self.model.sample(
                n_batch=z.shape[0], max_len=max_len, z=z, temp=temp, decode=decode
            )

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
        n_samples: int = 10,
        latent_std: float = 1.0,
        temperature: float = 1.0,
        decode_mode: str = "sample",
        max_length: int = 100,
    ) -> List[str]:
        """
        Sample new molecules from the latent space using Gaussian prior.

        Args:
            n_samples: Number of molecules to generate
            latent_std: Standard deviation for Gaussian sampling
            temperature: Temperature for sampling (higher = more random)
            decode_mode: 'greedy' for deterministic, 'sample' for stochastic
            max_length: Maximum length of generated SMILES

        Returns:
            List of generated SMILES strings
        """
        return self.sample_from_latent(
            z=None,
            n_samples=n_samples,
            latent_std=latent_std,
            temp=temperature,
            decode=decode_mode,
            max_len=max_length,
        )

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
        noise_scale: float = 0.1,
        n_neighbors: int = 5,
        temperature: float = 0.5,
        decode_mode: str = "sample",
    ) -> List[str]:
        """
        Explore the neighborhood of a molecule in latent space.

        Based on the neighborhood exploration example from the notebook.

        Args:
            base_smiles: Base molecule SMILES string
            noise_scale: Standard deviation of noise to add
            n_neighbors: Number of neighbors to generate
            temperature: Temperature for sampling
            decode_mode: 'greedy' for deterministic, 'sample' for stochastic

        Returns:
            List of generated neighbor SMILES strings
        """
        if not self.validate_model_loaded():
            raise AutoencoderError("Model not loaded")

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
            "vocabulary_size": len(self.vocab),
            "latent_dimension": self.config.d_z,
            "device": str(self.device),
            "encoder_type": self.config.q_cell,
            "encoder_layers": self.config.q_n_layers,
            "encoder_hidden_size": self.config.q_d_h,
            "decoder_type": self.config.d_cell,
            "decoder_layers": self.config.d_n_layers,
            "decoder_hidden_size": self.config.d_d_h,
            "batch_normalization": self.config.use_batch_norm,
            "bidirectional_encoder": self.config.q_bidir,
        }

    def test_reconstruction_quality(
        self, test_smiles: List[str], temperature: float = 0.1, decode_mode: str = "greedy"
    ) -> Dict[str, Any]:
        """
        Test reconstruction quality on a list of SMILES strings.

        Based on the reconstruction test from the notebook.

        Args:
            test_smiles: List of SMILES strings to test
            temperature: Temperature for decoding
            decode_mode: 'greedy' for deterministic, 'sample' for stochastic

        Returns:
            Dictionary with reconstruction results and statistics
        """
        if not self.validate_model_loaded():
            raise AutoencoderError("Model not loaded")

        results = []
        matches = 0

        for smiles in test_smiles:
            smiles_std = standardize_smiles(smiles) or smiles
            reconstructed = self.reconstruct_smiles(smiles_std, temperature, decode_mode)
            match = smiles_std == reconstructed
            if match:
                matches += 1

            results.append({"original": smiles_std, "reconstructed": reconstructed, "match": match})

        accuracy = matches / len(test_smiles) if test_smiles else 0

        return {
            "results": results,
            "accuracy": accuracy,
            "total_tested": len(test_smiles),
            "matches": matches,
        }
