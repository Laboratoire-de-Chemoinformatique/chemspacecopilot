#!/usr/bin/env python
# coding: utf-8
"""Refit the default GTM manifold on chembed latents.

The published manifold (``GTM_manifold_900_100_8_100.pkl.gz``) was fitted on
256-dimensional deepchemography LSTM embeddings. chembed's latent is also 256
wide, so projecting chembed vectors onto that manifold raises no error at all --
it simply returns meaningless coordinates. A refit is therefore mandatory, not
optional, before chembed can back the default map.

The GTM hyperparameters are taken verbatim from the published model's recorded
attributes (``docs/manuscript/gtm-model-attributes.json``). They are not
re-tuned: re-tuning would turn a port onto a new latent space into a different
experiment and destroy the comparison with the published map.

The fit mirrors ``gtm_operations.optimize_gtm`` rather than ``train_latent_gtm``.
The descriptor path that consumes this manifold calls ``gtm.project(X)`` on the
raw descriptor matrix with no external scaler, so the GTM must do its own
scaling, exactly as the published model does.

Usage::

    python scripts/refit_gtm_manifold.py --frameset gtm_frameset_training.csv \
        --output ~/.cache/cs_copilot/models/gtm/GTM_manifold_chembed_900_225.pkl.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import dill
import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("refit_gtm_manifold")

# Verbatim from the published model's recorded attributes. num_basis_functions
# is 15**2 and num_nodes is 30**2, matching phi of shape (900, 226).
PUBLISHED_HYPERPARAMETERS: Dict[str, Any] = {
    "num_nodes": 900,
    "num_basis_functions": 225,
    "basis_width": 1,
    "reg_coeff": 100,
    "max_iter": 200,
    "standardize": False,
    "pca_scale": True,
    "pca_engine": "torch",
}

#: The manifold records which latent model produced its training descriptors.
#: Both backends emit 256 columns, so width alone cannot tell them apart and a
#: mismatch would otherwise be undetectable.
DESCRIPTOR_BACKEND = "chembed"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frameset", type=Path, required=True, help="Frameset CSV")
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--output", type=Path, required=True, help="Destination .pkl.gz")
    parser.add_argument("--attributes", type=Path, default=None, help="Attributes JSON to write")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Encode only the first N rows")
    return parser.parse_args(argv)


def encode_frameset(smiles, *, device: str, batch_size: int):
    """Encode the frameset with chembed and report token substitutions."""
    from cs_copilot.tools.chemistry.chembed_engine import ChembedEngine

    engine = ChembedEngine()
    engine.load(model_path=None, device=torch.device(device))
    logger.info("Encoding %d molecules on %s", len(smiles), device)
    latents = engine.encode(list(smiles), batch_size=batch_size)
    logger.info(
        "Encoded to %s; %d SELFIES token substitution(s)",
        latents.shape,
        engine.unknown_token_substitutions,
    )
    return latents, engine


def fit_manifold(latents: np.ndarray, *, device: str):
    """Fit the GTM exactly as the descriptor path will later project onto it."""
    from chemographykit.gtm import GTM

    from cs_copilot.tools.chemography.gtm_operations import shannon_entropy

    gtm = GTM(device=device, **PUBLISHED_HYPERPARAMETERS)
    x = torch.from_numpy(latents.astype(np.float64))
    logger.info("Fitting GTM: %s", PUBLISHED_HYPERPARAMETERS)
    gtm.fit(x)
    resps, _ = gtm.project(x)
    entropy = shannon_entropy(resps.cpu().numpy())
    logger.info("Fitted. Shannon entropy %.4f", entropy)
    return gtm, entropy


def main(argv=None) -> int:
    args = parse_args(argv)

    frame = pd.read_csv(args.frameset)
    if args.smiles_column not in frame.columns:
        raise SystemExit(f"{args.frameset} has no {args.smiles_column!r} column")
    smiles = frame[args.smiles_column].astype(str).tolist()
    if args.limit:
        smiles = smiles[: args.limit]
    frameset_sha = hashlib.sha256(args.frameset.read_bytes()).hexdigest()
    logger.info("Frameset %s (%d rows, sha256 %s)", args.frameset, len(smiles), frameset_sha)

    latents, engine = encode_frameset(smiles, device=args.device, batch_size=args.batch_size)
    gtm, entropy = fit_manifold(latents, device=args.device)

    # Travels with the pickle so a loader can refuse a manifold fitted on the
    # other 256-wide latent space.
    gtm.descriptor_backend = DESCRIPTOR_BACKEND
    gtm.descriptor_checkpoint = engine.checkpoint_path

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wb") as handle:
        dill.dump(gtm, handle)
    output_sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    logger.info("Wrote %s (%d bytes)", args.output, args.output.stat().st_size)

    attributes = {
        "name": "molecular",
        "path": str(args.output),
        "sha256": output_sha,
        "bytes": args.output.stat().st_size,
        "model_class": "chemographykit.gtm.GTM",
        "descriptor_backend": DESCRIPTOR_BACKEND,
        "descriptor_checkpoint": engine.checkpoint_path,
        "descriptor_dimensions": int(latents.shape[1]),
        "training_frameset": {
            "path": str(args.frameset),
            "sha256": frameset_sha,
            "rows": len(smiles),
        },
        "unknown_token_substitutions": engine.unknown_token_substitutions,
        "hyperparameters": PUBLISHED_HYPERPARAMETERS,
        "hyperparameter_source": "docs/manuscript/gtm-model-attributes.json (not re-tuned)",
        "shannon_entropy": float(entropy),
    }
    attributes_path = args.attributes or args.output.with_suffix("").with_suffix(".json")
    attributes_path.write_text(json.dumps(attributes, indent=2) + "\n")
    logger.info("Wrote %s", attributes_path)
    print(json.dumps(attributes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
