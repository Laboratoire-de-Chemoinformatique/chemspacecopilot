#!/usr/bin/env python
# coding: utf-8
"""Measure neighbourhood preservation of a GTM manifold.

Neighbourhood preservation is the mean fraction of each molecule's k nearest
neighbours in descriptor space that remain among its k nearest on the map. It
asks whether map proximity means chemical similarity, which is the assumption
every landscape reading rests on.

Two controls make the number interpretable, and both are easy to omit:

* **A shuffled baseline.** Preservation is bounded below by chance, and chance
  depends on k and sample size. Without it, 0.27 could be anything.
* **Matched sample size.** Preservation falls as the sample grows, because
  there are more molecules competing to be near. Comparing a 50,000-molecule
  set against a 2,000-molecule one measures the sizes, not the maps.

Measures the published chembed manifold. ``--fit-on-each`` adds a manifold
fitted on the dataset itself, as an upper bound for the published one.

Usage::

    python scripts/evaluate_gtm_neighborhood.py \\
        --frameset gtm_frameset_training.csv \\
        --compounds reports/reviewer_revision/seh/clean_compounds.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_gtm_neighborhood")

DEFAULT_K = (5, 10, 20, 50)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frameset", type=Path, required=True)
    parser.add_argument("--frameset-smiles-column", default="SMILES")
    parser.add_argument("--compounds", type=Path, default=None)
    parser.add_argument("--smiles-column", default="canonical_smiles")
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--shuffles", type=int, default=5)
    parser.add_argument(
        "--fit-on-each",
        action="store_true",
        help="Also fit a fresh GTM on each dataset and score that, which separates how "
        "much preservation comes from the latent space and how much from the manifold "
        "having been fitted on this chemistry.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def preservation(
    descriptors: np.ndarray,
    gtm,
    metric: str,
    ks: Sequence[int],
    shuffles: int,
    rng: np.random.Generator,
) -> Dict[int, Dict[str, float]]:
    """Return observed and shuffled preservation for each k."""
    import torch

    from cs_copilot.tools.chemography.gtm_operations import (
        _knn_indices,
        neighborhood_preservation,
    )

    responsibilities, _ = gtm.project(torch.from_numpy(descriptors))
    latent = responsibilities.cpu().numpy() @ gtm.nodes.cpu().numpy()

    scores: Dict[int, Dict[str, float]] = {}
    for k in ks:
        neighbours = _knn_indices(descriptors, k, metric)
        chance = float(
            np.mean(
                [
                    neighborhood_preservation(neighbours, latent[rng.permutation(len(latent))])
                    for _ in range(shuffles)
                ]
            )
        )
        observed = neighborhood_preservation(neighbours, latent)
        scores[k] = {
            "observed": observed,
            "shuffled": chance,
            "ratio_to_chance": observed / chance if chance else float("inf"),
        }
    return scores


def main(argv: Optional[List[str]] = None) -> int:
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download

    from cs_copilot.tools.chemistry.chembed_engine import ChembedEngine
    from cs_copilot.tools.chemography.gtm_operations import (
        _knn_indices,
        load_gtm_model,
        neighborhood_preservation,
        resolve_neighborhood_metric,
    )
    from cs_copilot.tools.constants import (
        HUGGINGFACE_GTM_CHEMBED_REPO,
        HUGGINGFACE_GTM_CHEMBED_REVISION,
    )

    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    chembed_gtm = load_gtm_model(
        hf_hub_download(
            repo_id=HUGGINGFACE_GTM_CHEMBED_REPO,
            revision=HUGGINGFACE_GTM_CHEMBED_REVISION,
            filename="GTM_manifold_chembed_900_225.pkl.gz",
        )
    )
    engine = ChembedEngine()
    engine.load(model_path=None, device=torch.device(device))

    datasets = {
        "frameset": pd.read_csv(args.frameset)[args.frameset_smiles_column].astype(str).tolist()
    }
    if args.compounds:
        datasets["compounds"] = pd.read_csv(args.compounds)[args.smiles_column].astype(str).tolist()

    # Preservation falls as the sample grows, so every set is cut to one size.
    size = args.sample_size or min(len(v) for v in datasets.values())
    logger.info("Comparing at n=%d per dataset, k=%s", size, args.k)

    report: Dict[str, Dict] = {"sample_size": size, "k": list(args.k), "manifolds": {}}
    for index, (name, smiles) in enumerate(datasets.items()):
        # One generator per dataset, not one per run: a shared stream makes every
        # dataset after the first depend on how many manifold arms were scored
        # before it, so adding or removing an arm silently moves the subsample
        # and the shuffled baseline at an unchanged --seed.
        rng = np.random.default_rng([args.seed, index])
        if len(smiles) > size:
            smiles = [smiles[i] for i in rng.choice(len(smiles), size, replace=False)]
        chembed_x = engine.encode(smiles, batch_size=256).astype(np.float64)

        # An engine that substitutes a placeholder when encoding fails would give
        # every failed molecule the same latent and inflate preservation.
        distinct = len(np.unique(np.round(chembed_x, 6), axis=0))
        if distinct < len(chembed_x):
            logger.warning(
                "%s: only %d distinct chembed latents for %d molecules; "
                "identical rows inflate preservation",
                name,
                distinct,
                len(chembed_x),
            )

        if args.fit_on_each:
            from cs_copilot.tools.chemography.gtm_operations import (
                project_latent_on_gtm,
                train_latent_gtm,
            )

            # Same shape as the published maps, so only the fitting set differs.
            fitted, scaler, entropy = train_latent_gtm(
                chembed_x,
                config={
                    "n_basis_functions_sqrt": 15,
                    "n_nodes_n_basis_diff": 15,
                    "basis_width": 1.0,
                    "reg_coeff": 100.0,
                },
                descriptor_backend="chembed",
            )
            logger.info("%s: fitted a GTM on it, Shannon entropy %.4f", name, entropy)
            own = {}
            for k in args.k:
                neighbours = _knn_indices(chembed_x, k, resolve_neighborhood_metric("chembed"))
                latent = (
                    project_latent_on_gtm(chembed_x, fitted, scaler) @ fitted.nodes.cpu().numpy()
                )
                own[k] = {
                    "observed": neighborhood_preservation(neighbours, latent),
                    "shannon_entropy": float(entropy),
                }
            report["manifolds"].setdefault("chembed_fitted_on_this_set", {})[name] = own

        report["manifolds"].setdefault("chembed", {})[name] = preservation(
            chembed_x,
            chembed_gtm,
            resolve_neighborhood_metric("chembed"),
            args.k,
            args.shuffles,
            rng,
        )

    columns = [
        (manifold, dataset)
        for manifold in report["manifolds"]
        for dataset in datasets
        if dataset in report["manifolds"][manifold]
    ]
    header = f"{'k':>4} " + "".join(f"{m[:14] + '/' + d[:8]:>26}" for m, d in columns)
    print(f"\nneighbourhood preservation, n={size}\n{header}")
    for k in args.k:
        row = f"{k:>4} "
        for manifold, dataset in columns:
            entry: Any = report["manifolds"][manifold][dataset][k]
            chance = entry.get("ratio_to_chance")
            row += f"{entry['observed']:>18.3f}" + (f" (x{chance:>4.0f})" if chance else " " * 8)
        print(row)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
        logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
