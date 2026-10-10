#!/usr/bin/env python
# coding: utf-8
"""Render the chembed GTM manifold: density, a projected dataset, activity.

Three panels. The first is the manifold itself, as responsibility density of
the frameset it was fitted on. The second projects an activity dataset onto it.
The third is the responsibility-weighted activity per node.

Each panel reports a Moran's I with a permutation test, because "this map looks
structured" is not a claim a colour scale can support: a log scale flattens
real contrast and a narrow one invents it.

Usage::

    python scripts/plot_chembed_gtm_map.py \\
        --frameset gtm_frameset_training.csv \\
        --compounds reports/reviewer_revision/seh/clean_compounds.csv \\
        --output docs/manuscript/results/chembed-latent-gtm/chembed_gtm_map.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_chembed_gtm_map")


def morans_i(values: np.ndarray, permutations: int = 999, seed: int = 0) -> Tuple[float, float]:
    """Rook-contiguity Moran's I over a grid, with a permutation p-value.

    Args:
        values: 2D grid; non-finite cells are excluded.
        permutations: Null draws.
        seed: Permutation seed.

    Returns:
        ``(I, p)``. I near 0 means no spatial structure.
    """
    mask = np.isfinite(values)
    index = np.argwhere(mask)
    position = {tuple(cell): k for k, cell in enumerate(index)}
    observed = np.array([values[tuple(cell)] for cell in index], dtype=float)
    centred = observed - observed.mean()

    pairs = [
        (k, position[(row + dr, col + dc)])
        for k, (row, col) in enumerate(index)
        for dr, dc in ((0, 1), (1, 0))
        if (row + dr, col + dc) in position
    ]
    if not pairs or centred.std() == 0:
        return float("nan"), float("nan")
    pairs_array = np.array(pairs)

    def statistic(z: np.ndarray) -> float:
        cross = (z[pairs_array[:, 0]] * z[pairs_array[:, 1]]).sum() * 2
        return len(z) / (2 * len(pairs_array)) * cross / (z**2).sum()

    value = statistic(centred)
    rng = np.random.default_rng(seed)
    null = np.array([statistic(rng.permutation(centred)) for _ in range(permutations)])
    return float(value), float((1 + (null >= value).sum()) / (permutations + 1))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frameset", type=Path, required=True)
    parser.add_argument("--frameset-smiles-column", default="SMILES")
    parser.add_argument("--compounds", type=Path, default=None)
    parser.add_argument("--smiles-column", default="canonical_smiles")
    parser.add_argument("--activity-column", default="pIC50")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-node-mass", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download

    from cs_copilot.tools.chemistry.chembed_engine import ChembedEngine
    from cs_copilot.tools.chemography.gtm_operations import load_gtm_model
    from cs_copilot.tools.constants import (
        HUGGINGFACE_GTM_CHEMBED_REPO,
        HUGGINGFACE_GTM_CHEMBED_REVISION,
    )

    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    path = hf_hub_download(
        repo_id=HUGGINGFACE_GTM_CHEMBED_REPO,
        revision=HUGGINGFACE_GTM_CHEMBED_REVISION,
        filename="GTM_manifold_chembed_900_225.pkl.gz",
    )
    gtm = load_gtm_model(path)
    nodes = gtm.nodes.cpu().numpy()
    side = int(round(np.sqrt(nodes.shape[0])))
    order = np.lexsort((nodes[:, 0], nodes[:, 1]))
    extent = [nodes[:, 0].min(), nodes[:, 0].max(), nodes[:, 1].min(), nodes[:, 1].max()]

    def as_grid(values: np.ndarray) -> np.ndarray:
        return values[order].reshape(side, side)

    engine = ChembedEngine()
    engine.load(model_path=None, device=torch.device(device))

    def project(smiles) -> np.ndarray:
        latents = engine.encode(list(smiles), batch_size=256).astype(np.float64)
        responsibilities, _ = gtm.project(torch.from_numpy(latents))
        return responsibilities.cpu().numpy()

    frameset = pd.read_csv(args.frameset)
    density = project(frameset[args.frameset_smiles_column].astype(str)).sum(axis=0)
    density_i, density_p = morans_i(as_grid(density))
    logger.info(
        "density: max/median %.2f, Moran's I %+.3f p=%.3f",
        density.max() / np.median(density),
        density_i,
        density_p,
    )

    panels = 3 if args.compounds else 1
    figure, axes = plt.subplots(1, panels, figsize=(5.7 * panels, 5.6), constrained_layout=True)
    axes = np.atleast_1d(axes)

    image = axes[0].imshow(
        as_grid(density),
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=np.percentile(density, 2),
        vmax=np.percentile(density, 98),
    )
    axes[0].set_title(
        f"a. The manifold\n{len(frameset):,} frameset molecules per node\n"
        f"max/median {density.max()/np.median(density):.1f}x, "
        f"Moran's I {density_i:+.2f} (p={density_p:.3f})",
        fontsize=10,
    )
    figure.colorbar(image, ax=axes[0], label="molecules per node")

    if args.compounds:
        compounds = pd.read_csv(args.compounds).dropna(subset=[args.activity_column])
        responsibilities = project(compounds[args.smiles_column].astype(str))
        activity = compounds[args.activity_column].to_numpy(float)
        coordinates = responsibilities @ nodes
        mass = responsibilities.sum(axis=0)
        weighted = (responsibilities * activity[:, None]).sum(axis=0) / np.maximum(mass, 1e-12)
        weighted[mass < args.min_node_mass] = np.nan
        activity_i, activity_p = morans_i(as_grid(weighted))
        logger.info(
            "activity: %d supported nodes, Moran's I %+.3f p=%.3f",
            int((mass >= args.min_node_mass).sum()),
            activity_i,
            activity_p,
        )

        axes[1].imshow(
            as_grid(density),
            origin="lower",
            extent=extent,
            cmap="Greys",
            vmin=np.percentile(density, 2),
            vmax=np.percentile(density, 98),
            alpha=0.45,
        )
        ascending = np.argsort(activity)
        points = axes[1].scatter(
            coordinates[ascending, 0],
            coordinates[ascending, 1],
            c=activity[ascending],
            s=9,
            cmap="plasma",
            alpha=0.9,
            linewidths=0,
        )
        axes[1].set_title(
            f"b. {len(compounds):,} compounds projected on it\ncoloured by {args.activity_column}\n"
            f"they occupy {int((mass >= args.min_node_mass).sum())} of {side*side} nodes",
            fontsize=10,
        )
        figure.colorbar(points, ax=axes[1], label=args.activity_column)

        landscape = axes[2].imshow(as_grid(weighted), origin="lower", extent=extent, cmap="plasma")
        axes[2].set_title(
            f"c. Activity landscape\nresponsibility-weighted mean {args.activity_column}\n"
            f"Moran's I {activity_i:+.2f} (p={activity_p:.3f})",
            fontsize=10,
        )
        figure.colorbar(landscape, ax=axes[2], label=f"weighted mean {args.activity_column}")

    for axis in axes:
        axis.set_xlabel("GTM dimension 1")
        axis.set_ylabel("GTM dimension 2")
        axis.set_aspect("equal")
    figure.suptitle(
        f"chembed GTM manifold  -  {HUGGINGFACE_GTM_CHEMBED_REPO} @ "
        f"{HUGGINGFACE_GTM_CHEMBED_REVISION[:8]}  ({side*side} nodes, 256-d chembed latents)",
        fontsize=12,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, bbox_inches="tight")
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
