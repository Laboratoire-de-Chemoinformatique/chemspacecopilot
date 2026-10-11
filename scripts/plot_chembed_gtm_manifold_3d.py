#!/usr/bin/env python
# coding: utf-8
"""Render the chembed GTM manifold in the first three principal components.

A GTM manifold is a two-dimensional sheet embedded in the encoder's latent
space. This projects its 900 node centres into the leading principal components
of the data the map was fitted on, so the sheet can be seen inside the cloud it
describes.

Read it with the explained variance in mind. Three components hold well under a
third of the variance of a 256-dimensional latent, so the sheet will appear to
fold through itself. That is mostly the projection, not the manifold: the local
stretch between neighbouring nodes is reported alongside, and a sheet with a
narrow stretch distribution is locally regular however tangled it looks here.

Writes a static multi-angle PNG and, unless disabled, an interactive HTML that
can be rotated, which is the only way to read a self-intersecting surface
honestly.

Usage::

    python scripts/plot_chembed_gtm_manifold_3d.py \\
        --frameset gtm_frameset_training.csv \\
        --output docs/manuscript/results/chembed-latent-gtm/chembed_gtm_manifold_3d.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_chembed_gtm_manifold_3d")


def local_stretch(sheet: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return latent distance per unit map distance, for each grid edge.

    A constant value means the sheet maps the plane rigidly. A wide spread means
    equal steps on the map correspond to unequal moves in latent space, so map
    distance stops being a fair proxy for chemical distance.

    Args:
        sheet: ``(side, side, n_latent)`` node centres.
        grid: ``(side, side, 2)`` node coordinates on the map.

    Returns:
        1D array of per-edge stretch values.
    """
    rows = np.linalg.norm(sheet[1:] - sheet[:-1], axis=-1) / np.linalg.norm(
        grid[1:] - grid[:-1], axis=-1
    )
    cols = np.linalg.norm(sheet[:, 1:] - sheet[:, :-1], axis=-1) / np.linalg.norm(
        grid[:, 1:] - grid[:, :-1], axis=-1
    )
    return np.concatenate([rows.ravel(), cols.ravel()])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frameset", type=Path, required=True)
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument(
        "--style",
        choices=("surface", "points"),
        default="surface",
        help="surface draws the sheet as a translucent mesh; points draws the node "
        "centres as red markers over the compounds as black dots, showing where the "
        "nodes sit rather than interpolating between them.",
    )
    parser.add_argument("--cloud-stride", type=int, default=25)
    parser.add_argument("--node-size", type=float, default=16.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def _load(args) -> Tuple:
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.decomposition import PCA

    from cs_copilot.tools.chemistry.chembed_engine import ChembedEngine
    from cs_copilot.tools.chemography.gtm_operations import (
        latent_centers_from_gtm,
        load_gtm_model,
    )
    from cs_copilot.tools.constants import (
        HUGGINGFACE_GTM_CHEMBED_REPO,
        HUGGINGFACE_GTM_CHEMBED_REVISION,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gtm = load_gtm_model(
        hf_hub_download(
            repo_id=HUGGINGFACE_GTM_CHEMBED_REPO,
            revision=HUGGINGFACE_GTM_CHEMBED_REVISION,
            filename="GTM_manifold_chembed_900_225.pkl.gz",
        )
    )
    engine = ChembedEngine()
    engine.load(model_path=None, device=torch.device(device))

    frame = pd.read_csv(args.frameset)
    latents = engine.encode(frame[args.smiles_column].astype(str).tolist(), batch_size=256)
    latents = latents.astype(np.float64)

    # PCA on the data, so the sheet is seen inside the cloud it describes.
    pca = PCA(n_components=3).fit(latents)
    centers = latent_centers_from_gtm(gtm)
    nodes = gtm.nodes.cpu().numpy()
    side = int(round(np.sqrt(len(nodes))))
    order = np.lexsort((nodes[:, 0], nodes[:, 1]))

    responsibilities, _ = gtm.project(torch.from_numpy(latents))
    density = responsibilities.cpu().numpy().sum(axis=0)

    return (
        pca.transform(centers)[order].reshape(side, side, 3),
        pca.transform(latents)[:: args.cloud_stride],
        density[order].reshape(side, side),
        local_stretch(centers[order].reshape(side, side, -1), nodes[order].reshape(side, side, 2)),
        pca.explained_variance_ratio_,
        side,
        len(frame),
        f"{HUGGINGFACE_GTM_CHEMBED_REPO} @ {HUGGINGFACE_GTM_CHEMBED_REVISION[:8]}",
    )


def main(argv: Optional[list] = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    sheet, cloud, density, stretch, variance, side, n_molecules, label = _load(args)

    logger.info(
        "PC1-3 explain %.1f%% of variance; local stretch median %.1f, p90/p10 %.1fx",
        variance[:3].sum() * 100,
        np.median(stretch),
        np.percentile(stretch, 90) / np.percentile(stretch, 10),
    )

    norm = plt.Normalize(np.percentile(density, 2), np.percentile(density, 98))
    facecolors = plt.cm.viridis(norm(density))
    figure = plt.figure(figsize=(15, 13.5))
    for position, (elev, azim) in enumerate(((24, -62), (24, 28), (68, -62), (6, 0)), start=1):
        axis = figure.add_subplot(2, 2, position, projection="3d")
        if args.style == "points":
            # Where the nodes actually sit, rather than a surface interpolated
            # between them. The compounds go down first so the nodes read on top.
            axis.scatter(
                cloud[:, 0],
                cloud[:, 1],
                cloud[:, 2],
                s=2.5,
                c="black",
                alpha=0.30,
                linewidths=0,
                depthshade=False,
            )
            flat = sheet.reshape(-1, 3)
            axis.scatter(
                flat[:, 0],
                flat[:, 1],
                flat[:, 2],
                s=args.node_size,
                c="red",
                alpha=0.9,
                linewidths=0.25,
                edgecolors="darkred",
                depthshade=False,
                zorder=5,
            )
        else:
            axis.scatter(
                cloud[:, 0], cloud[:, 1], cloud[:, 2], s=2, c="0.55", alpha=0.25, linewidths=0
            )
            # Translucent facets plus grid lines: an opaque surface that folds
            # through itself hides exactly the structure worth seeing.
            axis.plot_surface(
                sheet[:, :, 0],
                sheet[:, :, 1],
                sheet[:, :, 2],
                facecolors=facecolors,
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=True,
                shade=False,
                alpha=0.55,
            )
            axis.plot_wireframe(
                sheet[:, :, 0],
                sheet[:, :, 1],
                sheet[:, :, 2],
                rstride=2,
                cstride=2,
                linewidth=0.4,
                color="0.15",
                alpha=0.65,
            )
        axis.set_xlabel(f"PC1 ({variance[0]*100:.1f}%)")
        axis.set_ylabel(f"PC2 ({variance[1]*100:.1f}%)")
        axis.set_zlabel(f"PC3 ({variance[2]*100:.1f}%)")
        axis.set_title(f"elevation {elev}°, azimuth {azim}°", fontsize=10)
        axis.view_init(elev=elev, azim=azim)

    if args.style == "points":
        figure.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    color="none",
                    markerfacecolor="red",
                    markeredgecolor="black",
                    markersize=11,
                    label="GTM node centres (900)",
                ),
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    color="none",
                    markerfacecolor="black",
                    markersize=5,
                    label=f"frameset molecules ({len(cloud):,} shown)",
                ),
            ],
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=12,
        )
    else:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        mappable.set_array([])
        figure.colorbar(mappable, ax=figure.axes, shrink=0.5, pad=0.02).set_label(
            "frameset molecules per node"
        )
    figure.suptitle(
        f"chembed GTM manifold in the first 3 principal components  -  {label}\n"
        f"the {side*side}-node sheet inside {n_molecules:,} frameset molecules "
        f"({'black dots' if args.style == 'points' else 'grey'}). "
        f"PC1-3 hold {variance[:3].sum()*100:.1f}% of the variance, so apparent folds are "
        "largely projection:\nlocal stretch has p90/p10 of only "
        f"{np.percentile(stretch,90)/np.percentile(stretch,10):.1f}x, i.e. the sheet is "
        "locally regular.",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140, bbox_inches="tight")
    logger.info("Wrote %s", args.output)

    if not args.no_html:
        import plotly.graph_objects as go

        html = args.output.with_suffix(".html")
        cloud_trace = go.Scatter3d(
            x=cloud[:, 0],
            y=cloud[:, 1],
            z=cloud[:, 2],
            mode="markers",
            marker=(
                {"size": 1.8, "color": "rgba(0,0,0,0.38)"}
                if args.style == "points"
                else {"size": 1.6, "color": "rgba(130,130,130,0.35)"}
            ),
            name="frameset molecules",
            hoverinfo="skip",
        )
        if args.style == "points":
            flat = sheet.reshape(-1, 3)
            manifold_trace = go.Scatter3d(
                x=flat[:, 0],
                y=flat[:, 1],
                z=flat[:, 2],
                mode="markers",
                marker={
                    "size": 5.5,
                    "color": "red",
                    "line": {"width": 0.5, "color": "black"},
                },
                name="GTM node centres",
                customdata=density.reshape(-1, 1),
                hovertemplate=(
                    "PC1 %{x:.2f}<br>PC2 %{y:.2f}<br>PC3 %{z:.2f}"
                    "<br>%{customdata[0]:.0f} molecules<extra></extra>"
                ),
            )
        else:
            manifold_trace = go.Surface(
                x=sheet[:, :, 0],
                y=sheet[:, :, 1],
                z=sheet[:, :, 2],
                surfacecolor=density,
                colorscale="Viridis",
                opacity=0.92,
                colorbar={"title": "molecules<br>per node"},
                name="GTM manifold",
                contours={
                    "x": {"show": True, "color": "rgba(0,0,0,0.25)", "width": 1},
                    "y": {"show": True, "color": "rgba(0,0,0,0.25)", "width": 1},
                },
            )
        fig = go.Figure([cloud_trace, manifold_trace])
        fig.update_layout(
            title=(
                f"chembed GTM manifold in PC1-3  -  {label}<br>"
                f"<sub>PC1-3 hold {variance[:3].sum()*100:.1f}% of variance; "
                "drag to rotate</sub>"
            ),
            scene={
                "xaxis_title": f"PC1 ({variance[0]*100:.1f}%)",
                "yaxis_title": f"PC2 ({variance[1]*100:.1f}%)",
                "zaxis_title": f"PC3 ({variance[2]*100:.1f}%)",
                "aspectmode": "data",
            },
            margin={"l": 0, "r": 0, "t": 70, "b": 0},
        )
        fig.write_html(html, include_plotlyjs="cdn")
        logger.info("Wrote %s", html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
