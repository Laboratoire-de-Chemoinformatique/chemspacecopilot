#!/usr/bin/env python
"""Render saved prospective sEH projections without fitting or model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


def render(input_dir, output_dir):
    names = ["seh_density.csv", "seh_activity_regression.csv", "projection_summary.json"]
    inputs = {name: input_dir / name for name in names}
    density = pd.read_csv(inputs[names[0]])
    activity = pd.read_csv(inputs[names[1]])
    summary = json.loads(inputs[names[2]].read_text())
    if len(density) != summary["node_count"] or density.duplicated(["x", "y"]).any():
        raise ValueError("Density coordinates do not uniquely cover the recorded grid")
    if not density[["x", "y", "nodes"]].equals(activity[["x", "y", "nodes"]]):
        raise ValueError("Density and activity coordinates differ")
    if not np.isclose(density["density"].sum(), summary["sample_count"]):
        raise ValueError("Responsibility mass does not match the sample count")
    if not np.allclose(density["density"], activity["density"]):
        raise ValueError("Activity uses a different density denominator")
    grid = density.pivot(index="y", columns="x", values="density")
    values = activity.pivot(index="y", columns="x", values="filtered_reg_density")
    if grid.isna().any().any() or values.shape != grid.shape:
        raise ValueError("Incomplete map grid")
    xs, ys = grid.columns.to_numpy(), grid.index.to_numpy()
    if not np.all(np.diff(xs) == 1) or not np.all(np.diff(ys) == 1):
        raise ValueError("Expected adjacent, one-based map coordinates")
    masked = grid.to_numpy() < 0.1
    if not np.array_equal(np.isnan(values.to_numpy()), masked):
        raise ValueError("Activity masking differs from the declared 0.1 density threshold")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), layout="constrained")
    for axis, matrix, title, palette, label, norm in (
        (
            axes[0],
            grid.to_numpy(),
            "A  Compound density",
            "viridis",
            "Responsibility mass",
            LogNorm(vmin=0.1, vmax=float(grid.max().max())),
        ),
        (
            axes[1],
            values.to_numpy(),
            "B  IC₅₀ activity landscape",
            "cividis",
            "Responsibility-weighted pIC₅₀",
            None,
        ),
    ):
        color_map = matplotlib.colormaps[palette].copy()
        color_map.set_bad("#e8e8e8")
        artist = axis.pcolormesh(
            xs,
            ys,
            np.ma.masked_where(masked, matrix),
            shading="nearest",
            cmap=color_map,
            norm=norm,
            rasterized=False,
        )
        axis.set(
            xlabel="GTM column",
            ylabel="GTM row",
            aspect="equal",
            xticks=[1, 10, 20, 30],
            yticks=[1, 10, 20, 30],
        )
        axis.set_title(title, loc="left", fontsize=11, pad=9)
        figure.colorbar(artist, ax=axis, location="bottom", fraction=0.08, pad=0.14, label=label)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("svg", "pdf", "png"):
        path = output_dir / f"seh_landscapes.{extension}"
        figure.savefig(path, dpi=300, facecolor="white")
        outputs.append(path)
    plt.close(figure)
    manifest = {
        "inputs": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()
        },
        "outputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs},
        "matplotlib_version": matplotlib.__version__,
        "sample_count": summary["sample_count"],
        "coordinate_convention": "One-based x/y from saved tables; node=(x-1)*30+y; y increases upwards.",
        "density": "Sum of responsibilities over all 2,212 standardized compounds; logarithmic color scale.",
        "activity": "Sum(r_ij*pIC50_i)/sum(r_ij) over all 2,212 compounds, including intermediate labels.",
        "mask": "Both panels gray where node responsibility mass is below 0.1; no interpolation.",
        "scope": "Projection onto a pinned pretrained model, not a newly fitted map or a predictive validation.",
    }
    (output_dir / "seh_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("reports/reviewer_revision/frozen_inputs_v1")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/manuscript/figures"))
    args = parser.parse_args()
    render(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
