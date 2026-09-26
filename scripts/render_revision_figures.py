#!/usr/bin/env python3
"""Render publication figures from the measured generation tables, without model calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, rdBase
from rdkit.Chem import Draw


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def render(input_dir: Path, output_dir: Path) -> None:
    paths = {
        name: input_dir / name
        for name in (
            "summary.json",
            "per_seed.csv",
            "unique_candidates.csv",
            "retrosynthesis_targets.csv",
        )
    }
    summary = json.loads(paths["summary.json"].read_text())
    batches = read_rows(paths["per_seed.csv"])
    candidates = read_rows(paths["unique_candidates.csv"])
    targets = read_rows(paths["retrosynthesis_targets.csv"])
    by_smiles = {row["canonical_smiles"]: row for row in candidates}
    if len(by_smiles) != summary["pooled_unique_standardized_count"]:
        raise ValueError("Candidate identities do not agree with the measured summary")
    if sum(int(row["observed_record_count"]) for row in batches) != summary["raw_output_count"]:
        raise ValueError("Batch counts do not agree with the measured summary")
    if any(row["smiles"] not in by_smiles for row in targets):
        raise ValueError("A selected target is absent from the measured candidate pool")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "svg.fonttype": "none"})
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.6), layout="constrained")
    labels = [str(row["seed"]) for row in batches]
    for axis, numerator, denominator, title, color in (
        (
            axes[0],
            "rdkit_valid_count",
            "observed_record_count",
            "A  Raw-output validity",
            "#28668c",
        ),
        (
            axes[1],
            "unique_standardized_count",
            "standardized_valid_count",
            "B  Uniqueness among valid outputs",
            "#328070",
        ),
    ):
        fractions = [int(row[numerator]) / int(row[denominator]) * 100 for row in batches]
        bars = axis.bar(labels, fractions, color=color, width=0.65)
        axis.set(ylim=(0, 100), xlabel="Random seed", ylabel="Percent (%)")
        axis.set_title(title, loc="left", fontsize=11, pad=12)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, row, value in zip(bars, batches, fractions, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 3,
                f"{row[numerator]}/{row[denominator]}",
                ha="center",
                fontsize=10,
            )
    for extension in ("svg", "pdf", "png"):
        figure.savefig(output_dir / f"generation_rates.{extension}", dpi=300)
    plt.close(figure)

    parent = summary["configuration"]["parent_smiles"]
    molecules = [Chem.MolFromSmiles(parent)] + [
        Chem.MolFromSmiles(row["smiles"]) for row in targets
    ]
    if any(molecule is None for molecule in molecules):
        raise ValueError("Invalid molecule in the figure inputs")
    legends = [f"Parent: {summary['configuration']['parent_id']}"] + [
        f"{row['target_id']}\nParent Tanimoto = {float(by_smiles[row['smiles']]['parent_tanimoto']):.2f}"
        for row in targets
    ]
    options = Draw.MolDrawOptions()
    options.legendFontSize = 22
    options.bondLineWidth = 2.0
    svg = Draw.MolsToGridImage(
        molecules,
        molsPerRow=3,
        subImgSize=(420, 285),
        legends=legends,
        useSVG=True,
        drawOptions=options,
    )
    # RDKit clears only the first panel in some SVG grid versions. Give every
    # panel an explicit white page so bonds and labels survive dark backgrounds.
    root = ET.fromstring(svg)
    root.insert(
        0,
        ET.Element(
            "{http://www.w3.org/2000/svg}rect",
            {
                "width": "100%",
                "height": "100%",
                "fill": "white",
            },
        ),
    )
    svg = ET.tostring(root, encoding="unicode")
    svg_path = output_dir / "selected_structures.svg"
    svg_path.write_text(svg)
    import cairosvg

    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(output_dir / "selected_structures.pdf"))
    cairosvg.svg2png(
        bytestring=svg.encode(), write_to=str(output_dir / "selected_structures.png"), scale=2
    )
    manifest = {
        "rdkit_version": rdBase.rdkitVersion,
        "matplotlib_version": matplotlib.__version__,
        "inputs": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
        },
        "outputs": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output_dir.iterdir())
            if path.suffix in {".svg", ".pdf", ".png"}
        },
        "interpretation": "Prospective local-latent generation; selected structures are unvalidated computational proposals. No training novelty or activity claim.",
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("docs/manuscript/results/generation")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/manuscript/figures"))
    args = parser.parse_args()
    render(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
