"""Transparent positional peptide analysis, adapted from branch 114 helpers.

The frequency and positional-identity definitions follow ee6156fdc9ee15b8169af8441c150e16b31ecf99.
Logos align sequences at their N termini; no multiple-sequence alignment is inferred.
"""

from __future__ import annotations

import io
from itertools import combinations

import numpy as np
import pandas as pd

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")


def compact_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def positional_identity(left: str, right: str) -> float:
    """N-terminal matching residues divided by the longer sequence length."""
    left, right = compact_sequence(left), compact_sequence(right)
    denominator = max(len(left), len(right))
    if not denominator:
        return 0.0
    return sum(a == b for a, b in zip(left, right, strict=False)) / denominator


def analyze_sequences(sequences: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Summarize every sequence and unordered pair, using observed-position counts."""
    sequences = [compact_sequence(sequence) for sequence in sequences]
    if not sequences or any(not s or not set(s).issubset(AMINO_ACIDS) for s in sequences):
        raise ValueError("Analysis requires nonempty sequences of the 20 standard amino acids")
    if len(sequences) > 1000:
        raise ValueError("At most 1000 sequences can be analyzed in one call")
    pairs = pd.DataFrame(
        [
            {
                "sequence_i": i,
                "sequence_j": j,
                "positional_identity": positional_identity(left, right),
            }
            for (i, left), (j, right) in combinations(enumerate(sequences), 2)
        ],
        columns=["sequence_i", "sequence_j", "positional_identity"],
    )
    rows = []
    for position in range(max(map(len, sequences))):
        residues = [sequence[position] for sequence in sequences if len(sequence) > position]
        rows.append(
            {
                "position": position + 1,
                "n_observed": len(residues),
                **{residue: residues.count(residue) / len(residues) for residue in AMINO_ACIDS},
            }
        )
    frequency = pd.DataFrame(rows)
    values = pairs["positional_identity"].to_numpy()
    metrics = {
        "sequence_count": len(sequences),
        "unique_sequence_count": len(set(sequences)),
        "exact_sequence_uniqueness": len(set(sequences)) / len(sequences),
        "pairwise_comparison_count": len(values),
        "mean_pairwise_similarity": float(np.mean(values)) if len(values) else None,
        "minimum_pairwise_similarity": float(np.min(values)) if len(values) else None,
        "maximum_pairwise_similarity": float(np.max(values)) if len(values) else None,
        "similarity_definition": "N-terminal positional residue matches / longer sequence length; missing tail positions are mismatches",
        "logo_definition": "N-terminal alignment, not multiple-sequence alignment; frequency at each position uses only sequences reaching that position",
        "sequences": sequences,
        "training_set_novelty": None,
    }
    return metrics, pairs, frequency


def render_frequency_logo(frequency: pd.DataFrame) -> dict[str, bytes]:
    """Render vector letter stacks using Matplotlib; preserve frequency denominators."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    figure = Figure(figsize=(max(8, len(frequency) * 0.4), 4.0))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    font = FontProperties(family="DejaVu Sans", weight="bold")
    colors = {
        **dict.fromkeys("KRH", "#1565c0"),
        **dict.fromkeys("DE", "#d32f2f"),
        **dict.fromkeys("STNQ", "#2e7d32"),
        **dict.fromkeys("CGP", "#8e24aa"),
    }
    for row in frequency.to_dict(orient="records"):
        bottom = 0.0
        for residue in sorted(AMINO_ACIDS, key=lambda letter: (row[letter], letter)):
            height = float(row[residue])
            if height <= 0:
                continue
            letter = TextPath((0, 0), residue, size=1, prop=font)
            bounds = letter.get_extents()
            transform = (
                Affine2D()
                .translate(-bounds.x0, -bounds.y0)
                .scale(0.8 / bounds.width, height / bounds.height)
                .translate(row["position"] - 0.4, bottom)
            )
            axis.add_patch(
                PathPatch(
                    letter,
                    transform=transform + axis.transData,
                    facecolor=colors.get(residue, "#424242"),
                    linewidth=0,
                )
            )
            bottom += height
    axis.set(
        xlim=(0.4, len(frequency) + 0.6),
        ylim=(0, 1.01),
        ylabel="Residue frequency",
        xlabel="N-terminal position (no multiple-sequence alignment)",
        title="Generated peptide positional sequence logo",
    )
    axis.set_xticks(frequency["position"].tolist())
    axis.tick_params(axis="x", labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.01,
        "Per-position sample counts: " + ", ".join(map(str, frequency["n_observed"])),
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    outputs = {}
    for extension in ("png", "svg"):
        stream = io.BytesIO()
        figure.savefig(stream, format=extension, dpi=300)
        outputs[extension] = stream.getvalue()
    return outputs
