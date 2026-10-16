"""Transparent positional peptide analysis, adapted from branch 114 helpers.

The frequency and positional-identity definitions follow ee6156fdc9ee15b8169af8441c150e16b31ecf99.
Logos align sequences at their N termini by default and infer no multiple-sequence alignment.
Passing ``align=True`` restores the PyFAMSA path introduced in 70b7442, which produced the
published node-level sequence logo; ``render_frequency_logo(engine="logomaker")`` restores
its Logomaker rendering.
"""

from __future__ import annotations

import io
import logging
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
GAP = "-"

# Below this many sequences a logo shows agreement among a handful of sequences rather than a
# position preference. Artifacts stay readable either way because every logo prints its counts.
LOW_SUPPORT_SEQUENCES = 5


def compact_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def positional_identity(left: str, right: str) -> float:
    """N-terminal matching residues divided by the longer sequence length."""
    left, right = compact_sequence(left), compact_sequence(right)
    denominator = max(len(left), len(right))
    if not denominator:
        return 0.0
    return sum(a == b for a, b in zip(left, right, strict=False)) / denominator


def _anchor_n_terminus(aligned: str) -> str:
    """Slide leading gaps to the C terminus so every sequence starts at position 1."""
    leading = len(aligned) - len(aligned.lstrip(GAP))
    return aligned if leading <= 0 else aligned[leading:] + GAP * leading


def align_sequences(
    sequences: list[str], *, anchor_n_terminus: bool = False
) -> tuple[list[str], dict]:
    """Align sequences with PyFAMSA, falling back to raw N-terminal positions.

    ``anchor_n_terminus`` re-anchors every sequence at position 1 after alignment. That
    discards the column correspondence PyFAMSA computed, so the result is reported as
    ``n_terminal`` rather than ``pyfamsa``: the two are different alignments and a logo
    built on them means different things.
    """
    sequences = [compact_sequence(sequence) for sequence in sequences]

    def unaligned(reason: str) -> tuple[list[str], dict]:
        return list(sequences), {
            "method": "none",
            "reason": reason,
            "n_sequences": len(sequences),
            "alignment_length": max(map(len, sequences), default=0),
            "gap_count": 0,
            "gap_fraction": 0.0,
            "anchor_n_terminus": anchor_n_terminus,
        }

    if len(sequences) < 2:
        return unaligned("Fewer than two sequences; nothing to align.")
    try:
        import pyfamsa
    except ImportError as exc:
        return unaligned(f"PyFAMSA is unavailable: {exc}.")
    try:
        aligner = pyfamsa.Aligner(guide_tree="upgma")
        alignment = aligner.align(
            [
                pyfamsa.Sequence(f"seq_{index}".encode("ascii"), sequence.encode("ascii"))
                for index, sequence in enumerate(sequences)
            ]
        )
    except Exception as exc:  # pragma: no cover - defensive, PyFAMSA raises many types
        logger.warning("PyFAMSA alignment failed; falling back to raw positions: %s", exc)
        return unaligned(f"PyFAMSA alignment failed: {exc}.")

    by_index = {
        int(record.id.decode("ascii").removeprefix("seq_")): record.sequence.decode("ascii")
        for record in alignment
    }
    aligned = [by_index.get(index, sequence) for index, sequence in enumerate(sequences)]
    if anchor_n_terminus:
        aligned = [_anchor_n_terminus(sequence) for sequence in aligned]
    gap_count = sum(sequence.count(GAP) for sequence in aligned)
    length = max(map(len, aligned), default=0)
    cells = length * len(aligned)
    return aligned, {
        "method": "n_terminal" if anchor_n_terminus else "pyfamsa",
        "reason": None,
        "guide_tree": "upgma",
        "n_sequences": len(aligned),
        "alignment_length": length,
        "gap_count": gap_count,
        "gap_fraction": round(gap_count / cells, 6) if cells else 0.0,
        "anchor_n_terminus": anchor_n_terminus,
    }


def analyze_sequences(
    sequences: list[str], *, align: bool = False, anchor_n_terminus: bool = False
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Summarize every sequence and unordered pair, using observed-position counts."""
    sequences = [compact_sequence(sequence) for sequence in sequences]
    if not sequences or any(not s or not set(s).issubset(AMINO_ACIDS) for s in sequences):
        raise ValueError("Analysis requires nonempty sequences of the 20 standard amino acids")
    if len(sequences) > 1000:
        raise ValueError("At most 1000 sequences can be analyzed in one call")
    if anchor_n_terminus and not align:
        raise ValueError("anchor_n_terminus only applies to align=True; without it, nothing aligns")
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
    if align:
        columns, alignment = align_sequences(sequences, anchor_n_terminus=anchor_n_terminus)
    else:
        columns, alignment = list(sequences), {
            "method": "n_terminal",
            "reason": "Alignment not requested.",
            "n_sequences": len(sequences),
            "alignment_length": max(map(len, sequences)),
            "gap_count": 0,
            "gap_fraction": 0.0,
            "anchor_n_terminus": False,
        }
    rows = []
    for position in range(max(map(len, columns))):
        residues = [
            sequence[position]
            for sequence in columns
            if len(sequence) > position and sequence[position] != GAP
        ]
        if not residues:
            continue
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
        "logo_definition": (
            "PyFAMSA multiple-sequence alignment; frequency at each position is taken over the "
            "non-gap residues in that column, so a column held by few sequences still sums to 1"
            if alignment["method"] == "pyfamsa"
            else "N-terminal alignment, not multiple-sequence alignment; frequency at each "
            "position uses only sequences reaching that position"
        ),
        "alignment": alignment,
        "aligned_sequences": list(columns) if alignment["method"] == "pyfamsa" else None,
        "low_support": len(sequences) < LOW_SUPPORT_SEQUENCES,
        "sequences": sequences,
        "training_set_novelty": None,
    }
    return metrics, pairs, frequency


def _save_figure(figure) -> dict[str, bytes]:
    outputs = {}
    for extension in ("png", "svg"):
        stream = io.BytesIO()
        figure.savefig(stream, format=extension, dpi=300)
        outputs[extension] = stream.getvalue()
    return outputs


def _probability_matrix(frequency: pd.DataFrame) -> pd.DataFrame:
    """Per-position residue probabilities, dropping positions with no residue mass."""
    matrix = frequency.set_index("position")[list(AMINO_ACIDS)].astype(float)
    totals = matrix.sum(axis=1)
    keep = totals > 1e-9
    return matrix[keep].div(totals[keep], axis=0)


def _render_logomaker(frequency: pd.DataFrame, title: str, xlabel: str) -> dict[str, bytes]:
    """Render with Logomaker, as the published node-level logo was rendered."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    try:
        import logomaker
    except ImportError as exc:  # do not silently swap in a different-looking renderer
        raise ImportError(
            "engine='logomaker' requires logomaker. Install it with: pip install logomaker"
        ) from exc

    matrix = _probability_matrix(frequency)
    if matrix.empty:
        raise ValueError("No position carries non-gap residue mass; nothing to render")
    figure = Figure(figsize=(max(8, min(18, len(matrix) * 0.45)), 3.6))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    logomaker.Logo(
        matrix,
        color_scheme="chemistry",
        stack_order="big_on_top",
        fade_probabilities=True,
        ax=axis,
        vpad=0.02,
        width=0.9,
    )
    axis.set(xlabel=xlabel, ylabel="Frequency", ylim=(0, 1), title=title)
    axis.set_xticks(matrix.index.tolist())
    axis.set_xticklabels([str(value) for value in matrix.index], fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.01,
        "Per-position sample counts: " + ", ".join(map(str, frequency["n_observed"])),
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return _save_figure(figure)


def render_frequency_logo(
    frequency: pd.DataFrame,
    *,
    engine: str = "matplotlib",
    title: str = "Generated peptide positional sequence logo",
    xlabel: str = "N-terminal position (no multiple-sequence alignment)",
) -> dict[str, bytes]:
    """Render vector letter stacks using Matplotlib; preserve frequency denominators."""
    if engine == "logomaker":
        return _render_logomaker(frequency, title, xlabel)
    if engine != "matplotlib":
        raise ValueError(f"Unknown logo engine '{engine}'; expected matplotlib or logomaker")

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
        xlabel=xlabel,
        title=title,
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
    return _save_figure(figure)
