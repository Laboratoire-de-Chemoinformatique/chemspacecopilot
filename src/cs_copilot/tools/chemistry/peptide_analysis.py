"""Transparent positional peptide analysis, adapted from branch 114 helpers.

The frequency and positional-identity definitions follow ee6156fdc9ee15b8169af8441c150e16b31ecf99.
Logos align sequences at their N termini by default and infer no multiple-sequence alignment.
``align=True`` restores branch 114's PyFAMSA alignment (9dce3e6) and
``render_frequency_logo(aligned=True)`` its Logomaker rendering (ee6156f), the method behind the
published node-level logos. Those were drawn one GTM node at a time, so reproducing one takes
that node's sequences only. PyFAMSA and Logomaker come from the ``peptide-logos`` extra.
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
INSTALL_HINT = "install the peptide-logos extra (uv sync --extra peptide-logos)"

# Logomaker's "chemistry" scheme. Both engines use it, so a residue keeps its colour whether
# or not the sequences were aligned.
RESIDUE_COLORS = {
    **dict.fromkeys("GSTYC", "#008000"),
    **dict.fromkeys("QN", "#800080"),
    **dict.fromkeys("KRH", "#0000ff"),
    **dict.fromkeys("DE", "#ff0000"),
    **dict.fromkeys("AVLIPWFM", "#000000"),
}

# Below this many distinct sequences a column shows agreement among a handful of sequences
# rather than a position preference. Aligned columns are renormalised over their residues, so a
# column held by one sequence still draws a full-height letter.
LOW_SUPPORT_SEQUENCES = 5


def compact_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def positional_identity(left: str, right: str) -> float:
    """Matching residues at shared positions divided by the longer sequence length.

    Gaps never match and do not count towards length, so aligned rows compare alignment columns
    and raw sequences compare N-terminal positions.
    """
    left, right = compact_sequence(left), compact_sequence(right)
    denominator = max(len(left.replace(GAP, "")), len(right.replace(GAP, "")))
    if not denominator:
        return 0.0
    return sum(a == b != GAP for a, b in zip(left, right, strict=False)) / denominator


def _alignment_summary(rows: list[str], method: str, reason: str | None, **details) -> dict:
    gap_count = sum(row.count(GAP) for row in rows)
    length = max(map(len, rows), default=0)
    cells = length * len(rows)
    return {
        "method": method,
        "reason": reason,
        **details,
        "n_sequences": len(rows),
        "alignment_length": length,
        "gap_count": gap_count,
        "gap_fraction": round(gap_count / cells, 6) if cells else 0.0,
    }


def align_sequences(sequences: list[str]) -> tuple[list[str], dict]:
    """Align sequences with PyFAMSA, falling back to raw N-terminal positions.

    The published logos used a UPGMA guide tree, but PyFAMSA's UPGMA segfaults (which no
    ``except`` can catch) when two sequences share no residue type, so such sets use single
    linkage instead. Refinement is off because it does not give the same alignment twice.
    """
    sequences = [compact_sequence(sequence) for sequence in sequences]
    if len(sequences) < 2:
        return sequences, _alignment_summary(
            sequences, "n_terminal", "Fewer than two sequences; nothing to align."
        )
    try:
        import pyfamsa
    except ImportError as exc:
        return sequences, _alignment_summary(
            sequences, "n_terminal", f"PyFAMSA is unavailable ({exc}); {INSTALL_HINT}."
        )
    residue_sets = {frozenset(sequence) for sequence in sequences}
    disjoint = any(a.isdisjoint(b) for a, b in combinations(residue_sets, 2))
    guide_tree = "sl" if disjoint else "upgma"
    try:
        alignment = pyfamsa.Aligner(guide_tree=guide_tree, threads=1, refine=False).align(
            [
                pyfamsa.Sequence(f"seq_{index}".encode("ascii"), sequence.encode("ascii"))
                for index, sequence in enumerate(sequences)
            ]
        )
    except Exception as exc:  # pragma: no cover - defensive, PyFAMSA raises many types
        logger.warning("PyFAMSA alignment failed; falling back to raw positions: %s", exc)
        return sequences, _alignment_summary(
            sequences, "n_terminal", f"PyFAMSA alignment failed: {exc}."
        )

    by_index = {
        int(record.id.decode("ascii").removeprefix("seq_")): record.sequence.decode("ascii")
        for record in alignment
    }
    aligned = [by_index[index] for index in range(len(sequences))]
    return aligned, _alignment_summary(
        aligned, "pyfamsa", None, guide_tree=guide_tree, refine=False
    )


def analyze_sequences(
    sequences: list[str], *, align: bool = False
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Summarize every sequence and unordered pair on the logo's own columns."""
    sequences = [compact_sequence(sequence) for sequence in sequences]
    if not sequences or any(not s or not set(s).issubset(AMINO_ACIDS) for s in sequences):
        raise ValueError("Analysis requires nonempty sequences of the 20 standard amino acids")
    if len(sequences) > 1000:
        raise ValueError("At most 1000 sequences can be analyzed in one call")
    if align:
        columns, alignment = align_sequences(sequences)
    else:
        columns, alignment = sequences, _alignment_summary(
            sequences, "n_terminal", "Alignment not requested."
        )
    aligned = alignment["method"] == "pyfamsa"
    pairs = pd.DataFrame(
        [
            {
                "sequence_i": i,
                "sequence_j": j,
                "positional_identity": positional_identity(left, right),
            }
            for (i, left), (j, right) in combinations(enumerate(columns), 2)
        ],
        columns=["sequence_i", "sequence_j", "positional_identity"],
    )
    rows, low_support_positions = [], []
    for position in range(max(map(len, columns))):
        holders = [
            sequence
            for sequence in columns
            if len(sequence) > position and sequence[position] != GAP
        ]
        if not holders:
            continue
        residues = [sequence[position] for sequence in holders]
        if len(set(holders)) < LOW_SUPPORT_SEQUENCES:
            low_support_positions.append(position + 1)
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
        "similarity_definition": (
            "Residue matches in PyFAMSA alignment columns (gaps never match) / longer sequence length"
            if aligned
            else "N-terminal positional residue matches / longer sequence length; missing tail positions are mismatches"
        ),
        "logo_definition": (
            "PyFAMSA multiple-sequence alignment; frequency at each position is taken over the "
            "non-gap residues in that column, so a column held by few sequences still sums to 1"
            if aligned
            else "N-terminal alignment, not multiple-sequence alignment; frequency at each "
            "position uses only sequences reaching that position"
        ),
        "alignment": alignment,
        "aligned_sequences": columns if aligned else None,
        "low_support_positions": low_support_positions,
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


def _draw_logomaker(axis, frequency: pd.DataFrame) -> None:
    """Draw letter stacks with Logomaker, as the published node-level logos were drawn."""
    try:
        import logomaker
    except ImportError as exc:  # do not silently swap in a different-looking renderer
        raise ImportError(f"Aligned logos need Logomaker; {INSTALL_HINT}") from exc

    logo = logomaker.Logo(
        frequency.set_index("position")[list(AMINO_ACIDS)],
        color_scheme=RESIDUE_COLORS,
        stack_order="big_on_top",
        ax=axis,
        vpad=0.02,
        width=0.9,
    )
    # Same alpha as fade_probabilities=True, which rebuilds every glyph to apply it.
    for glyph in logo.glyph_list:
        if glyph.patch is not None:
            glyph.patch.set_alpha(glyph.ceiling - glyph.floor)


def _draw_text_paths(axis, frequency: pd.DataFrame) -> None:
    """Draw letter stacks as Matplotlib text paths scaled to each residue frequency."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    font = FontProperties(family="DejaVu Sans", weight="bold")
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
                    facecolor=RESIDUE_COLORS[residue],
                    linewidth=0,
                )
            )
            bottom += height


def render_frequency_logo(
    frequency: pd.DataFrame,
    *,
    aligned: bool = False,
    title: str = "Generated peptide positional sequence logo",
) -> dict[str, bytes]:
    """Render vector letter stacks; preserve frequency denominators.

    Aligned tables are drawn with Logomaker and labelled as alignment positions; N-terminal
    tables are drawn as Matplotlib text paths. Both share colours, axes and per-position counts.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    last = int(frequency["position"].max())
    figure = Figure(figsize=(max(8, last * 0.4), 4.0))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    (_draw_logomaker if aligned else _draw_text_paths)(axis, frequency)
    for patch in axis.patches:  # glyphs stay inside the axes; measuring each one is slow
        patch.set_in_layout(False)
    axis.set(
        xlim=(0.4, last + 0.6),
        ylim=(0, 1.01),
        ylabel="Residue frequency",
        xlabel=(
            "Alignment position (PyFAMSA)"
            if aligned
            else "N-terminal position (no multiple-sequence alignment)"
        ),
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
