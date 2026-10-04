"""Chemical similarity MCP tool specs."""

from __future__ import annotations

from typing import List

from ..tool_adapter import ToolSpec
from .common import factory

_SIMILARITY = factory("cs_copilot.tools.chemistry.similarity_toolkit:ChemicalSimilarityToolkit")

_METHODS = [
    (
        "calculate_tanimoto_similarity",
        "Tanimoto similarity for a SMILES pair: rdkit/morgan/maccs binary fingerprints "
        "or fp_type='morgan_count' (radius 2, 2048 counts; dot-product Tanimoto).",
    ),
    ("calculate_dice_similarity", "Dice similarity for one or more SMILES pairs."),
    ("calculate_tversky_similarity", "Tversky similarity for one or more SMILES pairs."),
    (
        "calculate_cosine_similarity",
        "Cosine similarity of binary fingerprints from a SMILES pair. Prefer Tanimoto "
        "for fingerprints; use chem_calculate_similarity_matrix for embedding vectors.",
    ),
    (
        "calculate_euclidean_distance",
        "Euclidean distance of binary fingerprints from a SMILES pair (lower is closer). "
        "Prefer Tanimoto for fingerprints; use chem_calculate_similarity_matrix for embeddings.",
    ),
    ("calculate_all_similarities", "Compute Tanimoto / Dice / Tversky / cosine in a single call."),
    (
        "find_most_similar",
        "Find the most similar molecules to a query SMILES from a candidate list.",
    ),
    (
        "calculate_similarity_matrix",
        "Compare descriptor rows using descriptor_kind='binary', 'count', or 'embedding'. "
        "metric='auto' uses dot-product Tanimoto for fingerprints and Euclidean distance "
        "for embeddings; embeddings also support metric='cosine'. Returns matrix and "
        "is_distance (lower is closer for distance, higher for similarity). Preserve "
        "counts and descriptor provenance; submit bounded blocks of large datasets.",
    ),
]

SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=f"chem_{name}",
        toolkit_factory=_SIMILARITY,
        method=name,
        summary=summary,
        read_only=True,
    )
    for name, summary in _METHODS
]
