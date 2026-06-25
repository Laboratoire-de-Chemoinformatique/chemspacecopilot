"""Chemical similarity MCP tool specs."""

from __future__ import annotations

from typing import List

from ..tool_adapter import ToolSpec
from .common import factory

_SIMILARITY = factory("cs_copilot.tools.chemistry.similarity_toolkit:ChemicalSimilarityToolkit")

_METHODS = [
    ("calculate_tanimoto_similarity", "Tanimoto similarity for one or more SMILES pairs."),
    ("calculate_dice_similarity", "Dice similarity for one or more SMILES pairs."),
    ("calculate_tversky_similarity", "Tversky similarity for one or more SMILES pairs."),
    ("calculate_cosine_similarity", "Cosine similarity for one or more SMILES pairs."),
    (
        "calculate_euclidean_distance",
        "Euclidean distance between fingerprint vectors of SMILES pairs.",
    ),
    ("calculate_all_similarities", "Compute Tanimoto / Dice / Tversky / cosine in a single call."),
    (
        "find_most_similar",
        "Find the most similar molecules to a query SMILES from a candidate list.",
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
