"""Molecular and peptide design MCP tool specs."""

from __future__ import annotations

from typing import List

from ..facades.molecular_design import molecular_designer_facade
from ..facades.peptide_design import peptide_designer_facade
from ..tool_adapter import ToolSpec

_MOLECULAR_METHODS = [
    (
        "mol_list_design_engines",
        "list_design_engines",
        "List available molecular design engines and supported generation modes.",
        True,
        {},
        False,
    ),
    (
        "mol_design_molecules",
        "design_molecules",
        "Design small-molecule candidates with a selected molecular design engine.",
        False,
        {"_source_tool": "design_molecules"},
        True,
    ),
    (
        "mol_generate_analogs",
        "generate_analogs",
        "Generate small-molecule analogs around a seed SMILES.",
        False,
        {},
        True,
    ),
    (
        "mol_interpolate_molecules",
        "interpolate_molecules",
        "Interpolate between two molecules using the molecular autoencoder engine.",
        False,
        {},
        True,
    ),
    (
        "mol_validate_design_candidates",
        "validate_design_candidates",
        "Validate, standardize, and annotate proposed molecular design candidates.",
        True,
        {},
        False,
    ),
    (
        "mol_rank_design_candidates",
        "rank_design_candidates",
        "Rank validated molecular design candidates by seed similarity and quality.",
        True,
        {},
        False,
    ),
    (
        "mol_register_design_candidates",
        "register_design_candidates",
        "Persist final molecular design candidates as a generated candidate set.",
        False,
        {},
        True,
    ),
]

MOLECULAR_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=mcp_name,
        toolkit_factory=molecular_designer_facade,
        method=method,
        summary=summary,
        forces=forces,
        read_only=read_only,
        requires_network=requires_network,
    )
    for mcp_name, method, summary, read_only, forces, requires_network in _MOLECULAR_METHODS
]

_PEPTIDE_METHODS = [
    (
        "peptide_list_design_engines",
        "list_design_engines",
        "List available peptide design engines and supported generation modes.",
        True,
        {},
        False,
    ),
    (
        "peptide_design_peptides",
        "design_peptides",
        "Design peptide candidates with a selected peptide design engine.",
        False,
        {"_source_tool": "design_peptides"},
        True,
    ),
    (
        "peptide_generate_analogs",
        "generate_peptide_analogs",
        "Generate peptide analogs around a seed sequence.",
        False,
        {},
        True,
    ),
    (
        "peptide_design_interpolation",
        "design_peptide_interpolation",
        "Interpolate between two peptide sequences using the WAE engine.",
        False,
        {},
        True,
    ),
    (
        "peptide_validate_design_candidates",
        "validate_design_candidates",
        "Validate, normalize, and annotate proposed peptide design candidates.",
        True,
        {},
        False,
    ),
    (
        "peptide_rank_design_candidates",
        "rank_design_candidates",
        "Rank validated peptide design candidates by seed similarity and quality.",
        True,
        {},
        False,
    ),
    (
        "peptide_load_design_candidates",
        "load_peptide_design_candidates",
        "Load peptide design candidates from a session pointer or artifact path.",
        True,
        {},
        False,
    ),
    (
        "peptide_validate_model_loaded",
        "validate_model_loaded",
        "Check whether the Peptide WAE model is loaded and usable.",
        True,
        {},
        True,
    ),
    (
        "peptide_get_latent_dimension",
        "get_latent_dimension",
        "Return the Peptide WAE latent dimension.",
        True,
        {},
        True,
    ),
    (
        "peptide_encode_peptides",
        "encode_peptides",
        "Encode peptide sequences to latent vectors.",
        True,
        {},
        True,
    ),
    (
        "peptide_decode_latent",
        "decode_latent",
        "Decode latent vectors to peptide sequences.",
        True,
        {},
        True,
    ),
    (
        "peptide_sample_peptides",
        "sample_peptides",
        "Sample new peptides from the WAE latent space.",
        False,
        {},
        True,
    ),
    (
        "peptide_interpolate_peptides",
        "interpolate_peptides",
        "Interpolate between two peptides in WAE latent space.",
        True,
        {},
        True,
    ),
    (
        "peptide_reconstruct_sequence",
        "reconstruct_sequence",
        "Reconstruct a peptide sequence by encoding and decoding it.",
        True,
        {},
        True,
    ),
    (
        "peptide_explore_latent_neighborhood",
        "explore_latent_neighborhood",
        "Explore the WAE latent neighborhood around a peptide sequence.",
        True,
        {},
        True,
    ),
    (
        "peptide_get_model_info",
        "get_model_info",
        "Return metadata about the loaded Peptide WAE model.",
        True,
        {},
        True,
    ),
]

PEPTIDE_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=mcp_name,
        toolkit_factory=peptide_designer_facade,
        method=method,
        summary=summary,
        forces=forces,
        read_only=read_only,
        requires_network=requires_network,
    )
    for mcp_name, method, summary, read_only, forces, requires_network in _PEPTIDE_METHODS
]
