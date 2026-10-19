#!/usr/bin/env python
# coding: utf-8
"""
Constants and configuration values for the cs_copilot tools package.
"""

import os

# DataFrame preview settings
MAX_COL_WIDTH = 120
SAMPLE_ROWS = 3
SAMPLE_COLS = 6

# Standard column names
SMILES_COLUMN = "smi"  # Standard SMILES column name used throughout the codebase

# Chart settings
DEFAULT_CHART_WIDTH = 600
DEFAULT_CHART_HEIGHT = 600
DEFAULT_NODE_THRESHOLD = 0.1

# File extensions and formats
CSV_EXTENSION = ".csv"
HTML_EXTENSION = ".html"
PNG_EXTENSION = ".png"
PKL_GZ_EXTENSION = ".pkl.gz"

# GTM model and dataset discovery patterns
GTM_MODEL_SUFFIXES = (".pkl.gz", ".pkl", ".dill", ".pt")
GTM_MODEL_DOWNLOAD_PATTERNS = tuple(f"*{suffix}" for suffix in GTM_MODEL_SUFFIXES)

GTM_DATASET_SUFFIXES = (".csv", ".csv.gz", ".tsv", ".tsv.gz")
GTM_DATASET_DOWNLOAD_PATTERNS = tuple(f"*{suffix}" for suffix in GTM_DATASET_SUFFIXES)

DATASET_NAME_SUFFIX_CANDIDATES = ("", "_dataset", "_data", "_source", "_chembl")
FRAMESET_NAME_MARKERS = (
    "_source_mols",
    "_source_activity",
    "_density_table",
    "_node_lookup_by_coords",
    "_node_lookup_by_node",
)

# Image settings
DEFAULT_POINTS_SIZE = 30
DEFAULT_POINTS_OPACITY = 0.8
DEFAULT_LEGEND_FONT_SIZE = 20
DEFAULT_GRADIENT_MAX_LENGTH = 600
DEFAULT_GRADIENT_THICKNESS = 20
DEFAULT_TICK_COUNT = 6

# MIME types for images
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
}

# Default ChEMBL activity fields
CHEMBL_ACTIVITY_FIELDS = [
    "activity_id",
    "assay_chembl_id",
    "molecule_chembl_id",
    "activity_comment",
    "pchembl_value",
    "standard_type",
    "standard_value",
    "standard_units",
]

CHEMBL_MOLECULE_FIELDS = ["molecule_chembl_id", "canonical_smiles", "molecule_structures"]

# chembed SELFIES transformer VAE defaults. The checkpoint is downloaded and
# cached by huggingface_hub itself, so there is no cs_copilot model directory.
HUGGINGFACE_CHEMBED_REPO = "3BioCompBio/chembed-default"
CHEMBED_CHECKPOINT_FILENAME = "model.ckpt"

# Peptide Designer model defaults
DEFAULT_PEPTIDE_DESIGNER_MODEL_PATH = os.path.expanduser(
    "~/.cache/cs_copilot/models/peptide_designer"
)
HUGGINGFACE_PEPTIDE_WAE_REPO = "axelrolov/wae_peptides"

# Default GTM manifold, fitted on chembed latents of the pinned frameset. A
# manifold is only valid for the latent space it was fitted on, so it is pinned
# by revision rather than tracking a branch: a repository update cannot swap it
# under results that depend on it.
HUGGINGFACE_GTM_CHEMBED_REPO = "axelrolov/chemspacecopilot-gtm-chembed"
HUGGINGFACE_GTM_CHEMBED_REVISION = os.getenv(
    "HUGGINGFACE_GTM_CHEMBED_REVISION", "ca1d609ba58a0b2074cee8779d28af8ce435b6ef"
)
# Not ~/.cache/cs_copilot/models/gtm, which may still hold the retired LSTM
# manifold: the resolver returns any loadable file it finds in its directory,
# and both latents are 256 wide, so it would project in the wrong space silently.
DEFAULT_GTM_CHEMBED_MODEL_PATH = os.path.expanduser("~/.cache/cs_copilot/models/gtm-chembed")

# DBAASP antimicrobial peptide data
DEFAULT_DBAASP_DATA_PATH = os.path.expanduser(
    os.getenv("DBAASP_DATA_PATH", "~/.cache/cs_copilot/data/dbaasp_data.csv")
)

# Standard peptide sequence column name
SEQUENCE_COLUMN = "SEQUENCE"

# Minimum data points per organism to build a reliable activity landscape
MIN_ORGANISM_DATA_POINTS = 200
