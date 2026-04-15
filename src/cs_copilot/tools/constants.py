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

# Autoencoder model defaults
DEFAULT_AUTOENCODER_MODEL_PATH = os.path.expanduser("~/.cache/cs_copilot/models/autoencoder")
HUGGINGFACE_AUTOENCODER_REPO = "axelrolov/lstm_autoencoder"  # Hugging Face model repository

# Peptide WAE model defaults
DEFAULT_PEPTIDE_WAE_MODEL_PATH = os.path.expanduser("~/.cache/cs_copilot/models/peptide_wae")
HUGGINGFACE_PEPTIDE_WAE_REPO = "axelrolov/wae_peptides"

# GTM model defaults
DEFAULT_GTM_MODEL_PATH = os.path.expanduser("~/.cache/cs_copilot/models/gtm")
HUGGINGFACE_GTM_REPO = "axelrolov/chemspacecopilot-gtm"

# DBAASP antimicrobial peptide data
DEFAULT_DBAASP_DATA_PATH = os.path.expanduser("~/.cache/cs_copilot/data/dbaasp_data.csv")

# Standard peptide sequence column name
SEQUENCE_COLUMN = "SEQUENCE"

# Minimum data points per organism to build a reliable activity landscape
MIN_ORGANISM_DATA_POINTS = 200
