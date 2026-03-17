#!/usr/bin/env python
# coding: utf-8
"""
Cs_copilot Tools Package

This package provides a comprehensive set of tools for cheminformatics analysis,
organized into logical modules for better maintainability and usability.

Modules:
--------
- constants: Configuration values and constants
- utils: Shared utility functions
- pandas_tools: Enhanced pandas integration with S3 support
- databases: Database toolkits including ChEMBL (use ChemblToolkit class)
- gtm_tools: GTM analysis and optimization
- chemistry: Molecular analysis and similarity tools
- visualization: Plotting and visualization tools

Main Classes and Functions:
---------------------------
"""

# Analysis toolkits
from .analysis import RobustnessAnalysisToolkit
from .chemistry import (
    AutoencoderToolkit,
    BaseChemistryToolkit,
    ChemicalSimilarityToolkit,
    PeptideWAEToolkit,
    SynPlannerToolkit,
)

# GTM Toolkit
from .chemography.gtm import GTMToolkit
from .chemography.gtm_operations import save_gtm_plot
from .constants import *  # noqa: F403

# ChEMBL toolkit now accessed via ChemblToolkit class
from .databases.chembl import ChemblToolkit

# Import all the main classes and functions for the public API
from .io.pointer_pandas_tools import PointerPandasTools
from .io.session_toolkit import SessionToolkit

# Backwards compatibility alias
from .io.utils import image_to_base64, safe_file_operation, validate_positive_int

# Define what gets exported when using "from cs_copilot.tools import *"
__all__ = [
    # Classes
    "PointerPandasTools",
    "SessionToolkit",
    "GTMToolkit",
    "BaseChemistryToolkit",
    "ChemicalSimilarityToolkit",
    "AutoencoderToolkit",
    "PeptideWAEToolkit",
    "SynPlannerToolkit",
    "ChemblToolkit",
    "RobustnessAnalysisToolkit",
    # Visualization functions
    "save_gtm_plot",
    # I/O functions
    "image_to_base64",
    # Utility functions
    "validate_positive_int",
    "safe_file_operation",
    # Constants (imported via *)
]
