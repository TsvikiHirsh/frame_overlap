"""
frame_overlap - Neutron Time-of-Flight Frame Overlap Analysis

This package provides tools for analyzing frame overlap in neutron ToF data
using deconvolution techniques. It features a modern object-oriented API
with support for data processing, reconstruction, and analysis.

Main Classes
------------
Data : Load and process neutron ToF data with convolution and frame overlap
Reconstruct : Apply various deconvolution filters to reconstruct signals
Analysis : Fit reconstructed data to extract material parameters
ParametricScan : Perform parametric scans over processing parameters

Legacy Functions
----------------
The package also provides backward-compatible functional API:
- read_tof_data, prepare_full_frame
- wiener_deconvolution, generate_kernel, apply_filter
- chi2_analysis, optimize_parameters
- plot_analysis
"""

# New object-oriented API
from .data_class import Data
from .reconstruct import Reconstruct
from .analysis_class import Analysis, CrossSection
from .groupby import ParametricScan, compare_configurations

# Legacy functional API (for backward compatibility)
from .data import read_tof_data, prepare_full_frame
from .analysis import wiener_deconvolution, generate_kernel, apply_filter
from .optimization import chi2_analysis, optimize_parameters, deconvolution_model
from .visualization import plot_analysis

__all__ = [
    # New OOP API
    'Data',
    'Reconstruct',
    'Analysis',
    'CrossSection',
    'ParametricScan',
    'compare_configurations',

    # Legacy functional API
    'read_tof_data',
    'prepare_full_frame',
    'wiener_deconvolution',
    'generate_kernel',
    'apply_filter',
    'chi2_analysis',
    'optimize_parameters',
    'deconvolution_model',
    'plot_analysis',
]

__version__ = '0.2.0'