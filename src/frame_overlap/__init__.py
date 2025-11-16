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
Workflow : High-level interface for method chaining and parameter sweeps
ParametricScan : Perform parametric scans over processing parameters

Adaptive Bragg Edge Measurement
--------------------------------
BraggEdgeMeasurementSystem : Complete measurement system with TOF calibration
AdaptiveEdgeOptimizer : Bayesian optimization for adaptive measurements
BraggEdgeSample : Sample model with multiple Bragg edges
IncidentSpectrum : Incident neutron spectrum models
PatternLibrary : Collection of chopper pattern generation strategies
PerformanceEvaluator : Tools for evaluating measurement strategies

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
from .analysis_nbragg import Analysis
from .workflow import Workflow
from .analysis_class import Analysis as LegacyAnalysis, CrossSection
from .groupby import ParametricScan, compare_configurations

# Adaptive Bragg Edge Measurement API
from .bragg_edge_model import (
    BraggEdge,
    BraggEdgeSample,
    IncidentSpectrum,
    TOFCalibration,
    MeasurementSimulator
)
from .chopper_patterns import PatternLibrary, ForwardModel
from .adaptive_measurement import (
    BayesianEdgeOptimizer,
    GradientFocusedMeasurement,
    MultiResolutionEdgeSearch,
    RealTimeAdaptiveSystem,
    EdgePosterior
)
from .bragg_edge_optimizer import (
    BraggEdgeMeasurementSystem,
    AdaptiveEdgeOptimizer,
    MeasurementTarget,
    OptimizationResult,
    optimize_measurement_strategy
)
from .performance_metrics import (
    PerformanceEvaluator,
    PerformanceMetrics,
    ComparisonResult
)
from .two_stage_measurement import (
    TwoStageMeasurementStrategy,
    OpenbeamMeasurement,
    TwoStageResult,
    OpenbeamLibrary,
    estimate_openbeam_time_savings
)
from .tof_offset_correction import (
    TOFOffsetCorrector,
    OffsetCorrectionResult,
    apply_offset_correction_to_workflow,
    estimate_expected_offset
)

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
    'Workflow',
    'LegacyAnalysis',
    'CrossSection',
    'ParametricScan',
    'compare_configurations',

    # Adaptive Bragg Edge API
    'BraggEdge',
    'BraggEdgeSample',
    'IncidentSpectrum',
    'TOFCalibration',
    'MeasurementSimulator',
    'PatternLibrary',
    'ForwardModel',
    'BayesianEdgeOptimizer',
    'GradientFocusedMeasurement',
    'MultiResolutionEdgeSearch',
    'RealTimeAdaptiveSystem',
    'EdgePosterior',
    'BraggEdgeMeasurementSystem',
    'AdaptiveEdgeOptimizer',
    'MeasurementTarget',
    'OptimizationResult',
    'optimize_measurement_strategy',
    'PerformanceEvaluator',
    'PerformanceMetrics',
    'ComparisonResult',
    'TwoStageMeasurementStrategy',
    'OpenbeamMeasurement',
    'TwoStageResult',
    'OpenbeamLibrary',
    'estimate_openbeam_time_savings',
    'TOFOffsetCorrector',
    'OffsetCorrectionResult',
    'apply_offset_correction_to_workflow',
    'estimate_expected_offset',

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

__version__ = '0.3.0'