"""
Adaptive Frame Overlap TOF Reconstruction Module

This module provides tools for adaptive reconstruction of neutron Time-of-Flight
spectra from event-mode data with multiple pulse timestamps per event.

Key Features:
- Event-mode data handling
- Multiple reconstruction algorithms (Baseline, Wiener, EM, Kalman)
- Adaptive kernel selection
- Uncertainty quantification
- Real-time reconstruction capability

Example Usage:
    >>> from frame_overlap.adaptive import EventDataset, EMReconstructor
    >>> events = EventDataset.from_hdf5('data.h5')
    >>> recon = EMReconstructor(tof_range=(1000, 20000), n_bins=1000)
    >>> result = recon.reconstruct(events, max_iterations=50)
    >>> result.plot()
"""

__version__ = "0.1.0"

# Core data structures
from .event_data import NeutronEvent, EventDataset, ReconstructionResult

# Reconstructors
from .reconstructors import (
    BaseReconstructor,
    BaselineReconstructor,
    WienerEventReconstructor,
    EMReconstructor,
)

# Adaptive components
from .kernel_manager import KernelManager
from .adaptive_controller import AdaptiveController

# Utilities
from .simulation import generate_synthetic_events, simulate_bragg_edges
from .uncertainty import calculate_uncertainty, bootstrap_uncertainty
from .evaluation import Benchmark, compare_reconstructors

__all__ = [
    # Data structures
    'NeutronEvent',
    'EventDataset',
    'ReconstructionResult',
    # Reconstructors
    'BaseReconstructor',
    'BaselineReconstructor',
    'WienerEventReconstructor',
    'EMReconstructor',
    # Adaptive
    'KernelManager',
    'AdaptiveController',
    # Utilities
    'generate_synthetic_events',
    'simulate_bragg_edges',
    'calculate_uncertainty',
    'bootstrap_uncertainty',
    'Benchmark',
    'compare_reconstructors',
]
