"""
Adaptive Frame Overlap TOF Reconstruction Module

This module provides tools for adaptive reconstruction of neutron Time-of-Flight
spectra from event-mode data with multiple pulse timestamps per event.

Key Features:
- Event-mode data handling (NeutronEvent, EventDataset)
- Multiple reconstruction algorithms (Baseline, Wiener, EM, Kalman)
- Adaptive kernel selection
- Uncertainty quantification
- Real-time reconstruction capability

Example Usage:
    >>> from frame_overlap.adaptive import EventDataset
    >>> events = EventDataset.load_hdf5('data.h5')
    >>> # Reconstructors will be added in Phase 1
"""

__version__ = "0.1.0"

# Core data structures (implemented)
from .event_data import NeutronEvent, EventDataset, ReconstructionResult

# Reconstructors (will be imported as implemented)
from .reconstructors import BaseReconstructor

# Future imports (Phase 1-4):
# from .reconstructors import BaselineReconstructor, WienerEventReconstructor
# from .reconstructors import EMReconstructor
# from .kernel_manager import KernelManager
# from .adaptive_controller import AdaptiveController
# from .simulation import generate_synthetic_events
# from .evaluation import Benchmark

__all__ = [
    # Data structures (Phase 1)
    'NeutronEvent',
    'EventDataset',
    'ReconstructionResult',
    'BaseReconstructor',
]
