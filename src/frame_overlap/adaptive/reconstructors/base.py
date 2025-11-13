"""
Base class for all reconstruction algorithms.

This module defines the interface that all reconstructors must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
import time

from ..event_data import EventDataset, NeutronEvent, ReconstructionResult


class BaseReconstructor(ABC):
    """
    Abstract base class for TOF spectrum reconstructors.

    All reconstruction algorithms inherit from this class and implement
    the reconstruct() and update() methods.

    Parameters
    ----------
    tof_range : tuple of float
        (min_tof, max_tof) range in microseconds
    n_bins : int
        Number of TOF bins for reconstruction

    Attributes
    ----------
    tof_bins : np.ndarray
        TOF bin centers in µs
    spectrum : np.ndarray
        Current spectrum estimate
    uncertainty : np.ndarray
        Current uncertainty estimate
    """

    def __init__(self, tof_range: tuple, n_bins: int):
        """Initialize reconstructor with TOF binning."""
        self.tof_range = tof_range
        self.n_bins = n_bins

        # Create bin centers
        self.tof_bins = np.linspace(tof_range[0], tof_range[1], n_bins)

        # Current state
        self.spectrum = None
        self.uncertainty = None

        # Statistics
        self.n_events_processed = 0
        self.iterations = 0

    @abstractmethod
    def reconstruct(
        self,
        event_data: EventDataset,
        **kwargs
    ) -> ReconstructionResult:
        """
        Reconstruct spectrum from event dataset.

        This is the main entry point for batch reconstruction.

        Parameters
        ----------
        event_data : EventDataset
            Input events with multiple timestamps
        **kwargs
            Algorithm-specific parameters

        Returns
        -------
        ReconstructionResult
            Reconstructed spectrum with diagnostics
        """
        pass

    @abstractmethod
    def update(self, new_events: List[NeutronEvent], **kwargs):
        """
        Update reconstruction with new events (for online/streaming).

        Parameters
        ----------
        new_events : List[NeutronEvent]
            New events to incorporate
        **kwargs
            Algorithm-specific parameters
        """
        pass

    def reset(self):
        """Reset reconstructor to initial state."""
        self.spectrum = None
        self.uncertainty = None
        self.n_events_processed = 0
        self.iterations = 0

    def get_current_result(self) -> ReconstructionResult:
        """
        Get current reconstruction result.

        Returns
        -------
        ReconstructionResult
            Current state as ReconstructionResult
        """
        if self.spectrum is None:
            raise ValueError("No reconstruction available. Call reconstruct() first.")

        return ReconstructionResult(
            spectrum=self.spectrum.copy(),
            tof_bins=self.tof_bins.copy(),
            uncertainty=self.uncertainty.copy(),
            chi2=0.0,  # Not calculated in online mode
            iterations=self.iterations,
            convergence=True,
            computation_time=0.0
        )

    def compute_chi2(
        self,
        reconstructed: np.ndarray,
        reference: np.ndarray,
        uncertainty: np.ndarray
    ) -> float:
        """
        Compute chi-squared goodness-of-fit.

        Parameters
        ----------
        reconstructed : np.ndarray
            Reconstructed spectrum
        reference : np.ndarray
            Reference (true) spectrum
        uncertainty : np.ndarray
            Uncertainty in reference

        Returns
        -------
        float
            Chi-squared value
        """
        # Avoid division by zero
        uncertainty_safe = np.maximum(uncertainty, 1e-10)

        # Calculate chi-squared
        residuals = reconstructed - reference
        chi2 = np.sum((residuals / uncertainty_safe) ** 2)

        return chi2

    def compute_log_likelihood(
        self,
        observed: np.ndarray,
        expected: np.ndarray
    ) -> float:
        """
        Compute Poisson log-likelihood.

        For Poisson-distributed data: LL = Σ [n log(λ) - λ - log(n!)]

        Parameters
        ----------
        observed : np.ndarray
            Observed counts
        expected : np.ndarray
            Expected counts

        Returns
        -------
        float
            Log-likelihood value
        """
        # Avoid log(0)
        expected_safe = np.maximum(expected, 1e-10)
        observed_safe = np.maximum(observed, 0)

        # Poisson log-likelihood (ignore constant term log(n!))
        ll = np.sum(observed_safe * np.log(expected_safe) - expected_safe)

        return ll

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"tof_range={self.tof_range}, "
                f"n_bins={self.n_bins}, "
                f"events_processed={self.n_events_processed})")
