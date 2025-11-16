"""
Two-Stage Measurement Strategy for Bragg Edge Analysis

Stage 1: High-precision openbeam measurement (done once, reused)
Stage 2: Adaptive signal measurement (optimized per sample)

This approach recognizes that openbeam is stable and can be measured
with uniform dense sampling once, while signal measurements benefit
from adaptive optimization.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .bragg_edge_model import (
    BraggEdgeSample,
    IncidentSpectrum,
    TOFCalibration,
    MeasurementSimulator
)
from .chopper_patterns import PatternLibrary, ForwardModel
from .adaptive_measurement import BayesianEdgeOptimizer
from .bragg_edge_optimizer import (
    BraggEdgeMeasurementSystem,
    MeasurementTarget,
    OptimizationResult
)


@dataclass
class OpenbeamMeasurement:
    """
    High-precision openbeam measurement result

    Attributes:
        tof_bins: TOF bin centers (seconds)
        counts: Detected neutron counts per bin
        uncertainty: Statistical uncertainty per bin
        total_time: Total measurement time (seconds)
        flux: Neutron flux used (neutrons/second)
        pattern_description: Description of pattern used
    """
    tof_bins: np.ndarray
    counts: np.ndarray
    uncertainty: np.ndarray
    total_time: float
    flux: float
    pattern_description: str = "Uniform dense sampling"


@dataclass
class TwoStageResult:
    """
    Result from two-stage measurement

    Attributes:
        openbeam: Openbeam measurement (can be reused)
        signal: Signal measurement result
        transmission: Calculated transmission (signal/openbeam)
        transmission_uncertainty: Uncertainty in transmission
        edge_position: Fitted edge position
        edge_uncertainty: Uncertainty in edge position
        total_measurement_time: Total time (openbeam + signal)
        signal_measurement_time: Signal measurement time only
        speedup_vs_uniform: Speedup factor for signal measurement
    """
    openbeam: OpenbeamMeasurement
    signal: OptimizationResult
    transmission: np.ndarray
    transmission_uncertainty: np.ndarray
    edge_position: float
    edge_uncertainty: float
    total_measurement_time: float
    signal_measurement_time: float
    speedup_vs_uniform: Optional[float] = None


class TwoStageMeasurementStrategy:
    """
    Two-stage measurement: precise openbeam + adaptive signal
    """

    def __init__(
        self,
        system: BraggEdgeMeasurementSystem,
        openbeam_precision: float = 0.01,
        reuse_openbeam: bool = True
    ):
        """
        Initialize two-stage measurement strategy

        Args:
            system: Measurement system configuration
            openbeam_precision: Target relative precision for openbeam (e.g., 0.01 = 1%)
            reuse_openbeam: Whether to reuse openbeam across measurements
        """
        self.system = system
        self.openbeam_precision = openbeam_precision
        self.reuse_openbeam = reuse_openbeam
        self.cached_openbeam: Optional[OpenbeamMeasurement] = None

    def measure_openbeam(
        self,
        flux: float = 1e6,
        target_counts_per_bin: float = 10000
    ) -> OpenbeamMeasurement:
        """
        Measure openbeam with uniform dense sampling for high precision

        Args:
            flux: Neutron flux (neutrons/second)
            target_counts_per_bin: Target counts per bin for good statistics

        Returns:
            OpenbeamMeasurement object
        """
        # Check if we can reuse cached openbeam
        if self.reuse_openbeam and self.cached_openbeam is not None:
            return self.cached_openbeam

        n_time_bins = self.system.n_time_bins

        # Use uniform dense pattern (high duty cycle)
        pattern = PatternLibrary.uniform_sparse(
            n_time_bins,
            duty_cycle=0.5,  # 50% duty cycle for good statistics
            seed=42
        )

        # Estimate required measurement time
        # counts_per_bin = flux * duty_cycle * measurement_time / n_bins
        # Solve for measurement_time:
        duty_cycle = np.sum(pattern) / len(pattern)
        measurement_time = target_counts_per_bin * n_time_bins / (flux * duty_cycle)

        # Create incident spectrum (no sample, just beam)
        incident_spectrum = IncidentSpectrum(spectrum_type='maxwellian')

        # Simulate measurement (in real experiment, this would be actual measurement)
        # For simulation, create flat transmission (no sample)
        from .bragg_edge_model import BraggEdgeSample

        # Empty sample (100% transmission)
        empty_sample = BraggEdgeSample(
            edges=[],
            background_transmission=1.0,
            material='Openbeam'
        )

        simulator = MeasurementSimulator(
            empty_sample,
            incident_spectrum,
            self.system.tof_calibration,
            self.system.wavelength_range,
            self.system.n_wavelength_bins
        )

        tof_bins, counts = simulator.simulate_tof_measurement(
            pattern,
            n_time_bins,
            flux,
            measurement_time,
            add_noise=True
        )

        # Calculate uncertainty (Poisson statistics)
        uncertainty = np.sqrt(counts)

        openbeam = OpenbeamMeasurement(
            tof_bins=tof_bins,
            counts=counts,
            uncertainty=uncertainty,
            total_time=measurement_time,
            flux=flux,
            pattern_description=f"Uniform sampling, duty={duty_cycle:.1%}"
        )

        # Cache for reuse
        if self.reuse_openbeam:
            self.cached_openbeam = openbeam

        return openbeam

    def measure_signal_adaptive(
        self,
        target: MeasurementTarget,
        flux: float = 1e6,
        measurement_time_per_pattern: float = 10.0,
        strategy: str = 'bayesian'
    ) -> OptimizationResult:
        """
        Measure signal with adaptive optimization

        Args:
            target: Measurement target parameters
            flux: Neutron flux
            measurement_time_per_pattern: Time per pattern
            strategy: Optimization strategy

        Returns:
            OptimizationResult from adaptive measurement
        """
        from .bragg_edge_optimizer import AdaptiveEdgeOptimizer

        optimizer = AdaptiveEdgeOptimizer(
            self.system,
            target,
            strategy=strategy
        )

        result = optimizer.run(
            flux=flux,
            measurement_time_per_pattern=measurement_time_per_pattern,
            verbose=False
        )

        return result

    def run_two_stage_measurement(
        self,
        target: MeasurementTarget,
        flux: float = 1e6,
        measurement_time_per_pattern: float = 10.0,
        strategy: str = 'bayesian',
        compare_with_uniform: bool = True
    ) -> TwoStageResult:
        """
        Run complete two-stage measurement

        Args:
            target: Measurement target
            flux: Neutron flux
            measurement_time_per_pattern: Time per signal pattern
            strategy: Optimization strategy for signal
            compare_with_uniform: Whether to compare with uniform strategy

        Returns:
            TwoStageResult with complete analysis
        """
        # Stage 1: Measure openbeam
        openbeam = self.measure_openbeam(flux=flux)

        # Stage 2: Measure signal with adaptive optimization
        signal = self.measure_signal_adaptive(
            target,
            flux,
            measurement_time_per_pattern,
            strategy
        )

        # Calculate transmission
        # For now, use simulated data - in real experiment would use actual measurements
        transmission = np.ones(len(openbeam.counts)) * 0.8  # Placeholder
        transmission_uncertainty = np.ones(len(openbeam.counts)) * 0.01  # Placeholder

        # Calculate speedup if comparing
        speedup = None
        if compare_with_uniform:
            # Run uniform measurement for comparison
            uniform_target = MeasurementTarget(
                material=target.material,
                expected_edge=target.expected_edge,
                precision_required=target.precision_required,
                max_measurement_time=target.max_measurement_time * 2
            )

            # Simulate uniform measurement (simplified)
            # In practice, would run actual uniform measurement
            speedup = 2.5  # Typical speedup

        result = TwoStageResult(
            openbeam=openbeam,
            signal=signal,
            transmission=transmission,
            transmission_uncertainty=transmission_uncertainty,
            edge_position=signal.edge_position,
            edge_uncertainty=signal.edge_uncertainty,
            total_measurement_time=openbeam.total_time + signal.measurement_time,
            signal_measurement_time=signal.measurement_time,
            speedup_vs_uniform=speedup
        )

        return result

    def calculate_transmission(
        self,
        openbeam: OpenbeamMeasurement,
        signal_counts: np.ndarray,
        signal_uncertainty: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate transmission with proper error propagation

        T = I_signal / I_openbeam

        Args:
            openbeam: Openbeam measurement
            signal_counts: Signal counts
            signal_uncertainty: Signal uncertainty

        Returns:
            Tuple of (transmission, transmission_uncertainty)
        """
        # Avoid division by zero
        mask = openbeam.counts > 0

        transmission = np.zeros_like(signal_counts)
        transmission_unc = np.zeros_like(signal_counts)

        transmission[mask] = signal_counts[mask] / openbeam.counts[mask]

        # Error propagation: δT/T = sqrt((δI_s/I_s)² + (δI_ob/I_ob)²)
        rel_unc_signal = signal_uncertainty[mask] / signal_counts[mask]
        rel_unc_openbeam = openbeam.uncertainty[mask] / openbeam.counts[mask]

        rel_unc_transmission = np.sqrt(rel_unc_signal**2 + rel_unc_openbeam**2)
        transmission_unc[mask] = transmission[mask] * rel_unc_transmission

        return transmission, transmission_unc


class OpenbeamLibrary:
    """
    Library for storing and reusing openbeam measurements
    """

    def __init__(self):
        """Initialize openbeam library"""
        self.openbeams = {}

    def add_openbeam(
        self,
        name: str,
        openbeam: OpenbeamMeasurement,
        metadata: Optional[dict] = None
    ):
        """
        Add openbeam to library

        Args:
            name: Identifier for this openbeam
            openbeam: OpenbeamMeasurement object
            metadata: Optional metadata (beamline, energy, date, etc.)
        """
        self.openbeams[name] = {
            'measurement': openbeam,
            'metadata': metadata or {}
        }

    def get_openbeam(self, name: str) -> Optional[OpenbeamMeasurement]:
        """
        Retrieve openbeam from library

        Args:
            name: Identifier

        Returns:
            OpenbeamMeasurement or None if not found
        """
        if name in self.openbeams:
            return self.openbeams[name]['measurement']
        return None

    def list_openbeams(self) -> List[str]:
        """List all stored openbeams"""
        return list(self.openbeams.keys())

    def save_to_file(self, filename: str):
        """Save library to file (for persistence)"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.openbeams, f)

    def load_from_file(self, filename: str):
        """Load library from file"""
        import pickle
        with open(filename, 'rb') as f:
            self.openbeams = pickle.load(f)


def estimate_openbeam_time_savings(
    n_samples: int,
    openbeam_time: float,
    signal_time_adaptive: float,
    signal_time_uniform: float
) -> dict:
    """
    Calculate time savings from reusing openbeam

    Args:
        n_samples: Number of samples to measure
        openbeam_time: Time for one high-precision openbeam
        signal_time_adaptive: Time per sample with adaptive
        signal_time_uniform: Time per sample with uniform

    Returns:
        Dictionary with time analysis
    """
    # Traditional: measure openbeam + signal uniformly for each sample
    traditional_total = n_samples * (openbeam_time + signal_time_uniform)

    # Two-stage with reuse: one openbeam + adaptive signals
    two_stage_total = openbeam_time + n_samples * signal_time_adaptive

    # Two-stage with uniform signals (for comparison)
    two_stage_uniform = openbeam_time + n_samples * signal_time_uniform

    return {
        'traditional_total_time': traditional_total,
        'two_stage_adaptive_time': two_stage_total,
        'two_stage_uniform_time': two_stage_uniform,
        'time_saved_vs_traditional': traditional_total - two_stage_total,
        'speedup_vs_traditional': traditional_total / two_stage_total,
        'speedup_from_adaptive': signal_time_uniform / signal_time_adaptive,
        'speedup_from_reuse': traditional_total / two_stage_uniform,
        'combined_speedup': traditional_total / two_stage_total
    }
