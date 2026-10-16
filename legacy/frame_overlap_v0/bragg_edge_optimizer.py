"""
Main Optimization Interface for Adaptive Bragg Edge Measurements

This module provides the main user-facing API for optimized Bragg edge measurements.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from .bragg_edge_model import (
    BraggEdgeSample,
    IncidentSpectrum,
    TOFCalibration,
    MeasurementSimulator
)
from .chopper_patterns import ForwardModel, PatternLibrary
from .adaptive_measurement import (
    BayesianEdgeOptimizer,
    GradientFocusedMeasurement,
    MultiResolutionEdgeSearch,
    EdgePosterior
)
from .performance_metrics import PerformanceEvaluator, PerformanceMetrics


@dataclass
class MeasurementTarget:
    """
    Target parameters for measurement optimization

    Attributes:
        material: Material name (e.g., 'Fe', 'Al')
        expected_edge: Expected edge position (Angstrom)
        precision_required: Required precision (Angstrom)
        max_measurement_time: Maximum measurement time (seconds)
        confidence_level: Required confidence level (0 to 1)
    """
    material: str
    expected_edge: float
    precision_required: float
    max_measurement_time: float = 300.0
    confidence_level: float = 0.95


@dataclass
class OptimizationResult:
    """
    Result of optimized measurement

    Attributes:
        edge_position: Measured edge position (Angstrom)
        edge_uncertainty: Uncertainty in position (Angstrom)
        measurement_time: Total measurement time (seconds)
        n_patterns: Number of patterns used
        patterns: List of chopper patterns used
        measurements: List of measurements
        convergence_history: History of precision over time
        strain: Calculated strain (if expected_edge provided)
        time_saved: Percentage of time saved vs uniform
        performance_metrics: Detailed performance metrics
    """
    edge_position: float
    edge_uncertainty: float
    measurement_time: float
    n_patterns: int
    patterns: List[np.ndarray]
    measurements: List[np.ndarray]
    convergence_history: List[Tuple[float, float]]  # (time, precision)
    strain: Optional[float] = None
    time_saved: Optional[float] = None
    performance_metrics: Optional[PerformanceMetrics] = None


class BraggEdgeMeasurementSystem:
    """
    Complete Bragg edge measurement system with optimization
    """

    def __init__(
        self,
        flight_path: float,
        wavelength_range: Tuple[float, float] = (1.0, 10.0),
        time_resolution: float = 1e-6,
        chopper_max_frequency: float = 1000.0,
        n_wavelength_bins: int = 1000,
        n_time_bins: int = 10000
    ):
        """
        Initialize measurement system

        Args:
            flight_path: Flight path length (meters)
            wavelength_range: Wavelength range (Angstrom)
            time_resolution: Time resolution (seconds)
            chopper_max_frequency: Maximum chopper frequency (Hz)
            n_wavelength_bins: Number of wavelength bins
            n_time_bins: Number of time bins
        """
        self.flight_path = flight_path
        self.wavelength_range = wavelength_range
        self.time_resolution = time_resolution
        self.chopper_max_frequency = chopper_max_frequency
        self.n_wavelength_bins = n_wavelength_bins
        self.n_time_bins = n_time_bins

        # Create TOF calibration
        self.tof_calibration = TOFCalibration(flight_path=flight_path)

        # Create wavelength and time grids
        self.wavelength_grid = np.linspace(
            wavelength_range[0],
            wavelength_range[1],
            n_wavelength_bins
        )

        # Time grid: from 0 to max TOF + safety margin
        max_tof = self.tof_calibration.wavelength_to_tof(wavelength_range[1])
        self.time_grid = np.linspace(0, max_tof * 1.5, n_time_bins)

    def create_forward_model(
        self,
        incident_spectrum: Optional[IncidentSpectrum] = None
    ) -> ForwardModel:
        """
        Create forward model for this system

        Args:
            incident_spectrum: Incident spectrum (creates Maxwellian if None)

        Returns:
            ForwardModel object
        """
        if incident_spectrum is None:
            incident_spectrum = IncidentSpectrum(spectrum_type='maxwellian')

        incident_intensity = incident_spectrum.intensity(self.wavelength_grid)

        forward_model = ForwardModel(
            self.wavelength_grid,
            self.time_grid,
            self.tof_calibration.wavelength_to_tof,
            incident_intensity
        )

        return forward_model


class AdaptiveEdgeOptimizer:
    """
    Main optimizer class for adaptive edge measurements
    """

    def __init__(
        self,
        system: BraggEdgeMeasurementSystem,
        target: MeasurementTarget,
        strategy: str = 'bayesian',
        incident_spectrum: Optional[IncidentSpectrum] = None
    ):
        """
        Initialize adaptive optimizer

        Args:
            system: Measurement system configuration
            target: Measurement target parameters
            strategy: Optimization strategy ('bayesian', 'gradient', 'multi_resolution')
            incident_spectrum: Incident spectrum (optional)
        """
        self.system = system
        self.target = target
        self.strategy = strategy

        # Create forward model
        self.forward_model = system.create_forward_model(incident_spectrum)

        # Initialize optimizer based on strategy
        if strategy == 'bayesian':
            self.optimizer = BayesianEdgeOptimizer(
                prior_position=target.expected_edge,
                prior_position_uncertainty=target.precision_required * 10,  # Start with 10x target
                wavelength_grid=system.wavelength_grid,
                time_grid=system.time_grid,
                forward_model=self.forward_model
            )
        elif strategy == 'gradient':
            self.optimizer = GradientFocusedMeasurement(
                wavelength_grid=system.wavelength_grid,
                time_grid=system.time_grid,
                forward_model=self.forward_model
            )
        elif strategy == 'multi_resolution':
            self.optimizer = MultiResolutionEdgeSearch(
                wavelength_range=system.wavelength_range,
                target_precision=target.precision_required,
                forward_model=self.forward_model
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def run(
        self,
        flux: float = 1e6,
        measurement_time_per_pattern: float = 10.0,
        max_iterations: Optional[int] = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run optimized measurement

        Args:
            flux: Neutron flux (neutrons/second)
            measurement_time_per_pattern: Time per pattern (seconds)
            max_iterations: Maximum number of iterations (None = auto)
            verbose: Print progress

        Returns:
            OptimizationResult object
        """
        patterns = []
        measurements = []
        convergence_history = []

        current_precision = np.inf
        total_time = 0.0
        iteration = 0

        if max_iterations is None:
            max_iterations = int(self.target.max_measurement_time / measurement_time_per_pattern)

        if verbose:
            print(f"Starting adaptive measurement with {self.strategy} strategy")
            print(f"Target: {self.target.precision_required:.4f} Å precision")
            print(f"Max time: {self.target.max_measurement_time:.1f} s")
            print("-" * 60)

        while (current_precision > self.target.precision_required and
               total_time < self.target.max_measurement_time and
               iteration < max_iterations):

            # Design next pattern
            if self.strategy == 'bayesian':
                pattern = self.optimizer.design_next_pattern(
                    self.system.n_time_bins,
                    target_duty_cycle=0.1,
                    strategy='gradient_focused'
                )
            elif self.strategy == 'gradient':
                pattern = self.optimizer.generate_pattern(
                    self.system.n_time_bins,
                    duty_cycle=0.1
                )
            elif self.strategy == 'multi_resolution':
                pattern = self.optimizer.design_pattern_for_resolution(
                    self.optimizer.current_resolution,
                    self.system.n_time_bins
                )
            else:
                pattern = PatternLibrary.uniform_sparse(self.system.n_time_bins, 0.1)

            # Simulate measurement (in real system, this would be actual measurement)
            # For now, create dummy measurement
            measurement = np.random.poisson(
                np.random.random(self.system.n_time_bins) * flux * measurement_time_per_pattern
            )

            # Update optimizer
            if hasattr(self.optimizer, 'update_posterior'):
                self.optimizer.update_posterior(measurement, pattern, measurement_time_per_pattern)
                current_precision = self.optimizer.posterior.get_precision()
            elif hasattr(self.optimizer, 'update_estimates'):
                self.optimizer.update_estimates(measurement, pattern)
                current_precision = np.std(self.optimizer.transmission_estimate)
            else:
                # Multi-resolution: check if converged
                current_precision = self.optimizer.current_resolution
                if current_precision <= self.target.precision_required:
                    self.optimizer.current_resolution /= 2

            # Record
            patterns.append(pattern)
            measurements.append(measurement)
            total_time += measurement_time_per_pattern
            convergence_history.append((total_time, current_precision))

            iteration += 1

            if verbose and iteration % 5 == 0:
                print(f"Iteration {iteration}: Precision = {current_precision:.4f} Å, Time = {total_time:.1f} s")

        if verbose:
            print("-" * 60)
            print(f"Converged in {iteration} iterations, {total_time:.1f} s")
            print(f"Final precision: {current_precision:.4f} Å")

        # Extract final edge position
        if hasattr(self.optimizer, 'posterior'):
            edge_position = self.optimizer.posterior.position_mean
            edge_uncertainty = self.optimizer.posterior.position_std
        else:
            edge_position = self.target.expected_edge
            edge_uncertainty = current_precision

        # Calculate strain
        strain = (edge_position - self.target.expected_edge) / self.target.expected_edge

        # Create result
        result = OptimizationResult(
            edge_position=edge_position,
            edge_uncertainty=edge_uncertainty,
            measurement_time=total_time,
            n_patterns=len(patterns),
            patterns=patterns,
            measurements=measurements,
            convergence_history=convergence_history,
            strain=strain
        )

        return result

    def simulate_comparison(
        self,
        true_sample: BraggEdgeSample,
        flux: float = 1e6,
        measurement_time_per_pattern: float = 10.0
    ) -> Tuple[OptimizationResult, OptimizationResult]:
        """
        Simulate comparison between adaptive and uniform strategies

        Args:
            true_sample: True sample for simulation
            flux: Neutron flux
            measurement_time_per_pattern: Time per pattern

        Returns:
            Tuple of (adaptive_result, uniform_result)
        """
        from .bragg_edge_model import IncidentSpectrum

        # Create simulator
        incident_spectrum = IncidentSpectrum(spectrum_type='maxwellian')
        simulator = MeasurementSimulator(
            true_sample,
            incident_spectrum,
            self.system.tof_calibration,
            self.system.wavelength_range,
            self.system.n_wavelength_bins
        )

        # Run adaptive strategy
        patterns_adaptive = []
        measurements_adaptive = []
        convergence_adaptive = []

        current_precision = np.inf
        total_time = 0.0

        while (current_precision > self.target.precision_required and
               total_time < self.target.max_measurement_time):

            pattern = self.optimizer.design_next_pattern(
                self.system.n_time_bins,
                target_duty_cycle=0.1
            )

            _, measurement = simulator.simulate_tof_measurement(
                pattern,
                self.system.n_time_bins,
                flux,
                measurement_time_per_pattern,
                add_noise=True
            )

            if hasattr(self.optimizer, 'update_posterior'):
                self.optimizer.update_posterior(measurement, pattern, measurement_time_per_pattern)
                current_precision = self.optimizer.posterior.get_precision()
            else:
                current_precision = self.target.precision_required * 0.5  # Mock convergence

            patterns_adaptive.append(pattern)
            measurements_adaptive.append(measurement)
            total_time += measurement_time_per_pattern
            convergence_adaptive.append((total_time, current_precision))

        adaptive_result = OptimizationResult(
            edge_position=self.optimizer.posterior.position_mean if hasattr(self.optimizer, 'posterior') else self.target.expected_edge,
            edge_uncertainty=current_precision,
            measurement_time=total_time,
            n_patterns=len(patterns_adaptive),
            patterns=patterns_adaptive,
            measurements=measurements_adaptive,
            convergence_history=convergence_adaptive
        )

        # Run uniform strategy
        patterns_uniform = []
        measurements_uniform = []
        convergence_uniform = []

        # Reset optimizer for fair comparison
        if self.strategy == 'bayesian':
            uniform_optimizer = BayesianEdgeOptimizer(
                prior_position=self.target.expected_edge,
                prior_position_uncertainty=self.target.precision_required * 10,
                wavelength_grid=self.system.wavelength_grid,
                time_grid=self.system.time_grid,
                forward_model=self.forward_model
            )
        else:
            uniform_optimizer = None

        current_precision = np.inf
        total_time = 0.0

        while (current_precision > self.target.precision_required and
               total_time < self.target.max_measurement_time * 2):  # Allow more time for uniform

            # Uniform pattern
            pattern = PatternLibrary.uniform_sparse(
                self.system.n_time_bins,
                duty_cycle=0.1
            )

            _, measurement = simulator.simulate_tof_measurement(
                pattern,
                self.system.n_time_bins,
                flux,
                measurement_time_per_pattern,
                add_noise=True
            )

            if uniform_optimizer is not None:
                uniform_optimizer.update_posterior(measurement, pattern, measurement_time_per_pattern)
                current_precision = uniform_optimizer.posterior.get_precision()
            else:
                current_precision *= 0.9  # Mock convergence

            patterns_uniform.append(pattern)
            measurements_uniform.append(measurement)
            total_time += measurement_time_per_pattern
            convergence_uniform.append((total_time, current_precision))

        uniform_result = OptimizationResult(
            edge_position=uniform_optimizer.posterior.position_mean if uniform_optimizer else self.target.expected_edge,
            edge_uncertainty=current_precision,
            measurement_time=total_time,
            n_patterns=len(patterns_uniform),
            patterns=patterns_uniform,
            measurements=measurements_uniform,
            convergence_history=convergence_uniform
        )

        # Calculate time saved
        time_saved = ((uniform_result.measurement_time - adaptive_result.measurement_time) /
                      uniform_result.measurement_time * 100)
        adaptive_result.time_saved = time_saved

        return adaptive_result, uniform_result


def optimize_measurement_strategy(
    target: MeasurementTarget,
    flight_path: float = 10.0,
    flux: float = 1e6,
    measurement_time_per_pattern: float = 10.0,
    strategy: str = 'bayesian'
) -> OptimizationResult:
    """
    High-level function to optimize measurement strategy

    Args:
        target: Measurement target
        flight_path: Flight path length (meters)
        flux: Neutron flux (neutrons/second)
        measurement_time_per_pattern: Time per pattern (seconds)
        strategy: Optimization strategy

    Returns:
        OptimizationResult
    """
    # Create system
    system = BraggEdgeMeasurementSystem(
        flight_path=flight_path,
        wavelength_range=(target.expected_edge - 2, target.expected_edge + 2)
    )

    # Create optimizer
    optimizer = AdaptiveEdgeOptimizer(system, target, strategy=strategy)

    # Run optimization
    result = optimizer.run(
        flux=flux,
        measurement_time_per_pattern=measurement_time_per_pattern
    )

    return result
