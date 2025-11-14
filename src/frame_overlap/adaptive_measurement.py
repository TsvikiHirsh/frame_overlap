"""
Adaptive Measurement Strategies for Bragg Edge Optimization

This module implements various adaptive strategies for optimizing chopper patterns
based on Bayesian optimization, gradient focusing, and multi-resolution search.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass, field
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.signal import find_peaks

from .bragg_edge_model import (
    BraggEdge,
    BraggEdgeSample,
    estimate_edge_position,
    calculate_edge_gradient
)
from .chopper_patterns import PatternLibrary, ForwardModel


@dataclass
class EdgePosterior:
    """
    Posterior distribution for edge parameters

    Attributes:
        position_mean: Mean edge position (Angstrom)
        position_std: Standard deviation of position
        height_mean: Mean edge height
        height_std: Standard deviation of height
        width_mean: Mean edge width
        width_std: Standard deviation of width
        samples: Optional MCMC samples
    """
    position_mean: float
    position_std: float
    height_mean: float = 0.5
    height_std: float = 0.2
    width_mean: float = 0.1
    width_std: float = 0.05
    samples: Optional[np.ndarray] = None

    def sample(self, n_samples: int = 100, seed: Optional[int] = None) -> np.ndarray:
        """
        Draw samples from posterior

        Args:
            n_samples: Number of samples to draw
            seed: Random seed

        Returns:
            Array of shape (n_samples, 3) with columns [position, height, width]
        """
        if seed is not None:
            np.random.seed(seed)

        if self.samples is not None and len(self.samples) >= n_samples:
            # Use stored MCMC samples if available
            indices = np.random.choice(len(self.samples), n_samples, replace=False)
            return self.samples[indices]

        # Draw from Gaussian approximation
        positions = np.random.normal(self.position_mean, self.position_std, n_samples)
        heights = np.random.normal(self.height_mean, self.height_std, n_samples)
        widths = np.random.normal(self.width_mean, self.width_std, n_samples)

        # Clip to physical bounds
        heights = np.clip(heights, 0.01, 1.0)
        widths = np.clip(widths, 0.01, 1.0)

        return np.column_stack([positions, heights, widths])

    def get_precision(self) -> float:
        """Get current precision (uncertainty) in edge position"""
        return self.position_std


@dataclass
class MeasurementHistory:
    """
    History of measurements and patterns

    Attributes:
        patterns: List of chopper patterns used
        measurements: List of measured signals
        times: List of measurement times
        posteriors: List of posterior distributions after each measurement
    """
    patterns: List[np.ndarray] = field(default_factory=list)
    measurements: List[np.ndarray] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    posteriors: List[EdgePosterior] = field(default_factory=list)

    def add(
        self,
        pattern: np.ndarray,
        measurement: np.ndarray,
        time: float,
        posterior: EdgePosterior
    ):
        """Add measurement to history"""
        self.patterns.append(pattern.copy())
        self.measurements.append(measurement.copy())
        self.times.append(time)
        self.posteriors.append(posterior)

    def get_total_time(self) -> float:
        """Get total measurement time"""
        return sum(self.times)

    def get_final_precision(self) -> float:
        """Get final precision achieved"""
        if len(self.posteriors) > 0:
            return self.posteriors[-1].get_precision()
        return np.inf


class BayesianEdgeOptimizer:
    """
    Bayesian optimizer for adaptive edge measurement
    """

    def __init__(
        self,
        prior_position: float,
        prior_position_uncertainty: float,
        wavelength_grid: np.ndarray,
        time_grid: np.ndarray,
        forward_model: ForwardModel
    ):
        """
        Initialize Bayesian optimizer

        Args:
            prior_position: Prior mean for edge position (Angstrom)
            prior_position_uncertainty: Prior uncertainty (Angstrom)
            wavelength_grid: Wavelength grid for reconstruction
            time_grid: Time grid for measurements
            forward_model: Forward model for predictions
        """
        self.wavelength_grid = wavelength_grid
        self.time_grid = time_grid
        self.forward_model = forward_model

        # Initialize posterior with prior
        self.posterior = EdgePosterior(
            position_mean=prior_position,
            position_std=prior_position_uncertainty
        )

        self.history = MeasurementHistory()

    def calculate_information_gain(
        self,
        candidate_pattern: np.ndarray,
        n_samples: int = 50
    ) -> float:
        """
        Calculate expected information gain for a candidate pattern

        Args:
            candidate_pattern: Proposed chopper pattern
            n_samples: Number of posterior samples for expectation

        Returns:
            Expected information gain (bits)
        """
        # Sample from current posterior
        samples = self.posterior.sample(n_samples)

        info_gains = []

        for sample in samples:
            position, height, width = sample

            # Create hypothetical edge
            edge = BraggEdge(position=position, height=height, width=width)
            sample_obj = BraggEdgeSample([edge])

            # Predict measurement
            transmission = sample_obj.transmission(self.wavelength_grid)
            predicted_signal = self.forward_model.predict_measurement(
                transmission, candidate_pattern
            )

            # Estimate information gain (using gradient as proxy)
            # Higher gradients in measured signal = more information
            gradient = np.abs(np.gradient(predicted_signal))
            info_gain = np.sum(gradient)

            info_gains.append(info_gain)

        # Return expected information gain
        return np.mean(info_gains)

    def calculate_statistical_quality(self, pattern: np.ndarray) -> float:
        """
        Calculate expected statistical quality (neutron count)

        Args:
            pattern: Chopper pattern

        Returns:
            Quality metric
        """
        # More open time = more neutrons = better statistics
        duty_cycle = np.sum(pattern) / len(pattern)
        return np.sqrt(duty_cycle * len(pattern))

    def design_next_pattern(
        self,
        n_time_bins: int,
        target_duty_cycle: float = 0.1,
        strategy: str = 'gradient_focused'
    ) -> np.ndarray:
        """
        Design optimal pattern for next measurement

        Args:
            n_time_bins: Number of time bins
            target_duty_cycle: Target duty cycle
            strategy: Design strategy ('gradient_focused', 'information_gain', 'window')

        Returns:
            Optimized chopper pattern
        """
        if strategy == 'gradient_focused':
            # Focus on region with highest expected gradient
            # Sample from posterior to estimate gradient location
            samples = self.posterior.sample(100)

            # Convert edge positions to time bins
            tof_calibration = self.forward_model.tof_calibration
            edge_times = []

            for sample in samples:
                position = sample[0]
                tof = tof_calibration(position)
                edge_times.append(tof)

            # Create gradient estimate in time domain
            gradient_estimate = np.zeros(n_time_bins)

            for edge_time in edge_times:
                # Find time bin
                time_idx = np.searchsorted(self.time_grid, edge_time)
                if 0 <= time_idx < n_time_bins:
                    # Add Gaussian weight around edge
                    width = int(0.05 * n_time_bins)  # 5% of range
                    for i in range(max(0, time_idx - width), min(n_time_bins, time_idx + width)):
                        distance = abs(i - time_idx)
                        gradient_estimate[i] += np.exp(-distance**2 / (2 * (width / 3)**2))

            # Normalize
            if np.sum(gradient_estimate) > 0:
                gradient_estimate /= np.sum(gradient_estimate)

            # Generate pattern
            pattern = PatternLibrary.gradient_weighted(
                n_time_bins,
                gradient_estimate,
                target_duty_cycle=target_duty_cycle,
                power=2.0
            )

        elif strategy == 'window':
            # Focus on window around posterior mean
            tof_calibration = self.forward_model.tof_calibration
            mean_tof = tof_calibration(self.posterior.position_mean)
            std_tof = tof_calibration(self.posterior.position_std)

            # Convert to bins
            center_bin = np.searchsorted(self.time_grid, mean_tof)
            width_bins = int(3 * std_tof / (self.time_grid[1] - self.time_grid[0]))

            pattern = PatternLibrary.focused_window(
                n_time_bins,
                center_bin,
                width_bins,
                density=target_duty_cycle * 2  # Higher density in window
            )

        elif strategy == 'information_gain':
            # Generate multiple candidates and pick best
            best_pattern = None
            best_score = -np.inf

            for _ in range(10):
                candidate = PatternLibrary.uniform_sparse(
                    n_time_bins, target_duty_cycle
                )

                info_gain = self.calculate_information_gain(candidate)
                stat_quality = self.calculate_statistical_quality(candidate)

                score = info_gain * stat_quality

                if score > best_score:
                    best_score = score
                    best_pattern = candidate

            pattern = best_pattern

        else:
            # Default: uniform sparse
            pattern = PatternLibrary.uniform_sparse(n_time_bins, target_duty_cycle)

        return pattern

    def update_posterior(
        self,
        measurement: np.ndarray,
        pattern: np.ndarray,
        measurement_time: float
    ):
        """
        Update posterior distribution based on new measurement

        Args:
            measurement: Measured signal
            pattern: Chopper pattern used
            measurement_time: Duration of measurement
        """
        # Fit edge position from accumulated data
        # This is a simplified update - in practice would use MCMC or particle filter

        # Convert time bins to wavelength
        wavelength_estimate = np.zeros_like(self.wavelength_grid)

        # Simple reconstruction: backproject measurement
        for i, t in enumerate(self.time_grid):
            if measurement[i] > 0:
                # Find corresponding wavelength
                # (This is simplified - real implementation would use proper inversion)
                wl = self.forward_model.tof_calibration(t)
                wl_idx = np.searchsorted(self.wavelength_grid, wl)
                if 0 <= wl_idx < len(wavelength_estimate):
                    wavelength_estimate[wl_idx] += measurement[i]

        # Normalize
        if np.sum(wavelength_estimate) > 0:
            wavelength_estimate /= np.sum(wavelength_estimate)

        # Estimate edge position from gradient
        try:
            edge_pos, edge_unc = estimate_edge_position(
                self.wavelength_grid,
                1 - wavelength_estimate  # Convert to transmission-like
            )

            # Bayesian update (simplified - combine with prior)
            prior_precision = 1 / self.posterior.position_std**2
            likelihood_precision = 1 / edge_unc**2

            posterior_precision = prior_precision + likelihood_precision
            posterior_variance = 1 / posterior_precision

            posterior_mean = posterior_variance * (
                prior_precision * self.posterior.position_mean +
                likelihood_precision * edge_pos
            )

            self.posterior = EdgePosterior(
                position_mean=posterior_mean,
                position_std=np.sqrt(posterior_variance),
                height_mean=self.posterior.height_mean,
                height_std=self.posterior.height_std,
                width_mean=self.posterior.width_mean,
                width_std=self.posterior.width_std
            )

        except Exception:
            # If fitting fails, keep current posterior
            pass

        # Add to history
        self.history.add(pattern, measurement, measurement_time, self.posterior)


class GradientFocusedMeasurement:
    """
    Gradient-focused adaptive measurement strategy
    """

    def __init__(
        self,
        wavelength_grid: np.ndarray,
        time_grid: np.ndarray,
        forward_model: ForwardModel
    ):
        """
        Initialize gradient-focused measurement

        Args:
            wavelength_grid: Wavelength grid
            time_grid: Time grid
            forward_model: Forward model
        """
        self.wavelength_grid = wavelength_grid
        self.time_grid = time_grid
        self.forward_model = forward_model

        # Initialize estimates
        self.transmission_estimate = np.ones(len(wavelength_grid))
        self.uncertainty = np.ones(len(wavelength_grid))

    def update_estimates(self, measurement: np.ndarray, pattern: np.ndarray):
        """
        Update transmission estimate from measurement

        Args:
            measurement: Measured signal
            pattern: Chopper pattern used
        """
        # Simple backprojection update
        # In practice, use proper reconstruction algorithm

        for i, t in enumerate(self.time_grid):
            if pattern[i] > 0 and measurement[i] > 0:
                # Map to wavelength
                wl = self.forward_model.tof_calibration(t)
                wl_idx = np.searchsorted(self.wavelength_grid, wl)

                if 0 <= wl_idx < len(self.transmission_estimate):
                    # Update estimate (running average)
                    alpha = 0.3  # Learning rate
                    self.transmission_estimate[wl_idx] = (
                        (1 - alpha) * self.transmission_estimate[wl_idx] +
                        alpha * measurement[i]
                    )

                    # Decrease uncertainty
                    self.uncertainty[wl_idx] *= 0.9

    def generate_pattern(
        self,
        n_time_bins: int,
        duty_cycle: float = 0.1
    ) -> np.ndarray:
        """
        Generate pattern focused on high-gradient regions

        Args:
            n_time_bins: Number of time bins
            duty_cycle: Target duty cycle

        Returns:
            Chopper pattern
        """
        # Calculate gradient of transmission estimate
        gradient = calculate_edge_gradient(
            self.wavelength_grid,
            self.transmission_estimate
        )

        # Weight by uncertainty
        weight = gradient * self.uncertainty

        # Convert wavelength weights to time weights
        time_weights = np.zeros(n_time_bins)

        for i, wl in enumerate(self.wavelength_grid):
            tof = self.forward_model.tof_calibration(wl)
            time_idx = np.searchsorted(self.time_grid, tof)

            if 0 <= time_idx < n_time_bins:
                time_weights[time_idx] += weight[i]

        # Normalize
        if np.sum(time_weights) > 0:
            time_weights /= np.sum(time_weights)

        # Generate pattern
        pattern = PatternLibrary.gradient_weighted(
            n_time_bins,
            time_weights,
            target_duty_cycle=duty_cycle,
            power=2.0
        )

        return pattern


class MultiResolutionEdgeSearch:
    """
    Multi-resolution hierarchical search for edge location
    """

    def __init__(
        self,
        wavelength_range: Tuple[float, float],
        target_precision: float = 0.01,
        forward_model: Optional[ForwardModel] = None
    ):
        """
        Initialize multi-resolution search

        Args:
            wavelength_range: Range to search (Angstrom)
            target_precision: Target precision (Angstrom)
            forward_model: Forward model for predictions
        """
        self.wavelength_range = wavelength_range
        self.target_precision = target_precision
        self.forward_model = forward_model

        self.current_resolution = (wavelength_range[1] - wavelength_range[0]) / 10
        self.edge_candidates = []

    def design_pattern_for_resolution(
        self,
        resolution: float,
        n_time_bins: int,
        search_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Design pattern appropriate for current resolution

        Args:
            resolution: Current resolution (Angstrom)
            n_time_bins: Number of time bins
            search_range: Wavelength range to focus on

        Returns:
            Chopper pattern
        """
        if search_range is None:
            search_range = self.wavelength_range

        # For coarse resolution: uniform sampling
        # For fine resolution: focused sampling

        if resolution > 0.5:
            # Coarse: uniform
            pattern = PatternLibrary.uniform_sparse(n_time_bins, duty_cycle=0.2)
        else:
            # Fine: focused on search range
            if self.forward_model is not None:
                # Convert wavelength range to time range
                tof_cal = self.forward_model.tof_calibration
                time_range = [tof_cal(search_range[0]), tof_cal(search_range[1])]

                # Find time bins
                center_time = (time_range[0] + time_range[1]) / 2
                width_time = time_range[1] - time_range[0]

                if hasattr(self.forward_model, 'time_grid'):
                    time_grid = self.forward_model.time_grid
                    center_bin = np.searchsorted(time_grid, center_time)
                    width_bins = int(width_time / (time_grid[1] - time_grid[0]))
                else:
                    center_bin = n_time_bins // 2
                    width_bins = n_time_bins // 4

                pattern = PatternLibrary.focused_window(
                    n_time_bins,
                    center_bin,
                    width_bins,
                    density=0.5
                )
            else:
                pattern = PatternLibrary.uniform_sparse(n_time_bins, duty_cycle=0.15)

        return pattern

    def detect_edges(
        self,
        wavelength: np.ndarray,
        transmission: np.ndarray,
        resolution: float
    ) -> List[float]:
        """
        Detect edge candidates at current resolution

        Args:
            wavelength: Wavelength array
            transmission: Transmission array
            resolution: Current resolution

        Returns:
            List of edge positions
        """
        # Find peaks in gradient
        gradient = np.abs(np.gradient(transmission, wavelength))

        # Find peaks
        peaks, properties = find_peaks(
            gradient,
            height=np.max(gradient) * 0.3,
            distance=int(resolution / (wavelength[1] - wavelength[0]))
        )

        edge_positions = wavelength[peaks].tolist()
        return edge_positions

    def refine_range(
        self,
        edge_candidates: List[float]
    ) -> Tuple[float, float]:
        """
        Compute refined range around detected edges

        Args:
            edge_candidates: List of candidate edge positions

        Returns:
            Refined wavelength range
        """
        if len(edge_candidates) == 0:
            return self.wavelength_range

        # Focus on first candidate (can be extended to multiple)
        edge = edge_candidates[0]
        margin = 3 * self.current_resolution

        return (max(self.wavelength_range[0], edge - margin),
                min(self.wavelength_range[1], edge + margin))


class RealTimeAdaptiveSystem:
    """
    Real-time adaptive measurement system with event-by-event processing
    """

    def __init__(
        self,
        initial_pattern: np.ndarray,
        update_interval: int = 1000,
        optimizer: Optional[BayesianEdgeOptimizer] = None
    ):
        """
        Initialize real-time system

        Args:
            initial_pattern: Initial chopper pattern
            update_interval: Number of events between pattern updates
            optimizer: Bayesian optimizer (optional)
        """
        self.current_pattern = initial_pattern
        self.update_interval = update_interval
        self.optimizer = optimizer

        self.event_buffer = []
        self.total_events = 0

    def process_event(
        self,
        event_time: float,
        event_wavelength: float
    ):
        """
        Process single neutron detection event

        Args:
            event_time: Detection time
            event_wavelength: Wavelength (if known) or converted from TOF
        """
        self.event_buffer.append((event_time, event_wavelength))
        self.total_events += 1

        # Check if update needed
        if len(self.event_buffer) >= self.update_interval:
            self.adapt_pattern()
            self.event_buffer = []

    def adapt_pattern(self):
        """
        Adapt chopper pattern based on accumulated events
        """
        if self.optimizer is None:
            return

        # Build histogram from events
        times = np.array([e[0] for e in self.event_buffer])
        wavelengths = np.array([e[1] for e in self.event_buffer])

        # Create measurement histogram
        n_bins = len(self.current_pattern)
        measurement, _ = np.histogram(times, bins=n_bins)

        # Update optimizer
        self.optimizer.update_posterior(
            measurement,
            self.current_pattern,
            measurement_time=1.0
        )

        # Design new pattern
        self.current_pattern = self.optimizer.design_next_pattern(
            n_bins,
            strategy='gradient_focused'
        )

    def get_current_pattern(self) -> np.ndarray:
        """Get current chopper pattern"""
        return self.current_pattern.copy()
