"""
Performance Metrics and Evaluation for Adaptive Measurements

This module provides tools for evaluating and comparing measurement strategies.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.stats import entropy
from scipy.signal import correlate
from dataclasses import dataclass, field


@dataclass
class PerformanceMetrics:
    """
    Container for performance metrics

    Attributes:
        edge_position_error: Error in edge position (Angstrom)
        edge_position_precision: Standard deviation of estimates (Angstrom)
        measurement_time: Total measurement time (seconds)
        total_counts: Total neutron counts detected
        efficiency: Measurement efficiency (0 to 1)
        information_rate: Information gain per unit time (bits/second)
        snr: Signal-to-noise ratio
        convergence_time: Time to reach target precision (seconds)
    """
    edge_position_error: float = np.inf
    edge_position_precision: float = np.inf
    measurement_time: float = 0.0
    total_counts: float = 0.0
    efficiency: float = 0.0
    information_rate: float = 0.0
    snr: float = 0.0
    convergence_time: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'edge_position_error': self.edge_position_error,
            'edge_position_precision': self.edge_position_precision,
            'measurement_time': self.measurement_time,
            'total_counts': self.total_counts,
            'efficiency': self.efficiency,
            'information_rate': self.information_rate,
            'snr': self.snr,
            'convergence_time': self.convergence_time
        }


@dataclass
class ComparisonResult:
    """
    Result of comparing two measurement strategies

    Attributes:
        strategy1_name: Name of first strategy
        strategy2_name: Name of second strategy
        metrics1: Performance metrics for strategy 1
        metrics2: Performance metrics for strategy 2
        speedup: Speedup factor (time_1 / time_2)
        efficiency_gain: Efficiency improvement (efficiency_2 - efficiency_1)
        precision_gain: Precision improvement (precision_1 / precision_2)
    """
    strategy1_name: str
    strategy2_name: str
    metrics1: PerformanceMetrics
    metrics2: PerformanceMetrics
    speedup: float = field(init=False)
    efficiency_gain: float = field(init=False)
    precision_gain: float = field(init=False)

    def __post_init__(self):
        """Calculate derived metrics"""
        self.speedup = (
            self.metrics1.measurement_time / self.metrics2.measurement_time
            if self.metrics2.measurement_time > 0 else np.inf
        )

        self.efficiency_gain = self.metrics2.efficiency - self.metrics1.efficiency

        self.precision_gain = (
            self.metrics1.edge_position_precision / self.metrics2.edge_position_precision
            if self.metrics2.edge_position_precision > 0 else np.inf
        )

    def summary(self) -> str:
        """Generate summary string"""
        summary = f"Comparison: {self.strategy1_name} vs {self.strategy2_name}\n"
        summary += f"Speedup: {self.speedup:.2f}x\n"
        summary += f"Efficiency gain: {self.efficiency_gain:.2%}\n"
        summary += f"Precision gain: {self.precision_gain:.2f}x\n"
        return summary


class PerformanceEvaluator:
    """
    Evaluator for measurement strategy performance
    """

    @staticmethod
    def edge_position_precision(
        measurements: List[np.ndarray],
        wavelength_grids: List[np.ndarray],
        true_edge: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate precision and accuracy of edge position determination

        Args:
            measurements: List of measurement results
            wavelength_grids: Corresponding wavelength grids
            true_edge: True edge position (if known)

        Returns:
            Tuple of (mean_error, std_dev) if true_edge known, else (mean_position, std_dev)
        """
        from .bragg_edge_model import estimate_edge_position

        positions = []

        for measurement, wavelength in zip(measurements, wavelength_grids):
            try:
                pos, _ = estimate_edge_position(wavelength, measurement)
                positions.append(pos)
            except Exception:
                continue

        positions = np.array(positions)

        if len(positions) == 0:
            return np.inf, np.inf

        if true_edge is not None:
            # Calculate error and precision
            errors = positions - true_edge
            mean_error = np.mean(errors)
            precision = np.std(positions)
            return mean_error, precision
        else:
            # Return mean and precision
            mean_pos = np.mean(positions)
            precision = np.std(positions)
            return mean_pos, precision

    @staticmethod
    def measurement_efficiency(
        pattern: np.ndarray,
        edge_region_indices: np.ndarray
    ) -> float:
        """
        Calculate fraction of neutrons used in informative region

        Args:
            pattern: Chopper pattern
            edge_region_indices: Indices of time bins containing edge information

        Returns:
            Efficiency (0 to 1)
        """
        if np.sum(pattern) == 0:
            return 0.0

        # Count pulses in edge region
        edge_mask = np.zeros_like(pattern)
        edge_mask[edge_region_indices] = 1

        useful_pulses = np.sum(pattern * edge_mask)
        total_pulses = np.sum(pattern)

        return useful_pulses / total_pulses

    @staticmethod
    def calculate_snr(
        signal: np.ndarray,
        noise_region_indices: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate signal-to-noise ratio

        Args:
            signal: Measured signal
            noise_region_indices: Indices of background region for noise estimation

        Returns:
            SNR value
        """
        if noise_region_indices is None:
            # Estimate noise from signal variance
            noise_std = np.std(signal)
        else:
            # Estimate noise from specified region
            noise_std = np.std(signal[noise_region_indices])

        if noise_std == 0:
            return np.inf

        signal_mean = np.mean(signal)
        return signal_mean / noise_std

    @staticmethod
    def information_rate(
        measurements: List[np.ndarray],
        time_stamps: List[float]
    ) -> np.ndarray:
        """
        Calculate information gain per unit time

        Args:
            measurements: Sequential measurements
            time_stamps: Time at each measurement

        Returns:
            Array of information rates (bits/second)
        """
        info_rates = []

        for i in range(1, len(measurements)):
            # Calculate mutual information between consecutive measurements
            info = calculate_mutual_information(
                measurements[i - 1],
                measurements[i]
            )

            dt = time_stamps[i] - time_stamps[i - 1]

            if dt > 0:
                info_rates.append(info / dt)
            else:
                info_rates.append(0.0)

        return np.array(info_rates)

    @staticmethod
    def convergence_analysis(
        precisions: List[float],
        times: List[float],
        target_precision: float
    ) -> Optional[float]:
        """
        Determine convergence time to target precision

        Args:
            precisions: List of precision values over time
            times: Corresponding time stamps
            target_precision: Target precision threshold

        Returns:
            Time to reach target, or None if not reached
        """
        precisions = np.array(precisions)
        times = np.array(times)

        # Find first time precision drops below target
        converged = precisions <= target_precision

        if np.any(converged):
            idx = np.argmax(converged)
            return times[idx]
        else:
            return None

    @staticmethod
    def evaluate_strategy(
        measurements: List[np.ndarray],
        patterns: List[np.ndarray],
        wavelength_grids: List[np.ndarray],
        time_stamps: List[float],
        true_edge: Optional[float] = None,
        edge_region: Optional[Tuple[float, float]] = None
    ) -> PerformanceMetrics:
        """
        Comprehensive evaluation of measurement strategy

        Args:
            measurements: List of measurements
            patterns: Chopper patterns used
            wavelength_grids: Wavelength grids
            time_stamps: Time stamps
            true_edge: True edge position (if known)
            edge_region: Wavelength range of edge (for efficiency calculation)

        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()

        # Edge position precision
        if true_edge is not None:
            error, precision = PerformanceEvaluator.edge_position_precision(
                measurements, wavelength_grids, true_edge
            )
            metrics.edge_position_error = abs(error)
            metrics.edge_position_precision = precision
        else:
            _, precision = PerformanceEvaluator.edge_position_precision(
                measurements, wavelength_grids
            )
            metrics.edge_position_precision = precision

        # Measurement time
        metrics.measurement_time = time_stamps[-1] if len(time_stamps) > 0 else 0.0

        # Total counts
        metrics.total_counts = sum(np.sum(m) for m in measurements)

        # Efficiency
        if edge_region is not None and len(patterns) > 0:
            # Convert edge region to time bins (approximate)
            # This is simplified - would need proper TOF calibration
            efficiencies = []
            for pattern in patterns:
                # Assume edge region is in middle third of pattern
                n_bins = len(pattern)
                edge_indices = np.arange(n_bins // 3, 2 * n_bins // 3)
                eff = PerformanceEvaluator.measurement_efficiency(pattern, edge_indices)
                efficiencies.append(eff)
            metrics.efficiency = np.mean(efficiencies)

        # Information rate
        if len(measurements) > 1:
            info_rates = PerformanceEvaluator.information_rate(measurements, time_stamps)
            metrics.information_rate = np.mean(info_rates) if len(info_rates) > 0 else 0.0

        # SNR
        if len(measurements) > 0:
            snrs = [PerformanceEvaluator.calculate_snr(m) for m in measurements]
            metrics.snr = np.mean(snrs)

        return metrics


def calculate_mutual_information(
    signal1: np.ndarray,
    signal2: np.ndarray,
    bins: int = 50
) -> float:
    """
    Calculate mutual information between two signals

    Args:
        signal1: First signal
        signal2: Second signal
        bins: Number of bins for histogram

    Returns:
        Mutual information (bits)
    """
    # Create 2D histogram
    hist_2d, _, _ = np.histogram2d(signal1, signal2, bins=bins)

    # Normalize to probability
    hist_2d = hist_2d / np.sum(hist_2d)

    # Marginal distributions
    p_x = np.sum(hist_2d, axis=1)
    p_y = np.sum(hist_2d, axis=0)

    # Calculate mutual information
    mi = 0.0

    for i in range(bins):
        for j in range(bins):
            if hist_2d[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (p_x[i] * p_y[j]))

    return mi


def calculate_fisher_information(
    wavelength: np.ndarray,
    transmission: np.ndarray,
    edge_position: float,
    noise_level: float
) -> float:
    """
    Calculate Fisher information for edge position estimation

    Args:
        wavelength: Wavelength array
        transmission: Transmission curve
        edge_position: Current edge position estimate
        noise_level: Noise level (std dev)

    Returns:
        Fisher information
    """
    # Calculate derivative of transmission w.r.t. edge position
    # Approximate by finite difference

    eps = 0.01  # Small shift
    idx = np.searchsorted(wavelength, edge_position)

    if idx > 0 and idx < len(transmission) - 1:
        derivative = (transmission[idx + 1] - transmission[idx - 1]) / (2 * eps)
    else:
        derivative = 0.0

    # Fisher information: I(θ) = (1/σ²) * (dμ/dθ)²
    if noise_level > 0:
        fisher_info = (derivative ** 2) / (noise_level ** 2)
    else:
        fisher_info = 0.0

    return fisher_info


def cramer_rao_bound(
    wavelength: np.ndarray,
    transmission: np.ndarray,
    edge_position: float,
    noise_level: float,
    n_measurements: int = 1
) -> float:
    """
    Calculate Cramér-Rao lower bound for edge position estimation

    Args:
        wavelength: Wavelength array
        transmission: Transmission curve
        edge_position: Edge position
        noise_level: Noise level
        n_measurements: Number of measurements

    Returns:
        Cramér-Rao bound (minimum achievable variance)
    """
    fisher_info = calculate_fisher_information(
        wavelength, transmission, edge_position, noise_level
    )

    if fisher_info > 0:
        return 1.0 / (n_measurements * fisher_info)
    else:
        return np.inf


class SimulationComparison:
    """
    Compare different measurement strategies via simulation
    """

    def __init__(
        self,
        true_sample,
        incident_spectrum,
        tof_calibration,
        wavelength_range: Tuple[float, float] = (1.0, 10.0),
        n_wavelength_bins: int = 1000
    ):
        """
        Initialize simulation comparison

        Args:
            true_sample: True Bragg edge sample
            incident_spectrum: Incident spectrum
            tof_calibration: TOF calibration
            wavelength_range: Wavelength range
            n_wavelength_bins: Number of wavelength bins
        """
        from .bragg_edge_model import MeasurementSimulator

        self.simulator = MeasurementSimulator(
            true_sample,
            incident_spectrum,
            tof_calibration,
            wavelength_range,
            n_wavelength_bins
        )

        self.true_edge = true_sample.edges[0].position if len(true_sample.edges) > 0 else None

    def compare_strategies(
        self,
        strategy1_patterns: List[np.ndarray],
        strategy2_patterns: List[np.ndarray],
        strategy1_name: str = "Strategy 1",
        strategy2_name: str = "Strategy 2",
        n_time_bins: int = 1000,
        flux: float = 1e6,
        measurement_time: float = 1.0
    ) -> ComparisonResult:
        """
        Compare two measurement strategies

        Args:
            strategy1_patterns: Patterns for strategy 1
            strategy2_patterns: Patterns for strategy 2
            strategy1_name: Name of strategy 1
            strategy2_name: Name of strategy 2
            n_time_bins: Number of time bins
            flux: Neutron flux
            measurement_time: Measurement time per pattern

        Returns:
            ComparisonResult object
        """
        # Simulate strategy 1
        measurements1 = []
        times1 = []
        total_time1 = 0.0

        for pattern in strategy1_patterns:
            _, counts = self.simulator.simulate_tof_measurement(
                pattern, n_time_bins, flux, measurement_time, add_noise=True
            )
            measurements1.append(counts)
            total_time1 += measurement_time
            times1.append(total_time1)

        # Simulate strategy 2
        measurements2 = []
        times2 = []
        total_time2 = 0.0

        for pattern in strategy2_patterns:
            _, counts = self.simulator.simulate_tof_measurement(
                pattern, n_time_bins, flux, measurement_time, add_noise=True
            )
            measurements2.append(counts)
            total_time2 += measurement_time
            times2.append(total_time2)

        # Create wavelength grids for evaluation
        wavelength_grids = [self.simulator.wavelength] * max(len(measurements1), len(measurements2))

        # Evaluate both strategies
        metrics1 = PerformanceEvaluator.evaluate_strategy(
            measurements1,
            strategy1_patterns,
            wavelength_grids[:len(measurements1)],
            times1,
            true_edge=self.true_edge
        )

        metrics2 = PerformanceEvaluator.evaluate_strategy(
            measurements2,
            strategy2_patterns,
            wavelength_grids[:len(measurements2)],
            times2,
            true_edge=self.true_edge
        )

        return ComparisonResult(
            strategy1_name,
            strategy2_name,
            metrics1,
            metrics2
        )
