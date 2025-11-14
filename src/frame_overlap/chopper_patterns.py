"""
Chopper Pattern Generation and Forward Model

This module provides various strategies for generating chopper patterns
and implements the forward model for converting patterns to measurements.
"""

import numpy as np
from scipy.linalg import hadamard
from scipy.sparse import csr_matrix
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass


@dataclass
class ChopperConstraints:
    """
    Physical constraints for chopper operation

    Attributes:
        max_frequency: Maximum chopper rotation frequency (Hz)
        min_pulse_width: Minimum pulse width (seconds)
        max_duty_cycle: Maximum duty cycle (fraction of time open)
        dead_time: Dead time between pulses (seconds)
    """
    max_frequency: float = 1000.0
    min_pulse_width: float = 1e-6
    max_duty_cycle: float = 0.5
    dead_time: float = 0.0


class PatternLibrary:
    """
    Library of chopper pattern generation strategies
    """

    @staticmethod
    def uniform_sparse(
        n_bins: int,
        duty_cycle: float = 0.1,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate uniform random sparse pattern

        Args:
            n_bins: Number of time bins
            duty_cycle: Fraction of bins that are open (0 to 1)
            seed: Random seed for reproducibility

        Returns:
            Binary pattern array (1 = open, 0 = closed)
        """
        if seed is not None:
            np.random.seed(seed)

        pattern = np.random.random(n_bins) < duty_cycle
        return pattern.astype(int)

    @staticmethod
    def focused_window(
        n_bins: int,
        center: int,
        width: int,
        density: float = 0.5,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate pattern focused on specific time window

        Args:
            n_bins: Number of time bins
            center: Center bin of focused region
            width: Width of focused region (bins)
            density: Sampling density within window (0 to 1)
            seed: Random seed

        Returns:
            Binary pattern array
        """
        if seed is not None:
            np.random.seed(seed)

        pattern = np.zeros(n_bins, dtype=int)
        start = max(0, center - width // 2)
        end = min(n_bins, center + width // 2)

        pattern[start:end] = (np.random.random(end - start) < density).astype(int)

        return pattern

    @staticmethod
    def multi_window(
        n_bins: int,
        centers: List[int],
        widths: List[int],
        densities: Optional[List[float]] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate pattern with multiple focused windows

        Args:
            n_bins: Number of time bins
            centers: List of window centers
            widths: List of window widths
            densities: List of sampling densities (optional)
            seed: Random seed

        Returns:
            Binary pattern array
        """
        if seed is not None:
            np.random.seed(seed)

        if densities is None:
            densities = [0.5] * len(centers)

        pattern = np.zeros(n_bins, dtype=int)

        for center, width, density in zip(centers, widths, densities):
            start = max(0, center - width // 2)
            end = min(n_bins, center + width // 2)
            window_pattern = (np.random.random(end - start) < density).astype(int)
            pattern[start:end] = np.maximum(pattern[start:end], window_pattern)

        return pattern

    @staticmethod
    def adaptive_density(
        n_bins: int,
        density_function: np.ndarray,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate pattern with spatially varying density

        Args:
            n_bins: Number of time bins
            density_function: Array of sampling probabilities for each bin
            seed: Random seed

        Returns:
            Binary pattern array
        """
        if seed is not None:
            np.random.seed(seed)

        # Normalize density function
        density = np.array(density_function)
        if len(density) != n_bins:
            # Interpolate to correct size
            density = np.interp(
                np.arange(n_bins),
                np.linspace(0, n_bins - 1, len(density)),
                density
            )

        # Clip to valid probability range
        density = np.clip(density, 0, 1)

        pattern = (np.random.random(n_bins) < density).astype(int)
        return pattern

    @staticmethod
    def hadamard_based(
        n_bins: int,
        order: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Hadamard matrix-based pattern for compressed sensing

        Args:
            n_bins: Number of time bins
            order: Hadamard matrix order (must be power of 2)

        Returns:
            Binary pattern array
        """
        if order is None:
            # Find nearest power of 2
            order = 2 ** int(np.ceil(np.log2(n_bins / 10)))

        # Generate Hadamard matrix
        H = hadamard(order)

        # Use first row, tile to fill n_bins
        pattern = np.tile(H[0], (n_bins // order) + 1)[:n_bins]

        # Convert to binary (Hadamard has -1, 1)
        return (pattern > 0).astype(int)

    @staticmethod
    def periodic_pulse(
        n_bins: int,
        period: int,
        pulse_width: int = 1
    ) -> np.ndarray:
        """
        Generate periodic pulse train

        Args:
            n_bins: Number of time bins
            period: Period between pulses (bins)
            pulse_width: Width of each pulse (bins)

        Returns:
            Binary pattern array
        """
        pattern = np.zeros(n_bins, dtype=int)

        for i in range(0, n_bins, period):
            end = min(i + pulse_width, n_bins)
            pattern[i:end] = 1

        return pattern

    @staticmethod
    def chirped_pulse(
        n_bins: int,
        start_period: int,
        end_period: int,
        pulse_width: int = 1
    ) -> np.ndarray:
        """
        Generate chirped pulse train with varying frequency

        Args:
            n_bins: Number of time bins
            start_period: Initial period (bins)
            end_period: Final period (bins)
            pulse_width: Width of each pulse (bins)

        Returns:
            Binary pattern array
        """
        pattern = np.zeros(n_bins, dtype=int)

        # Generate periods with linear chirp
        n_pulses = int(n_bins / ((start_period + end_period) / 2))
        periods = np.linspace(start_period, end_period, n_pulses).astype(int)

        pos = 0
        for period in periods:
            if pos >= n_bins:
                break
            end = min(pos + pulse_width, n_bins)
            pattern[pos:end] = 1
            pos += period

        return pattern

    @staticmethod
    def gradient_weighted(
        n_bins: int,
        gradient_estimate: np.ndarray,
        target_duty_cycle: float = 0.1,
        power: float = 2.0,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate pattern weighted by gradient (edge importance)

        Args:
            n_bins: Number of time bins
            gradient_estimate: Estimated gradient at each bin
            target_duty_cycle: Target fraction of open bins
            power: Exponent for weighting (higher = more focused)
            seed: Random seed

        Returns:
            Binary pattern array
        """
        if seed is not None:
            np.random.seed(seed)

        # Ensure gradient is the right size
        if len(gradient_estimate) != n_bins:
            gradient = np.interp(
                np.arange(n_bins),
                np.linspace(0, n_bins - 1, len(gradient_estimate)),
                gradient_estimate
            )
        else:
            gradient = gradient_estimate.copy()

        # Apply power law weighting
        weight = np.abs(gradient) ** power

        # Normalize to probabilities
        weight = weight / np.sum(weight) if np.sum(weight) > 0 else np.ones_like(weight)

        # Scale to achieve target duty cycle
        weight = weight * target_duty_cycle * n_bins

        # Clip to valid probabilities
        weight = np.clip(weight, 0, 1)

        # Generate pattern
        pattern = (np.random.random(n_bins) < weight).astype(int)

        return pattern


class ForwardModel:
    """
    Forward model: transmission → chopper pattern → measured signal
    """

    def __init__(
        self,
        wavelength_grid: np.ndarray,
        time_grid: np.ndarray,
        tof_calibration: Callable,
        incident_spectrum: np.ndarray
    ):
        """
        Initialize forward model

        Args:
            wavelength_grid: Wavelength values (Angstrom)
            time_grid: Time values for chopper/detection (seconds)
            tof_calibration: Function converting wavelength to TOF
            incident_spectrum: Incident spectrum intensity at wavelength_grid
        """
        self.wavelength_grid = wavelength_grid
        self.time_grid = time_grid
        self.tof_calibration = tof_calibration
        self.incident_spectrum = incident_spectrum

        # Precompute TOF for each wavelength
        self.tof_values = tof_calibration(wavelength_grid)

    def predict_measurement(
        self,
        transmission: np.ndarray,
        chopper_pattern: np.ndarray,
        flux: float = 1e6,
        measurement_time: float = 1.0
    ) -> np.ndarray:
        """
        Predict measured signal from transmission and chopper pattern

        Args:
            transmission: Transmission curve (same length as wavelength_grid)
            chopper_pattern: Binary chopper pattern (same length as time_grid)
            flux: Neutron flux (neutrons/second)
            measurement_time: Total measurement time (seconds)

        Returns:
            Detected signal (counts per time bin)
        """
        n_time_bins = len(self.time_grid)
        detected = np.zeros(n_time_bins)

        dt = self.time_grid[1] - self.time_grid[0] if len(self.time_grid) > 1 else measurement_time
        d_wavelength = (
            self.wavelength_grid[1] - self.wavelength_grid[0]
            if len(self.wavelength_grid) > 1
            else 1.0
        )

        # For each open chopper bin
        open_indices = np.where(chopper_pattern > 0)[0]

        for chop_idx in open_indices:
            t_chop = self.time_grid[chop_idx]

            # For each wavelength
            for wl_idx, tof in enumerate(self.tof_values):
                t_detect = t_chop + tof

                # Find detection bin
                det_idx = np.searchsorted(self.time_grid, t_detect)

                if 0 <= det_idx < n_time_bins:
                    # Add contribution
                    intensity = (
                        self.incident_spectrum[wl_idx] *
                        transmission[wl_idx] *
                        flux * dt * d_wavelength
                    )
                    detected[det_idx] += intensity

        return detected

    def build_measurement_matrix(
        self,
        chopper_patterns: List[np.ndarray],
        flux: float = 1e6,
        measurement_time: float = 1.0,
        sparse: bool = True
    ) -> np.ndarray:
        """
        Build measurement matrix A such that y = A·x

        where:
            y = measured signals (concatenated)
            x = transmission curve
            A = measurement matrix

        Args:
            chopper_patterns: List of chopper patterns
            flux: Neutron flux
            measurement_time: Measurement time per pattern
            sparse: Whether to return sparse matrix

        Returns:
            Measurement matrix (n_measurements × n_wavelengths)
        """
        n_patterns = len(chopper_patterns)
        n_time_bins = len(self.time_grid)
        n_wavelengths = len(self.wavelength_grid)
        n_measurements = n_patterns * n_time_bins

        dt = self.time_grid[1] - self.time_grid[0] if len(self.time_grid) > 1 else measurement_time
        d_wavelength = (
            self.wavelength_grid[1] - self.wavelength_grid[0]
            if len(self.wavelength_grid) > 1
            else 1.0
        )

        if sparse:
            # Build sparse matrix
            row_indices = []
            col_indices = []
            values = []

            for pattern_idx, pattern in enumerate(chopper_patterns):
                open_indices = np.where(pattern > 0)[0]

                for chop_idx in open_indices:
                    t_chop = self.time_grid[chop_idx]

                    for wl_idx, tof in enumerate(self.tof_values):
                        t_detect = t_chop + tof
                        det_idx = np.searchsorted(self.time_grid, t_detect)

                        if 0 <= det_idx < n_time_bins:
                            row = pattern_idx * n_time_bins + det_idx
                            col = wl_idx
                            value = self.incident_spectrum[wl_idx] * flux * dt * d_wavelength

                            row_indices.append(row)
                            col_indices.append(col)
                            values.append(value)

            A = csr_matrix(
                (values, (row_indices, col_indices)),
                shape=(n_measurements, n_wavelengths)
            )

        else:
            # Build dense matrix
            A = np.zeros((n_measurements, n_wavelengths))

            for pattern_idx, pattern in enumerate(chopper_patterns):
                open_indices = np.where(pattern > 0)[0]

                for chop_idx in open_indices:
                    t_chop = self.time_grid[chop_idx]

                    for wl_idx, tof in enumerate(self.tof_values):
                        t_detect = t_chop + tof
                        det_idx = np.searchsorted(self.time_grid, t_detect)

                        if 0 <= det_idx < n_time_bins:
                            row = pattern_idx * n_time_bins + det_idx
                            col = wl_idx
                            A[row, col] += self.incident_spectrum[wl_idx] * flux * dt * d_wavelength

        return A


def calculate_pattern_efficiency(
    pattern: np.ndarray,
    importance_weights: np.ndarray
) -> float:
    """
    Calculate efficiency of pattern for measuring specific regions

    Args:
        pattern: Binary chopper pattern
        importance_weights: Importance weights for each time bin

    Returns:
        Efficiency score (0 to 1)
    """
    if len(pattern) != len(importance_weights):
        raise ValueError("Pattern and weights must have same length")

    total_pulses = np.sum(pattern)
    if total_pulses == 0:
        return 0.0

    weighted_pulses = np.sum(pattern * importance_weights)
    total_weight = np.sum(importance_weights)

    if total_weight == 0:
        return 0.0

    return weighted_pulses / total_weight


def validate_pattern(
    pattern: np.ndarray,
    constraints: ChopperConstraints,
    time_resolution: float
) -> Tuple[bool, str]:
    """
    Validate chopper pattern against physical constraints

    Args:
        pattern: Binary chopper pattern
        constraints: Physical constraints
        time_resolution: Time per bin (seconds)

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check duty cycle
    duty_cycle = np.sum(pattern) / len(pattern)
    if duty_cycle > constraints.max_duty_cycle:
        return False, f"Duty cycle {duty_cycle:.2%} exceeds maximum {constraints.max_duty_cycle:.2%}"

    # Check minimum pulse width
    # Find transitions
    diff = np.diff(np.concatenate([[0], pattern, [0]]))
    rising = np.where(diff == 1)[0]
    falling = np.where(diff == -1)[0]

    for start, end in zip(rising, falling):
        pulse_width = (end - start) * time_resolution
        if pulse_width < constraints.min_pulse_width:
            return False, f"Pulse width {pulse_width*1e6:.1f}μs below minimum {constraints.min_pulse_width*1e6:.1f}μs"

    # Check dead time
    if constraints.dead_time > 0 and len(rising) > 1:
        for i in range(len(rising) - 1):
            gap = (rising[i + 1] - falling[i]) * time_resolution
            if gap < constraints.dead_time:
                return False, f"Gap {gap*1e6:.1f}μs below dead time {constraints.dead_time*1e6:.1f}μs"

    return True, "Pattern is valid"
