"""
Bragg Edge Physical Model

This module implements the physical model for Bragg edge transmission measurements
in neutron imaging, including edge functions, transmission calculations, and TOF calibration.

References:
    - Santisteban et al. (2001) - Bragg edge analysis for strain mapping
    - Woracek et al. (2018) - Neutron Bragg edge imaging review
"""

import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List


# Physical constants
PLANCK_CONSTANT = 6.62607015e-34  # J·s
NEUTRON_MASS = 1.67492749804e-27  # kg
ANGSTROM_TO_METER = 1e-10


@dataclass
class TOFCalibration:
    """
    Time-of-Flight calibration parameters

    Attributes:
        flight_path: Distance from chopper to detector (meters)
        time_offset: Time offset correction (seconds)
        wavelength_to_tof: Function converting wavelength to TOF
        tof_to_wavelength: Function converting TOF to wavelength
    """
    flight_path: float
    time_offset: float = 0.0

    def __post_init__(self):
        """Initialize conversion functions"""
        self.wavelength_to_tof = self._create_wavelength_to_tof()
        self.tof_to_wavelength = self._create_tof_to_wavelength()

    def _create_wavelength_to_tof(self) -> Callable:
        """Create wavelength to TOF conversion function

        Formula: t = (m_n * L / h) * λ + t_offset
        Simplified: t(μs) ≈ 252.778 * L(m) * λ(Å) + t_offset
        """
        conversion_factor = (NEUTRON_MASS * self.flight_path) / (PLANCK_CONSTANT / ANGSTROM_TO_METER)

        def converter(wavelength_angstrom: np.ndarray) -> np.ndarray:
            """Convert wavelength (Angstrom) to TOF (seconds)"""
            return conversion_factor * wavelength_angstrom * 1e-6 + self.time_offset

        return converter

    def _create_tof_to_wavelength(self) -> Callable:
        """Create TOF to wavelength conversion function

        Formula: λ = (h / m_n * L) * (t - t_offset)
        Simplified: λ(Å) ≈ 3.956 * t(μs) / L(m)
        """
        conversion_factor = (PLANCK_CONSTANT / ANGSTROM_TO_METER) / (NEUTRON_MASS * self.flight_path)

        def converter(tof_seconds: np.ndarray) -> np.ndarray:
            """Convert TOF (seconds) to wavelength (Angstrom)"""
            return conversion_factor * (tof_seconds - self.time_offset) * 1e6

        return converter


class BraggEdge:
    """
    Representation of a Bragg edge with physical parameters
    """

    def __init__(
        self,
        position: float,
        height: float = 0.5,
        width: float = 0.1,
        edge_type: str = 'erf'
    ):
        """
        Initialize Bragg edge

        Args:
            position: Edge position in wavelength (Angstrom)
            height: Edge height/contrast (0 to 1)
            width: Edge width/broadening (Angstrom)
            edge_type: Type of edge function ('erf', 'tanh', 'step')
        """
        self.position = position
        self.height = height
        self.width = width
        self.edge_type = edge_type

    def edge_function(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Calculate edge function E(λ)

        Args:
            wavelength: Wavelength array (Angstrom)

        Returns:
            Edge function values (0 to 1)
        """
        if self.edge_type == 'erf':
            # Error function edge: E(λ) = 0.5 * (1 + erf((λ - λ_edge) / (√2 * σ)))
            x = (wavelength - self.position) / (np.sqrt(2) * self.width)
            return 0.5 * (1 + erf(x))

        elif self.edge_type == 'tanh':
            # Hyperbolic tangent edge
            x = (wavelength - self.position) / self.width
            return 0.5 * (1 + np.tanh(x))

        elif self.edge_type == 'step':
            # Sharp step function
            return (wavelength >= self.position).astype(float)

        else:
            raise ValueError(f"Unknown edge type: {self.edge_type}")

    def transmission_contribution(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Calculate transmission contribution from this edge

        Args:
            wavelength: Wavelength array (Angstrom)

        Returns:
            Transmission values (0 to 1)
        """
        edge = self.edge_function(wavelength)
        return 1 - self.height * edge


class BraggEdgeSample:
    """
    Sample with multiple Bragg edges
    """

    def __init__(
        self,
        edges: List[BraggEdge],
        background_transmission: float = 1.0,
        material: Optional[str] = None
    ):
        """
        Initialize Bragg edge sample

        Args:
            edges: List of Bragg edges
            background_transmission: Background transmission level
            material: Material name (optional)
        """
        self.edges = edges
        self.background_transmission = background_transmission
        self.material = material

    def transmission(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Calculate total transmission through sample

        T(λ) = T_bg × ∏_i [1 - A_i × E(λ, λ_edge_i, σ_i)]

        Args:
            wavelength: Wavelength array (Angstrom)

        Returns:
            Transmission values (0 to 1)
        """
        trans = np.ones_like(wavelength) * self.background_transmission

        for edge in self.edges:
            trans *= edge.transmission_contribution(wavelength)

        return trans

    @classmethod
    def create_iron_sample(cls, strain: float = 0.0) -> 'BraggEdgeSample':
        """
        Create a typical iron (BCC) sample with strain

        Args:
            strain: Applied strain (dimensionless, e.g., 1e-3 for 1000 microstrain)

        Returns:
            BraggEdgeSample configured for iron
        """
        # Iron Bragg edge positions (unstrained, in Angstrom)
        # Fe(110): 4.05 Å, Fe(200): 2.87 Å, Fe(211): 2.34 Å
        base_edges = [
            {'position': 4.05, 'height': 0.6, 'width': 0.08},  # (110)
            {'position': 2.87, 'height': 0.4, 'width': 0.06},  # (200)
            {'position': 2.34, 'height': 0.3, 'width': 0.05},  # (211)
        ]

        # Apply strain: λ_strained = λ_0 * (1 + ε)
        edges = []
        for params in base_edges:
            strained_position = params['position'] * (1 + strain)
            edge = BraggEdge(
                position=strained_position,
                height=params['height'],
                width=params['width'],
                edge_type='erf'
            )
            edges.append(edge)

        return cls(edges=edges, background_transmission=0.95, material='Fe')


class IncidentSpectrum:
    """
    Model for incident neutron spectrum
    """

    def __init__(
        self,
        spectrum_type: str = 'maxwellian',
        temperature: float = 300.0,
        custom_spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """
        Initialize incident spectrum

        Args:
            spectrum_type: Type of spectrum ('maxwellian', 'flat', 'custom')
            temperature: Temperature for Maxwellian spectrum (K)
            custom_spectrum: Tuple of (wavelength, intensity) for custom spectrum
        """
        self.spectrum_type = spectrum_type
        self.temperature = temperature

        if spectrum_type == 'custom' and custom_spectrum is not None:
            wl, intensity = custom_spectrum
            self._custom_interpolator = interp1d(
                wl, intensity,
                kind='cubic',
                bounds_error=False,
                fill_value=0.0
            )
        else:
            self._custom_interpolator = None

    def intensity(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Calculate incident spectrum intensity

        Args:
            wavelength: Wavelength array (Angstrom)

        Returns:
            Intensity values (arbitrary units)
        """
        if self.spectrum_type == 'maxwellian':
            # Maxwellian distribution: I(λ) ∝ λ^(-5) * exp(-a/λ²)
            # where a = h²/(2*m*k*T)
            a = (PLANCK_CONSTANT**2) / (2 * NEUTRON_MASS * 1.380649e-23 * self.temperature)
            a_angstrom = a / (ANGSTROM_TO_METER**2)

            intensity = wavelength**(-5) * np.exp(-a_angstrom / wavelength**2)
            return intensity / np.max(intensity)  # Normalize

        elif self.spectrum_type == 'flat':
            # Flat (white beam) spectrum
            return np.ones_like(wavelength)

        elif self.spectrum_type == 'custom' and self._custom_interpolator is not None:
            return self._custom_interpolator(wavelength)

        else:
            raise ValueError(f"Unknown spectrum type: {self.spectrum_type}")


class MeasurementSimulator:
    """
    Simulate neutron measurements with Bragg edge samples
    """

    def __init__(
        self,
        sample: BraggEdgeSample,
        incident_spectrum: IncidentSpectrum,
        tof_calibration: TOFCalibration,
        wavelength_range: Tuple[float, float] = (1.0, 10.0),
        n_wavelength_bins: int = 1000
    ):
        """
        Initialize measurement simulator

        Args:
            sample: Bragg edge sample
            incident_spectrum: Incident neutron spectrum
            tof_calibration: TOF calibration parameters
            wavelength_range: Wavelength range to simulate (Angstrom)
            n_wavelength_bins: Number of wavelength bins
        """
        self.sample = sample
        self.incident_spectrum = incident_spectrum
        self.tof_calibration = tof_calibration
        self.wavelength_range = wavelength_range
        self.n_wavelength_bins = n_wavelength_bins

        # Create wavelength grid
        self.wavelength = np.linspace(
            wavelength_range[0],
            wavelength_range[1],
            n_wavelength_bins
        )

        # Pre-calculate transmission and incident spectrum
        self.transmission = sample.transmission(self.wavelength)
        self.incident_intensity = incident_spectrum.intensity(self.wavelength)
        self.transmitted_intensity = self.incident_intensity * self.transmission

    def simulate_tof_measurement(
        self,
        chopper_pattern: np.ndarray,
        n_time_bins: int,
        flux: float = 1e6,
        measurement_time: float = 1.0,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a TOF measurement with a given chopper pattern

        Args:
            chopper_pattern: Binary chopper pattern (1 = open, 0 = closed)
            n_time_bins: Number of time bins for detection
            flux: Neutron flux (neutrons/second)
            measurement_time: Total measurement time (seconds)
            add_noise: Whether to add Poisson noise

        Returns:
            Tuple of (time_bins, detected_counts)
        """
        # Initialize detected signal
        detected_counts = np.zeros(n_time_bins)

        # Time bins for chopper and detection
        n_chop_bins = len(chopper_pattern)
        chop_time_max = measurement_time
        det_time_max = measurement_time + self.tof_calibration.wavelength_to_tof(
            self.wavelength_range[1]
        )

        chop_times = np.linspace(0, chop_time_max, n_chop_bins)
        det_times = np.linspace(0, det_time_max, n_time_bins)

        dt_chop = chop_times[1] - chop_times[0] if len(chop_times) > 1 else measurement_time

        # For each open chopper slot
        open_indices = np.where(chopper_pattern > 0)[0]

        for chop_idx in open_indices:
            t_chop = chop_times[chop_idx]

            # For each wavelength, calculate when it arrives at detector
            for i, wl in enumerate(self.wavelength):
                tof = self.tof_calibration.wavelength_to_tof(wl)
                t_detect = t_chop + tof

                # Find detection time bin
                det_idx = np.searchsorted(det_times, t_detect)

                if 0 <= det_idx < n_time_bins:
                    # Add contribution weighted by intensity and flux
                    intensity = self.transmitted_intensity[i]
                    d_wavelength = (self.wavelength_range[1] - self.wavelength_range[0]) / self.n_wavelength_bins

                    # Neutron count contribution
                    counts = intensity * flux * dt_chop * d_wavelength
                    detected_counts[det_idx] += counts

        # Add Poisson noise
        if add_noise:
            detected_counts = np.random.poisson(detected_counts)

        return det_times, detected_counts

    def get_wavelength_histogram(self, n_bins: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get histogram of transmitted intensity vs wavelength

        Args:
            n_bins: Number of bins (uses default if None)

        Returns:
            Tuple of (wavelength, intensity)
        """
        if n_bins is None:
            return self.wavelength, self.transmitted_intensity
        else:
            # Rebin to requested resolution
            new_wavelength = np.linspace(
                self.wavelength_range[0],
                self.wavelength_range[1],
                n_bins
            )
            new_intensity = np.interp(
                new_wavelength,
                self.wavelength,
                self.transmitted_intensity
            )
            return new_wavelength, new_intensity


def calculate_edge_gradient(
    wavelength: np.ndarray,
    transmission: np.ndarray
) -> np.ndarray:
    """
    Calculate gradient (derivative) of transmission curve

    This identifies regions with sharp features (edges) where
    measurements are most informative.

    Args:
        wavelength: Wavelength array
        transmission: Transmission array

    Returns:
        Absolute gradient of transmission
    """
    gradient = np.gradient(transmission, wavelength)
    return np.abs(gradient)


def estimate_edge_position(
    wavelength: np.ndarray,
    transmission: np.ndarray,
    window: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
    """
    Estimate Bragg edge position from transmission data

    Args:
        wavelength: Wavelength array
        transmission: Transmission array
        window: Optional wavelength window to search in

    Returns:
        Tuple of (edge_position, uncertainty)
    """
    # Apply window if specified
    if window is not None:
        mask = (wavelength >= window[0]) & (wavelength <= window[1])
        wl = wavelength[mask]
        trans = transmission[mask]
    else:
        wl = wavelength
        trans = transmission

    # Find position of maximum gradient
    gradient = np.abs(np.gradient(trans, wl))
    max_grad_idx = np.argmax(gradient)
    edge_position = wl[max_grad_idx]

    # Estimate uncertainty from width of gradient peak
    # Find half-maximum points
    half_max = gradient[max_grad_idx] / 2
    above_half = gradient > half_max

    # Find width of region above half maximum
    if np.any(above_half):
        indices = np.where(above_half)[0]
        width = wl[indices[-1]] - wl[indices[0]]
        uncertainty = width / 2
    else:
        uncertainty = (wl[-1] - wl[0]) / 10  # Default to 10% of range

    return edge_position, uncertainty
