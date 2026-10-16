"""
TOF Offset Correction After Wiener Deconvolution

After Wiener reconstruction, the retrieved data has an offset in the TOF domain
that depends on the peak of the source spectrum. This module provides methods
to detect and correct this systematic shift.

Reference:
    Tremsin et al. - FOBI: Frame-Overlap Bragg-Edge Imaging
    The offset is related to the position of the peak in the kernel (source spectrum)
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import correlate, find_peaks
from scipy.interpolate import interp1d
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class OffsetCorrectionResult:
    """
    Result of TOF offset correction

    Attributes:
        offset: Detected TOF offset (bins or time units)
        offset_uncertainty: Uncertainty in offset
        corrected_data: Data after offset correction
        correction_method: Method used for detection
        quality_metric: Quality of the correction (e.g., correlation coefficient)
        reference_used: What reference was used (edge position, cross-correlation, etc.)
    """
    offset: float
    offset_uncertainty: float
    corrected_data: np.ndarray
    correction_method: str
    quality_metric: float
    reference_used: str


class TOFOffsetCorrector:
    """
    Detect and correct TOF offset after Wiener deconvolution
    """

    def __init__(
        self,
        tof_bins: np.ndarray,
        kernel: Optional[np.ndarray] = None
    ):
        """
        Initialize offset corrector

        Args:
            tof_bins: TOF bin centers
            kernel: Convolution kernel (source spectrum shape)
        """
        self.tof_bins = tof_bins
        self.kernel = kernel

        # Estimate expected offset from kernel peak if provided
        self.kernel_peak_offset = None
        if kernel is not None:
            self.kernel_peak_offset = self._estimate_kernel_offset(kernel)

    def _estimate_kernel_offset(self, kernel: np.ndarray) -> float:
        """
        Estimate offset from kernel peak position

        The offset is typically related to where the kernel (source spectrum) peaks

        Args:
            kernel: Convolution kernel

        Returns:
            Estimated offset in bins
        """
        # Find peak of kernel
        peak_idx = np.argmax(kernel)

        # Center of kernel
        center = len(kernel) // 2

        # Offset is difference from center
        offset = peak_idx - center

        return offset

    def correct_by_cross_correlation(
        self,
        reconstructed: np.ndarray,
        reference: np.ndarray,
        max_shift: Optional[int] = None
    ) -> OffsetCorrectionResult:
        """
        Correct offset by cross-correlation with reference

        This is useful when you have a reference measurement or known signal shape

        Args:
            reconstructed: Reconstructed data (potentially shifted)
            reference: Reference data (correctly aligned)
            max_shift: Maximum shift to search (bins)

        Returns:
            OffsetCorrectionResult
        """
        if max_shift is None:
            max_shift = len(reconstructed) // 10  # Search up to 10% of length

        # Compute cross-correlation
        correlation = correlate(reference, reconstructed, mode='same')

        # Find peak of correlation
        center = len(correlation) // 2
        search_region = slice(
            max(0, center - max_shift),
            min(len(correlation), center + max_shift)
        )

        peak_idx = np.argmax(correlation[search_region])
        peak_idx += search_region.start

        # Offset from center
        offset = peak_idx - center

        # Apply offset correction
        corrected = self._apply_shift(reconstructed, -offset)

        # Calculate quality metric (correlation coefficient at peak)
        quality = correlation[peak_idx] / (np.linalg.norm(reference) * np.linalg.norm(reconstructed))

        # Estimate uncertainty from correlation peak width
        peak_vals = correlation[search_region]
        half_max = np.max(peak_vals) / 2
        above_half = peak_vals > half_max
        if np.any(above_half):
            uncertainty = np.sum(above_half) / 4  # Quarter of peak width
        else:
            uncertainty = 1.0

        return OffsetCorrectionResult(
            offset=offset,
            offset_uncertainty=uncertainty,
            corrected_data=corrected,
            correction_method='cross_correlation',
            quality_metric=quality,
            reference_used='reference_signal'
        )

    def correct_by_edge_position(
        self,
        reconstructed: np.ndarray,
        expected_edge_wavelength: float,
        tof_to_wavelength: Callable,
        search_window: float = 0.5
    ) -> OffsetCorrectionResult:
        """
        Correct offset using known edge position

        Finds the edge in reconstructed data and shifts to match expected position

        Args:
            reconstructed: Reconstructed transmission data
            expected_edge_wavelength: Expected edge position (Angstrom)
            tof_to_wavelength: Function to convert TOF to wavelength
            search_window: Search window around expected position (Angstrom)

        Returns:
            OffsetCorrectionResult
        """
        # Convert TOF bins to wavelength
        wavelengths = tof_to_wavelength(self.tof_bins)

        # Find edge in reconstructed data
        # Edge is where transmission drops (maximum negative gradient)
        gradient = -np.gradient(reconstructed, wavelengths)

        # Search in window around expected position
        window_mask = np.abs(wavelengths - expected_edge_wavelength) < search_window
        window_gradient = gradient.copy()
        window_gradient[~window_mask] = 0

        # Find peak gradient (edge location)
        measured_edge_idx = np.argmax(window_gradient)
        measured_edge_wavelength = wavelengths[measured_edge_idx]

        # Calculate wavelength offset
        wavelength_offset = measured_edge_wavelength - expected_edge_wavelength

        # Convert wavelength offset to TOF offset
        # Approximate: δλ/λ ≈ δt/t
        # More accurate: use inverse function
        expected_idx = np.argmin(np.abs(wavelengths - expected_edge_wavelength))
        offset_bins = measured_edge_idx - expected_idx

        # Apply correction
        corrected = self._apply_shift(reconstructed, -offset_bins)

        # Quality metric: edge sharpness after correction
        corrected_gradient = np.abs(np.gradient(corrected, wavelengths))
        quality = np.max(corrected_gradient)

        # Uncertainty based on gradient peak width
        gradient_at_edge = window_gradient[window_mask]
        if len(gradient_at_edge) > 0:
            half_max = np.max(gradient_at_edge) / 2
            above_half = gradient_at_edge > half_max
            uncertainty = np.sum(above_half) / 2  # Half of peak width in bins
        else:
            uncertainty = 1.0

        return OffsetCorrectionResult(
            offset=offset_bins,
            offset_uncertainty=uncertainty,
            corrected_data=corrected,
            correction_method='edge_position',
            quality_metric=quality,
            reference_used=f'edge_at_{expected_edge_wavelength:.3f}A'
        )

    def correct_by_kernel_peak(
        self,
        reconstructed: np.ndarray
    ) -> OffsetCorrectionResult:
        """
        Correct offset using kernel peak position

        Uses the pre-calculated offset from kernel peak

        Args:
            reconstructed: Reconstructed data

        Returns:
            OffsetCorrectionResult
        """
        if self.kernel_peak_offset is None:
            raise ValueError("Kernel not provided during initialization")

        offset = self.kernel_peak_offset

        # Apply correction
        corrected = self._apply_shift(reconstructed, -offset)

        # Quality metric: not applicable for this method
        quality = 1.0

        return OffsetCorrectionResult(
            offset=offset,
            offset_uncertainty=0.5,  # Uncertainty is about half a bin
            corrected_data=corrected,
            correction_method='kernel_peak',
            quality_metric=quality,
            reference_used='kernel_peak_position'
        )

    def correct_by_optimization(
        self,
        reconstructed: np.ndarray,
        reference_features: dict,
        objective: str = 'edge_sharpness'
    ) -> OffsetCorrectionResult:
        """
        Correct offset by optimizing an objective function

        Args:
            reconstructed: Reconstructed data
            reference_features: Dictionary of reference features (edges, peaks, etc.)
            objective: Objective to optimize ('edge_sharpness', 'mse', 'correlation')

        Returns:
            OffsetCorrectionResult
        """
        def objective_function(shift: float) -> float:
            """Objective to minimize"""
            shifted = self._apply_shift(reconstructed, shift)

            if objective == 'edge_sharpness':
                # Maximize edge sharpness (minimize negative)
                gradient = np.abs(np.gradient(shifted))
                return -np.max(gradient)

            elif objective == 'mse':
                # Minimize MSE with reference
                if 'reference' in reference_features:
                    return np.mean((shifted - reference_features['reference'])**2)
                else:
                    raise ValueError("Reference signal required for MSE objective")

            elif objective == 'correlation':
                # Maximize correlation (minimize negative)
                if 'reference' in reference_features:
                    ref = reference_features['reference']
                    corr = np.corrcoef(shifted, ref)[0, 1]
                    return -corr
                else:
                    raise ValueError("Reference signal required for correlation objective")

            return 0.0

        # Optimize shift
        max_shift = len(reconstructed) // 10
        result = minimize_scalar(
            objective_function,
            bounds=(-max_shift, max_shift),
            method='bounded'
        )

        optimal_shift = result.x

        # Apply correction
        corrected = self._apply_shift(reconstructed, optimal_shift)

        # Quality metric
        quality = -result.fun  # Negate since we minimized negative

        return OffsetCorrectionResult(
            offset=-optimal_shift,  # Negate for correction
            offset_uncertainty=1.0,
            corrected_data=corrected,
            correction_method=f'optimization_{objective}',
            quality_metric=quality,
            reference_used=objective
        )

    def _apply_shift(
        self,
        data: np.ndarray,
        shift: float
    ) -> np.ndarray:
        """
        Apply shift to data with interpolation

        Args:
            data: Data to shift
            shift: Shift amount (can be fractional bins)

        Returns:
            Shifted data
        """
        # Create interpolator
        x = np.arange(len(data))
        interpolator = interp1d(
            x,
            data,
            kind='cubic',
            bounds_error=False,
            fill_value=0.0
        )

        # Shifted x values
        x_shifted = x - shift

        # Interpolate
        shifted_data = interpolator(x_shifted)

        return shifted_data

    def auto_correct(
        self,
        reconstructed: np.ndarray,
        method: str = 'auto',
        **kwargs
    ) -> OffsetCorrectionResult:
        """
        Automatically correct offset using best available method

        Args:
            reconstructed: Reconstructed data
            method: Method to use ('auto', 'kernel', 'edge', 'correlation', 'optimization')
            **kwargs: Additional arguments for specific methods

        Returns:
            OffsetCorrectionResult
        """
        if method == 'auto':
            # Choose best method based on available information
            if self.kernel is not None:
                method = 'kernel'
            elif 'expected_edge_wavelength' in kwargs:
                method = 'edge'
            elif 'reference' in kwargs:
                method = 'correlation'
            else:
                method = 'optimization'

        if method == 'kernel':
            return self.correct_by_kernel_peak(reconstructed)

        elif method == 'edge':
            return self.correct_by_edge_position(
                reconstructed,
                kwargs.get('expected_edge_wavelength'),
                kwargs.get('tof_to_wavelength')
            )

        elif method == 'correlation':
            return self.correct_by_cross_correlation(
                reconstructed,
                kwargs.get('reference')
            )

        elif method == 'optimization':
            return self.correct_by_optimization(
                reconstructed,
                kwargs.get('reference_features', {}),
                kwargs.get('objective', 'edge_sharpness')
            )

        else:
            raise ValueError(f"Unknown method: {method}")


def estimate_expected_offset(
    kernel: np.ndarray,
    source_type: str = 'maxwellian'
) -> Tuple[float, str]:
    """
    Estimate expected TOF offset based on source spectrum shape

    Args:
        kernel: Convolution kernel (source spectrum)
        source_type: Type of source ('maxwellian', 'flat', 'custom')

    Returns:
        Tuple of (expected_offset_bins, explanation)
    """
    # Find peak of kernel
    peak_idx = np.argmax(kernel)
    center = len(kernel) // 2
    offset = peak_idx - center

    if source_type == 'maxwellian':
        explanation = (
            f"Maxwellian source peaks at bin {peak_idx}, "
            f"causing systematic shift of {offset} bins from center. "
            f"This offset should be subtracted from reconstructed TOF data."
        )
    elif source_type == 'flat':
        explanation = (
            f"Flat source should have minimal offset, but measured peak is at bin {peak_idx}. "
            f"Offset is {offset} bins."
        )
    else:
        explanation = (
            f"Source spectrum peaks at bin {peak_idx}, "
            f"resulting in {offset} bin offset."
        )

    return offset, explanation


def apply_offset_correction_to_workflow(
    reconstructed_data: np.ndarray,
    tof_bins: np.ndarray,
    kernel: Optional[np.ndarray] = None,
    expected_edge_wavelength: Optional[float] = None,
    tof_to_wavelength: Optional[Callable] = None,
    method: str = 'auto'
) -> Tuple[np.ndarray, OffsetCorrectionResult]:
    """
    Convenience function to apply offset correction in a workflow

    Args:
        reconstructed_data: Data after Wiener deconvolution
        tof_bins: TOF bin centers
        kernel: Convolution kernel (if available)
        expected_edge_wavelength: Expected edge position (if known)
        tof_to_wavelength: TOF to wavelength conversion function
        method: Correction method

    Returns:
        Tuple of (corrected_data, correction_result)
    """
    corrector = TOFOffsetCorrector(tof_bins, kernel)

    kwargs = {}
    if expected_edge_wavelength is not None:
        kwargs['expected_edge_wavelength'] = expected_edge_wavelength
    if tof_to_wavelength is not None:
        kwargs['tof_to_wavelength'] = tof_to_wavelength

    result = corrector.auto_correct(reconstructed_data, method=method, **kwargs)

    return result.corrected_data, result
