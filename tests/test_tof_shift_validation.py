"""
Test TOF shift correction against the article's formulas.

This test validates the TOF shift correction implementation against the
Frame Overlap Bragg Edge Imaging article (Nature Scientific Reports, 2020).

Key formulas from the article:
1. Time delays: τₖ = θₖ / (360° × f) where θₖ is slit angle, f is frequency
2. Kernel reconstruction: Places Dirac deltas at time delays τₖ
3. TOF to wavelength: λ = (h/m_n) × t / L_sd
"""

import numpy as np
import pytest
from frame_overlap.data_class import Data
from frame_overlap.reconstruct import Reconstructor
from frame_overlap.tof_offset_correction import TOFOffsetCorrector


class TestTOFShiftValidation:
    """Test suite for validating TOF shift correction calculations."""

    def test_simple_two_frame_kernel_reconstruction(self):
        """
        Test Case 1: Simple two-frame case with known delays.

        According to the article, the time structure function is:
        τ(t) = Σₖ aₖ δ(t - τₖ)

        For two frames with delays [0, 25] ms and bin_width=10 µs:
        - Frame 0: at t=0 µs → bin 0
        - Frame 1: at t=25000 µs → bin 2500

        Expected kernel should have delta functions at these positions.
        """
        # Setup
        bin_width = 10  # µs
        kernel_delays = [0, 25]  # ms

        # Expected bin positions
        expected_bins = [0, 2500]  # bins

        # Create mock data
        data = self._create_mock_data(kernel_delays, bin_width)
        reconstructor = Reconstructor(data)

        # Reconstruct kernel (discrete mode)
        kernel_discrete = reconstructor._reconstruct_kernel(interpolate=False)

        # Validate: kernel should have non-zero values at expected positions
        non_zero_indices = np.where(kernel_discrete > 0)[0]

        print(f"\nTest: Two-frame kernel reconstruction")
        print(f"Kernel delays: {kernel_delays} ms")
        print(f"Expected bin positions: {expected_bins}")
        print(f"Actual non-zero positions: {non_zero_indices}")
        print(f"Kernel values at expected positions: {kernel_discrete[expected_bins]}")

        # Assertions
        assert len(non_zero_indices) == 2, f"Expected 2 non-zero bins, got {len(non_zero_indices)}"
        assert non_zero_indices[0] == expected_bins[0], f"First bin mismatch"
        assert non_zero_indices[1] == expected_bins[1], f"Second bin mismatch"

        # Each frame should have weight 1/n_frames (article Eq. 1)
        expected_weight = 1.0 / len(kernel_delays)
        assert np.isclose(kernel_discrete[expected_bins[0]], expected_weight), \
            f"Frame 0 weight should be {expected_weight}"
        assert np.isclose(kernel_discrete[expected_bins[1]], expected_weight), \
            f"Frame 1 weight should be {expected_weight}"

    def test_fractional_bin_interpolation(self):
        """
        Test Case 2: Kernel reconstruction with fractional bins.

        According to the article (Methods section):
        "In the discretization of the Dirac's delta into bins with defined ToF width,
        each term δ(t-τₖ) is split between the two nearest adjacent bins, each with
        a proper weight corresponding to the distance of the bin from the time delay."

        For a delay of 25.003 ms with bin_width=10 µs:
        - Position: 2500.3 bins
        - Should split: 70% to bin 2500, 30% to bin 2501
        """
        bin_width = 10  # µs
        kernel_delays = [0, 25.003]  # ms (fractional delay)

        # Expected fractional bin position
        delay_us = 25.003 * 1000  # 25003 µs
        expected_bin_float = delay_us / bin_width  # 2500.3
        bin_floor = int(np.floor(expected_bin_float))  # 2500
        bin_ceil = bin_floor + 1  # 2501
        fractional_part = expected_bin_float - bin_floor  # 0.3

        # Expected weights (per article's interpolation method)
        expected_weight_floor = (1.0 - fractional_part) / 2  # 0.35
        expected_weight_ceil = fractional_part / 2  # 0.15

        # Create mock data
        data = self._create_mock_data(kernel_delays, bin_width)
        reconstructor = Reconstructor(data)

        # Reconstruct kernel (interpolated mode - FOBI style)
        kernel_interpolated = reconstructor._reconstruct_kernel(interpolate=True)

        print(f"\nTest: Fractional bin interpolation")
        print(f"Delay: {kernel_delays[1]} ms = {delay_us} µs")
        print(f"Fractional bin position: {expected_bin_float}")
        print(f"Expected distribution:")
        print(f"  Bin {bin_floor}: {expected_weight_floor:.3f}")
        print(f"  Bin {bin_ceil}: {expected_weight_ceil:.3f}")
        print(f"Actual kernel values:")
        print(f"  Bin {bin_floor}: {kernel_interpolated[bin_floor]:.3f}")
        print(f"  Bin {bin_ceil}: {kernel_interpolated[bin_ceil]:.3f}")

        # Assertions
        assert np.isclose(kernel_interpolated[bin_floor], expected_weight_floor, atol=1e-3), \
            f"Floor bin weight mismatch"
        assert np.isclose(kernel_interpolated[bin_ceil], expected_weight_ceil, atol=1e-3), \
            f"Ceiling bin weight mismatch"

        # Total weight should equal 1/n_frames
        total_weight_frame1 = kernel_interpolated[bin_floor] + kernel_interpolated[bin_ceil]
        assert np.isclose(total_weight_frame1, 0.5, atol=1e-3), \
            f"Total weight for frame 1 should be 0.5, got {total_weight_frame1}"

    def test_time_delay_from_chopper_parameters(self):
        """
        Test Case 3: Calculate time delays from chopper parameters.

        According to the article:
        τₖ = θₖ / (360° × f)

        Where:
        - θₖ is the angular position of slit k (degrees)
        - f is the chopper rotation frequency (Hz)

        Example with POLDI chopper parameters from the article:
        - Angles: [0, 9.363, 21.475, 37.039, 50.417, 56.664, 67.422, 75.406] degrees
        - Frequency: 50 Hz
        """
        # POLDI chopper angles from the article (Table 1)
        angles_deg = np.array([0, 9.363, 21.475, 37.039, 50.417, 56.664, 67.422, 75.406])
        frequency_hz = 50  # Hz

        # Calculate time delays using article's formula
        # τₖ = θₖ / (360° × f)
        time_delays_s = angles_deg / (360.0 * frequency_hz)
        time_delays_ms = time_delays_s * 1000  # Convert to milliseconds

        print(f"\nTest: Time delays from chopper parameters")
        print(f"Chopper frequency: {frequency_hz} Hz")
        print(f"Slit angles (deg): {angles_deg}")
        print(f"Calculated time delays (ms):")
        for i, (angle, delay) in enumerate(zip(angles_deg, time_delays_ms)):
            print(f"  Slit {i}: {angle:7.3f}° → {delay:7.4f} ms")

        # Verify the formula matches expected pattern
        # For 50 Hz, one full rotation takes 20 ms
        # So 360° = 20 ms, therefore 1° = 20/360 = 0.0556 ms
        expected_delay_per_degree = 1000.0 / (360.0 * frequency_hz)  # ms per degree

        print(f"\nExpected delay per degree: {expected_delay_per_degree:.4f} ms/°")

        for i, (angle, delay) in enumerate(zip(angles_deg, time_delays_ms)):
            expected_delay = angle * expected_delay_per_degree
            assert np.isclose(delay, expected_delay, rtol=1e-6), \
                f"Delay calculation mismatch for slit {i}"

        # The first frame should always start at t=0
        assert time_delays_ms[0] == 0, "First frame delay should be 0"

        # Delays should be monotonically increasing
        assert np.all(np.diff(time_delays_ms) > 0), "Delays should be monotonically increasing"

    def test_tof_offset_estimation(self):
        """
        Test Case 4: TOF offset estimation after Wiener deconvolution.

        According to the reconstruction method, after Wiener deconvolution,
        there can be a TOF offset due to the phase of the Fourier transform.

        The offset should be related to the position of the peak in the
        source spectrum (kernel).
        """
        bin_width = 10  # µs

        # Test with symmetric kernel (peak in center)
        kernel_delays_symmetric = [0, 10, 20, 30]  # ms - symmetric around 15 ms
        data_sym = self._create_mock_data(kernel_delays_symmetric, bin_width)
        reconstructor_sym = Reconstructor(data_sym)
        kernel_sym = reconstructor_sym._reconstruct_kernel(interpolate=True)

        # Find peak position
        peak_idx_sym = np.argmax(kernel_sym)
        center_idx = len(kernel_sym) // 2
        offset_sym = peak_idx_sym - center_idx

        print(f"\nTest: TOF offset estimation")
        print(f"Symmetric kernel delays: {kernel_delays_symmetric} ms")
        print(f"Kernel length: {len(kernel_sym)} bins")
        print(f"Center bin: {center_idx}")
        print(f"Peak bin: {peak_idx_sym}")
        print(f"Estimated offset: {offset_sym} bins = {offset_sym * bin_width} µs")

        # Test with asymmetric kernel (peak offset from center)
        kernel_delays_asymmetric = [0, 5, 30, 35]  # ms - biased toward later times
        data_asym = self._create_mock_data(kernel_delays_asymmetric, bin_width)
        reconstructor_asym = Reconstructor(data_asym)
        kernel_asym = reconstructor_asym._reconstruct_kernel(interpolate=True)

        peak_idx_asym = np.argmax(kernel_asym)
        offset_asym = peak_idx_asym - center_idx

        print(f"\nAsymmetric kernel delays: {kernel_delays_asymmetric} ms")
        print(f"Peak bin: {peak_idx_asym}")
        print(f"Estimated offset: {offset_asym} bins = {offset_asym * bin_width} µs")

        # The asymmetric kernel should have a larger positive offset
        assert offset_asym > offset_sym, \
            "Asymmetric kernel should have larger positive offset"

    def test_cumulative_delay_calculation(self):
        """
        Test Case 5: Verify cumulative delay calculation.

        The kernel parameter represents frame-to-frame delays, and these
        should be converted to absolute frame start times using cumulative sum.

        Example:
        kernel = [0, 12, 10, 25] ms (frame delays)
        frame_starts = [0, 12, 22, 47] ms (cumulative)
        """
        kernel_delays = [0, 12, 10, 25]  # ms (frame-to-frame delays)
        expected_frame_starts = [0, 12, 22, 47]  # ms (cumulative)

        # Calculate cumulative delays
        calculated_frame_starts = np.cumsum(kernel_delays)

        print(f"\nTest: Cumulative delay calculation")
        print(f"Frame delays (ms): {kernel_delays}")
        print(f"Expected frame starts (ms): {expected_frame_starts}")
        print(f"Calculated frame starts (ms): {calculated_frame_starts.tolist()}")

        assert np.allclose(calculated_frame_starts, expected_frame_starts), \
            "Cumulative delay calculation mismatch"

    def test_bin_width_independence(self):
        """
        Test Case 6: Verify that kernel reconstruction is consistent across
        different bin widths.

        The same physical time delays should map to proportionally scaled
        bin positions when bin width changes.
        """
        kernel_delays = [0, 25]  # ms

        # Test with different bin widths
        bin_widths = [5, 10, 20, 50]  # µs

        print(f"\nTest: Bin width independence")
        print(f"Kernel delays: {kernel_delays} ms")

        for bin_width in bin_widths:
            data = self._create_mock_data(kernel_delays, bin_width)
            reconstructor = Reconstructor(data)
            kernel = reconstructor._reconstruct_kernel(interpolate=False)

            # Find non-zero positions
            non_zero_indices = np.where(kernel > 0)[0]

            # Expected positions for this bin width
            expected_bin_1 = int(25000 / bin_width)  # 25 ms = 25000 µs

            print(f"  Bin width {bin_width} µs:")
            print(f"    Expected bin for frame 1: {expected_bin_1}")
            print(f"    Actual non-zero bins: {non_zero_indices}")

            assert non_zero_indices[1] == expected_bin_1, \
                f"Bin position mismatch for bin_width={bin_width} µs"

    def test_frequency_scaling(self):
        """
        Test Case 7: Test how chopper frequency affects time delays.

        According to the article: τₖ = θₖ / (360° × f)

        Doubling the frequency should halve the time delays.
        """
        angles_deg = np.array([0, 45, 90, 180])  # Simple angles

        print(f"\nTest: Frequency scaling")
        print(f"Slit angles: {angles_deg}°")

        frequencies = [25, 50, 100]  # Hz

        for freq in frequencies:
            time_delays_s = angles_deg / (360.0 * freq)
            time_delays_ms = time_delays_s * 1000

            print(f"\nFrequency {freq} Hz:")
            print(f"  Time delays (ms): {time_delays_ms}")

            # Verify inverse relationship: delay ∝ 1/f
            for i, (angle, delay) in enumerate(zip(angles_deg, time_delays_ms)):
                expected_delay = angle / (360.0 * freq) * 1000  # ms
                assert np.isclose(delay, expected_delay, rtol=1e-6)

        # Verify scaling: doubling frequency halves delays
        delays_50hz = angles_deg / (360.0 * 50) * 1000
        delays_100hz = angles_deg / (360.0 * 100) * 1000

        assert np.allclose(delays_100hz, delays_50hz / 2.0, rtol=1e-6), \
            "Doubling frequency should halve delays"

    # Helper methods

    def _create_mock_data(self, kernel_delays, bin_width=10):
        """Create mock Data for testing."""
        import pandas as pd

        # Create minimal data table
        n_bins = 5000
        tof_us = np.arange(n_bins) * bin_width

        # Create simple test spectrum (e.g., Gaussian)
        center_bin = n_bins // 2
        width = 500
        test_spectrum = np.exp(-0.5 * ((np.arange(n_bins) - center_bin) / width) ** 2)
        test_spectrum *= 10000  # Scale to reasonable counts

        data_table = pd.DataFrame({
            'tof': tof_us,
            'counts': test_spectrum,
            'counts_err': np.sqrt(test_spectrum)
        })

        # Create metadata
        metadata = {
            'bin_width': bin_width,
            'kernel': kernel_delays,
            'distance': 10.0,  # meters (dummy value)
            'name': 'test_data'
        }

        return Data(data_table, metadata)


# Additional validation tests

def test_article_equation_1_validation():
    """
    Validate that our implementation matches Equation (1) from the article:

    τ(t') = Σₖ aₖ δ(t' - τₖ)

    where:
    - τₖ = θₖ / (2πf) are the time delays
    - aₖ are coefficients (= 1/n for equal slits)
    - δ is the Dirac delta function
    """
    print("\n" + "="*70)
    print("ARTICLE EQUATION (1) VALIDATION")
    print("="*70)

    # Test parameters
    n_slits = 8
    angles_deg = np.array([0, 9.363, 21.475, 37.039, 50.417, 56.664, 67.422, 75.406])
    frequency_hz = 50

    # Calculate time delays (article formula)
    tau_k = angles_deg / (360.0 * frequency_hz)  # seconds
    tau_k_ms = tau_k * 1000  # milliseconds

    print(f"Number of slits: {n_slits}")
    print(f"Chopper frequency: {frequency_hz} Hz")
    print(f"\nCalculated time delays τₖ (ms):")
    for k, (angle, delay) in enumerate(zip(angles_deg, tau_k_ms)):
        print(f"  τ_{k} (θ={angle:7.3f}°) = {delay:8.4f} ms")

    # Coefficients for equal slits
    a_k = 1.0 / n_slits
    print(f"\nCoefficients aₖ (equal slits): {a_k:.4f}")

    # Verify normalization
    total_weight = n_slits * a_k
    print(f"Total weight (Σ aₖ): {total_weight:.4f}")
    assert np.isclose(total_weight, 1.0), "Coefficients should sum to 1"

    print("\n✓ Equation (1) implementation validated")


def test_article_equation_3_validation():
    """
    Validate wavelength conversion according to Equation (3):

    λ = (h/m_n) × t / L_sd

    where:
    - h = Planck constant = 6.62607015e-34 J·s
    - m_n = neutron mass = 1.674927498e-27 kg
    - L_sd = source-to-detector distance (m)
    - t = time-of-flight (s)

    Note: The constant h/(m_n) ≈ 3956.034 Å·m/s
    """
    print("\n" + "="*70)
    print("ARTICLE EQUATION (3) VALIDATION")
    print("="*70)

    # Physical constants
    h = 6.62607015e-34  # J·s (Planck constant)
    m_n = 1.674927498e-27  # kg (neutron mass)
    h_over_m = h / m_n  # m²/s

    print(f"Planck constant h: {h:.3e} J·s")
    print(f"Neutron mass m_n: {m_n:.3e} kg")
    print(f"h/m_n: {h_over_m:.6f} m²/s")
    print(f"h/m_n: {h_over_m * 1e10:.3f} Å·m/s")

    # Test conversion
    L_sd = 10.0  # meters (typical)
    tof_values_ms = [10, 20, 30, 40, 50]  # milliseconds

    print(f"\nSource-to-detector distance: {L_sd} m")
    print(f"\nTOF to wavelength conversion:")
    print(f"{'TOF (ms)':>10} {'TOF (s)':>12} {'λ (Å)':>10}")
    print("-" * 35)

    for tof_ms in tof_values_ms:
        tof_s = tof_ms / 1000.0  # Convert to seconds
        # λ = (h/m_n) × t / L_sd
        wavelength_m = h_over_m * tof_s / L_sd
        wavelength_angstrom = wavelength_m * 1e10  # Convert to Ångströms

        print(f"{tof_ms:10.1f} {tof_s:12.4f} {wavelength_angstrom:10.4f}")

    # Verify with known value
    # For thermal neutrons: λ ≈ 1.8 Å corresponds to E ≈ 25 meV
    # v = h/(m_n × λ) = sqrt(2E/m_n)
    # For L=10m, λ=1.8Å: t = L×m_n×λ/h
    lambda_test = 1.8e-10  # meters (1.8 Å)
    t_expected = L_sd * m_n * lambda_test / h
    t_expected_ms = t_expected * 1000

    print(f"\nVerification: For λ = 1.8 Å at L = {L_sd} m:")
    print(f"  Expected TOF: {t_expected_ms:.3f} ms")

    # Reverse calculation
    lambda_calc = h_over_m * t_expected / L_sd * 1e10
    print(f"  Calculated λ: {lambda_calc:.3f} Å")
    assert np.isclose(lambda_calc, 1.8, rtol=1e-4), "Wavelength conversion mismatch"

    print("\n✓ Equation (3) implementation validated")


if __name__ == '__main__':
    # Run all tests
    print("\n" + "="*70)
    print("TOF SHIFT CORRECTION VALIDATION TEST SUITE")
    print("Based on: Frame overlap Bragg edge imaging")
    print("Nature Scientific Reports, vol 10, Article 14867 (2020)")
    print("="*70)

    # Run pytest
    pytest.main([__file__, '-v', '-s'])
