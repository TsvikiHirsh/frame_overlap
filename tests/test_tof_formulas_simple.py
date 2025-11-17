"""
Simple TOF shift correction formula validation.

This test validates the TOF shift correction formulas against the
Frame Overlap Bragg Edge Imaging article (Nature Scientific Reports, 2020)
WITHOUT relying on the complex Data/Reconstructor infrastructure.

Key formulas from the article:
1. Time delays: τₖ = θₖ / (360° × f) where θₖ is slit angle, f is frequency
2. Kernel reconstruction: Places Dirac deltas at time delays τₖ
3. TOF to wavelength: λ = (h/m_n) × t / L_sd
"""

import numpy as np


def test_article_equation_1_time_delays():
    """
    Test Equation (1): Time structure function τ(t') = Σₖ aₖ δ(t' - τₖ)

    Formula for time delays: τₖ = θₖ / (360° × f)
    """
    print("\n" + "="*70)
    print("TEST 1: Article Equation (1) - Time Delays Calculation")
    print("="*70)

    # Test parameters from the article (POLDI chopper)
    angles_deg = np.array([0, 9.363, 21.475, 37.039, 50.417, 56.664, 67.422, 75.406])
    frequency_hz = 50  # Hz

    # Calculate time delays using article's formula
    tau_k_s = angles_deg / (360.0 * frequency_hz)  # seconds
    tau_k_ms = tau_k_s * 1000  # milliseconds

    print(f"\nChopper frequency: {frequency_hz} Hz")
    print(f"Number of slits: {len(angles_deg)}")
    print(f"\nCalculated time delays τₖ:")
    print(f"{'Slit':<5} {'Angle (°)':>10} {'τₖ (s)':>12} {'τₖ (ms)':>12}")
    print("-" * 45)

    for k, (angle, tau_s, tau_ms) in enumerate(zip(angles_deg, tau_k_s, tau_k_ms)):
        print(f"{k:<5} {angle:>10.3f} {tau_s:>12.6f} {tau_ms:>12.4f}")

    # Verify the formula
    expected_delay_per_degree = 1.0 / (360.0 * frequency_hz)  # s per degree
    print(f"\nExpected delay per degree: {expected_delay_per_degree*1000:.4f} ms/°")

    # Test cases
    print("\nValidation tests:")

    # Test 1: First slit should be at t=0
    assert tau_k_ms[0] == 0, "First slit delay should be 0"
    print("✓ First slit at t=0")

    # Test 2: Delays should be monotonically increasing
    assert np.all(np.diff(tau_k_ms) > 0), "Delays should be monotonically increasing"
    print("✓ Delays are monotonically increasing")

    # Test 3: Verify formula for each angle
    for k, (angle, tau_calc) in enumerate(zip(angles_deg, tau_k_s)):
        tau_expected = angle * expected_delay_per_degree
        assert np.isclose(tau_calc, tau_expected, rtol=1e-10), \
            f"Delay calculation mismatch for slit {k}"
    print("✓ All delays match formula τₖ = θₖ / (360° × f)")

    # Test 4: For 50 Hz, one full rotation = 20 ms
    full_rotation_time = 1.0 / frequency_hz  # seconds
    print(f"\nFull rotation time: {full_rotation_time*1000:.1f} ms (360°)")
    assert np.isclose(full_rotation_time, 0.02, rtol=1e-10)
    print("✓ Full rotation time verified")

    print("\n" + "="*70)
    print("✓ TEST 1 PASSED\n")


def test_frequency_scaling():
    """
    Test how chopper frequency affects time delays.
    τₖ ∝ 1/f (inverse relationship)
    """
    print("="*70)
    print("TEST 2: Frequency Scaling")
    print("="*70)

    angles_deg = np.array([0, 45, 90, 180])
    frequencies = [25, 50, 100]  # Hz

    print(f"\nSlit angles: {angles_deg}°")
    print(f"\nTesting frequencies: {frequencies} Hz")

    results = {}
    for freq in frequencies:
        tau_k_ms = angles_deg / (360.0 * freq) * 1000
        results[freq] = tau_k_ms
        print(f"\nFrequency = {freq} Hz:")
        print(f"  Time delays (ms): {tau_k_ms}")

    # Verify inverse relationship: doubling frequency halves delays
    tau_50 = results[50]
    tau_100 = results[100]

    print(f"\nVerifying inverse relationship:")
    print(f"  Delays at 50 Hz:  {tau_50}")
    print(f"  Delays at 100 Hz: {tau_100}")
    print(f"  Ratio (50/100):   {tau_50 / tau_100}")

    assert np.allclose(tau_100, tau_50 / 2.0, rtol=1e-10), \
        "Doubling frequency should halve delays"
    print("✓ Doubling frequency halves delays")

    print("\n" + "="*70)
    print("✓ TEST 2 PASSED\n")


def test_kernel_reconstruction_discrete():
    """
    Test discrete kernel reconstruction from time delays.

    The kernel should have delta functions at bin positions corresponding
    to the time delays, with equal weights (1/n_frames) for equal slits.
    """
    print("="*70)
    print("TEST 3: Discrete Kernel Reconstruction")
    print("="*70)

    # Simple test case: 2 frames
    kernel_delays_ms = np.array([0, 25])  # milliseconds
    bin_width_us = 10  # microseconds
    n_frames = len(kernel_delays_ms)

    # Convert to bin positions
    kernel_delays_us = kernel_delays_ms * 1000  # Convert ms to µs
    frame_starts_us = np.cumsum(kernel_delays_us)
    frame_starts_bins = frame_starts_us / bin_width_us

    print(f"\nFrame delays: {kernel_delays_ms} ms")
    print(f"Bin width: {bin_width_us} µs")
    print(f"Frame start times: {frame_starts_us} µs")
    print(f"Frame start bins: {frame_starts_bins}")

    # Create kernel
    kernel_length = 5000
    kernel = np.zeros(kernel_length)

    # Place delta functions at integer bin positions
    for bin_float in frame_starts_bins:
        bin_int = int(np.round(bin_float))
        kernel[bin_int] = 1.0 / n_frames  # Equal weight per article Eq. 1

    # Find non-zero positions
    non_zero_indices = np.where(kernel > 0)[0]

    print(f"\nKernel length: {kernel_length} bins")
    print(f"Non-zero bin positions: {non_zero_indices}")
    print(f"Kernel values at these positions: {kernel[non_zero_indices]}")

    # Validation
    expected_bins = [0, 2500]
    assert len(non_zero_indices) == n_frames, f"Expected {n_frames} non-zero bins"
    assert list(non_zero_indices) == expected_bins, "Bin positions mismatch"
    print(f"✓ Kernel has {n_frames} non-zero bins at correct positions")

    # Check weights
    for i in non_zero_indices:
        assert np.isclose(kernel[i], 1.0/n_frames), "Weight should be 1/n_frames"
    print(f"✓ Each frame has weight 1/n_frames = {1.0/n_frames}")

    # Check normalization
    total_weight = np.sum(kernel)
    assert np.isclose(total_weight, 1.0), "Kernel should be normalized to 1"
    print(f"✓ Total kernel weight = {total_weight:.6f} (normalized)")

    print("\n" + "="*70)
    print("✓ TEST 3 PASSED\n")


def test_kernel_reconstruction_interpolated():
    """
    Test interpolated kernel reconstruction with fractional bins.

    According to the article:
    "In the discretization of the Dirac's delta into bins with defined ToF width,
    each term δ(t-τₖ) is split between the two nearest adjacent bins, each with
    a proper weight corresponding to the distance of the bin from the time delay."
    """
    print("="*70)
    print("TEST 4: Interpolated Kernel Reconstruction (FOBI Method)")
    print("="*70)

    # Test case with fractional delay
    kernel_delays_ms = np.array([0, 25.003])  # Fractional delay
    bin_width_us = 10  # microseconds
    n_frames = len(kernel_delays_ms)

    # Convert to bin positions
    kernel_delays_us = kernel_delays_ms * 1000
    frame_starts_us = np.cumsum(kernel_delays_us)
    frame_starts_bins_float = frame_starts_us / bin_width_us

    print(f"\nFrame delays: {kernel_delays_ms} ms")
    print(f"Bin width: {bin_width_us} µs")
    print(f"Frame start bins (fractional): {frame_starts_bins_float}")

    # Analyze fractional position
    for i, bin_float in enumerate(frame_starts_bins_float):
        bin_floor = int(np.floor(bin_float))
        bin_ceil = bin_floor + 1
        frac = bin_float - bin_floor

        print(f"\nFrame {i} (delay={kernel_delays_ms[i]} ms):")
        print(f"  Fractional bin position: {bin_float}")
        print(f"  Floor bin: {bin_floor}")
        print(f"  Ceiling bin: {bin_ceil}")
        print(f"  Fractional part: {frac}")

        # Weights according to article's interpolation method
        weight_floor = (1.0 - frac) / n_frames
        weight_ceil = frac / n_frames

        print(f"  Weight for floor bin:   {weight_floor:.4f}")
        print(f"  Weight for ceiling bin: {weight_ceil:.4f}")
        print(f"  Total weight:           {weight_floor + weight_ceil:.4f}")

    # Create interpolated kernel
    kernel_length = 5000
    kernel = np.zeros(kernel_length)

    for bin_float in frame_starts_bins_float:
        bin_floor = int(np.floor(bin_float))
        frac = bin_float - bin_floor

        # Distribute weight between adjacent bins
        kernel[bin_floor] += (1.0 - frac) / n_frames
        kernel[bin_floor + 1] += frac / n_frames

    # Find non-zero positions
    non_zero_indices = np.where(kernel > 1e-10)[0]

    print(f"\nInterpolated kernel:")
    print(f"Non-zero bin positions: {non_zero_indices}")
    print(f"Kernel values:")
    for idx in non_zero_indices:
        print(f"  Bin {idx}: {kernel[idx]:.6f}")

    # Validation
    # For frame 1 (bin 2500.3):
    # - Floor bin 2500 should have weight (1-0.3)/2 = 0.35
    # - Ceil bin 2501 should have weight 0.3/2 = 0.15
    expected_weights = {
        0: 0.5,      # Frame 0: full weight at bin 0
        2500: 0.35,  # Frame 1: floor part
        2501: 0.15   # Frame 1: ceiling part
    }

    print(f"\nValidation:")
    for bin_idx, expected_weight in expected_weights.items():
        actual_weight = kernel[bin_idx]
        print(f"  Bin {bin_idx}: expected {expected_weight:.4f}, got {actual_weight:.4f}")
        assert np.isclose(actual_weight, expected_weight, atol=1e-6), \
            f"Weight mismatch at bin {bin_idx}"

    print("✓ Interpolated weights match expected values")

    # Check normalization
    total_weight = np.sum(kernel)
    assert np.isclose(total_weight, 1.0, atol=1e-6), "Kernel should be normalized"
    print(f"✓ Total kernel weight = {total_weight:.6f} (normalized)")

    print("\n" + "="*70)
    print("✓ TEST 4 PASSED\n")


def test_tof_to_wavelength_conversion():
    """
    Test Equation (3): λ = (h/m_n) × t / L_sd

    where:
    - h = Planck constant = 6.62607015e-34 J·s
    - m_n = neutron mass = 1.674927498e-27 kg
    - L_sd = source-to-detector distance (m)
    - t = time-of-flight (s)
    """
    print("="*70)
    print("TEST 5: TOF to Wavelength Conversion (Article Equation 3)")
    print("="*70)

    # Physical constants
    h = 6.62607015e-34  # J·s (Planck constant)
    m_n = 1.674927498e-27  # kg (neutron mass)
    h_over_m = h / m_n  # m²/s

    print(f"\nPhysical constants:")
    print(f"  Planck constant h:  {h:.6e} J·s")
    print(f"  Neutron mass m_n:   {m_n:.6e} kg")
    print(f"  h/m_n:              {h_over_m:.6f} m²/s")
    print(f"  h/m_n:              {h_over_m * 1e10:.3f} Å·m/s")

    # Test conversion
    L_sd = 10.0  # meters (source-to-detector distance)
    tof_values_ms = [10, 20, 30, 40, 50]

    print(f"\nSource-to-detector distance: {L_sd} m")
    print(f"\nTOF to wavelength conversion:")
    print(f"{'TOF (ms)':>10} {'TOF (s)':>12} {'λ (m)':>12} {'λ (Å)':>10}")
    print("-" * 50)

    for tof_ms in tof_values_ms:
        tof_s = tof_ms / 1000.0
        # Apply article's formula: λ = (h/m_n) × t / L_sd
        wavelength_m = (h / m_n) * tof_s / L_sd
        wavelength_angstrom = wavelength_m * 1e10

        print(f"{tof_ms:10.1f} {tof_s:12.5f} {wavelength_m:12.6e} {wavelength_angstrom:10.4f}")

    # Verification: For thermal neutrons (λ ≈ 1.8 Å), calculate expected TOF
    lambda_thermal = 1.8e-10  # meters (1.8 Ångströms)
    # From λ = (h/m_n) × t / L_sd, we get t = λ × L_sd × m_n / h
    t_thermal = lambda_thermal * L_sd * m_n / h
    t_thermal_ms = t_thermal * 1000

    print(f"\nVerification:")
    print(f"  For thermal neutrons (λ = 1.8 Å) at L = {L_sd} m:")
    print(f"    Expected TOF: {t_thermal_ms:.3f} ms")

    # Reverse calculation
    lambda_check = (h / m_n) * t_thermal / L_sd * 1e10
    print(f"    Reverse calculation: {lambda_check:.3f} Å")
    assert np.isclose(lambda_check, 1.8, rtol=1e-4), "Wavelength conversion error"
    print("  ✓ Conversion verified")

    print("\n" + "="*70)
    print("✓ TEST 5 PASSED\n")


def test_cumulative_delay_calculation():
    """
    Test that frame delays are correctly converted to absolute frame start times
    using cumulative sum.
    """
    print("="*70)
    print("TEST 6: Cumulative Delay Calculation")
    print("="*70)

    # Test case from the article methodology
    kernel_delays_ms = np.array([0, 12, 10, 25])
    expected_frame_starts = np.array([0, 12, 22, 47])

    print(f"\nFrame-to-frame delays (kernel): {kernel_delays_ms} ms")
    print(f"Expected cumulative frame starts: {expected_frame_starts} ms")

    # Calculate cumulative delays
    calculated_frame_starts = np.cumsum(kernel_delays_ms)

    print(f"Calculated frame starts: {calculated_frame_starts} ms")

    assert np.allclose(calculated_frame_starts, expected_frame_starts), \
        "Cumulative delay calculation mismatch"
    print("✓ Cumulative delays match expected values")

    # Test the relationship
    print(f"\nVerification:")
    for i, (delay, start) in enumerate(zip(kernel_delays_ms, calculated_frame_starts)):
        print(f"  Frame {i}: delay = {delay} ms → start time = {start} ms")

    print("\n" + "="*70)
    print("✓ TEST 6 PASSED\n")


def test_bin_width_scaling():
    """
    Test that kernel reconstruction scales correctly with different bin widths.
    """
    print("="*70)
    print("TEST 7: Bin Width Scaling")
    print("="*70)

    kernel_delays_ms = np.array([0, 25])
    bin_widths_us = [5, 10, 20, 50]

    print(f"\nFrame delays: {kernel_delays_ms} ms")
    print(f"\nTesting different bin widths:")

    for bin_width in bin_widths_us:
        # Calculate bin positions
        kernel_delays_us = kernel_delays_ms * 1000
        frame_starts_us = np.cumsum(kernel_delays_us)
        frame_starts_bins = frame_starts_us / bin_width

        expected_bin_1 = int(25000 / bin_width)

        print(f"\n  Bin width: {bin_width} µs")
        print(f"    Frame 1 delay: 25 ms = 25000 µs")
        print(f"    Expected bin: {expected_bin_1}")
        print(f"    Calculated bin: {int(frame_starts_bins[1])}")

        assert int(frame_starts_bins[1]) == expected_bin_1, \
            f"Bin position mismatch for bin_width={bin_width} µs"

    print(f"\n✓ Kernel scales correctly with bin width")

    print("\n" + "="*70)
    print("✓ TEST 7 PASSED\n")


def run_all_tests():
    """Run all TOF formula validation tests."""
    print("\n" + "#"*70)
    print("# TOF SHIFT CORRECTION FORMULA VALIDATION TEST SUITE")
    print("# Based on: Frame overlap Bragg edge imaging")
    print("# Nature Scientific Reports, vol 10, Article 14867 (2020)")
    print("#"*70 + "\n")

    tests = [
        test_article_equation_1_time_delays,
        test_frequency_scaling,
        test_kernel_reconstruction_discrete,
        test_kernel_reconstruction_interpolated,
        test_tof_to_wavelength_conversion,
        test_cumulative_delay_calculation,
        test_bin_width_scaling,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {test_func.__name__}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {test_func.__name__}")
            print(f"   Error: {e}")
            failed += 1

    print("\n" + "#"*70)
    print(f"# TEST SUMMARY: {passed} PASSED, {failed} FAILED")
    print("#"*70 + "\n")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
