"""
TOF Shift Correction Diagnostic Tool

This tool helps diagnose potential issues with TOF shift correction
for different parameter sets.

Usage:
    python tof_diagnostic.py

Or import and use:
    from tof_diagnostic import diagnose_parameters
    diagnose_parameters(kernel=[0, 25], bin_width=10)
"""

import numpy as np


def diagnose_parameters(kernel, bin_width=10, frequency=None, angles=None):
    """
    Diagnose potential TOF shift correction issues for given parameters.

    Parameters
    ----------
    kernel : list
        Frame delays in milliseconds, e.g., [0, 12, 10, 25]
    bin_width : float
        Bin width in microseconds (default: 10)
    frequency : float, optional
        Chopper frequency in Hz (for reference)
    angles : list, optional
        Slit angles in degrees (for reference)

    Returns
    -------
    dict
        Diagnostic results with potential issues
    """
    print("\n" + "="*70)
    print("TOF SHIFT CORRECTION DIAGNOSTIC")
    print("="*70)

    results = {
        "warnings": [],
        "errors": [],
        "info": []
    }

    # Input validation
    print("\n1. INPUT PARAMETERS:")
    print(f"   Kernel (frame delays): {kernel} ms")
    print(f"   Bin width: {bin_width} µs")
    if frequency:
        print(f"   Chopper frequency: {frequency} Hz")
    if angles:
        print(f"   Slit angles: {angles}°")

    # Check kernel validity
    print("\n2. KERNEL VALIDATION:")

    if kernel[0] != 0:
        results["errors"].append(
            "First kernel element must be 0 (first frame starts at t=0)"
        )
        print("   ❌ ERROR: First kernel element should be 0")
    else:
        print("   ✓ First kernel element is 0")

    if any(d < 0 for d in kernel):
        results["errors"].append("Kernel contains negative delays")
        print("   ❌ ERROR: Kernel contains negative delays")
    else:
        print("   ✓ All delays are non-negative")

    # Calculate frame starts
    print("\n3. FRAME START TIMES:")
    kernel_us = np.array(kernel) * 1000  # Convert to µs
    frame_starts_us = np.cumsum(kernel_us)
    frame_starts_bins = frame_starts_us / bin_width

    for i, (start_us, start_bin) in enumerate(zip(frame_starts_us, frame_starts_bins)):
        print(f"   Frame {i}: {start_us:8.2f} µs = bin {start_bin:8.2f}")

    # Check for fractional bins
    print("\n4. BIN ALIGNMENT:")
    fractional_parts = frame_starts_bins - np.floor(frame_starts_bins)
    has_fractional = any(frac > 1e-6 for frac in fractional_parts)

    if has_fractional:
        print("   ⚠ NOTICE: Delays include fractional bins")
        print("   → Use interpolate_kernel=True for FOBI-style reconstruction")
        results["info"].append(
            "Fractional bins detected - interpolation recommended"
        )

        for i, frac in enumerate(fractional_parts):
            if frac > 1e-6:
                print(f"     Frame {i}: fractional part = {frac:.6f}")
    else:
        print("   ✓ All delays align to integer bins")
        print("   → Both discrete and interpolated modes will work")

    # Check kernel length requirements
    print("\n5. KERNEL LENGTH REQUIREMENTS:")
    max_delay_us = np.max(frame_starts_us)
    max_bin = int(np.ceil(max_delay_us / bin_width))
    recommended_length = max_bin + 1000  # Add buffer

    print(f"   Maximum delay: {max_delay_us:.2f} µs (bin {max_bin})")
    print(f"   Recommended kernel length: ≥ {recommended_length} bins")
    results["info"].append(f"Recommended kernel length: {recommended_length} bins")

    # Check for potential TOF offset issues
    print("\n6. TOF OFFSET ANALYSIS:")

    # Create a test kernel to analyze peak position
    test_length = max(5000, recommended_length)
    kernel_test = np.zeros(test_length)
    n_frames = len(kernel)

    # Interpolated kernel (FOBI-style)
    for bin_float in frame_starts_bins:
        bin_floor = int(np.floor(bin_float))
        frac = bin_float - bin_floor
        if bin_floor < test_length:
            kernel_test[bin_floor] += (1.0 - frac) / n_frames
        if bin_floor + 1 < test_length:
            kernel_test[bin_floor + 1] += frac / n_frames

    # Find peak
    peak_idx = np.argmax(kernel_test)
    center = test_length // 2
    offset = peak_idx - center

    print(f"   Kernel length (test): {test_length} bins")
    print(f"   Center position: bin {center}")
    print(f"   Peak position: bin {peak_idx}")
    print(f"   Expected TOF offset: {offset} bins = {offset * bin_width} µs")

    if abs(offset) > 1000:
        results["warnings"].append(
            f"Large TOF offset expected: {offset} bins ({offset * bin_width} µs)"
        )
        print(f"   ⚠ WARNING: Large offset may cause alignment issues")

    # Check normalization
    print("\n7. KERNEL NORMALIZATION:")
    total_weight = np.sum(kernel_test)
    if not np.isclose(total_weight, 1.0, atol=1e-6):
        results["errors"].append(f"Kernel not normalized: sum = {total_weight}")
        print(f"   ❌ ERROR: Kernel sum = {total_weight} (should be 1.0)")
    else:
        print(f"   ✓ Kernel is normalized (sum = {total_weight:.6f})")

    # Check if angles match frequency
    if frequency and angles:
        print("\n8. CHOPPER PARAMETER VALIDATION:")
        expected_delays_s = np.array(angles) / (360.0 * frequency)
        expected_delays_ms = expected_delays_s * 1000
        calculated_cumulative = np.cumsum(kernel)

        if np.allclose(calculated_cumulative, expected_delays_ms, rtol=1e-4):
            print("   ✓ Kernel matches chopper parameters")
            print(f"     Formula: τₖ = θₖ / (360° × f)")
        else:
            results["warnings"].append(
                "Kernel doesn't match expected values from chopper parameters"
            )
            print("   ⚠ WARNING: Kernel doesn't match chopper parameters")
            print(f"     Expected (from angles/frequency): {expected_delays_ms}")
            print(f"     Actual (cumulative kernel): {calculated_cumulative}")

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY:")
    print("="*70)

    if results["errors"]:
        print(f"\n❌ ERRORS ({len(results['errors'])}):")
        for err in results["errors"]:
            print(f"   - {err}")

    if results["warnings"]:
        print(f"\n⚠ WARNINGS ({len(results['warnings'])}):")
        for warn in results["warnings"]:
            print(f"   - {warn}")

    if results["info"]:
        print(f"\nℹ INFO ({len(results['info'])}):")
        for info in results["info"]:
            print(f"   - {info}")

    if not results["errors"] and not results["warnings"]:
        print("\n✓ No issues detected. Parameters look good!")

    print("\n" + "="*70 + "\n")

    return results


def diagnose_from_chopper(angles, frequency, bin_width=10):
    """
    Diagnose TOF correction for chopper-based parameters.

    Parameters
    ----------
    angles : list
        Slit angles in degrees, e.g., [0, 45, 90, 180]
    frequency : float
        Chopper frequency in Hz
    bin_width : float
        Bin width in microseconds

    Returns
    -------
    dict
        Diagnostic results
    """
    # Calculate time delays from chopper parameters
    # Formula from article: τₖ = θₖ / (360° × f)
    time_delays_s = np.array(angles) / (360.0 * frequency)
    time_delays_ms = time_delays_s * 1000

    # Convert to frame-to-frame delays (kernel format)
    kernel = np.diff(np.concatenate([[0], time_delays_ms]))

    print(f"\nCalculated kernel from chopper parameters:")
    print(f"  Angles: {angles}°")
    print(f"  Frequency: {frequency} Hz")
    print(f"  → Kernel: {kernel} ms")

    return diagnose_parameters(kernel, bin_width, frequency, angles)


def main():
    """Run diagnostics on example parameter sets."""
    print("\n" + "#"*70)
    print("# TOF SHIFT CORRECTION DIAGNOSTIC TOOL")
    print("# Based on: Frame overlap Bragg edge imaging")
    print("# Nature Scientific Reports, vol 10, Article 14867 (2020)")
    print("#"*70)

    # Example 1: Simple 2-frame case
    print("\n\nEXAMPLE 1: Simple 2-frame case")
    diagnose_parameters(kernel=[0, 25], bin_width=10)

    # Example 2: 4-frame case from article
    print("\n\nEXAMPLE 2: 4-frame case from article")
    diagnose_parameters(kernel=[0, 12, 10, 25], bin_width=10)

    # Example 3: Fractional delays
    print("\n\nEXAMPLE 3: Fractional delays")
    diagnose_parameters(kernel=[0, 25.003], bin_width=10)

    # Example 4: POLDI chopper parameters
    print("\n\nEXAMPLE 4: POLDI chopper from article")
    angles = [0, 9.363, 21.475, 37.039, 50.417, 56.664, 67.422, 75.406]
    diagnose_from_chopper(angles, frequency=50, bin_width=10)

    # Example 5: Problematic case - very large delays
    print("\n\nEXAMPLE 5: Large delays (potential issue)")
    diagnose_parameters(kernel=[0, 100, 200, 300], bin_width=10)

    # Example 6: Invalid kernel (doesn't start at 0)
    print("\n\nEXAMPLE 6: Invalid kernel (for demonstration)")
    diagnose_parameters(kernel=[5, 12, 10, 25], bin_width=10)


if __name__ == '__main__':
    main()
