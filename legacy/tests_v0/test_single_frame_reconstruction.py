"""
Test that single-frame reconstruction returns identity (same data back).

When using only one frame (no overlap), the reconstruction should return
exactly the same data because there's nothing to deconvolve.
"""

import sys
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct
import numpy as np


def test_single_frame_wiener():
    """Test that single-frame reconstruction with Wiener filter returns identity."""
    print("\n" + "="*70)
    print("TEST 1: Single-frame reconstruction with Wiener filter")
    print("="*70)

    # Load data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)

    # Apply convolution
    data.convolute_response(200, bin_width=10)

    # Apply Poisson sampling
    data.poisson_sample(flux=1e6, freq=20, measurement_time=30, seed=42)

    # Store the signal before overlap (this is what we should get back after reconstruction)
    reference_signal = data.poissoned_data['counts'].values.copy()
    reference_time = data.poissoned_data['time'].values.copy()

    print(f"\nBefore overlap:")
    print(f"  Reference signal (poissoned): {len(reference_signal)} points, mean={reference_signal.mean():.2f}")

    # Apply single-frame "overlap"
    data.overlap(kernel=[0], total_time=50)  # Single frame at t=0

    overlapped_signal = data.overlapped_data['counts'].values
    overlapped_time = data.overlapped_data['time'].values

    print(f"\nAfter overlap:")
    print(f"  Overlapped signal: {len(overlapped_signal)} points, mean={overlapped_signal.mean():.2f}")

    # The overlap extends the time range, so we need to compare only the overlapping region
    # Find common time range
    min_len = min(len(reference_signal), len(overlapped_signal))
    print(f"  Comparing first {min_len} points...")

    # Now reconstruct - should get back the reference signal
    recon = Reconstruct(data)
    recon.filter(kind='wiener', noise_power=0.01)

    reconstructed_signal = recon.reconstructed_data['counts'].values

    print(f"\nAfter reconstruction:")
    print(f"  Reconstructed signal: {len(reconstructed_signal)} points, mean={reconstructed_signal.mean():.2f}")

    # Compare reconstruction to reference (in the overlapping region)
    # The reconstructed signal should match the reference (poissoned) signal
    # We compare only the first min_len points
    ref_comp = reference_signal[:min_len]
    recon_comp = reconstructed_signal[:min_len]

    print(f"  Reference mean (first {min_len} pts): {ref_comp.mean():.2f}")
    print(f"  Reconstructed mean (first {min_len} pts): {recon_comp.mean():.2f}")

    # Calculate percent difference
    percent_diff = 100 * np.abs(ref_comp - recon_comp).mean() / ref_comp.mean()
    max_percent_diff = 100 * np.abs(ref_comp - recon_comp).max() / ref_comp.mean()

    print(f"  Mean percent difference: {percent_diff:.2f}%")
    print(f"  Max percent difference: {max_percent_diff:.2f}%")

    # Check chi2
    chi2_dof = recon.statistics['chi2_per_dof']
    print(f"  χ²/dof: {chi2_dof:.3f}")

    # For single frame, reconstruction should be near-perfect
    if percent_diff < 1.0:  # Less than 1% difference
        print(f"\n✓ TEST PASSED: Single-frame reconstruction is identity (diff < 1%)")
        return True
    else:
        print(f"\n✗ TEST FAILED: Single-frame reconstruction has {percent_diff:.2f}% difference")
        print(f"   Expected < 1%, got {percent_diff:.2f}%")
        return False


def test_single_frame_all_methods():
    """Test single-frame reconstruction with all methods."""
    print("\n" + "="*70)
    print("TEST 2: Single-frame reconstruction with all methods")
    print("="*70)

    methods = [
        ('wiener', {'noise_power': 0.01}),
        ('wiener_smooth', {'noise_power': 0.01, 'smooth_window': 5}),
        ('lucy', {'iterations': 20}),
        ('tikhonov', {'noise_power': 0.01})
    ]

    results = []

    for method_name, params in methods:
        # Load data fresh for each test
        data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                    flux=5e6, duration=0.5, freq=20)
        data.convolute_response(200, bin_width=10)
        data.poisson_sample(flux=1e6, freq=20, measurement_time=30, seed=42)

        reference_signal = data.poissoned_data['counts'].values.copy()

        # Single frame overlap
        data.overlap(kernel=[0], total_time=50)

        # Reconstruct
        recon = Reconstruct(data)
        recon.filter(kind=method_name, **params)

        reconstructed_signal = recon.reconstructed_data['counts'].values

        # Compare only overlapping region
        min_len = min(len(reference_signal), len(reconstructed_signal))
        ref_comp = reference_signal[:min_len]
        recon_comp = reconstructed_signal[:min_len]

        # Calculate difference
        percent_diff = 100 * np.abs(ref_comp - recon_comp).mean() / ref_comp.mean()
        chi2_dof = recon.statistics['chi2_per_dof']

        results.append((method_name, percent_diff, chi2_dof))

        print(f"\n{method_name}:")
        print(f"  Mean % diff: {percent_diff:.2f}%")
        print(f"  χ²/dof: {chi2_dof:.3f}")
        print(f"  Status: {'✓ PASS' if percent_diff < 1.0 else '✗ FAIL'}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    all_passed = True
    for method_name, percent_diff, chi2_dof in results:
        status = "✓ PASS" if percent_diff < 1.0 else "✗ FAIL"
        print(f"{method_name:20s}: {percent_diff:6.2f}% diff, χ²/dof={chi2_dof:6.2f} - {status}")
        if percent_diff >= 1.0:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing Single-Frame Reconstruction (Should Return Identity)")
    print("="*70)

    try:
        # Run tests
        test1_passed = test_single_frame_wiener()
        test2_passed = test_single_frame_all_methods()

        if test1_passed and test2_passed:
            print("\n" + "="*70)
            print("✅ ALL TESTS PASSED!")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("❌ SOME TESTS FAILED!")
            print("="*70)
            print("\nThe reconstruction should return the same signal for single-frame case.")
            print("This is a critical bug that needs to be fixed.")
            sys.exit(1)

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST EXECUTION FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
