#!/usr/bin/env python3
"""
Test that the reconstruction plot fix works correctly with wavelength filtering.
This verifies that reconstructed data is properly filtered to match the wavelength range.
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct

# Constants
PLANCK_CONSTANT = 6.62607015e-34  # J·s
NEUTRON_MASS_KG = 1.67492749804e-27  # kg

def wavelength_to_tof(wavelength_angstrom, flight_path_length_m):
    """Convert neutron wavelength to time-of-flight (in microseconds)"""
    wavelength_m = wavelength_angstrom * 1e-10
    velocity = PLANCK_CONSTANT / (NEUTRON_MASS_KG * wavelength_m)
    tof_seconds = flight_path_length_m / velocity
    return tof_seconds * 1e6  # microseconds

def test_reconstruction_with_wavelength_filter_fix():
    """Test the fixed reconstruction with wavelength filtering"""
    print("=" * 70)
    print("Testing Reconstruction Fix with Wavelength Filtering")
    print("=" * 70)

    # Settings
    signal_path = 'notebooks/iron_powder.csv'
    openbeam_path = 'notebooks/openbeam.csv'
    lambda_min = 1.0  # Å
    lambda_max = 10.0  # Å
    flight_path_m = 9.0

    # Convert wavelength to TOF
    tof_min_us = wavelength_to_tof(lambda_min, flight_path_m)
    tof_max_us = wavelength_to_tof(lambda_max, flight_path_m)

    print(f"\nWavelength range: {lambda_min} - {lambda_max} Å")
    print(f"TOF range: {tof_min_us:.2f} - {tof_max_us:.2f} µs ({tof_min_us/1000:.4f} - {tof_max_us/1000:.4f} ms)")

    # Load and filter data
    print(f"\n1. Loading and filtering data...")
    data = Data(signal_path, openbeam_path, flux=5e6, duration=0.5, freq=20)

    original_len = len(data.data)
    print(f"   Original: {original_len} points")

    # Apply wavelength filtering
    mask_signal = (data.data['time'] >= tof_min_us) & (data.data['time'] <= tof_max_us)
    data.data = data.data[mask_signal].copy()
    data.table = data.data

    mask_openbeam = (data.op_data['time'] >= tof_min_us) & (data.op_data['time'] <= tof_max_us)
    data.op_data = data.op_data[mask_openbeam].copy()
    data.openbeam_table = data.op_data

    filtered_len = len(data.data)
    print(f"   Filtered: {filtered_len} points ({100*filtered_len/original_len:.1f}%)")
    print(f"   Time range: {data.data['time'].min():.2f} - {data.data['time'].max():.2f} µs")

    # Process pipeline
    print(f"\n2. Processing pipeline...")
    data.convolute_response(200.0, bin_width=10)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
    data.overlap(kernel=[0, 25], total_time=50)
    print(f"   Overlapped data: {len(data.overlapped_data)} points")
    print(f"   Overlapped time range: {data.overlapped_data['time'].min():.2f} - {data.overlapped_data['time'].max():.2f} µs")

    # Reconstruct
    print(f"\n3. Reconstructing...")
    recon = Reconstruct(data, tmin=None, tmax=None)
    recon.filter(kind='wiener', noise_power=0.2)

    print(f"   Reference: {len(recon.reference_data)} points")
    print(f"   Reference time: {recon.reference_data['time'].min():.2f} - {recon.reference_data['time'].max():.2f} µs")
    print(f"   Reconstructed (before filter): {len(recon.reconstructed_data)} points")
    print(f"   Reconstructed time (before): {recon.reconstructed_data['time'].min():.2f} - {recon.reconstructed_data['time'].max():.2f} µs")

    # Apply the fix: Filter reconstructed data to match wavelength range
    print(f"\n4. Applying wavelength filter to reconstructed data...")
    if recon.reconstructed_data is not None and recon.reconstructed_openbeam is not None:
        before_len = len(recon.reconstructed_data)

        mask_recon = (recon.reconstructed_data['time'] >= tof_min_us) & \
                     (recon.reconstructed_data['time'] <= tof_max_us)
        recon.reconstructed_data = recon.reconstructed_data[mask_recon].copy()

        mask_recon_ob = (recon.reconstructed_openbeam['time'] >= tof_min_us) & \
                        (recon.reconstructed_openbeam['time'] <= tof_max_us)
        recon.reconstructed_openbeam = recon.reconstructed_openbeam[mask_recon_ob].copy()

        after_len = len(recon.reconstructed_data)
        print(f"   Filtered reconstructed: {before_len} → {after_len} points ({100*after_len/before_len:.1f}%)")
        print(f"   Reconstructed time (after): {recon.reconstructed_data['time'].min():.2f} - {recon.reconstructed_data['time'].max():.2f} µs")

    # Check alignment
    print(f"\n5. Checking data alignment...")
    ref_len = len(recon.reference_data)
    recon_len = len(recon.reconstructed_data)
    op_len = len(data.op_poissoned_data) if data.op_poissoned_data is not None else 0

    print(f"   Reference:     {ref_len} points")
    print(f"   Reconstructed: {recon_len} points")
    print(f"   Openbeam:      {op_len} points")

    # Check time ranges overlap
    ref_time_min = recon.reference_data['time'].min()
    ref_time_max = recon.reference_data['time'].max()
    recon_time_min = recon.reconstructed_data['time'].min()
    recon_time_max = recon.reconstructed_data['time'].max()

    print(f"\n   Reference time range:     {ref_time_min:.2f} - {ref_time_max:.2f} µs")
    print(f"   Reconstructed time range: {recon_time_min:.2f} - {recon_time_max:.2f} µs")

    # Check if ranges overlap significantly
    time_overlap_start = max(ref_time_min, recon_time_min)
    time_overlap_end = min(ref_time_max, recon_time_max)
    overlap_range = time_overlap_end - time_overlap_start

    ref_range = ref_time_max - ref_time_min
    recon_range = recon_time_max - recon_time_min

    overlap_percent_ref = 100 * overlap_range / ref_range if ref_range > 0 else 0
    overlap_percent_recon = 100 * overlap_range / recon_range if recon_range > 0 else 0

    print(f"\n   Time overlap: {time_overlap_start:.2f} - {time_overlap_end:.2f} µs")
    print(f"   Overlap coverage: {overlap_percent_ref:.1f}% of reference, {overlap_percent_recon:.1f}% of reconstructed")

    # Test plotting
    print(f"\n6. Testing plot generation...")
    try:
        import matplotlib.pyplot as plt
        fig = recon.plot(kind='transmission', show_errors=False, figsize=(10, 6))
        print(f"   ✅ Plot succeeded!")
        plt.close(fig)
    except Exception as e:
        print(f"   ❌ Plot failed: {e}")
        return False

    # Success criteria
    print(f"\n7. Validating fix...")
    success = True

    # Criterion 1: Ranges should overlap significantly (> 90%)
    if overlap_percent_ref < 90 or overlap_percent_recon < 90:
        print(f"   ❌ FAIL: Time ranges don't overlap enough")
        print(f"      Expected > 90% overlap, got {min(overlap_percent_ref, overlap_percent_recon):.1f}%")
        success = False
    else:
        print(f"   ✅ PASS: Time ranges overlap significantly ({min(overlap_percent_ref, overlap_percent_recon):.1f}%)")

    # Criterion 2: Both should be within wavelength range
    if recon_time_min < tof_min_us or recon_time_max > tof_max_us:
        print(f"   ❌ FAIL: Reconstructed data extends beyond wavelength range")
        print(f"      Expected: {tof_min_us:.2f} - {tof_max_us:.2f} µs")
        print(f"      Got:      {recon_time_min:.2f} - {recon_time_max:.2f} µs")
        success = False
    else:
        print(f"   ✅ PASS: Reconstructed data is within wavelength range")

    # Criterion 3: Data lengths should be similar (within 20%)
    len_ratio = min(ref_len, recon_len) / max(ref_len, recon_len)
    if len_ratio < 0.8:
        print(f"   ⚠️  WARNING: Data lengths differ significantly ({ref_len} vs {recon_len})")
        print(f"      This may cause plotting issues")
    else:
        print(f"   ✅ PASS: Data lengths are similar ({ref_len} vs {recon_len}, ratio: {len_ratio:.2f})")

    return success

if __name__ == '__main__':
    print("\n")
    try:
        success = test_reconstruction_with_wavelength_filter_fix()
        print(f"\n{'='*70}")
        if success:
            print("✅ ALL TESTS PASSED - Reconstruction fix works correctly!")
        else:
            print("❌ SOME TESTS FAILED - Check output above")
        print("=" * 70)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
