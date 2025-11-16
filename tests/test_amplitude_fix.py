#!/usr/bin/env python3
"""
Test that reconstructed signal has correct amplitude (scaled by n_frames).
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
    return tof_seconds * 1e6

def test_amplitude_scaling():
    """Test that reconstruction properly scales by n_frames"""
    print("=" * 70)
    print("Testing Reconstruction Amplitude Scaling")
    print("=" * 70)

    # Load and filter data
    signal_path = 'notebooks/iron_powder.csv'
    openbeam_path = 'notebooks/openbeam.csv'
    lambda_min = 1.0
    lambda_max = 10.0
    flight_path_m = 9.0

    tof_min_us = wavelength_to_tof(lambda_min, flight_path_m)
    tof_max_us = wavelength_to_tof(lambda_max, flight_path_m)

    print(f"\nWavelength range: {lambda_min} - {lambda_max} Å")

    # Load data
    data = Data(signal_path, openbeam_path, flux=5e6, duration=0.5, freq=20)

    # Apply wavelength filtering
    mask_signal = (data.data['time'] >= tof_min_us) & (data.data['time'] <= tof_max_us)
    data.data = data.data[mask_signal].copy()
    data.table = data.data

    mask_openbeam = (data.op_data['time'] >= tof_min_us) & (data.op_data['time'] <= tof_max_us)
    data.op_data = data.op_data[mask_openbeam].copy()
    data.openbeam_table = data.op_data

    print(f"Filtered to {len(data.data)} points")

    # Process pipeline
    data.convolute_response(200.0, bin_width=10)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)

    # Save reference BEFORE overlap
    reference_signal = data.poissoned_data['counts'].values.copy()
    reference_mean = reference_signal.mean()
    reference_max = reference_signal.max()

    print(f"\nReference (before overlap):")
    print(f"  Mean counts: {reference_mean:.2f}")
    print(f"  Max counts:  {reference_max:.2f}")

    # Apply overlap with 2 frames
    kernel = [0, 25]
    n_frames = len(kernel)
    data.overlap(kernel=kernel, total_time=50)

    print(f"\nAfter overlap ({n_frames} frames):")
    print(f"  Overlapped mean: {data.overlapped_data['counts'].mean():.2f}")

    # Reconstruct
    recon = Reconstruct(data, tmin=None, tmax=None)
    recon.filter(kind='wiener', noise_power=0.2)

    # Filter reconstructed data to wavelength range
    mask_recon = (recon.reconstructed_data['time'] >= tof_min_us) & \
                 (recon.reconstructed_data['time'] <= tof_max_us)
    recon.reconstructed_data = recon.reconstructed_data[mask_recon].copy()

    mask_recon_ob = (recon.reconstructed_openbeam['time'] >= tof_min_us) & \
                    (recon.reconstructed_openbeam['time'] <= tof_max_us)
    recon.reconstructed_openbeam = recon.reconstructed_openbeam[mask_recon_ob].copy()

    # Get reconstructed statistics
    recon_signal = recon.reconstructed_data['counts'].values
    recon_mean = recon_signal.mean()
    recon_max = recon_signal.max()

    print(f"\nReconstructed (after deconvolution):")
    print(f"  Mean counts: {recon_mean:.2f}")
    print(f"  Max counts:  {recon_max:.2f}")

    # Calculate ratios
    mean_ratio = recon_mean / reference_mean
    max_ratio = recon_max / reference_max

    print(f"\nAmplitude comparison:")
    print(f"  Reconstructed mean / Reference mean: {mean_ratio:.3f}")
    print(f"  Reconstructed max / Reference max:   {max_ratio:.3f}")
    print(f"  Expected ratio (should be ~1.0):     1.000")

    # Check transmission
    ref_transmission = reference_signal / data.op_poissoned_data['counts'].values
    recon_transmission = recon_signal / recon.reconstructed_openbeam['counts'].values

    ref_trans_mean = ref_transmission.mean()
    recon_trans_mean = recon_transmission.mean()

    print(f"\nTransmission comparison:")
    print(f"  Reference transmission mean:     {ref_trans_mean:.4f}")
    print(f"  Reconstructed transmission mean: {recon_trans_mean:.4f}")
    print(f"  Difference: {abs(ref_trans_mean - recon_trans_mean):.4f}")

    # Success criteria
    print(f"\n{'='*70}")
    print("VALIDATION")
    print("=" * 70)

    success = True

    # The reconstructed signal should have similar amplitude to reference
    # Allow 20% tolerance due to deconvolution artifacts
    if 0.8 <= mean_ratio <= 1.2:
        print(f"✅ PASS: Mean amplitude ratio is {mean_ratio:.3f} (within 0.8-1.2)")
    else:
        print(f"❌ FAIL: Mean amplitude ratio is {mean_ratio:.3f} (expected 0.8-1.2)")
        success = False

    if 0.8 <= max_ratio <= 1.2:
        print(f"✅ PASS: Max amplitude ratio is {max_ratio:.3f} (within 0.8-1.2)")
    else:
        print(f"❌ FAIL: Max amplitude ratio is {max_ratio:.3f} (expected 0.8-1.2)")
        success = False

    # Transmission should be very similar
    trans_diff = abs(ref_trans_mean - recon_trans_mean)
    if trans_diff < 0.05:
        print(f"✅ PASS: Transmission difference is {trans_diff:.4f} (< 0.05)")
    else:
        print(f"❌ FAIL: Transmission difference is {trans_diff:.4f} (expected < 0.05)")
        success = False

    return success

if __name__ == '__main__':
    try:
        success = test_amplitude_scaling()
        print(f"\n{'='*70}")
        if success:
            print("✅ ALL TESTS PASSED - Amplitude scaling is correct!")
        else:
            print("❌ SOME TESTS FAILED")
        print("=" * 70)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
