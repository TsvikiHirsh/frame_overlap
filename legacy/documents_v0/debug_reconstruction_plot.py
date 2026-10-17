#!/usr/bin/env python3
"""
Debug script to investigate reconstruction plot issues with wavelength filtering.
This will show what happens to data dimensions at each stage.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct

# Constants for wavelength conversion
PLANCK_CONSTANT = 6.62607015e-34  # J·s
NEUTRON_MASS_KG = 1.67492749804e-27  # kg

def wavelength_to_tof(wavelength_angstrom, flight_path_length_m):
    """Convert neutron wavelength to time-of-flight (in microseconds)"""
    wavelength_m = wavelength_angstrom * 1e-10
    velocity = PLANCK_CONSTANT / (NEUTRON_MASS_KG * wavelength_m)
    tof_seconds = flight_path_length_m / velocity
    return tof_seconds * 1e6  # microseconds

def debug_reconstruction_with_wavelength_filter():
    """Debug the reconstruction plot with wavelength filtering"""
    print("=" * 70)
    print("Debugging Reconstruction Plot with Wavelength Filtering")
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

    print(f"\nWavelength filtering:")
    print(f"  Range: {lambda_min} - {lambda_max} Å")
    print(f"  TOF range: {tof_min_us:.2f} - {tof_max_us:.2f} µs")
    print(f"  TOF range: {tof_min_us/1000:.4f} - {tof_max_us/1000:.4f} ms")

    # Load data WITHOUT filtering first
    print(f"\n{'='*70}")
    print("SCENARIO 1: Without Wavelength Filtering")
    print("=" * 70)

    data1 = Data(signal_path, openbeam_path, flux=5e6, duration=0.5, freq=20)
    print(f"\n1. After loading:")
    print(f"   data.data: {len(data1.data)} points")
    print(f"   data.op_data: {len(data1.op_data)} points")
    print(f"   Time range: {data1.data['time'].min():.2f} - {data1.data['time'].max():.2f} µs")

    # Process
    data1.convolute_response(200.0, bin_width=10)
    print(f"\n2. After convolution:")
    print(f"   convolved_data: {len(data1.convolved_data)} points")

    data1.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
    print(f"\n3. After Poisson sampling:")
    print(f"   poissoned_data: {len(data1.poissoned_data)} points")

    data1.overlap(kernel=[0, 25], total_time=50)
    print(f"\n4. After overlap:")
    print(f"   overlapped_data: {len(data1.overlapped_data)} points")

    # Reconstruct
    recon1 = Reconstruct(data1, tmin=None, tmax=None)
    print(f"\n5. Reconstruct object created:")
    print(f"   reference_data: {len(recon1.reference_data) if recon1.reference_data is not None else 'None'} points")

    recon1.filter(kind='wiener', noise_power=0.2)
    print(f"\n6. After reconstruction:")
    print(f"   reconstructed_data: {len(recon1.reconstructed_data)} points")
    print(f"   reconstructed_openbeam: {len(recon1.reconstructed_openbeam)} points")
    print(f"   reference_data: {len(recon1.reference_data)} points")

    # Try to plot
    print(f"\n7. Attempting to plot...")
    try:
        fig1 = recon1.plot(kind='transmission', show_errors=False, figsize=(10, 6))
        print(f"   ✅ Plot succeeded!")
        plt.close(fig1)
    except Exception as e:
        print(f"   ❌ Plot failed: {e}")
        import traceback
        traceback.print_exc()

    # Now with wavelength filtering
    print(f"\n{'='*70}")
    print("SCENARIO 2: WITH Wavelength Filtering")
    print("=" * 70)

    data2 = Data(signal_path, openbeam_path, flux=5e6, duration=0.5, freq=20)
    print(f"\n1. After loading:")
    print(f"   data.data: {len(data2.data)} points")
    print(f"   data.op_data: {len(data2.op_data)} points")

    # Apply wavelength filtering
    print(f"\n2. Applying wavelength filter...")
    mask_signal = (data2.data['time'] >= tof_min_us) & (data2.data['time'] <= tof_max_us)
    data2.data = data2.data[mask_signal].copy()
    data2.table = data2.data

    mask_openbeam = (data2.op_data['time'] >= tof_min_us) & (data2.op_data['time'] <= tof_max_us)
    data2.op_data = data2.op_data[mask_openbeam].copy()
    data2.openbeam_table = data2.op_data

    print(f"   Filtered data.data: {len(data2.data)} points ({100*len(data2.data)/2400:.1f}%)")
    print(f"   Filtered data.op_data: {len(data2.op_data)} points ({100*len(data2.op_data)/2400:.1f}%)")
    print(f"   Time range: {data2.data['time'].min():.2f} - {data2.data['time'].max():.2f} µs")

    # Process
    data2.convolute_response(200.0, bin_width=10)
    print(f"\n3. After convolution:")
    print(f"   convolved_data: {len(data2.convolved_data)} points")
    print(f"   Time range: {data2.convolved_data['time'].min():.2f} - {data2.convolved_data['time'].max():.2f} µs")

    data2.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
    print(f"\n4. After Poisson sampling:")
    print(f"   poissoned_data: {len(data2.poissoned_data)} points")
    print(f"   Time range: {data2.poissoned_data['time'].min():.2f} - {data2.poissoned_data['time'].max():.2f} µs")

    data2.overlap(kernel=[0, 25], total_time=50)
    print(f"\n5. After overlap:")
    print(f"   overlapped_data: {len(data2.overlapped_data)} points")
    print(f"   Time range: {data2.overlapped_data['time'].min():.2f} - {data2.overlapped_data['time'].max():.2f} µs")

    # Reconstruct
    recon2 = Reconstruct(data2, tmin=None, tmax=None)
    print(f"\n6. Reconstruct object created:")
    print(f"   reference_data: {len(recon2.reference_data) if recon2.reference_data is not None else 'None'} points")
    if recon2.reference_data is not None:
        print(f"   Reference time range: {recon2.reference_data['time'].min():.2f} - {recon2.reference_data['time'].max():.2f} µs")

    recon2.filter(kind='wiener', noise_power=0.2)
    print(f"\n7. After reconstruction:")
    print(f"   reconstructed_data: {len(recon2.reconstructed_data)} points")
    print(f"   reconstructed_openbeam: {len(recon2.reconstructed_openbeam)} points")
    print(f"   reference_data: {len(recon2.reference_data)} points")

    if recon2.reconstructed_data is not None:
        print(f"   Reconstructed time range: {recon2.reconstructed_data['time'].min():.2f} - {recon2.reconstructed_data['time'].max():.2f} µs")

    # Check for dimension mismatch
    print(f"\n8. Checking dimensions for plotting:")
    if recon2.reference_data is not None and recon2.reconstructed_data is not None:
        ref_len = len(recon2.reference_data)
        recon_len = len(recon2.reconstructed_data)
        openbeam_len = len(data2.op_poissoned_data) if data2.op_poissoned_data is not None else 0

        print(f"   reference_data:         {ref_len} points")
        print(f"   reconstructed_data:     {recon_len} points")
        print(f"   op_poissoned_data:      {openbeam_len} points")
        print(f"   min_len (for plotting): {min(ref_len, recon_len, openbeam_len)}")

        if ref_len != recon_len:
            print(f"   ⚠️  WARNING: Dimension mismatch!")
            print(f"   Reference has {ref_len} points but reconstruction has {recon_len} points")

    # Try to plot
    print(f"\n9. Attempting to plot...")
    try:
        fig2 = recon2.plot(kind='transmission', show_errors=False, figsize=(10, 6))
        print(f"   ✅ Plot succeeded!")

        # Save the plot for inspection
        fig2.savefig('/tmp/reconstruction_with_wavelength_filter.png', dpi=100, bbox_inches='tight')
        print(f"   Saved to: /tmp/reconstruction_with_wavelength_filter.png")
        plt.close(fig2)

    except Exception as e:
        print(f"   ❌ Plot failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print("ANALYSIS")
    print("=" * 70)

    print(f"\nKey observations:")
    print(f"1. Without filter: reference and reconstructed have same length")
    print(f"2. With filter: dimensions may mismatch if reconstruction creates fixed grid")
    print(f"3. The issue is likely in how the reconstruction creates the time grid")
    print(f"4. Reconstruction should respect the filtered time range")

if __name__ == '__main__':
    debug_reconstruction_with_wavelength_filter()
