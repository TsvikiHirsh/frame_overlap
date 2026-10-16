#!/usr/bin/env python3
"""Debug TOF shift issue with tof_min filtering and single-frame reconstruction"""

import numpy as np
import matplotlib.pyplot as plt
from frame_overlap import Data, Reconstruct

def test_tof_shift():
    """Reproduce the TOF shift issue with tof_min=1000"""

    # Load data with tof_min filtering
    data = Data(
        signal_file='notebooks/iron_powder.csv',
        openbeam_file='notebooks/openbeam.csv',
        flux=5e6,
        duration=0.5,
        freq=20,
        tof_min=1000  # Filter to times >= 1000 µs
    )

    print("="*70)
    print("After loading with tof_min=1000:")
    print(f"  data.data time range: {data.data['time'].min():.1f} - {data.data['time'].max():.1f} µs")

    # Convolve
    data.convolute_response(pulse_duration=20.0)
    print(f"  convolved_data time range: {data.convolved_data['time'].min():.1f} - {data.convolved_data['time'].max():.1f} µs")

    # Poisson sample
    data.poisson_sample(flux=1e6, freq=20, measurement_time=24.0)
    print(f"  poissoned_data time range: {data.poissoned_data['time'].min():.1f} - {data.poissoned_data['time'].max():.1f} µs")

    # Overlap with single frame
    data.overlap(kernel=1, total_time=50, mode='blue_noise')
    print(f"  overlapped_data time range: {data.table['time'].min():.1f} - {data.table['time'].max():.1f} µs")
    print(f"  Kernel: {data.kernel}")

    # Reconstruct
    recon = Reconstruct(data, tmin=5, tmax=None)
    recon.filter(kind='wiener', noise_power=1.0)

    print("\n" + "="*70)
    print("After reconstruction:")
    print(f"  reference_data time range: {recon.reference_data['time'].min():.1f} - {recon.reference_data['time'].max():.1f} µs")
    print(f"  reconstructed_data time range: {recon.reconstructed_data['time'].min():.1f} - {recon.reconstructed_data['time'].max():.1f} µs")

    # Check if they align
    time_diff = recon.reconstructed_data['time'].values[0] - recon.reference_data['time'].values[0]
    print(f"\n  Time shift: {time_diff:.1f} µs")

    if abs(time_diff) > 1:
        print("  ❌ PROBLEM: Reconstructed and reference data are shifted!")
    else:
        print("  ✅ OK: Time axes are aligned")

    # Print first few time points
    print("\n" + "="*70)
    print("First 5 time points comparison:")
    print(f"  Reference:     {recon.reference_data['time'].values[:5]}")
    print(f"  Reconstructed: {recon.reconstructed_data['time'].values[:5]}")

if __name__ == "__main__":
    test_tof_shift()
