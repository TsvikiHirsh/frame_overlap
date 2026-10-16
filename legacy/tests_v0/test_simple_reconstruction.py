#!/usr/bin/env python3
"""Test simple reconstruction with equal spacing"""

import numpy as np
from frame_overlap import Data, Reconstruct, Analysis

def test_simple_reconstruction():
    """Test with equal spacing kernels which should work better"""

    # Load data
    data = Data(
        signal_file='notebooks/iron_powder.csv',
        openbeam_file='notebooks/openbeam.csv',
        flux=5e6,
        duration=0.5,
        freq=20
    )

    # Process pipeline
    data.convolute_response(pulse_duration=20.0)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=8.0)  # 8 hours
    data.overlap(kernel=5, total_time=50, mode='equal')  # Equal spacing should work better

    print(f"Generated equal kernel: {data.kernel}")

    # Reconstruct
    recon = Reconstruct(data, tmin=None, tmax=None)
    recon.filter(kind='wiener', noise_power=1.0)

    print("\n" + "="*70)
    print("Reconstruction statistics:")
    stats = recon.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Convert to nbragg
    print("\n" + "="*70)
    print("Converting to nbragg format...")
    nbragg_data = recon.to_nbragg(L=9.0, tstep=10e-6)

    print("\nChecking for NaN/Inf in nbragg data:")
    print(f"  trans: min={np.min(nbragg_data.table['trans']):.6f}, max={np.max(nbragg_data.table['trans']):.6f}, has_nan={np.any(np.isnan(nbragg_data.table['trans']))}, has_inf={np.any(np.isinf(nbragg_data.table['trans']))}")
    print(f"  err: min={np.min(nbragg_data.table['err']):.6f}, max={np.max(nbragg_data.table['err']):.6f}, has_nan={np.any(np.isnan(nbragg_data.table['err']))}, has_inf={np.any(np.isinf(nbragg_data.table['err']))}")

    # Try nbragg fit if no inf
    if not np.any(np.isinf(nbragg_data.table['trans'])):
        print("\n" + "="*70)
        print("Attempting nbragg fit...")
        analysis = Analysis(
            xs='iron_with_cellulose',
            vary_background=False,
            vary_response=False,
            vary_weights=True
        )

        try:
            result = analysis.fit(recon, L=9.0, tstep=10e-6, wlmin=1.0, wlmax=5.0)
            print("✅ Fit succeeded!")
            print(f"  Reduced χ²: {result.redchi:.4f}")
        except Exception as e:
            print(f"❌ Fit failed: {e}")
    else:
        print("\n⚠️  Inf values detected - skipping nbragg fit")

if __name__ == "__main__":
    test_simple_reconstruction()
