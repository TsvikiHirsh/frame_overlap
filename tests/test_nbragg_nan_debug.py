#!/usr/bin/env python3
"""Debug NaN error in nbragg fitting"""

import numpy as np
from frame_overlap import Data, Reconstruct, Analysis

def test_nbragg_nan():
    """Reproduce and debug the NaN error in nbragg fitting"""

    # Load data
    data = Data(
        signal_file='notebooks/iron_powder.csv',
        openbeam_file='notebooks/openbeam.csv',
        flux=5e6,
        duration=0.5,
        freq=20
    )

    # Process pipeline (matching the notebook)
    data.convolute_response(pulse_duration=20.0)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=8.0)  # 8 hours
    data.overlap(kernel=5, total_time=50, mode='random', kernel_seed=42)

    # Reconstruct
    recon = Reconstruct(data, tmin=None, tmax=None)
    recon.filter(kind='wiener', noise_power=1.0)

    print("="*70)
    print("Reconstruction statistics:")
    stats = recon.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("Checking reconstruction data for NaN/Inf:")
    time = recon.reconstructed_data['time'].values
    counts = recon.reconstructed_data['counts'].values
    err = recon.reconstructed_data['err'].values
    print(f"  time: min={np.min(time):.6f}, max={np.max(time):.6f}, has_nan={np.any(np.isnan(time))}, has_inf={np.any(np.isinf(time))}")
    print(f"  counts: min={np.min(counts):.6f}, max={np.max(counts):.6f}, has_nan={np.any(np.isnan(counts))}, has_inf={np.any(np.isinf(counts))}")
    print(f"  err: min={np.min(err):.6f}, max={np.max(err):.6f}, has_nan={np.any(np.isnan(err))}, has_inf={np.any(np.isinf(err))}")

    # Create Analysis object
    print("\n" + "="*70)
    print("Creating Analysis object...")
    analysis = Analysis(
        xs='iron_with_cellulose',
        vary_background=False,
        vary_response=False,
        vary_weights=True
    )

    print("Model parameters:")
    for param_name, param in analysis.model.params.items():
        print(f"  {param_name}: value={param.value}, vary={param.vary}")

    # Convert to nbragg format
    print("\n" + "="*70)
    print("Converting to nbragg format...")
    nbragg_data = recon.to_nbragg(L=9.0, tstep=10e-6)

    print("nbragg data table:")
    print(nbragg_data.table)

    print("\n" + "="*70)
    print("Checking nbragg data for NaN/Inf:")
    print(f"  wavelength: min={np.min(nbragg_data.table['wavelength']):.6f}, max={np.max(nbragg_data.table['wavelength']):.6f}, has_nan={np.any(np.isnan(nbragg_data.table['wavelength']))}, has_inf={np.any(np.isinf(nbragg_data.table['wavelength']))}")
    print(f"  trans: min={np.min(nbragg_data.table['trans']):.6f}, max={np.max(nbragg_data.table['trans']):.6f}, has_nan={np.any(np.isnan(nbragg_data.table['trans']))}, has_inf={np.any(np.isinf(nbragg_data.table['trans']))}")
    print(f"  err: min={np.min(nbragg_data.table['err']):.6f}, max={np.max(nbragg_data.table['err']):.6f}, has_nan={np.any(np.isnan(nbragg_data.table['err']))}, has_inf={np.any(np.isinf(nbragg_data.table['err']))}")

    # Filter wavelength range
    print("\n" + "="*70)
    print("Filtering wavelength range wlmin=1.0, wlmax=5.0...")
    mask = (nbragg_data.table['wavelength'] >= 1.0) & (nbragg_data.table['wavelength'] <= 5.0)
    print(f"Number of points in range: {np.sum(mask)} / {len(mask)}")

    filtered_data = nbragg_data.table[mask]
    print("\nFiltered data:")
    print(f"  wavelength: min={np.min(filtered_data['wavelength']):.6f}, max={np.max(filtered_data['wavelength']):.6f}, has_nan={np.any(np.isnan(filtered_data['wavelength']))}, has_inf={np.any(np.isinf(filtered_data['wavelength']))}")
    print(f"  trans: min={np.min(filtered_data['trans']):.6f}, max={np.max(filtered_data['trans']):.6f}, has_nan={np.any(np.isnan(filtered_data['trans']))}, has_inf={np.any(np.isinf(filtered_data['trans']))}")
    print(f"  err: min={np.min(filtered_data['err']):.6f}, max={np.max(filtered_data['err']):.6f}, has_nan={np.any(np.isnan(filtered_data['err']))}, has_inf={np.any(np.isinf(filtered_data['err']))}")

    # Try fitting
    print("\n" + "="*70)
    print("Attempting nbragg fit...")
    try:
        result = analysis.fit(recon, L=9.0, tstep=10e-6, wlmin=1.0, wlmax=5.0)
        print("✅ Fit succeeded!")
        print(f"  Reduced χ²: {result.redchi:.4f}")
    except Exception as e:
        print(f"❌ Fit failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nbragg_nan()
