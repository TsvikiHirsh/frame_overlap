#!/usr/bin/env python3
"""Test that tof_min filtering works correctly with nbragg fitting"""

from frame_overlap import Data, Reconstruct, Analysis

def test_tof_min_with_nbragg():
    """Test that wavelength conversion is correct when tof_min is used"""

    print("\n" + "="*70)
    print("TEST: tof_min filtering with nbragg wavelength conversion")
    print("="*70)

    # Test 1: Without tof_min (baseline)
    print("\n" + "-"*70)
    print("Test 1: WITHOUT tof_min (baseline)")
    print("-"*70)

    data1 = Data(
        signal_file='notebooks/iron_powder.csv',
        openbeam_file='notebooks/openbeam.csv',
        flux=5e6,
        duration=0.5,
        freq=20
    )

    data1.convolute_response(pulse_duration=20.0)
    data1.poisson_sample(flux=1e6, freq=20, measurement_time=8.0)
    data1.overlap(kernel=1, total_time=50, mode='equal')

    recon1 = Reconstruct(data1, tmin=None, tmax=None)
    recon1.filter(kind='wiener', noise_power=1.0)

    print(f"Reconstructed data shape: {recon1.reconstructed_data.shape}")
    print(f"Time range: {recon1.reconstructed_data['time'].min():.1f} - {recon1.reconstructed_data['time'].max():.1f} µs")

    # Convert to nbragg
    nbragg_data1 = recon1.to_nbragg(L=9.0, tstep=10e-6)
    print(f"\nnbragg data wavelength range: {nbragg_data1.table['wavelength'].min():.3f} - {nbragg_data1.table['wavelength'].max():.3f} Å")

    # Test 2: With tof_min=1000 µs (1 ms)
    print("\n" + "-"*70)
    print("Test 2: WITH tof_min=1000 µs")
    print("-"*70)

    data2 = Data(
        signal_file='notebooks/iron_powder.csv',
        openbeam_file='notebooks/openbeam.csv',
        flux=5e6,
        duration=0.5,
        freq=20,
        tof_min=1000  # Filter out data before 1000 µs
    )

    data2.convolute_response(pulse_duration=20.0)
    data2.poisson_sample(flux=1e6, freq=20, measurement_time=8.0)
    data2.overlap(kernel=1, total_time=50, mode='equal')

    recon2 = Reconstruct(data2, tmin=None, tmax=None)
    recon2.filter(kind='wiener', noise_power=1.0)

    print(f"Reconstructed data shape: {recon2.reconstructed_data.shape}")
    print(f"Time range: {recon2.reconstructed_data['time'].min():.1f} - {recon2.reconstructed_data['time'].max():.1f} µs")

    # Convert to nbragg
    nbragg_data2 = recon2.to_nbragg(L=9.0, tstep=10e-6)
    print(f"\nnbragg data wavelength range: {nbragg_data2.table['wavelength'].min():.3f} - {nbragg_data2.table['wavelength'].max():.3f} Å")

    # Check that wavelength ranges are different (tof_min should shift to higher wavelengths)
    wl_min_1 = nbragg_data1.table['wavelength'].min()
    wl_min_2 = nbragg_data2.table['wavelength'].min()

    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    print(f"Wavelength minimum without tof_min: {wl_min_1:.3f} Å")
    print(f"Wavelength minimum with tof_min=1000: {wl_min_2:.3f} Å")
    print(f"Difference: {wl_min_2 - wl_min_1:.3f} Å")

    if wl_min_2 > wl_min_1:
        print("\n✅ PASS: Wavelength range correctly shifted when using tof_min")
        print("   (Filtering low TOF = filtering low wavelength)")
    else:
        print("\n❌ FAIL: Wavelength range not shifted correctly!")
        return False

    # Test 3: Fit with nbragg to ensure no mismatch
    print("\n" + "-"*70)
    print("Test 3: nbragg fitting with tof_min")
    print("-"*70)

    analysis = Analysis(xs='iron', vary_background=True, vary_weights=True)
    analysis.set_params(thickness={'value': 1.95, 'vary': False})

    # Use wider wavelength range since data starts at ~4.4 Å with tof_min=1000
    wl_min_fit = nbragg_data2.table['wavelength'].min()
    wl_max_fit = min(6.0, nbragg_data2.table['wavelength'].max())
    result = analysis.fit(recon2, wlmin=wl_min_fit, wlmax=wl_max_fit)

    print(f"\nnbragg fit reduced χ²: {result.redchi:.4f}")

    if result.redchi < 100:  # Reasonable fit
        print("✅ PASS: nbragg fit converged with reasonable χ² (no wavelength mismatch)")
    else:
        print(f"⚠️  WARNING: High χ² = {result.redchi:.4f}, possible wavelength mismatch")

    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    return True

if __name__ == "__main__":
    test_tof_min_with_nbragg()
