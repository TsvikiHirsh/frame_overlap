#!/usr/bin/env python3
"""
Test that the Streamlit app wavelength filtering works end-to-end.
This simulates what happens when a user sets wavelength ranges in the UI.
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct, Analysis

# Constants from streamlit_app.py
PLANCK_CONSTANT = 6.62607015e-34  # J·s
NEUTRON_MASS_KG = 1.67492749804e-27  # kg

def wavelength_to_tof(wavelength_angstrom, flight_path_length_m):
    """Convert neutron wavelength to time-of-flight (in microseconds)"""
    wavelength_m = wavelength_angstrom * 1e-10
    velocity = PLANCK_CONSTANT / (NEUTRON_MASS_KG * wavelength_m)
    tof_seconds = flight_path_length_m / velocity
    return tof_seconds * 1e6  # microseconds

def test_streamlit_pipeline_with_wavelength_filter():
    """Test complete pipeline with wavelength filtering"""
    print("=" * 60)
    print("Streamlit Pipeline Test with Wavelength Filtering")
    print("=" * 60)

    # Simulate Streamlit app settings
    signal_path = 'notebooks/iron_powder.csv'
    openbeam_path = 'notebooks/openbeam.csv'
    flux_orig = 5e6
    duration_orig = 0.5
    freq_orig = 20

    # Wavelength range settings (DEFAULT VALUES)
    lambda_min = 1.0  # Å
    lambda_max = 10.0  # Å
    flight_path_m = 9.0

    print(f"\nSettings:")
    print(f"  Wavelength range: {lambda_min} - {lambda_max} Å")
    print(f"  Flight path: {flight_path_m} m")

    # Convert wavelength to TOF
    tof_min_us = wavelength_to_tof(lambda_min, flight_path_m)
    tof_max_us = wavelength_to_tof(lambda_max, flight_path_m)

    print(f"  TOF range: {tof_min_us/1000:.4f} - {tof_max_us/1000:.4f} ms")

    # Load data
    print(f"\n1. Loading data...")
    data = Data(signal_path, openbeam_path,
               flux=flux_orig, duration=duration_orig, freq=freq_orig)

    original_points = len(data.data)
    print(f"   Original data points: {original_points}")

    # Apply wavelength filtering
    print(f"\n2. Applying wavelength filter...")
    if data.data is not None and data.op_data is not None:
        # Filter signal data
        mask_signal = (data.data['time'] >= tof_min_us) & (data.data['time'] <= tof_max_us)
        data.data = data.data[mask_signal].copy()
        data.table = data.data

        # Filter openbeam data
        mask_openbeam = (data.op_data['time'] >= tof_min_us) & (data.op_data['time'] <= tof_max_us)
        data.op_data = data.op_data[mask_openbeam].copy()
        data.openbeam_table = data.op_data

        filtered_points = len(data.data)
        print(f"   Filtered data points: {filtered_points} (kept {100*filtered_points/original_points:.1f}%)")
        print(f"   ✓ Wavelength filtered: {lambda_min:.1f}-{lambda_max:.1f} Å")

    # Apply processing stages
    print(f"\n3. Processing pipeline...")

    # Convolution
    pulse_duration = 200.0
    bin_width = 10
    data.convolute_response(pulse_duration, bin_width=bin_width)
    print(f"   ✓ Convolved (pulse: {pulse_duration} µs)")

    # Poisson sampling
    flux_new = 1e6
    freq_new = 20
    measurement_time = 8 * 60  # 8 hours in minutes
    seed_poisson = 42
    data.poisson_sample(flux=flux_new, freq=freq_new,
                       measurement_time=measurement_time, seed=seed_poisson)
    print(f"   ✓ Poisson (flux: {flux_new:.1e})")

    # Frame overlap
    kernel = [0, 25]
    total_time = 50
    data.overlap(kernel=kernel, total_time=total_time)
    n_frames = len(kernel)
    print(f"   ✓ Overlap ({n_frames} frames)")

    # Reconstruction
    print(f"\n4. Reconstruction...")
    tmin = None
    tmax = None
    recon = Reconstruct(data, tmin=tmin, tmax=tmax)
    recon_method = 'wiener'
    noise_power = 0.2
    recon.filter(kind=recon_method, noise_power=noise_power)

    stats = recon.get_statistics()
    print(f"   ✓ Reconstructed (χ²/dof: {stats['chi2_per_dof']:.1f})")

    # Check if we have valid reconstructed data
    has_recon_data = recon.reconstructed_data is not None and len(recon.reconstructed_data) > 0
    print(f"   Reconstructed data points: {len(recon.reconstructed_data) if has_recon_data else 0}")

    # Try nbragg analysis if available
    print(f"\n5. Testing nbragg analysis (optional)...")
    try:
        from frame_overlap import Analysis

        analysis = Analysis(xs='iron', vary_background=True, vary_response=True)

        # Prepare nbragg data with cleaning
        nbragg_data = recon.to_nbragg(L=9.0, tstep=10e-6)
        nbragg_data.table = nbragg_data.table.dropna()
        nbragg_data.table = nbragg_data.table[~np.isinf(nbragg_data.table['trans'])]
        nbragg_data.table = nbragg_data.table[~np.isinf(nbragg_data.table['err'])]
        nbragg_data.table = nbragg_data.table[nbragg_data.table['err'] > 0]

        result = analysis.model.fit(nbragg_data)

        if hasattr(result, 'redchi') and not np.isnan(result.redchi):
            print(f"   ✓ nbragg fit (χ²/dof: {result.redchi:.2f})")
            print(f"   nbragg data points used: {len(nbragg_data.table)}")
        else:
            print(f"   ⚠️ nbragg fit produced invalid results")

    except ImportError:
        print(f"   ⚠️ nbragg not available (optional feature)")
    except Exception as e:
        print(f"   ⚠️ nbragg fit failed: {e}")

    print(f"\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)

    return True

if __name__ == '__main__':
    try:
        success = test_streamlit_pipeline_with_wavelength_filter()
        if success:
            print("\n✅ Streamlit wavelength filtering test passed!")
            sys.exit(0)
        else:
            print("\n❌ Streamlit wavelength filtering test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
