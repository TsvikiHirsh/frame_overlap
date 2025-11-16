#!/usr/bin/env python3
"""
Debug script to investigate amplitude scaling in reconstruction plots.
This simulates what the Streamlit app does and checks the actual plot data.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct

# Constants
PLANCK_CONSTANT = 6.62607015e-34
NEUTRON_MASS_KG = 1.67492749804e-27

def wavelength_to_tof(wavelength_angstrom, flight_path_length_m):
    wavelength_m = wavelength_angstrom * 1e-10
    velocity = PLANCK_CONSTANT / (NEUTRON_MASS_KG * wavelength_m)
    tof_seconds = flight_path_length_m / velocity
    return tof_seconds * 1e6

def debug_plot_amplitude():
    print("=" * 70)
    print("Debug: Reconstruction Plot Amplitude Scaling")
    print("=" * 70)

    # Simulate Streamlit app settings
    signal_path = 'notebooks/iron_powder.csv'
    openbeam_path = 'notebooks/openbeam.csv'
    lambda_min = 1.0
    lambda_max = 10.0
    flight_path_m = 9.0

    tof_min_us = wavelength_to_tof(lambda_min, flight_path_m)
    tof_max_us = wavelength_to_tof(lambda_max, flight_path_m)

    print(f"\nSettings:")
    print(f"  Wavelength range: {lambda_min} - {lambda_max} Å")
    print(f"  TOF range: {tof_min_us/1000:.2f} - {tof_max_us/1000:.2f} ms")

    # Load and filter data
    data = Data(signal_path, openbeam_path, flux=5e6, duration=0.5, freq=20)

    mask_signal = (data.data['time'] >= tof_min_us) & (data.data['time'] <= tof_max_us)
    data.data = data.data[mask_signal].copy()
    data.table = data.data

    mask_openbeam = (data.op_data['time'] >= tof_min_us) & (data.op_data['time'] <= tof_max_us)
    data.op_data = data.op_data[mask_openbeam].copy()
    data.openbeam_table = data.op_data

    print(f"\n1. After wavelength filtering:")
    print(f"   Data points: {len(data.data)}")

    # Process pipeline
    data.convolute_response(200.0, bin_width=10)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)

    # Store reference before overlap
    reference_data = data.poissoned_data.copy()
    reference_openbeam = data.op_poissoned_data.copy()

    print(f"\n2. Before overlap (reference):")
    print(f"   Signal mean: {reference_data['counts'].mean():.2f}")
    print(f"   Signal max: {reference_data['counts'].max():.2f}")
    print(f"   Openbeam mean: {reference_openbeam['counts'].mean():.2f}")

    # Calculate reference transmission
    ref_trans = reference_data['counts'] / reference_openbeam['counts']
    print(f"   Transmission mean: {ref_trans.mean():.4f}")
    print(f"   Transmission range: [{ref_trans.min():.4f}, {ref_trans.max():.4f}]")

    # Apply overlap
    kernel = [0, 25]
    n_frames = len(kernel)
    data.overlap(kernel=kernel, total_time=50)

    print(f"\n3. After overlap ({n_frames} frames):")
    print(f"   Overlapped signal mean: {data.overlapped_data['counts'].mean():.2f}")
    print(f"   Overlapped openbeam mean: {data.op_overlapped_data['counts'].mean():.2f}")

    # Reconstruct
    recon = Reconstruct(data, tmin=None, tmax=None)
    recon.filter(kind='wiener', noise_power=0.2)

    print(f"\n4. Reference data in Reconstruct (after filter):")
    print(f"   Reference signal mean: {recon.reference_data['counts'].mean():.2f}")
    print(f"   Reference signal max: {recon.reference_data['counts'].max():.2f}")

    print(f"\n5. After reconstruction (before wavelength filter):")
    print(f"   Reconstructed signal points: {len(recon.reconstructed_data)}")
    print(f"   Reconstructed signal mean: {recon.reconstructed_data['counts'].mean():.2f}")
    print(f"   Reconstructed signal max: {recon.reconstructed_data['counts'].max():.2f}")
    print(f"   Reconstructed openbeam mean: {recon.reconstructed_openbeam['counts'].mean():.2f}")

    # Filter reconstructed data to wavelength range (like Streamlit app does)
    mask_recon = (recon.reconstructed_data['time'] >= tof_min_us) & \
                 (recon.reconstructed_data['time'] <= tof_max_us)
    recon.reconstructed_data = recon.reconstructed_data[mask_recon].copy()

    mask_recon_ob = (recon.reconstructed_openbeam['time'] >= tof_min_us) & \
                    (recon.reconstructed_openbeam['time'] <= tof_max_us)
    recon.reconstructed_openbeam = recon.reconstructed_openbeam[mask_recon_ob].copy()

    print(f"\n6. After wavelength filtering reconstructed data:")
    print(f"   Reconstructed signal points: {len(recon.reconstructed_data)}")
    print(f"   Reconstructed signal mean: {recon.reconstructed_data['counts'].mean():.2f}")
    print(f"   Reconstructed signal max: {recon.reconstructed_data['counts'].max():.2f}")
    print(f"   Reconstructed openbeam mean: {recon.reconstructed_openbeam['counts'].mean():.2f}")

    # Calculate reconstructed transmission
    recon_trans = recon.reconstructed_data['counts'] / recon.reconstructed_openbeam['counts']
    print(f"   Transmission mean: {recon_trans.mean():.4f}")
    print(f"   Transmission range: [{recon_trans.min():.4f}, {recon_trans.max():.4f}]")

    # Now simulate what plot() does
    print(f"\n7. What recon.plot() will show:")

    # Get openbeam for transmission calculation (from plot method logic)
    if data.op_poissoned_data is not None:
        ref_openbeam_data = data.op_poissoned_data
    else:
        ref_openbeam_data = data.op_overlapped_data

    # Match lengths (what plot does)
    min_len = min(len(recon.reference_data), len(recon.reconstructed_data), len(ref_openbeam_data))
    print(f"   min_len for plot: {min_len}")

    # Calculate transmissions (what plot does)
    ref_signal = recon.reference_data['counts'].values[:min_len]
    recon_signal = recon.reconstructed_data['counts'].values[:min_len]
    ref_ob = ref_openbeam_data['counts'].values[:min_len]

    plot_ref_trans = ref_signal / np.maximum(ref_ob, 1)
    plot_recon_trans = recon_signal / np.maximum(ref_ob, 1)

    print(f"\n   Plot reference transmission:")
    print(f"     Mean: {plot_ref_trans.mean():.4f}")
    print(f"     Range: [{plot_ref_trans.min():.4f}, {plot_ref_trans.max():.4f}]")

    print(f"\n   Plot reconstructed transmission:")
    print(f"     Mean: {plot_recon_trans.mean():.4f}")
    print(f"     Range: [{plot_recon_trans.min():.4f}, {plot_recon_trans.max():.4f}]")

    # Check amplitude ratios
    print(f"\n8. Amplitude ratios:")
    signal_ratio = recon_signal.mean() / ref_signal.mean()
    print(f"   Reconstructed signal / Reference signal: {signal_ratio:.3f}")
    print(f"   Expected: ~1.0")

    trans_diff = abs(plot_recon_trans.mean() - plot_ref_trans.mean())
    print(f"   Transmission difference: {trans_diff:.6f}")
    print(f"   Expected: ~0.0")

    # Create actual plot to verify visually
    print(f"\n9. Creating plot...")
    fig = recon.plot(kind='transmission', show_errors=False, figsize=(12, 6))
    fig.savefig('/tmp/debug_reconstruction_amplitude.png', dpi=100, bbox_inches='tight')
    print(f"   Saved to: /tmp/debug_reconstruction_amplitude.png")
    plt.close(fig)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    if 0.9 <= signal_ratio <= 1.1:
        print(f"✅ Signal amplitude scaling looks correct (ratio: {signal_ratio:.3f})")
    else:
        print(f"❌ Signal amplitude scaling issue (ratio: {signal_ratio:.3f}, expected ~1.0)")
        print(f"   This means reconstructed signal is {signal_ratio:.1f}x the reference")

    if trans_diff < 0.01:
        print(f"✅ Transmission is correctly preserved (diff: {trans_diff:.6f})")
    else:
        print(f"⚠️  Transmission has small difference (diff: {trans_diff:.6f})")

if __name__ == '__main__':
    debug_plot_amplitude()
