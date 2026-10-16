"""
Final test to verify nbragg integration works with NaN cleaning.
This tests the exact code path used in the Streamlit app.
"""

import sys
sys.path.insert(0, 'src')
import numpy as np
import matplotlib.pyplot as plt

from frame_overlap import Data, Reconstruct, Analysis

# Neutron constants
PLANCK_CONSTANT = 6.62607015e-34  # J·s
NEUTRON_MASS_KG = 1.67492749804e-27  # kg

def wavelength_to_tof(wavelength_angstrom, flight_path_length_m):
    """Convert neutron wavelength to time-of-flight (matches streamlit_app.py)."""
    wavelength_m = wavelength_angstrom * 1e-10
    velocity = PLANCK_CONSTANT / (NEUTRON_MASS_KG * wavelength_m)
    tof_seconds = flight_path_length_m / velocity
    return tof_seconds * 1e6  # Convert to microseconds

print("="*70)
print("TESTING STREAMLIT NBRAGG INTEGRATION (with NaN cleaning fix)")
print("="*70)

# Pipeline setup (exactly as in streamlit with default settings)
print("\n1. Running pipeline...")
data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
           flux=5e6, duration=0.5, freq=20)
data.convolute_response(200, bin_width=10)
data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
data.overlap(kernel=[0, 25])
print("   ✓ Data processing complete")

# Reconstruction (with time filter - as in streamlit defaults)
print("\n2. Running reconstruction...")
recon = Reconstruct(data, tmin=3.7, tmax=11.0)
recon.filter(kind='wiener', noise_power=0.2)
stats = recon.get_statistics()
print(f"   ✓ Reconstruction complete (χ²/dof: {stats['chi2_per_dof']:.1f})")

# nbragg Analysis (EXACTLY as updated in streamlit_app.py)
print("\n3. Running nbragg analysis (WITH NaN cleaning)...")
try:
    analysis = Analysis(xs='iron', vary_background=True, vary_response=True)

    # Prepare nbragg data and clean NaN values (CRITICAL!)
    nbragg_data = recon.to_nbragg(L=9.0, tstep=10e-6)
    print(f"   - Before dropna: {nbragg_data.table.shape[0]} rows")

    nbragg_data.table = nbragg_data.table.dropna()
    print(f"   - After dropna: {nbragg_data.table.shape[0]} rows")

    # Fit using the cleaned data
    result = analysis.model.fit(nbragg_data)
    analysis.result = result
    analysis.data = nbragg_data

    # Check if result is valid
    if hasattr(result, 'redchi') and not np.isnan(result.redchi):
        print(f"   ✓ nbragg fit succeeded (χ²/dof: {result.redchi:.2f})")
    else:
        print(f"   ✗ nbragg fit produced invalid results")
        sys.exit(1)

except Exception as e:
    print(f"   ✗ nbragg fit failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test plotting (EXACTLY as in streamlit_app.py)
print("\n4. Testing plot with nbragg overlay...")
try:
    # Create reconstruction plot
    mpl_fig = recon.plot(kind='transmission', show_errors=False, figsize=(12, 8), ylim=(0, 1))

    # Add nbragg fit curve
    axes = mpl_fig.get_axes()
    ax_data = axes[0]

    # Get nbragg best fit data
    wavelength_angstrom = result.userkws['wl']
    best_fit_transmission = result.best_fit

    # Convert wavelength to time-of-flight
    L = 9.0  # Flight path in meters
    time_us = wavelength_to_tof(wavelength_angstrom, L)
    time_ms = time_us / 1000

    # Plot nbragg fit
    ax_data.plot(time_ms, best_fit_transmission,
               label='nbragg fit', color='green', linewidth=2, linestyle='--')
    ax_data.legend()

    print(f"   ✓ nbragg fit overlay added ({len(best_fit_transmission)} points)")
    print(f"   ✓ Time range: {time_ms.min():.2f}-{time_ms.max():.2f} ms")

    # Count lines in plot
    n_lines = len(ax_data.get_lines())
    print(f"   ✓ Plot has {n_lines} lines (should be 5: Convolved, Reconstructed, tmin, tmax, nbragg fit)")

    plt.savefig('/tmp/streamlit_nbragg_test.png', dpi=150, bbox_inches='tight')
    print("   ✓ Plot saved to /tmp/streamlit_nbragg_test.png")
    plt.close()

except Exception as e:
    print(f"   ✗ Plotting failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test statistics display (EXACTLY as in streamlit_app.py)
print("\n5. Testing statistics display...")
try:
    # Check result attributes
    print(f"   ✓ Reduced χ² (nbragg): {result.redchi:.4f}")

    # Quality indicator
    if result.redchi < 2:
        quality = "Excellent"
    elif result.redchi < 5:
        quality = "Good"
    else:
        quality = "Poor"
    print(f"   ✓ Quality: {quality}")

    # Parameters table
    print(f"   ✓ Parameters ({len(result.params)} total):")
    for param_name, param in result.params.items():
        stderr_str = f"{param.stderr:.4e}" if param.stderr is not None else "N/A"
        print(f"     - {param_name}: {param.value:.4e} ± {stderr_str}")

except Exception as e:
    print(f"   ✗ Statistics display failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nThe Streamlit app should now work correctly:")
print("  1. Enable 'Apply nbragg Analysis' in sidebar (Stage 6)")
print("  2. Click 'Run Pipeline'")
print("  3. Reconstruction tab will show green dashed nbragg fit line")
print("  4. Statistics tab will show nbragg fit results with parameters table")
