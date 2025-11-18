"""
Test that the epsilon clipping fix prevents NaN errors in nbragg fitting.
"""

import sys
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct, Analysis
import warnings


def test_no_nan_in_fitting():
    """Test that nbragg fitting doesn't produce NaN errors."""
    print("\n" + "="*70)
    print("TEST: NaN Fix in nbragg Fitting")
    print("="*70)

    # Load and process data
    print("\n1. Loading and processing data...")
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)
    data.convolute_response(200, bin_width=10)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=30, seed=42)
    data.overlap(kernel=[0, 25], total_time=50)

    print("   ✓ Data processed")

    # Reconstruct
    print("\n2. Reconstructing...")
    recon = Reconstruct(data)
    recon.filter(kind='fobi', noise_power=0.1, interpolate_kernel=True)

    print("   ✓ Reconstruction complete")
    print(f"     Reconstructed signal range: [{recon.reconstructed_data['counts'].min():.3f}, "
          f"{recon.reconstructed_data['counts'].max():.3f}]")

    # Convert to nbragg format
    print("\n3. Converting to nbragg format...")
    try:
        nbragg_data = recon.to_nbragg(L=9.0, tstep=10e-6)
        print("   ✓ Conversion successful")

        # Check for NaN in nbragg data
        import numpy as np
        has_nan = np.any(np.isnan(nbragg_data.table['transmission']))
        has_inf = np.any(np.isinf(nbragg_data.table['transmission']))

        print(f"\n   nbragg data quality:")
        print(f"     Length: {len(nbragg_data.table)}")
        print(f"     Transmission range: [{nbragg_data.table['transmission'].min():.6f}, "
              f"{nbragg_data.table['transmission'].max():.6f}]")
        print(f"     Contains NaN: {has_nan}")
        print(f"     Contains Inf: {has_inf}")

        if has_nan or has_inf:
            print("   ❌ Still has NaN/Inf!")
            return False

    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")
        return False

    # Try fitting
    print("\n4. Testing nbragg fitting...")

    try:
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Create analysis and fit
            analysis = Analysis('Fe_alpha')
            result = analysis.fit(recon, L=9.0, tstep=10e-6)

            # Check for NaN warnings
            nan_warnings = [warning for warning in w if 'NaN' in str(warning.message)]

            if nan_warnings:
                print(f"   ⚠ Got {len(nan_warnings)} NaN warnings:")
                for warning in nan_warnings[:3]:  # Show first 3
                    print(f"     - {warning.message}")
                return False
            else:
                print("   ✓ Fitting successful with NO NaN warnings!")
                print(f"     χ²/dof: {result.redchi:.3f}")
                return True

    except Exception as e:
        print(f"   ❌ Fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = test_no_nan_in_fitting()

        if success:
            print("\n" + "="*70)
            print("✅ TEST PASSED! NaN issue is fixed!")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("⚠ TEST SHOWS REMAINING ISSUES")
            print("="*70)
            print("\nThe epsilon clipping may not be sufficient.")
            print("Check if there are other sources of zeros in the data.")

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
