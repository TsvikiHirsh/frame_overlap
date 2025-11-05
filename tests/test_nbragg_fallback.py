#!/usr/bin/env python3
"""
Test that the app gracefully handles when nbragg is not available.
This simulates the Streamlit Cloud deployment scenario where nbragg fails to install.
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_import_without_nbragg():
    """Test that frame_overlap imports work even if nbragg is not available"""
    try:
        from frame_overlap import Data, Reconstruct, Workflow
        print("✅ Core imports successful (Data, Reconstruct, Workflow)")

        # Try to import Analysis
        try:
            from frame_overlap import Analysis
            print("✅ Analysis import successful (nbragg is available)")
            nbragg_available = True
        except ImportError as e:
            print(f"⚠️ Analysis import failed (nbragg not available): {e}")
            nbragg_available = False

        return nbragg_available

    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        return False

def test_data_processing_without_nbragg():
    """Test that basic data processing works without nbragg"""
    try:
        from frame_overlap import Data, Reconstruct

        # Create basic data object
        data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                   flux=5e6, duration=0.5, freq=20)
        print("✅ Data object created successfully")

        # Apply processing steps
        data.convolute_response(200.0, bin_width=10)
        print("✅ Convolution applied successfully")

        data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
        print("✅ Poisson sampling applied successfully")

        data.overlap(kernel=[0, 25], total_time=50)
        print("✅ Frame overlap applied successfully")

        # Reconstruction
        recon = Reconstruct(data, tmin=None, tmax=None)
        recon.filter(kind='wiener', noise_power=0.2)
        print("✅ Reconstruction completed successfully")

        # Get statistics
        stats = recon.get_statistics()
        print(f"✅ Statistics computed: χ²/dof = {stats['chi2_per_dof']:.2f}")

        return True

    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Testing nbragg Fallback Behavior")
    print("=" * 60)

    print("\n1. Testing imports...")
    nbragg_available = test_import_without_nbragg()

    print("\n2. Testing core functionality without nbragg...")
    core_works = test_data_processing_without_nbragg()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"nbragg available: {nbragg_available}")
    print(f"Core functionality: {'✅ PASS' if core_works else '❌ FAIL'}")

    if nbragg_available:
        print("\n✅ Full functionality available (including nbragg analysis)")
    else:
        print("\n⚠️ Limited functionality (nbragg analysis disabled)")
        print("   The app will work but nbragg features will be unavailable.")

    print("\nTo test without nbragg, temporarily rename/remove the nbragg package:")
    print("  pip uninstall nbragg")
    print("\nThen run this test again to verify the fallback behavior.")
