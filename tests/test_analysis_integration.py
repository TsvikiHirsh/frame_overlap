"""
Test script to verify nbragg Analysis integration works correctly.
"""

import sys
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct, Analysis
import matplotlib.pyplot as plt

def test_basic_analysis():
    """Test basic analysis workflow."""
    print("=" * 60)
    print("Testing nbragg Analysis Integration")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    signal_path = 'notebooks/iron_powder.csv'
    openbeam_path = 'notebooks/openbeam.csv'

    try:
        data = Data(signal_path, openbeam_path, flux=5e6, duration=0.5, freq=20)
        print("   ✓ Data loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load data: {e}")
        return False

    # Apply processing stages
    print("\n2. Applying processing stages...")
    try:
        data.convolute_response(200, bin_width=10)
        print("   ✓ Convolution applied")

        data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
        print("   ✓ Poisson sampling applied")

        data.overlap(kernel=[0, 25])
        print("   ✓ Frame overlap applied")
    except Exception as e:
        print(f"   ✗ Failed in processing: {e}")
        return False

    # Reconstruction
    print("\n3. Running reconstruction...")
    try:
        recon = Reconstruct(data, tmin=3.7, tmax=11.0)
        recon.filter(kind='wiener', noise_power=0.2)
        stats = recon.get_statistics()
        print(f"   ✓ Reconstruction complete (χ²/dof: {stats['chi2_per_dof']:.2f})")
    except Exception as e:
        print(f"   ✗ Failed in reconstruction: {e}")
        return False

    # nbragg Analysis
    print("\n4. Running nbragg analysis...")
    try:
        analysis = Analysis(xs='iron', vary_background=True, vary_response=True)
        print("   ✓ Analysis object created")

        result = analysis.fit(recon)
        print(f"   ✓ Fitting complete (reduced χ²: {result.redchi:.4f})")

        # Check if result has required attributes
        print("\n5. Checking result attributes...")
        assert hasattr(result, 'redchi'), "Missing 'redchi' attribute"
        print(f"   ✓ result.redchi = {result.redchi:.4f}")

        assert hasattr(result, 'best_fit'), "Missing 'best_fit' attribute"
        print(f"   ✓ result.best_fit exists (length: {len(result.best_fit)})")

        assert hasattr(result, 'params'), "Missing 'params' attribute"
        print(f"   ✓ result.params exists ({len(result.params)} parameters)")

        # Print parameters
        print("\n6. Fitted parameters:")
        for param_name, param in result.params.items():
            print(f"   - {param_name}: {param.value:.4e} ± {param.stderr:.4e if param.stderr else 'N/A'}")

        # Check data attribute
        print("\n7. Checking analysis.data attribute...")
        assert analysis.data is not None, "analysis.data is None"
        print(f"   ✓ analysis.data exists")
        assert hasattr(analysis.data, 'table'), "analysis.data missing 'table' attribute"
        print(f"   ✓ analysis.data.table exists (shape: {analysis.data.table.shape})")

        # Test plotting
        print("\n8. Testing plot generation...")
        fig = analysis.plot()
        print(f"   ✓ Plot generated successfully (type: {type(fig)})")
        plt.close(fig)

        return True

    except Exception as e:
        print(f"   ✗ Failed in nbragg analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_best_fit_extraction():
    """Test extracting best fit data for plotting."""
    print("\n" + "=" * 60)
    print("Testing Best Fit Data Extraction")
    print("=" * 60)

    try:
        # Quick setup
        data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                   flux=5e6, duration=0.5, freq=20)
        data.convolute_response(200, bin_width=10)
        data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
        data.overlap(kernel=[0, 25])

        recon = Reconstruct(data, tmin=3.7, tmax=11.0)
        recon.filter(kind='wiener', noise_power=0.2)

        analysis = Analysis(xs='iron', vary_background=True, vary_response=True)
        result = analysis.fit(recon)

        # Extract data for plotting
        print("\nExtracting data for plotting...")
        nbragg_data = analysis.data
        time_stack = nbragg_data.table['stack'].values
        best_fit_transmission = result.best_fit

        # Convert stack to time in ms
        time_ms = (time_stack - 1) * 0.01

        print(f"   ✓ time_stack shape: {time_stack.shape}")
        print(f"   ✓ best_fit shape: {best_fit_transmission.shape}")
        print(f"   ✓ time_ms range: [{time_ms.min():.2f}, {time_ms.max():.2f}] ms")
        print(f"   ✓ best_fit range: [{best_fit_transmission.min():.4f}, {best_fit_transmission.max():.4f}]")

        # Verify data is plottable
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(time_ms, best_fit_transmission, 'g--', label='nbragg fit')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Transmission')
        ax.legend()
        plt.close(fig)
        print("   ✓ Data is plottable")

        return True

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_basic_analysis()
    success2 = test_best_fit_extraction()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        exit(1)
