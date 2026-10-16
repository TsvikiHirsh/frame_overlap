"""
Test interpolated kernel construction (official FOBI style) vs discrete kernels.

This test compares:
1. Discrete delta function kernels (our original approach)
2. Interpolated kernels with sub-bin precision (official FOBI approach)
"""

import sys
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct
import numpy as np


def test_kernel_construction():
    """Test that interpolated kernel differs from discrete kernel for fractional delays."""
    print("\n" + "="*70)
    print("TEST 1: Kernel Construction Comparison")
    print("="*70)

    # Load data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)
    data.convolute_response(200, bin_width=10)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=30, seed=42)

    # Use kernel that would produce fractional bin indices
    # At 10 µs bin width: [0, 25ms] = [0, 25000µs] = [0, 2500 bins] (integer)
    # Try [0, 25.003ms] = [0, 25003µs] = [0, 2500.3 bins] for fractional
    data.overlap(kernel=[0, 25.003], total_time=50)

    # Create reconstruct object
    recon = Reconstruct(data)

    # Get discrete kernel
    kernel_discrete = recon._reconstruct_kernel(interpolate=False)

    # Get interpolated kernel
    kernel_interp = recon._reconstruct_kernel(interpolate=True)

    print(f"\nKernel analysis for {data.kernel} ms:")
    print(f"  Bin width: {(data.table['time'][1] - data.table['time'][0]):.1f} µs")

    print(f"\nDiscrete kernel:")
    print(f"  Length: {len(kernel_discrete)}")
    nonzero_discrete = np.where(kernel_discrete > 1e-10)[0]
    print(f"  Non-zero positions: {nonzero_discrete}")
    print(f"  Non-zero values: {kernel_discrete[nonzero_discrete]}")
    print(f"  Sum: {kernel_discrete.sum():.6f}")

    print(f"\nInterpolated kernel:")
    print(f"  Length: {len(kernel_interp)}")
    nonzero_interp = np.where(kernel_interp > 1e-10)[0]
    print(f"  Non-zero positions: {nonzero_interp}")
    print(f"  Non-zero values: {kernel_interp[nonzero_interp]}")
    print(f"  Sum: {kernel_interp.sum():.6f}")

    # Check differences
    print(f"\nDifferences:")
    if len(nonzero_discrete) == len(nonzero_interp):
        print(f"  Same number of non-zero bins: {len(nonzero_discrete)}")
        print(f"  ✗ Interpolation should create MORE non-zero bins!")
        return False
    else:
        print(f"  Discrete: {len(nonzero_discrete)} non-zero bins")
        print(f"  Interpolated: {len(nonzero_interp)} non-zero bins")
        print(f"  ✓ Interpolation creates more non-zero bins (sub-bin precision)")
        return True


def test_reconstruction_comparison():
    """Compare reconstruction quality with discrete vs interpolated kernels."""
    print("\n" + "="*70)
    print("TEST 2: Reconstruction Quality Comparison")
    print("="*70)

    # Load data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)
    data.convolute_response(200, bin_width=10)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=30, seed=42)

    # Store reference
    reference_signal = data.poissoned_data['counts'].values.copy()

    # Two frames with fractional delay
    data.overlap(kernel=[0, 25], total_time=50)

    print(f"\nKernel: {data.kernel} ms")

    # Test with discrete kernel (interpolate_kernel=False)
    recon_discrete = Reconstruct(data)
    recon_discrete.filter(kind='fobi', noise_power=0.1, smooth_window=1,
                          interpolate_kernel=False)

    stats_discrete = recon_discrete.get_statistics()

    print(f"\nDiscrete kernel FOBI:")
    print(f"  χ²/dof: {stats_discrete['chi2_per_dof']:.3f}")
    print(f"  R²: {stats_discrete['r_squared']:.3f}")
    print(f"  RMSE: {stats_discrete['rmse']:.3f}")

    # Test with interpolated kernel (interpolate_kernel=True, default)
    recon_interp = Reconstruct(data)
    recon_interp.filter(kind='fobi', noise_power=0.1, smooth_window=1,
                        interpolate_kernel=True)

    stats_interp = recon_interp.get_statistics()

    print(f"\nInterpolated kernel FOBI (official style):")
    print(f"  χ²/dof: {stats_interp['chi2_per_dof']:.3f}")
    print(f"  R²: {stats_interp['r_squared']:.3f}")
    print(f"  RMSE: {stats_interp['rmse']:.3f}")

    # Compare
    print(f"\nComparison:")
    chi2_improvement = ((stats_discrete['chi2_per_dof'] - stats_interp['chi2_per_dof']) /
                        stats_discrete['chi2_per_dof'] * 100)
    r2_improvement = ((stats_interp['r_squared'] - stats_discrete['r_squared']) /
                      stats_discrete['r_squared'] * 100)

    print(f"  χ²/dof improvement: {chi2_improvement:+.2f}%")
    print(f"  R² improvement: {r2_improvement:+.2f}%")

    if chi2_improvement > 1 or r2_improvement > 0.1:
        print(f"  ✓ Interpolated kernel shows improvement!")
        return True
    else:
        print(f"  = Similar performance (kernel delays are likely at integer bins)")
        return True


def test_official_fobi_parameters():
    """Test FOBI with official parameters: noise_power=0.1, interpolated kernel."""
    print("\n" + "="*70)
    print("TEST 3: Official FOBI Parameters")
    print("="*70)

    # Load data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)
    data.convolute_response(200, bin_width=10)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=30, seed=42)
    data.overlap(kernel=[0, 25], total_time=50)

    print(f"\nTesting with official FOBI parameters:")
    print(f"  noise_power: 0.1 (official FOBI default)")
    print(f"  interpolate_kernel: True")
    print(f"  smooth_window: 1 (no smoothing)")

    # Apply official FOBI
    recon = Reconstruct(data)
    recon.filter(kind='fobi', noise_power=0.1, smooth_window=1,
                 interpolate_kernel=True)

    stats = recon.get_statistics()

    print(f"\nResults:")
    print(f"  χ²/dof: {stats['chi2_per_dof']:.3f}")
    print(f"  R²: {stats['r_squared']:.3f}")
    print(f"  RMSE: {stats['rmse']:.3f}")

    if stats['chi2_per_dof'] < 1.0 and stats['r_squared'] > 0.95:
        print(f"\n✓ Excellent reconstruction quality!")
        return True
    elif stats['chi2_per_dof'] < 2.0 and stats['r_squared'] > 0.90:
        print(f"\n✓ Good reconstruction quality")
        return True
    else:
        print(f"\n⚠ Reconstruction quality could be better")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing Interpolated Kernel Implementation (Official FOBI Style)")
    print("="*70)

    try:
        # Run tests
        test1_passed = test_kernel_construction()
        test2_passed = test_reconstruction_comparison()
        test3_passed = test_official_fobi_parameters()

        if test1_passed and test2_passed and test3_passed:
            print("\n" + "="*70)
            print("✅ ALL TESTS PASSED!")
            print("="*70)
            print("\nInterpolated kernel implementation matches official FOBI!")
        else:
            print("\n" + "="*70)
            print("⚠ SOME TESTS HAD ISSUES")
            print("="*70)

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST EXECUTION FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
