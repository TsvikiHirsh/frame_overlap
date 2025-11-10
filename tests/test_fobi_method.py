"""
Test FOBI reconstruction method and compare with other methods.

This tests the FOBI-style Wiener deconvolution adapted from the original
FOBI code, which uses conjugate in the frequency domain.
"""

import sys
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct
import numpy as np
import matplotlib.pyplot as plt


def test_two_frame_comparison():
    """Compare all reconstruction methods for two frames at 0 and 25 ms."""
    print("\n" + "="*70)
    print("TEST: Two-frame reconstruction comparison (0ms and 25ms)")
    print("="*70)

    # Load data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)

    # Apply stages
    data.convolute_response(200, bin_width=10)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=30, seed=42)

    # Two frames with NO overlap (0 and 25 ms)
    data.overlap(kernel=[0, 25], total_time=50)

    print(f"\nKernel: {data.kernel} ms")
    print(f"Frame spacing: 25 ms (no temporal overlap)")

    # Test different methods
    methods = [
        ('wiener', {'noise_power': 0.2}),
        ('wiener_smooth', {'noise_power': 0.2, 'smooth_window': 5}),
        ('fobi', {'noise_power': 0.2, 'smooth_window': 5, 'sg_order': 1}),
        ('fobi', {'noise_power': 0.1, 'smooth_window': 5, 'sg_order': 1}, 'fobi_low_noise'),
        ('fobi', {'noise_power': 0.01, 'smooth_window': 5, 'sg_order': 1}, 'fobi_vlow_noise'),
        ('lucy', {'iterations': 20}),
        ('tikhonov', {'noise_power': 0.2})
    ]

    results = []

    for method_spec in methods:
        if len(method_spec) == 3:
            method_name, params, label = method_spec
        else:
            method_name, params = method_spec
            label = method_name

        # Reconstruct
        recon = Reconstruct(data)
        recon.filter(kind=method_name, **params)

        stats = recon.get_statistics()
        chi2_dof = stats['chi2_per_dof']
        r2 = stats['r_squared']

        results.append({
            'method': label,
            'chi2_dof': chi2_dof,
            'r2': r2,
            'recon': recon
        })

        print(f"\n{label:20s}:")
        print(f"  χ²/dof: {chi2_dof:.3f}")
        print(f"  R²: {r2:.3f}")
        params_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        print(f"  Params: {params_str}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Method':<20s} {'χ²/dof':>10s} {'R²':>10s} {'Quality':<10s}")
    print("-"*70)

    for r in results:
        quality = "Excellent" if r['chi2_dof'] < 1.0 else "Good" if r['chi2_dof'] < 2.0 else "Fair"
        print(f"{r['method']:<20s} {r['chi2_dof']:>10.3f} {r['r2']:>10.3f} {quality:<10s}")

    # Find best method
    best = min(results, key=lambda x: x['chi2_dof'])
    print(f"\n✓ Best method: {best['method']} (χ²/dof = {best['chi2_dof']:.3f})")

    return results


def test_fobi_parameters():
    """Test FOBI method with different parameters."""
    print("\n" + "="*70)
    print("TEST: FOBI parameter sensitivity")
    print("="*70)

    # Load data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)
    data.convolute_response(200, bin_width=10)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=30, seed=42)
    data.overlap(kernel=[0, 25], total_time=50)

    # Test different noise_power values
    print("\n--- Testing noise_power parameter ---")
    noise_powers = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]

    for noise_power in noise_powers:
        recon = Reconstruct(data)
        recon.filter(kind='fobi', noise_power=noise_power, smooth_window=5, sg_order=1)

        stats = recon.get_statistics()
        print(f"noise_power={noise_power:6.3f}: χ²/dof={stats['chi2_per_dof']:.3f}, R²={stats['r_squared']:.3f}")

    # Test different smoothing window sizes
    print("\n--- Testing smooth_window parameter ---")
    smooth_windows = [1, 3, 5, 7, 9, 11]

    for sw in smooth_windows:
        recon = Reconstruct(data)
        recon.filter(kind='fobi', noise_power=0.2, smooth_window=sw, sg_order=1)

        stats = recon.get_statistics()
        print(f"smooth_window={sw:2d}: χ²/dof={stats['chi2_per_dof']:.3f}, R²={stats['r_squared']:.3f}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing FOBI Reconstruction Method")
    print("="*70)

    try:
        # Run comparison test
        results = test_two_frame_comparison()

        # Run parameter sensitivity test
        test_fobi_parameters()

        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED!")
        print("="*70)

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST EXECUTION FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
