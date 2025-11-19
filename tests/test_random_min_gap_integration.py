#!/usr/bin/env python3
"""Integration test for random_min_gap mode with full pipeline"""

import numpy as np
from frame_overlap import Data, Reconstruct

def test_full_pipeline_with_random_min_gap():
    """Test full pipeline: load -> convolve -> poisson -> overlap (random_min_gap) -> reconstruct"""

    print("\n" + "="*70)
    print("INTEGRATION TEST: Full pipeline with random_min_gap mode")
    print("="*70)

    # Load data
    data = Data(
        signal_file='notebooks/iron_powder.csv',
        openbeam_file='notebooks/openbeam.csv',
        flux=5e6,
        duration=0.5,
        freq=20
    )
    print("✓ Data loaded")

    # Apply convolution
    data.convolute_response(pulse_duration=20.0)
    print("✓ Convolution applied")

    # Apply poisson sampling
    data.poisson_sample(flux=1e6, freq=20, measurement_time=8.0)
    print("✓ Poisson sampling applied")

    # Apply overlap with random_min_gap mode
    min_gap = 5.0  # ms
    n_frames = 5
    total_time = 50.0  # ms

    print(f"\nApplying overlap with random_min_gap:")
    print(f"  n_frames = {n_frames}")
    print(f"  total_time = {total_time} ms")
    print(f"  min_gap = {min_gap} ms")

    data.overlap(
        kernel=n_frames,
        total_time=total_time,
        mode='random_min_gap',
        min_gap=min_gap,
        kernel_seed=42
    )
    print(f"✓ Overlap applied")
    print(f"  Generated kernel: {data.kernel}")

    # Verify gaps
    if len(data.kernel) > 1:
        gaps = np.diff(data.kernel)
        print(f"\n  Gaps: {gaps}")
        print(f"  Min gap: {np.min(gaps):.3f} ms")
        print(f"  Max gap: {np.max(gaps):.3f} ms")

        if np.all(gaps >= min_gap):
            print(f"  ✅ All gaps >= {min_gap} ms")
        else:
            print(f"  ❌ Some gaps < {min_gap} ms")
            return False

    # Apply reconstruction
    print(f"\nApplying reconstruction:")
    recon = Reconstruct(data, tmin=None, tmax=None)
    recon.filter(kind='wiener', noise_power=1.0)
    print("✓ Reconstruction applied")

    # Check statistics
    stats = recon.get_statistics()
    print(f"\nReconstruction statistics:")
    print(f"  R²: {stats.get('r2', 'N/A')}")
    print(f"  χ²/dof: {stats.get('chi2_dof', 'N/A')}")

    print("\n" + "="*70)
    print("✅ Integration test PASSED")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_full_pipeline_with_random_min_gap()
    exit(0 if success else 1)
