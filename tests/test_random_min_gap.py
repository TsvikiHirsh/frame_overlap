#!/usr/bin/env python3
"""Test random_min_gap kernel generation mode"""

import numpy as np
from frame_overlap import Data

def test_random_min_gap():
    """Test that random_min_gap mode respects minimum gap constraint"""

    # Load test data
    data = Data(
        signal_file='notebooks/iron_powder.csv',
        openbeam_file='notebooks/openbeam.csv',
        flux=5e6,
        duration=0.5,
        freq=20
    )

    # Apply convolution
    data.convolute_response(pulse_duration=20.0)

    # Apply poisson sampling
    data.poisson_sample(flux=1e6, freq=20, measurement_time=8.0)

    print("\n" + "="*70)
    print("TEST: random_min_gap kernel generation")
    print("="*70)

    # Test with 5 frames, 50 ms total time, 5 ms minimum gap
    min_gap = 5.0  # ms
    n_frames = 5
    total_time = 50.0  # ms

    print(f"\nGenerating kernel with:")
    print(f"  n_frames = {n_frames}")
    print(f"  total_time = {total_time} ms")
    print(f"  min_gap = {min_gap} ms")
    print(f"  mode = 'random_min_gap'")

    data.overlap(
        kernel=n_frames,
        total_time=total_time,
        mode='random_min_gap',
        min_gap=min_gap,
        kernel_seed=42
    )

    print(f"\nGenerated kernel: {data.kernel}")

    # Verify gaps
    if len(data.kernel) > 1:
        gaps = np.diff(data.kernel)
        print(f"\nGaps between frames: {gaps}")
        print(f"Minimum gap: {np.min(gaps):.3f} ms")
        print(f"Maximum gap: {np.max(gaps):.3f} ms")

        # Check all gaps meet minimum requirement
        if np.all(gaps >= min_gap):
            print(f"\n✅ SUCCESS: All gaps >= {min_gap} ms")
        else:
            print(f"\n❌ FAILURE: Some gaps < {min_gap} ms")
            failing_gaps = gaps[gaps < min_gap]
            print(f"  Failing gaps: {failing_gaps}")
    else:
        print("\n✅ Single frame kernel (no gaps to check)")

    # Test edge case: impossible configuration
    print("\n" + "="*70)
    print("TEST: Impossible configuration (should raise error)")
    print("="*70)

    try:
        data2 = Data(
            signal_file='notebooks/iron_powder.csv',
            openbeam_file='notebooks/openbeam.csv',
            flux=5e6,
            duration=0.5,
            freq=20
        )
        data2.convolute_response(pulse_duration=20.0)
        data2.poisson_sample(flux=1e6, freq=20, measurement_time=8.0)

        # Try to fit 20 frames with 5 ms gap in 50 ms (impossible)
        data2.overlap(kernel=20, total_time=50, mode='random_min_gap', min_gap=5.0)
        print("❌ FAILURE: Should have raised ValueError")
    except ValueError as e:
        print(f"✅ SUCCESS: Correctly raised ValueError")
        print(f"  Error message: {e}")

    # Test with different min_gap values
    print("\n" + "="*70)
    print("TEST: Different min_gap values")
    print("="*70)

    for test_min_gap in [2.0, 5.0, 8.0]:
        data3 = Data(
            signal_file='notebooks/iron_powder.csv',
            openbeam_file='notebooks/openbeam.csv',
            flux=5e6,
            duration=0.5,
            freq=20
        )
        data3.convolute_response(pulse_duration=20.0)
        data3.poisson_sample(flux=1e6, freq=20, measurement_time=8.0)

        data3.overlap(
            kernel=5,
            total_time=50,
            mode='random_min_gap',
            min_gap=test_min_gap,
            kernel_seed=42
        )

        gaps = np.diff(data3.kernel)
        min_actual = np.min(gaps)
        print(f"\nmin_gap={test_min_gap} ms: kernel={data3.kernel}")
        print(f"  Actual minimum gap: {min_actual:.3f} ms")
        print(f"  {'✅ PASS' if min_actual >= test_min_gap else '❌ FAIL'}")

if __name__ == "__main__":
    test_random_min_gap()
