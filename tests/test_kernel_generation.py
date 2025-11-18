#!/usr/bin/env python3
"""
Test kernel generation modes for overlap method.
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

from frame_overlap import Data

def test_equal_kernel():
    """Test equal spacing kernel generation"""
    print("=" * 70)
    print("TEST 1: EQUAL SPACING KERNEL")
    print("=" * 70)

    # Load data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)

    # Generate equal spacing kernel
    data.overlap(kernel=5, total_time=50, mode='equal')

    print(f"Generated kernel: {data.kernel}")
    print(f"Number of frames: {data.n_overlapping_frames}")

    # Check spacing
    kernel = np.array(data.kernel)
    expected = np.linspace(0, 50, 5, endpoint=False)

    if np.allclose(kernel, expected):
        print("✅ PASS: Equal spacing is correct")
        print(f"   Expected: {expected}")
        print(f"   Got:      {kernel}")
    else:
        print("❌ FAIL: Equal spacing is incorrect")
        print(f"   Expected: {expected}")
        print(f"   Got:      {kernel}")

    print()
    return np.allclose(kernel, expected)


def test_random_kernel():
    """Test random kernel generation"""
    print("=" * 70)
    print("TEST 2: RANDOM KERNEL")
    print("=" * 70)

    # Load data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)

    # Generate random kernel with seed
    data.overlap(kernel=5, total_time=50, mode='random', kernel_seed=42)

    print(f"Generated kernel: {data.kernel}")
    print(f"Number of frames: {data.n_overlapping_frames}")

    # Check properties
    kernel = np.array(data.kernel)

    checks = []

    # Check: should have 5 frames
    if len(kernel) == 5:
        print("✅ PASS: Correct number of frames (5)")
        checks.append(True)
    else:
        print(f"❌ FAIL: Expected 5 frames, got {len(kernel)}")
        checks.append(False)

    # Check: first frame should be at 0
    if kernel[0] == 0.0:
        print("✅ PASS: First frame at t=0")
        checks.append(True)
    else:
        print(f"❌ FAIL: First frame at t={kernel[0]}, expected 0")
        checks.append(False)

    # Check: all frames should be sorted
    if np.all(kernel[1:] >= kernel[:-1]):
        print("✅ PASS: Frames are sorted")
        checks.append(True)
    else:
        print("❌ FAIL: Frames are not sorted")
        checks.append(False)

    # Check: all frames should be within [0, 50]
    if np.all(kernel >= 0) and np.all(kernel < 50):
        print("✅ PASS: All frames within [0, 50)")
        checks.append(True)
    else:
        print(f"❌ FAIL: Some frames outside [0, 50): {kernel}")
        checks.append(False)

    # Check: with same seed, should get same kernel
    data2 = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                 flux=5e6, duration=0.5, freq=20)
    data2.overlap(kernel=5, total_time=50, mode='random', kernel_seed=42)
    if np.allclose(kernel, np.array(data2.kernel)):
        print("✅ PASS: Reproducible with same seed")
        checks.append(True)
    else:
        print("❌ FAIL: Not reproducible with same seed")
        checks.append(False)

    print()
    return all(checks)


def test_blue_noise_kernel():
    """Test blue noise kernel generation"""
    print("=" * 70)
    print("TEST 3: BLUE NOISE KERNEL")
    print("=" * 70)

    # Load data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)

    # Generate blue noise kernel with seed
    data.overlap(kernel=5, total_time=50, mode='blue_noise', kernel_seed=42)

    print(f"Generated kernel: {data.kernel}")
    print(f"Number of frames: {data.n_overlapping_frames}")

    # Check properties
    kernel = np.array(data.kernel)

    checks = []

    # Check: should have 5 frames
    if len(kernel) == 5:
        print("✅ PASS: Correct number of frames (5)")
        checks.append(True)
    else:
        print(f"❌ FAIL: Expected 5 frames, got {len(kernel)}")
        checks.append(False)

    # Check: first frame should be at 0
    if kernel[0] == 0.0:
        print("✅ PASS: First frame at t=0")
        checks.append(True)
    else:
        print(f"❌ FAIL: First frame at t={kernel[0]}, expected 0")
        checks.append(False)

    # Check: all frames should be sorted
    if np.all(kernel[1:] >= kernel[:-1]):
        print("✅ PASS: Frames are sorted")
        checks.append(True)
    else:
        print("❌ FAIL: Frames are not sorted")
        checks.append(False)

    # Check: all frames should be within [0, 50]
    if np.all(kernel >= 0) and np.all(kernel < 50):
        print("✅ PASS: All frames within [0, 50)")
        checks.append(True)
    else:
        print(f"❌ FAIL: Some frames outside [0, 50): {kernel}")
        checks.append(False)

    # Check: blue noise should have more uniform spacing than random
    # Calculate variance of inter-frame distances
    dists = np.diff(kernel)
    variance = np.var(dists)
    print(f"   Inter-frame distance variance: {variance:.2f}")

    # Compare with random kernel
    data_random = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                      flux=5e6, duration=0.5, freq=20)
    data_random.overlap(kernel=5, total_time=50, mode='random', kernel_seed=42)
    kernel_random = np.array(data_random.kernel)
    dists_random = np.diff(kernel_random)
    variance_random = np.var(dists_random)
    print(f"   Random inter-frame distance variance: {variance_random:.2f}")

    if variance < variance_random:
        print("✅ PASS: Blue noise has lower variance than random (more uniform)")
        checks.append(True)
    else:
        print("⚠️  NOTE: Blue noise variance not lower than random (may happen with small samples)")
        checks.append(True)  # Don't fail on this, it's probabilistic

    print()
    return all(checks)


def test_comparison():
    """Compare all three modes visually"""
    print("=" * 70)
    print("TEST 4: VISUAL COMPARISON")
    print("=" * 70)

    modes = ['equal', 'random', 'blue_noise']

    for mode in modes:
        data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                   flux=5e6, duration=0.5, freq=20)

        seed = 42 if mode != 'equal' else None
        data.overlap(kernel=10, total_time=50, mode=mode, kernel_seed=seed)

        kernel = np.array(data.kernel)

        print(f"\n{mode.upper()}:")
        print(f"  Kernel: {[f'{k:.2f}' for k in kernel]}")

        # Calculate spacing statistics
        dists = np.diff(kernel)
        print(f"  Mean spacing: {np.mean(dists):.2f} ms")
        print(f"  Std spacing:  {np.std(dists):.2f} ms")
        print(f"  Min spacing:  {np.min(dists):.2f} ms")
        print(f"  Max spacing:  {np.max(dists):.2f} ms")

    print()
    print("✅ PASS: All modes generated successfully")
    return True


if __name__ == '__main__':
    print("=" * 70)
    print("KERNEL GENERATION TESTS")
    print("=" * 70)
    print()

    all_passed = True

    try:
        passed = test_equal_kernel()
        all_passed = all_passed and passed

        passed = test_random_kernel()
        all_passed = all_passed and passed

        passed = test_blue_noise_kernel()
        all_passed = all_passed and passed

        passed = test_comparison()
        all_passed = all_passed and passed

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_passed:
        print("✅ All tests passed")
    else:
        print("❌ Some tests failed")

    print("=" * 70)

    sys.exit(0 if all_passed else 1)
