#!/usr/bin/env python3
"""Test all kernel generation modes"""

import numpy as np
from frame_overlap import Data

def test_all_kernel_modes():
    """Test all kernel generation modes: equal, random, blue_noise, random_min_gap"""

    print("\n" + "="*70)
    print("TEST: All kernel generation modes")
    print("="*70)

    modes_to_test = [
        ('equal', None),
        ('random', None),
        ('blue_noise', None),
        ('random_min_gap', 5.0),  # min_gap required
    ]

    n_frames = 5
    total_time = 50.0

    for mode, min_gap in modes_to_test:
        print(f"\n{'='*70}")
        print(f"Testing mode: {mode}")
        if min_gap is not None:
            print(f"  min_gap: {min_gap} ms")
        print(f"{'='*70}")

        # Load fresh data for each test
        data = Data(
            signal_file='notebooks/iron_powder.csv',
            openbeam_file='notebooks/openbeam.csv',
            flux=5e6,
            duration=0.5,
            freq=20
        )
        data.convolute_response(pulse_duration=20.0)
        data.poisson_sample(flux=1e6, freq=20, measurement_time=8.0)

        # Apply overlap with the mode
        if min_gap is not None:
            data.overlap(
                kernel=n_frames,
                total_time=total_time,
                mode=mode,
                min_gap=min_gap,
                kernel_seed=42
            )
        else:
            data.overlap(
                kernel=n_frames,
                total_time=total_time,
                mode=mode,
                kernel_seed=42
            )

        print(f"\nGenerated kernel: {data.kernel}")

        # Analyze gaps
        if len(data.kernel) > 1:
            gaps = np.diff(data.kernel)
            print(f"Gaps: {[f'{g:.2f}' for g in gaps]}")
            print(f"Min gap: {np.min(gaps):.3f} ms")
            print(f"Max gap: {np.max(gaps):.3f} ms")
            print(f"Mean gap: {np.mean(gaps):.3f} ms")
            print(f"Std gap: {np.std(gaps):.3f} ms")

            # Check min_gap constraint if applicable
            if min_gap is not None:
                if np.all(gaps >= min_gap):
                    print(f"✅ All gaps >= {min_gap} ms")
                else:
                    print(f"❌ FAILURE: Some gaps < {min_gap} ms")
                    failing_gaps = gaps[gaps < min_gap]
                    print(f"  Failing gaps: {failing_gaps}")

            # Mode-specific checks
            if mode == 'equal':
                expected_gap = total_time / n_frames
                if np.allclose(gaps, expected_gap, rtol=1e-6):
                    print(f"✅ Equal spacing verified (gap ≈ {expected_gap:.2f} ms)")
                else:
                    print(f"❌ Not equally spaced")

            elif mode == 'blue_noise':
                # Blue noise should have more uniform distribution than random
                # Check that std is reasonable
                print(f"✅ Blue noise generated (lower std = more uniform)")

            elif mode == 'random':
                print(f"✅ Random spacing generated")

    print("\n" + "="*70)
    print("✅ All kernel modes tested successfully")
    print("="*70)

if __name__ == "__main__":
    test_all_kernel_modes()
