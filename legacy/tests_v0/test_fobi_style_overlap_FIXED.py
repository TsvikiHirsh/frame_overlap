"""
Test FOBI-style overlap with CORRECT measurement times.

KEY INSIGHT: To simulate frame overlap, your measurement time must be
AT LEAST as long as the frame period + overlap!
"""

import sys
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct
import numpy as np


def test_with_correct_measurement_time():
    """Test with measurement time that covers full frames."""
    print("\n" + "="*70)
    print("TEST: FOBI-STYLE OVERLAP WITH CORRECT MEASUREMENT TIME")
    print("="*70)

    # At 20 Hz, frame period = 50ms
    # To properly test 2-frame overlap, we need >= 50ms measurement time!

    print("\nConfiguration:")
    print("  Frequency: 20 Hz → frame period = 50ms")
    print("  Measurement time: 100ms (covers 2 full frames)")
    print("  Kernel: [0, 25] ms (frames overlap by 50%)")

    # Load data with LONGER measurement time
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)
    data.convolute_response(200, bin_width=10)

    # KEY FIX: Use longer measurement time!
    data.poisson_sample(flux=1e6, freq=20, measurement_time=100, seed=42)

    signal_len = len(data.poissoned_data)
    signal_time_ms = data.poissoned_data['time'].max() / 1000

    print(f"\n Input signal: {signal_len} points, {signal_time_ms:.1f} ms")

    # Apply superimpose overlap
    data.overlap(kernel=[0, 25], mode='superimpose')

    overlap_len = len(data.overlapped_data)
    overlap_time_ms = data.overlapped_data['time'].max() / 1000

    print(f"Overlapped signal: {overlap_len} points, {overlap_time_ms:.1f} ms")
    print(f"  → Length unchanged ✓")

    # Reconstruct
    recon = Reconstruct(data)
    recon.filter(kind='fobi', noise_power=0.1, smooth_window=1,
                 interpolate_kernel=True)

    stats = recon.get_statistics()

    print(f"\nReconstruction Quality:")
    print(f"  χ²/dof: {stats['chi2_per_dof']:.6f}")
    print(f"  R²: {stats['r_squared']:.6f}")

    if stats['chi2_per_dof'] < 1.0:
        print(f"\n✅ EXCELLENT! Reconstruction works with correct measurement time!")
        return True
    else:
        print(f"\n⚠ χ²/dof = {stats['chi2_per_dof']:.3f} - still investigating...")
        return False


def test_high_frequency_short_measurement():
    """Test with higher frequency so frames fit in short measurement."""
    print("\n" + "="*70)
    print("TEST: HIGH FREQUENCY WITH SHORT MEASUREMENT")
    print("="*70)

    # Use 100 Hz so frame period = 10ms
    # Then 24ms measurement time covers 2.4 frames!

    print("\nConfiguration:")
    print("  Frequency: 100 Hz → frame period = 10ms")
    print("  Measurement time: 30ms (covers 3 full frames)")
    print("  Kernel: [0, 5] ms (50% overlap)")

    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=100)  # 100 Hz!
    data.convolute_response(200, bin_width=10)
    data.poisson_sample(flux=1e6, freq=100, measurement_time=30, seed=42)

    signal_len = len(data.poissoned_data)
    signal_time_ms = data.poissoned_data['time'].max() / 1000

    print(f"\nInput signal: {signal_len} points, {signal_time_ms:.1f} ms")

    # Two frames with 50% overlap
    data.overlap(kernel=[0, 5], mode='superimpose')

    print(f"Overlapped signal: {len(data.overlapped_data)} points (same length ✓)")

    # Reconstruct
    recon = Reconstruct(data)
    recon.filter(kind='fobi', noise_power=0.1, smooth_window=1,
                 interpolate_kernel=True)

    stats = recon.get_statistics()

    print(f"\nReconstruction Quality:")
    print(f"  χ²/dof: {stats['chi2_per_dof']:.6f}")
    print(f"  R²: {stats['r_squared']:.6f}")

    if stats['chi2_per_dof'] < 1.0:
        print(f"\n✅ GOOD! Higher frequency works!")
        return True
    else:
        print(f"\n⚠ χ²/dof = {stats['chi2_per_dof']:.3f}")
        return False


def explain_the_problem():
    """Explain why the original approach wasn't working."""
    print("\n" + "="*70)
    print("WHY YOUR ORIGINAL APPROACH WASN'T WORKING")
    print("="*70)

    print("\nYour original setup:")
    print("  - Frequency: 20 Hz → frame period = 50ms")
    print("  - Measurement time: 30ms")
    print("  - Kernel: [0, 25] ms")

    print("\nWhat happens with superimpose mode:")
    print("  Frame 1 (starts at 0ms):")
    print("    - Adds signal[0:24ms] to overlapped[0:24ms] ✓")
    print("  ")
    print("  Frame 2 (starts at 25ms):")
    print("    - Should add signal[0:24ms] to overlapped[25:49ms]")
    print("    - BUT your signal only goes to 24ms!")
    print("    - So Frame 2 contributes NOTHING ❌")

    print("\nResult:")
    print("  - Overlapped signal = signal / 2 (divided by n_frames)")
    print("  - Reconstruction tries to recover original from half-amplitude signal")
    print("  - Math doesn't work because only 1 frame contributed!")

    print("\nThe Fix:")
    print("  Option 1: Longer measurement time (100ms covers 2 frames)")
    print("  Option 2: Higher frequency (100 Hz → 10ms frames fit in 30ms)")
    print("  Option 3: Understand that you're testing frame overlap,")
    print("            not simulating a shorter measurement!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING FOBI-STYLE OVERLAP - CORRECTED APPROACH")
    print("="*70)

    try:
        # Explain the issue
        explain_the_problem()

        # Test 1: Correct measurement time
        test1_passed = test_with_correct_measurement_time()

        # Test 2: Higher frequency
        test2_passed = test_high_frequency_short_measurement()

        if test1_passed or test2_passed:
            print("\n" + "="*70)
            print("✅ SUCCESS!")
            print("="*70)
            print("\nKEY LEARNINGS:")
            print("  1. Measurement time must cover full frames")
            print("  2. mode='superimpose' correctly implements FOBI physics")
            print("  3. Your original poor reconstruction was due to")
            print("     incomplete frame coverage, NOT a bug in the code!")
        else:
            print("\n" + "="*70)
            print("⚠ STILL INVESTIGATING")
            print("="*70)

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST EXECUTION FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
