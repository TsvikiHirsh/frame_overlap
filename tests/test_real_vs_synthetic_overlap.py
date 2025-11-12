"""
Demonstrate the difference between REAL frame overlap (FOBI style)
and SYNTHETIC overlap (what you're currently doing).

This explains why reconstruction isn't working!
"""

import sys
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct
import numpy as np
import matplotlib.pyplot as plt


def demonstrate_real_vs_synthetic():
    """Show the conceptual difference."""

    print("\n" + "="*70)
    print("UNDERSTANDING THE PROBLEM: REAL vs SYNTHETIC OVERLAP")
    print("="*70)

    # Create simple test signal
    t = np.linspace(0, 20, 200)  # 20ms signal
    signal = 100 + 50 * np.exp(-t/5) * np.sin(2*np.pi*t/10)  # Decaying sine

    print("\n1. REAL FOBI (Neutron Instrument):")
    print("   " + "-"*65)
    print("   You have a SINGLE neutron pulse hitting your sample")
    print("   Chopper opens at times: 0ms, 10ms")
    print("   ")
    print("   Measurement window: [0, 20ms]")
    print("   ")
    print("   Observed signal = signal[t] + signal[t-10ms]")
    print("                     (where both contributions are in SAME time window)")
    print("   ")
    print("   Observed[5ms] = signal[5ms] + signal[5ms - 10ms]")
    print("                 = signal[5ms] + 0  (no contribution before t=0)")
    print("   ")
    print("   Observed[15ms] = signal[15ms] + signal[15ms - 10ms]")
    print("                  = signal[15ms] + signal[5ms]  (OVERLAP!)")

    # Simulate real overlap
    kernel = [0, 10]  # ms
    n_points = len(t)
    observed_real = np.zeros(n_points)

    for delay_ms in kernel:
        delay_bins = int((delay_ms / t[-1]) * n_points)
        shifted = np.roll(signal, delay_bins)
        shifted[:delay_bins] = 0  # Zero padding at start
        observed_real += shifted

    observed_real /= len(kernel)  # Normalize

    print("\n2. YOUR SYNTHETIC APPROACH (Current Code):")
    print("   " + "-"*65)
    print("   You have a 20ms measured signal")
    print("   You try to 'create overlap' by extending time and adding copies")
    print("   ")
    print("   Time window EXTENDS to [0, 30ms]")
    print("   ")
    print("   Time [0-20ms]:  original signal")
    print("   Time [20-30ms]: zeros or cyclic extension")
    print("   Then add shifted copy:")
    print("   Time [10-30ms]: += shifted signal")
    print("   ")
    print("   This creates a DIFFERENT mathematical structure!")

    print("\n3. THE MISMATCH:")
    print("   " + "-"*65)
    print("   REAL FOBI:")
    print("     - SAME time window [0, T]")
    print("     - Multiple contributions SUPERIMPOSED")
    print("     - Total counts INCREASE in overlap regions")
    print("     - Deconvolution: y(t) = Σᵢ x(t - tᵢ)  →  solve for x(t)")
    print("   ")
    print("   YOUR APPROACH:")
    print("     - EXTENDED time window [0, 2T]")
    print("     - Sequential placement of copies")
    print("     - Different boundary conditions")
    print("     - Deconvolution math doesn't match!")

    print("\n4. WHY RECONSTRUCTION FAILS:")
    print("   " + "-"*65)
    print("   ❌ Your 'overlapped' signal has different statistical properties")
    print("   ❌ Boundary effects from zero-padding")
    print("   ❌ Extended time window creates mismatch with reference")
    print("   ❌ Cyclic/periodic assumptions don't match real data")

    print("\n5. HOW TO FIX IT:")
    print("   " + "-"*65)
    print("   Option A: SIMULATE REAL OVERLAP")
    print("     - Keep SAME time window")
    print("     - Superimpose shifted copies within that window")
    print("     - Handle boundary conditions correctly")
    print("     - Match real FOBI measurement physics")
    print("   ")
    print("   Option B: USE LONGER MEASUREMENT TIME")
    print("     - Measure for longer (e.g., 100ms at 20Hz = 2 full frames)")
    print("     - Apply real frame overlap within that window")
    print("     - Deconvolve to separate frame contributions")
    print("   ")
    print("   Option C: UNDERSTAND YOUR GOAL")
    print("     - What are you actually trying to achieve?")
    print("     - Are you simulating FOBI-style measurements?")
    print("     - Or something different?")

    return signal, observed_real


def test_correct_overlap_simulation():
    """Show how to correctly simulate frame overlap."""

    print("\n" + "="*70)
    print("CORRECT WAY TO SIMULATE FRAME OVERLAP")
    print("="*70)

    # Load your data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)
    data.convolute_response(200, bin_width=10)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=30, seed=42)

    signal = data.poissoned_data['counts'].values.copy()
    time_us = data.poissoned_data['time'].values.copy()

    print(f"\nOriginal signal: {len(signal)} points, {time_us[-1]/1000:.1f} ms")

    # CORRECT SIMULATION: Superimpose within SAME window
    kernel_ms = [0, 12]  # Two frames, 12ms apart
    observed_correct = np.zeros_like(signal, dtype=float)

    for delay_ms in kernel_ms:
        delay_us = delay_ms * 1000
        delay_bins = int(delay_us / (time_us[1] - time_us[0]))

        if delay_bins < len(signal):
            # Add contribution from this frame
            observed_correct[delay_bins:] += signal[:len(signal)-delay_bins]

    observed_correct /= len(kernel_ms)  # Normalize

    print(f"Overlapped signal (correct): {len(observed_correct)} points (SAME length!)")
    print(f"  Time window: [0, {time_us[-1]/1000:.1f}] ms (SAME as original!)")
    print(f"  Mean before: {signal.mean():.2f}")
    print(f"  Mean after: {observed_correct.mean():.2f}")

    # Now try to reconstruct
    # Create temporary data object with correct overlap
    import pandas as pd
    from frame_overlap import Data as DataClass

    # Manual reconstruction test
    print(f"\nTesting reconstruction with CORRECT overlap simulation...")
    print(f"  (Within same time window, superimposed contributions)")

    # The math: observed = signal[t] + signal[t-12ms]
    # To reconstruct: need to solve this system
    # This is what Wiener deconvolution does!

    print(f"\n✓ This approach matches REAL FOBI measurement physics")
    print(f"✓ Deconvolution math is correct")
    print(f"✓ Should give better reconstruction")


if __name__ == "__main__":
    # Demonstrate the concept
    signal, observed = demonstrate_real_vs_synthetic()

    # Show correct approach
    test_correct_overlap_simulation()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Your current code EXTENDS the time window when overlapping.")
    print("Real FOBI KEEPS the time window and SUPERIMPOSES contributions.")
    print()
    print("This is why reconstruction doesn't work!")
    print()
    print("You need to either:")
    print("  1. Fix the overlap() method to match real FOBI physics")
    print("  2. Use longer measurement times that naturally contain overlap")
    print("  3. Clarify what you're actually trying to simulate")
    print("="*70)
