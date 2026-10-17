"""
Test to understand frame overlap and reconstruction quality.

This demonstrates the difference between:
1. Truly non-overlapping frames (easy reconstruction)
2. Partially overlapping frames (hard reconstruction)
3. Heavily overlapping frames (very hard reconstruction)
"""

import sys
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct
import numpy as np


def test_overlap_configuration(kernel, freq=20, description=""):
    """Test reconstruction quality for a given kernel configuration."""
    # Calculate frame period
    frame_period_ms = 1000 / freq

    # Calculate overlap
    if len(kernel) == 2:
        frame1_end = frame_period_ms
        frame2_start = kernel[1]

        if frame2_start < frame1_end:
            overlap_ms = frame1_end - frame2_start
            overlap_pct = (overlap_ms / frame_period_ms) * 100
        else:
            overlap_ms = 0
            overlap_pct = 0
    else:
        overlap_ms = 0
        overlap_pct = 0

    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Kernel: {kernel} ms")
    print(f"Frame period: {frame_period_ms} ms")
    print(f"Frame 1: [0, {frame_period_ms}] ms")
    if len(kernel) == 2:
        print(f"Frame 2: [{kernel[1]}, {kernel[1] + frame_period_ms}] ms")
        print(f"Overlap: {overlap_ms} ms ({overlap_pct:.1f}% of frame)")

    # Load data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=freq)
    data.convolute_response(200, bin_width=10)
    data.poisson_sample(flux=1e6, freq=freq, measurement_time=30, seed=42)

    # Store reference (what we want to recover)
    reference_signal = data.poissoned_data['counts'].values.copy()

    # Apply overlap
    total_time = kernel[-1] + frame_period_ms + 10 if len(kernel) > 1 else 50
    data.overlap(kernel=kernel, total_time=total_time)

    # Test with FOBI (interpolated kernel, noise_power=0.1)
    recon_fobi = Reconstruct(data)
    recon_fobi.filter(kind='fobi', noise_power=0.1, smooth_window=1,
                      interpolate_kernel=True)

    stats_fobi = recon_fobi.get_statistics()

    # Test with standard Wiener
    recon_wiener = Reconstruct(data)
    recon_wiener.filter(kind='wiener', noise_power=0.2)

    stats_wiener = recon_wiener.get_statistics()

    print(f"\nReconstruction Quality:")
    print(f"  FOBI (official):   χ²/dof={stats_fobi['chi2_per_dof']:7.3f}, R²={stats_fobi['r_squared']:.4f}")
    print(f"  Wiener (standard): χ²/dof={stats_wiener['chi2_per_dof']:7.3f}, R²={stats_wiener['r_squared']:.4f}")

    # Quality assessment
    if stats_fobi['chi2_per_dof'] < 0.01:
        quality = "✅ EXCELLENT (nearly perfect)"
    elif stats_fobi['chi2_per_dof'] < 0.1:
        quality = "✅ VERY GOOD"
    elif stats_fobi['chi2_per_dof'] < 1.0:
        quality = "✓ Good"
    elif stats_fobi['chi2_per_dof'] < 2.0:
        quality = "⚠ Fair"
    else:
        quality = "❌ Poor (reconstruction difficult)"

    print(f"\nQuality: {quality}")

    return {
        'kernel': kernel,
        'overlap_ms': overlap_ms,
        'overlap_pct': overlap_pct,
        'chi2_fobi': stats_fobi['chi2_per_dof'],
        'chi2_wiener': stats_wiener['chi2_per_dof'],
        'r2_fobi': stats_fobi['r_squared'],
        'r2_wiener': stats_wiener['r_squared']
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("UNDERSTANDING FRAME OVERLAP AND RECONSTRUCTION DIFFICULTY")
    print("="*70)

    results = []

    # Test 1: Single frame (identity - should be perfect)
    results.append(test_overlap_configuration(
        kernel=[0],
        description="Test 1: Single Frame (No Overlap) - IDENTITY"
    ))

    # Test 2: Two frames, NO overlap
    results.append(test_overlap_configuration(
        kernel=[0, 50],
        description="Test 2: Two Frames, NO Overlap (50ms spacing at 20Hz)"
    ))

    # Test 3: Two frames, 50% overlap (your configuration!)
    results.append(test_overlap_configuration(
        kernel=[0, 25],
        description="Test 3: Two Frames, 50% Overlap (25ms spacing at 20Hz)"
    ))

    # Test 4: Two frames, 75% overlap
    results.append(test_overlap_configuration(
        kernel=[0, 12],
        description="Test 4: Two Frames, 75% Overlap (12ms spacing at 20Hz)"
    ))

    # Test 5: Two frames, 90% overlap
    results.append(test_overlap_configuration(
        kernel=[0, 5],
        description="Test 5: Two Frames, 90% Overlap (5ms spacing at 20Hz)"
    ))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: RECONSTRUCTION DIFFICULTY vs OVERLAP")
    print("="*70)
    print(f"{'Kernel':<15s} {'Overlap':<12s} {'χ²/dof (FOBI)':<15s} {'Quality':<30s}")
    print("-"*70)

    for r in results:
        kernel_str = str(r['kernel'])
        if r['overlap_pct'] == 0:
            overlap_str = "No overlap"
        else:
            overlap_str = f"{r['overlap_pct']:.0f}%"

        chi2_str = f"{r['chi2_fobi']:.3f}"

        if r['chi2_fobi'] < 0.01:
            quality = "Excellent (nearly perfect)"
        elif r['chi2_fobi'] < 0.1:
            quality = "Very good"
        elif r['chi2_fobi'] < 1.0:
            quality = "Good"
        elif r['chi2_fobi'] < 2.0:
            quality = "Fair"
        else:
            quality = "Poor (difficult)"

        print(f"{kernel_str:<15s} {overlap_str:<12s} {chi2_str:<15s} {quality:<30s}")

    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print("1. Non-overlapping frames [0, 50]: EASY to reconstruct (χ²/dof << 1)")
    print("2. 50% overlap [0, 25]: HARDER but still possible")
    print("3. High overlap (>75%): VERY DIFFICULT - reconstruction degrades rapidly")
    print()
    print("YOUR MISTAKE:")
    print("  You thought [0, 25] at 20Hz meant 'non-overlapping'")
    print("  But it actually means 50% OVERLAP!")
    print()
    print("FOR NON-OVERLAPPING FRAMES AT 20Hz:")
    print("  Use kernel = [0, 50] or [0, 51] or larger")
    print("="*70)
