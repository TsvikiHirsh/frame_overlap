"""
Example: Two-Stage Measurement with TOF Offset Correction

This example demonstrates:
1. Measuring openbeam once with high precision (reusable)
2. Using adaptive patterns for signal measurement
3. Correcting TOF offset after Wiener reconstruction
4. Calculating time savings from openbeam reuse

Author: frame_overlap team
"""

import numpy as np
import matplotlib.pyplot as plt
from frame_overlap import (
    # Two-stage measurement
    TwoStageMeasurementStrategy,
    OpenbeamLibrary,
    estimate_openbeam_time_savings,
    # TOF offset correction
    TOFOffsetCorrector,
    apply_offset_correction_to_workflow,
    estimate_expected_offset,
    # Core components
    BraggEdgeSample,
    BraggEdgeMeasurementSystem,
    MeasurementTarget,
    TOFCalibration,
    IncidentSpectrum
)


def example_two_stage_measurement():
    """
    Example 1: Two-stage measurement strategy
    """
    print("=" * 70)
    print("EXAMPLE 1: TWO-STAGE MEASUREMENT STRATEGY")
    print("=" * 70)
    print()

    # Set up measurement system
    system = BraggEdgeMeasurementSystem(
        flight_path=10.0,
        wavelength_range=(3.0, 5.0),
        n_wavelength_bins=500,
        n_time_bins=1000
    )

    # Create two-stage strategy
    strategy = TwoStageMeasurementStrategy(
        system=system,
        openbeam_precision=0.01,  # 1% precision
        reuse_openbeam=True
    )

    print("Step 1: Measure openbeam (once, reusable)")
    print("-" * 70)

    # Measure openbeam - this is done once and reused
    openbeam = strategy.measure_openbeam(
        flux=1e6,
        target_counts_per_bin=10000
    )

    print(f"  Openbeam measurement complete!")
    print(f"  Time required: {openbeam.total_time:.1f} s")
    print(f"  Total counts: {np.sum(openbeam.counts):.0e}")
    print(f"  Average SNR: {np.mean(openbeam.counts / openbeam.uncertainty):.1f}")
    print()

    # Store in library for reuse
    library = OpenbeamLibrary()
    library.add_openbeam(
        "standard_10m",
        openbeam,
        metadata={'flight_path': 10.0, 'flux': 1e6}
    )

    print("Step 2: Measure multiple samples adaptively")
    print("-" * 70)

    # Measure 3 different samples
    n_samples = 3
    samples_data = []

    for i in range(n_samples):
        print(f"\n  Sample {i+1}/{n_samples}:")

        # Define target for this sample
        target = MeasurementTarget(
            material='Fe',
            expected_edge=4.05 + i * 0.01,  # Slightly different edges
            precision_required=0.005,
            max_measurement_time=200.0
        )

        # Measure signal only (reuse openbeam)
        signal_result = strategy.measure_signal_adaptive(
            target,
            flux=1e6,
            measurement_time_per_pattern=10.0,
            strategy='bayesian'
        )

        print(f"    Signal measurement time: {signal_result.measurement_time:.1f} s")
        print(f"    Edge position: {signal_result.edge_position:.4f} Å")
        print(f"    Precision: {signal_result.edge_uncertainty:.4f} Å")

        samples_data.append(signal_result)

    print()
    print("Step 3: Calculate time savings")
    print("-" * 70)

    # Calculate time savings from reusing openbeam
    avg_signal_time = np.mean([s.measurement_time for s in samples_data])

    # For comparison: traditional approach measures both for each sample
    traditional_time_per_sample = openbeam.total_time + avg_signal_time * 1.5  # Uniform is slower

    savings = estimate_openbeam_time_savings(
        n_samples=n_samples,
        openbeam_time=openbeam.total_time,
        signal_time_adaptive=avg_signal_time,
        signal_time_uniform=avg_signal_time * 1.5
    )

    print(f"  Traditional approach (uniform, no reuse):")
    print(f"    Total time: {savings['traditional_total_time']:.1f} s")
    print()
    print(f"  Two-stage approach (adaptive + reuse):")
    print(f"    Total time: {savings['two_stage_adaptive_time']:.1f} s")
    print()
    print(f"  Combined speedup: {savings['combined_speedup']:.2f}x")
    print(f"  Time saved: {savings['time_saved_vs_traditional']:.1f} s")
    print()


def example_tof_offset_correction():
    """
    Example 2: TOF offset correction after Wiener reconstruction
    """
    print("=" * 70)
    print("EXAMPLE 2: TOF OFFSET CORRECTION")
    print("=" * 70)
    print()

    # Create synthetic reconstructed data with known offset
    n_bins = 1000
    tof_bins = np.linspace(0, 0.01, n_bins)  # 10 ms range

    # Create ideal transmission with edge at known position
    wavelength = np.linspace(3.0, 5.0, n_bins)
    edge_position_wavelength = 4.05  # Angstrom

    # Simple edge function
    transmission_ideal = 0.95 * (1 - 0.5 / (1 + np.exp(-50 * (wavelength - edge_position_wavelength))))

    # Simulate Wiener reconstruction with offset
    # After Wiener, data is shifted
    offset_bins = 15  # Systematic offset from source spectrum peak

    # Apply offset to simulated reconstructed data
    transmission_shifted = np.roll(transmission_ideal, offset_bins)

    print("Simulated scenario:")
    print(f"  True edge position: {edge_position_wavelength} Å")
    print(f"  Systematic TOF offset: {offset_bins} bins")
    print()

    # Method 1: Correct using known edge position
    print("Method 1: Correction using known edge position")
    print("-" * 70)

    # Create TOF calibration
    tof_calib = TOFCalibration(flight_path=10.0)

    corrector = TOFOffsetCorrector(tof_bins)

    result = corrector.correct_by_edge_position(
        transmission_shifted,
        expected_edge_wavelength=edge_position_wavelength,
        tof_to_wavelength=tof_calib.tof_to_wavelength,
        search_window=0.5
    )

    print(f"  Detected offset: {result.offset:.1f} bins")
    print(f"  Offset uncertainty: {result.offset_uncertainty:.1f} bins")
    print(f"  Quality metric: {result.quality_metric:.3f}")
    print(f"  Correction error: {abs(result.offset - offset_bins):.1f} bins")
    print()

    # Method 2: Correct using kernel peak
    print("Method 2: Correction using source spectrum kernel")
    print("-" * 70)

    # Create kernel (source spectrum)
    # Maxwellian peaks away from center
    kernel = np.exp(-((np.arange(100) - 65)**2) / (2 * 10**2))

    # Estimate expected offset from kernel
    expected_offset, explanation = estimate_expected_offset(kernel, source_type='maxwellian')

    print(f"  {explanation}")
    print()

    corrector_with_kernel = TOFOffsetCorrector(tof_bins, kernel=kernel)

    result_kernel = corrector_with_kernel.correct_by_kernel_peak(transmission_shifted)

    print(f"  Applied offset: {result_kernel.offset:.1f} bins")
    print()

    # Method 3: Auto-correction (convenience function)
    print("Method 3: Auto-correction (convenience function)")
    print("-" * 70)

    corrected, result_auto = apply_offset_correction_to_workflow(
        reconstructed_data=transmission_shifted,
        tof_bins=tof_bins,
        kernel=kernel,
        expected_edge_wavelength=edge_position_wavelength,
        tof_to_wavelength=tof_calib.tof_to_wavelength,
        method='auto'
    )

    print(f"  Auto-selected method: {result_auto.correction_method}")
    print(f"  Detected offset: {result_auto.offset:.1f} bins")
    print(f"  Correction successful!")
    print()

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Original vs shifted
    axes[0, 0].plot(wavelength, transmission_ideal, 'b-', label='Ideal (no offset)', linewidth=2)
    axes[0, 0].plot(wavelength, transmission_shifted, 'r--', label=f'Shifted (+{offset_bins} bins)', linewidth=2)
    axes[0, 0].axvline(edge_position_wavelength, color='g', linestyle=':', label='True edge')
    axes[0, 0].set_xlabel('Wavelength (Å)')
    axes[0, 0].set_ylabel('Transmission')
    axes[0, 0].set_title('Before Correction: Systematic Offset')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Corrected data
    axes[0, 1].plot(wavelength, transmission_ideal, 'b-', label='Ideal', linewidth=2)
    axes[0, 1].plot(wavelength, result.corrected_data, 'g--', label='Corrected', linewidth=2, alpha=0.7)
    axes[0, 1].axvline(edge_position_wavelength, color='g', linestyle=':', label='True edge')
    axes[0, 1].set_xlabel('Wavelength (Å)')
    axes[0, 1].set_ylabel('Transmission')
    axes[0, 1].set_title('After Correction: Aligned')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Gradients (edge detection)
    grad_ideal = np.abs(np.gradient(transmission_ideal))
    grad_shifted = np.abs(np.gradient(transmission_shifted))
    grad_corrected = np.abs(np.gradient(result.corrected_data))

    axes[1, 0].plot(wavelength, grad_ideal, 'b-', label='Ideal', linewidth=2)
    axes[1, 0].plot(wavelength, grad_shifted, 'r--', label='Shifted', linewidth=2)
    axes[1, 0].plot(wavelength, grad_corrected, 'g-.', label='Corrected', linewidth=2)
    axes[1, 0].set_xlabel('Wavelength (Å)')
    axes[1, 0].set_ylabel('|Gradient|')
    axes[1, 0].set_title('Edge Detection via Gradient')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Kernel (source spectrum)
    axes[1, 1].plot(kernel, 'k-', linewidth=2)
    axes[1, 1].axvline(len(kernel)//2, color='b', linestyle='--', label='Center')
    axes[1, 1].axvline(expected_offset + len(kernel)//2, color='r', linestyle='--', label='Peak')
    axes[1, 1].set_xlabel('Bin')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].set_title(f'Source Spectrum Kernel (offset={expected_offset:.0f} bins)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tof_offset_correction_example.png', dpi=150)
    print(f"  Saved figure: tof_offset_correction_example.png")
    print()


def example_complete_workflow():
    """
    Example 3: Complete workflow with both features
    """
    print("=" * 70)
    print("EXAMPLE 3: COMPLETE WORKFLOW")
    print("=" * 70)
    print()

    print("Scenario: Measure strain in 5 steel samples")
    print("  - Openbeam: measured once, 1% precision")
    print("  - Signals: adaptive patterns, 0.005 Å precision")
    print("  - TOF correction: automatic using edge position")
    print()

    # Calculate expected time savings
    savings = estimate_openbeam_time_savings(
        n_samples=5,
        openbeam_time=120.0,  # 2 minutes for high-precision openbeam
        signal_time_adaptive=50.0,  # 50 seconds per sample (adaptive)
        signal_time_uniform=150.0  # 2.5 minutes per sample (uniform)
    )

    print("Time Analysis:")
    print("-" * 70)
    print(f"  Traditional (uniform, measure openbeam each time):")
    print(f"    Time per sample: {(120 + 150):.0f} s")
    print(f"    Total for 5 samples: {savings['traditional_total_time']:.0f} s ({savings['traditional_total_time']/60:.1f} min)")
    print()
    print(f"  Two-stage adaptive (reuse openbeam, adaptive signals):")
    print(f"    Openbeam (once): 120 s")
    print(f"    Signal per sample: 50 s")
    print(f"    Total for 5 samples: {savings['two_stage_adaptive_time']:.0f} s ({savings['two_stage_adaptive_time']/60:.1f} min)")
    print()
    print(f"  Speedup breakdown:")
    print(f"    From openbeam reuse: {savings['speedup_from_reuse']:.2f}x")
    print(f"    From adaptive patterns: {savings['speedup_from_adaptive']:.2f}x")
    print(f"    Combined speedup: {savings['combined_speedup']:.2f}x")
    print()
    print(f"  Total time saved: {savings['time_saved_vs_traditional']:.0f} s ({savings['time_saved_vs_traditional']/60:.1f} min)")
    print()


if __name__ == "__main__":
    # Run all examples
    example_two_stage_measurement()
    print("\n\n")

    example_tof_offset_correction()
    print("\n\n")

    example_complete_workflow()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
