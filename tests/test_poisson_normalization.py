"""
Test Poisson sampling normalization with different measurement times.

This test verifies that changing measurement_time affects the statistics
when using poisson_sample().
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from frame_overlap import Data, Reconstruct


def test_poisson_measurement_time_effect():
    """
    Test that different measurement times produce different count levels.

    For a pulsed source, longer measurement time should result in higher counts
    (more pulses collected), which should affect the Poisson-sampled counts.
    """
    signal_path = 'notebooks/iron_powder.csv'
    openbeam_path = 'notebooks/openbeam.csv'

    # Original measurement parameters
    flux_orig = 5e6
    duration_orig = 0.5  # hours
    freq_orig = 20

    # New measurement conditions (same for both tests, except time)
    flux_new = 1e6
    freq_new = 60
    pulse_duration = 200  # µs
    seed = 42

    # Test 1: Short measurement time (0.5 hours = 30 min)
    data_short = Data(signal_path, openbeam_path,
                     flux=flux_orig, duration=duration_orig, freq=freq_orig)
    data_short.convolute_response(pulse_duration, bin_width=10)
    data_short.poisson_sample(flux=flux_new, freq=freq_new,
                             measurement_time=30, seed=seed)  # 30 minutes

    # Test 2: Long measurement time (8 hours = 480 min)
    data_long = Data(signal_path, openbeam_path,
                    flux=flux_orig, duration=duration_orig, freq=freq_orig)
    data_long.convolute_response(pulse_duration, bin_width=10)
    data_long.poisson_sample(flux=flux_new, freq=freq_new,
                            measurement_time=480, seed=seed)  # 480 minutes

    # Get mean counts
    mean_counts_short = data_short.poissoned_data['counts'].mean()
    mean_counts_long = data_long.poissoned_data['counts'].mean()

    print(f"\nMeasurement time effect test:")
    print(f"  30 min measurement: mean counts = {mean_counts_short:.1f}")
    print(f"  480 min measurement: mean counts = {mean_counts_long:.1f}")
    print(f"  Ratio (long/short): {mean_counts_long / mean_counts_short:.2f}")
    print(f"  Expected ratio: {480 / 30:.2f}")

    # The counts should scale with measurement time
    # With 16x longer measurement, we should get ~16x more counts
    expected_ratio = 480 / 30  # = 16
    actual_ratio = mean_counts_long / mean_counts_short

    # Check if the ratio is close to expected (within 20% due to Poisson noise)
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.2, \
        f"Counts didn't scale with measurement time: got {actual_ratio:.2f}, expected {expected_ratio:.2f}"


def test_poisson_duty_cycle_calculation():
    """
    Test the duty cycle calculation for pulsed sources.

    According to the current implementation:
    duty_cycle = (flux_new / flux_orig) × freq_new × pulse_duration

    This doesn't include measurement_time, which may be incorrect.
    """
    signal_path = 'notebooks/iron_powder.csv'
    openbeam_path = 'notebooks/openbeam.csv'

    flux_orig = 5e6
    duration_orig = 0.5
    freq_orig = 20

    flux_new = 1e6
    freq_new = 60
    pulse_duration = 200  # µs

    # Expected duty cycle (current formula)
    flux_ratio = flux_new / flux_orig  # = 0.2
    duty_cycle_expected = flux_ratio * freq_new * (pulse_duration / 1e6)
    # = 0.2 × 60 × 0.0002 = 0.0024

    print(f"\nDuty cycle calculation:")
    print(f"  Flux ratio: {flux_ratio}")
    print(f"  Freq new: {freq_new}")
    print(f"  Pulse duration: {pulse_duration} µs = {pulse_duration/1e6} s")
    print(f"  Duty cycle (current formula): {duty_cycle_expected}")

    # Create data and check the calculated duty cycle
    data = Data(signal_path, openbeam_path,
               flux=flux_orig, duration=duration_orig, freq=freq_orig)
    data.convolute_response(pulse_duration, bin_width=10)

    # Capture the print output to check duty cycle
    # (The current implementation prints the duty cycle)
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        data.poisson_sample(flux=flux_new, freq=freq_new, measurement_time=30, seed=42)
    output = f.getvalue()

    print("\nPoisson sample output:")
    print(output)

    # Check that measurement_time is NOT in the current formula
    assert "measurement_time" not in output or "time_new / time_orig" not in output, \
        "The current implementation should NOT use measurement_time for pulsed sources"


def test_poisson_continuous_vs_pulsed():
    """
    Test the difference between continuous and pulsed source formulas.
    """
    signal_path = 'notebooks/iron_powder.csv'
    openbeam_path = 'notebooks/openbeam.csv'

    flux_orig = 5e6
    duration_orig = 0.5
    freq_orig = 20

    flux_new = 1e6
    freq_new = 60
    measurement_time = 30  # minutes

    # Test 1: Continuous source (no pulse_duration)
    data_continuous = Data(signal_path, openbeam_path,
                          flux=flux_orig, duration=duration_orig, freq=freq_orig)
    # Don't call convolute_response - this makes it continuous

    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        data_continuous.poisson_sample(flux=flux_new, freq=freq_new,
                                      measurement_time=measurement_time, seed=42)
    continuous_output = f.getvalue()

    print("\nContinuous source output:")
    print(continuous_output)

    # Test 2: Pulsed source (with pulse_duration)
    data_pulsed = Data(signal_path, openbeam_path,
                      flux=flux_orig, duration=duration_orig, freq=freq_orig)
    data_pulsed.convolute_response(200, bin_width=10)

    f = io.StringIO()
    with redirect_stdout(f):
        data_pulsed.poisson_sample(flux=flux_new, freq=freq_new,
                                  measurement_time=measurement_time, seed=42)
    pulsed_output = f.getvalue()

    print("\nPulsed source output:")
    print(pulsed_output)

    # Continuous should include time_ratio
    assert "time_new / time_orig" in continuous_output, \
        "Continuous source should include time ratio"

    # Pulsed should include pulse_duration
    assert "pulse_duration" in pulsed_output.lower(), \
        "Pulsed source should include pulse duration"


def test_reconstruction_with_different_times():
    """
    Test that reconstruction quality changes with measurement time.

    Longer measurement times should give better statistics (lower noise).
    """
    signal_path = 'notebooks/iron_powder.csv'
    openbeam_path = 'notebooks/openbeam.csv'

    flux_orig = 5e6
    duration_orig = 0.5
    freq_orig = 20

    flux_new = 1e6
    freq_new = 60
    pulse_duration = 200
    kernel = [0, 25]

    # Test with 0.5 hour measurement
    data_short = Data(signal_path, openbeam_path,
                     flux=flux_orig, duration=duration_orig, freq=freq_orig)
    data_short.convolute_response(pulse_duration, bin_width=10)
    data_short.poisson_sample(flux=flux_new, freq=freq_new,
                             measurement_time=30, seed=42)
    data_short.overlap(kernel=kernel)

    recon_short = Reconstruct(data_short)
    recon_short.filter(kind='wiener', noise_power=0.01)
    stats_short = recon_short.get_statistics()

    # Test with 8 hour measurement
    data_long = Data(signal_path, openbeam_path,
                    flux=flux_orig, duration=duration_orig, freq=freq_orig)
    data_long.convolute_response(pulse_duration, bin_width=10)
    data_long.poisson_sample(flux=flux_new, freq=freq_new,
                            measurement_time=480, seed=42)
    data_long.overlap(kernel=kernel)

    recon_long = Reconstruct(data_long)
    recon_long.filter(kind='wiener', noise_power=0.01)
    stats_long = recon_long.get_statistics()

    print(f"\nReconstruction statistics comparison:")
    print(f"  30 min: χ²/dof = {stats_short['chi2_per_dof']:.2f}, RMSE = {stats_short['rmse']:.4f}")
    print(f"  480 min: χ²/dof = {stats_long['chi2_per_dof']:.2f}, RMSE = {stats_long['rmse']:.4f}")

    # Longer measurement should have different (likely better) statistics
    # Note: We can't guarantee "better" due to the duty cycle issue,
    # but they should definitely be different
    assert stats_short['chi2_per_dof'] != stats_long['chi2_per_dof'], \
        "Statistics should change with measurement time"


if __name__ == "__main__":
    # Run tests
    print("=" * 70)
    print("Testing Poisson normalization with measurement time")
    print("=" * 70)

    try:
        test_poisson_measurement_time_effect()
        print("\n✓ test_poisson_measurement_time_effect PASSED")
    except AssertionError as e:
        print(f"\n✗ test_poisson_measurement_time_effect FAILED: {e}")

    try:
        test_poisson_duty_cycle_calculation()
        print("\n✓ test_poisson_duty_cycle_calculation PASSED")
    except AssertionError as e:
        print(f"\n✗ test_poisson_duty_cycle_calculation FAILED: {e}")

    try:
        test_poisson_continuous_vs_pulsed()
        print("\n✓ test_poisson_continuous_vs_pulsed PASSED")
    except AssertionError as e:
        print(f"\n✗ test_poisson_continuous_vs_pulsed FAILED: {e}")

    try:
        test_reconstruction_with_different_times()
        print("\n✓ test_reconstruction_with_different_times PASSED")
    except AssertionError as e:
        print(f"\n✗ test_reconstruction_with_different_times FAILED: {e}")
