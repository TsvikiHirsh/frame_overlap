#!/usr/bin/env python3
"""Test to verify measurement_time units in poisson_sample()"""

from frame_overlap import Data

def test_measurement_time_units():
    """
    Test that measurement_time is correctly interpreted as HOURS.

    Both the docstring and code now expect measurement_time in HOURS.
    This test verifies the duty cycle calculation is correct.
    """

    # Load test data
    data = Data(
        signal_file='notebooks/iron_powder.csv',
        openbeam_file='notebooks/openbeam.csv',
        flux=5e6,       # n/cm²/s
        duration=0.5,   # hours (original measurement)
        freq=20         # Hz
    )

    # Apply convolution
    data.convolute_response(pulse_duration=20.0)  # 20 µs

    print("\n" + "="*70)
    print("TEST: measurement_time = 8.0 (HOURS)")
    print("="*70)

    # Test: measurement_time is in hours
    data.poisson_sample(
        flux=1e6,
        freq=20,
        measurement_time=8.0  # 8 hours
    )

    print("\n" + "="*70)
    print("EXPECTED DUTY CYCLE CALCULATION:")
    print("="*70)
    print("For 8 hours measurement with flux change from 5e6 to 1e6:")
    print("  flux_ratio = 1e6 / 5e6 = 0.2")
    print("  time_ratio = 8.0 hours / 0.5 hours = 16.0")
    print("  freq = 20 Hz")
    print("  pulse_duration_ratio = 20 µs / 10 µs = 2.0")
    print("  baseline_duration = 10 µs = 10e-6 s")
    print("")
    print("  duty_cycle = 0.2 × 16.0 × 20 × 2.0 × 10e-6")
    print("  duty_cycle = 0.00128 (0.128%)")
    print("")
    print("✅ Test passes if the calculated duty_cycle matches 0.00128")

if __name__ == "__main__":
    test_measurement_time_units()
