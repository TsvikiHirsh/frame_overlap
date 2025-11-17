#!/usr/bin/env python3
"""
Test duty cycle calculation with specific parameters to compare Python API vs Streamlit app.
User reports: 24 hours, 10 µs pulse duration, 20 Hz, flux 1e6 → app shows 0.192% duty cycle
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

from frame_overlap import Data

def test_duty_cycle_24h_10us():
    """Test with user's specific parameters: 24h, 10µs, 20Hz, 1e6 flux"""
    print("=" * 70)
    print("Duty Cycle Test: User's Parameters")
    print("=" * 70)

    print("\nUser's parameters from Streamlit app:")
    print("  Original: flux=5e6, duration=0.5 hours, freq=20 Hz")
    print("  New: measurement_time=24 hours, freq=20 Hz, flux=1e6")
    print("  Pulse duration: 10 µs")
    print("  App shows: 0.192% duty cycle")

    # Load data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)

    # Convolve with 10 µs pulse duration
    print("\n" + "="*70)
    print("TEST 1: Using Python API (measurement_time in MINUTES)")
    print("="*70)
    data.convolute_response(10.0)  # 10 µs pulse duration

    # measurement_time in MINUTES for Python API
    measurement_time_minutes = 24 * 60  # 24 hours = 1440 minutes
    print(f"\nApplying Poisson with:")
    print(f"  flux = 1e6")
    print(f"  measurement_time = {measurement_time_minutes} minutes ({measurement_time_minutes/60} hours)")
    print(f"  freq = 20 Hz")

    data.poisson_sample(flux=1e6, freq=20, measurement_time=measurement_time_minutes, seed=42)

    # Manual calculation
    print("\n" + "="*70)
    print("Manual Calculation:")
    print("="*70)
    flux_ratio = 1e6 / 5e6
    time_ratio = (measurement_time_minutes / 60) / 0.5  # Convert to hours, then ratio
    pulse_factor = 20 * (10.0 / 1e6)  # freq × (pulse_duration / 1e6)
    duty_cycle_manual = flux_ratio * time_ratio * pulse_factor
    duty_cycle_percent = duty_cycle_manual * 100

    print(f"  flux_ratio = {flux_ratio:.6f}")
    print(f"  time_ratio = ({measurement_time_minutes}/60) / 0.5 = {time_ratio:.6f}")
    print(f"  pulse_factor = 20 × (10/1e6) = {pulse_factor:.6f}")
    print(f"  duty_cycle = {flux_ratio:.6f} × {time_ratio:.6f} × {pulse_factor:.6f}")
    print(f"  duty_cycle = {duty_cycle_manual:.6f} = {duty_cycle_percent:.3f}%")

    print(f"\n  Expected from app: 0.192%")
    print(f"  Python API gives:  {duty_cycle_percent:.3f}%")

    if abs(duty_cycle_percent - 0.192) < 0.001:
        print(f"  ✅ MATCH!")
    else:
        print(f"  ❌ MISMATCH! Difference: {abs(duty_cycle_percent - 0.192):.3f}%")

    # Show statistics
    print(f"\nPoisson sampling results:")
    print(f"  Signal mean: {data.poissoned_data['counts'].mean():.2f}")
    print(f"  Signal max: {data.poissoned_data['counts'].max():.2f}")
    print(f"  Signal std: {data.poissoned_data['counts'].std():.2f}")

    # Test 2: What if Streamlit app interprets measurement_time differently?
    print("\n" + "="*70)
    print("TEST 2: Alternative interpretation (measurement_time already in HOURS)")
    print("="*70)

    data2 = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                 flux=5e6, duration=0.5, freq=20)
    data2.convolute_response(10.0)

    # What if Streamlit passes hours directly as "minutes"?
    measurement_time_as_hours = 24  # Just the number 24
    print(f"\nWhat if Streamlit passes measurement_time={measurement_time_as_hours} (meaning hours)?")

    # Manual calculation with this interpretation
    flux_ratio2 = 1e6 / 5e6
    time_ratio2 = (measurement_time_as_hours / 60) / 0.5  # Interpreting 24 as minutes
    pulse_factor2 = 20 * (10.0 / 1e6)
    duty_cycle_manual2 = flux_ratio2 * time_ratio2 * pulse_factor2
    duty_cycle_percent2 = duty_cycle_manual2 * 100

    print(f"  If measurement_time={measurement_time_as_hours} is interpreted as minutes:")
    print(f"    time_ratio = ({measurement_time_as_hours}/60) / 0.5 = {time_ratio2:.6f}")
    print(f"    duty_cycle = {duty_cycle_manual2:.6f} = {duty_cycle_percent2:.3f}%")

    if abs(duty_cycle_percent2 - 0.192) < 0.001:
        print(f"    ✅ MATCH! This might be the issue!")
    else:
        print(f"    ❌ Still doesn't match")

    # Test 3: Check Streamlit app code
    print("\n" + "="*70)
    print("TEST 3: Checking Streamlit app conversion")
    print("="*70)

    # Streamlit might convert hours to minutes internally
    # measurement_time slider in Streamlit shows HOURS but might pass MINUTES
    streamlit_slider_value = 24  # User sees 24 hours
    streamlit_internal_minutes = streamlit_slider_value * 60  # App converts to minutes?

    flux_ratio3 = 1e6 / 5e6
    time_ratio3 = (streamlit_internal_minutes / 60) / 0.5
    pulse_factor3 = 20 * (10.0 / 1e6)
    duty_cycle_manual3 = flux_ratio3 * time_ratio3 * pulse_factor3
    duty_cycle_percent3 = duty_cycle_manual3 * 100

    print(f"  If Streamlit slider shows {streamlit_slider_value} hours:")
    print(f"  And internally converts to {streamlit_internal_minutes} minutes:")
    print(f"    duty_cycle = {duty_cycle_percent3:.3f}%")

    if abs(duty_cycle_percent3 - 0.192) < 0.001:
        print(f"    ✅ MATCH!")
    else:
        print(f"    ❌ Still doesn't match")

    # Test 4: Maybe it's the pulse duration interpretation?
    print("\n" + "="*70)
    print("TEST 4: Check pulse duration scaling")
    print("="*70)

    # What if pulse duration is interpreted differently?
    # Some systems use pulse width as fraction of period instead of absolute time
    period_ms = 1000 / 20  # 20 Hz = 50 ms period
    pulse_duration_us = 10
    pulse_fraction = pulse_duration_us / (period_ms * 1000)  # fraction of period

    print(f"  Period at 20 Hz: {period_ms} ms = {period_ms * 1000} µs")
    print(f"  Pulse duration: {pulse_duration_us} µs")
    print(f"  Pulse fraction: {pulse_duration_us}/{period_ms * 1000} = {pulse_fraction:.6f}")
    print(f"  As percentage: {pulse_fraction * 100:.3f}%")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nExpected (from Streamlit app): 0.192%")
    print(f"Python API calculation:        {duty_cycle_percent:.3f}%")
    print(f"\nDifference: {abs(duty_cycle_percent - 0.192):.3f}%")

    if abs(duty_cycle_percent - 0.192) > 0.001:
        print(f"\n⚠️  WARNING: Duty cycles don't match!")
        print(f"\nPossible reasons:")
        print(f"1. Streamlit app might interpret measurement_time units differently")
        print(f"2. Different pulse duration values (10 vs something else)")
        print(f"3. Different original parameters (flux, duration, freq)")
        print(f"\nPlease check:")
        print(f"- In Streamlit sidebar, Stage 2 'Poisson Sampling':")
        print(f"  - Measurement Time slider value")
        print(f"  - Flux value")
        print(f"  - Frequency value")
        print(f"- In Streamlit sidebar, Stage 1 'Instrument Response':")
        print(f"  - Pulse Duration value")

def test_reverse_engineer_duty_cycle():
    """Try to reverse-engineer what parameters give 0.192%"""
    print("\n\n" + "="*70)
    print("Reverse Engineering: What gives 0.192% duty cycle?")
    print("="*70)

    target_duty_cycle = 0.192 / 100  # 0.192% = 0.00192

    # Known parameters
    flux_orig = 5e6
    duration_orig = 0.5
    flux_new = 1e6
    freq = 20

    print(f"\nTarget duty cycle: {target_duty_cycle:.6f} (0.192%)")
    print(f"Known parameters:")
    print(f"  flux_orig = {flux_orig}")
    print(f"  duration_orig = {duration_orig} hours")
    print(f"  flux_new = {flux_new}")
    print(f"  freq = {freq} Hz")

    # Test different pulse durations and measurement times
    print(f"\nTrying different combinations:")
    print(f"{'Pulse (µs)':<12} {'Meas Time (h)':<15} {'Duty Cycle %':<15} {'Match?'}")
    print("-" * 60)

    for pulse_us in [10, 20, 50, 100, 200]:
        for meas_time_h in [1, 2, 4, 8, 12, 24, 48]:
            flux_ratio = flux_new / flux_orig
            time_ratio = meas_time_h / duration_orig
            pulse_factor = freq * (pulse_us / 1e6)
            duty_cycle = flux_ratio * time_ratio * pulse_factor
            duty_cycle_percent = duty_cycle * 100

            match = "✓" if abs(duty_cycle_percent - 0.192) < 0.001 else ""
            print(f"{pulse_us:<12} {meas_time_h:<15} {duty_cycle_percent:<15.3f} {match}")

            if match:
                print(f"  → FOUND: pulse={pulse_us}µs, meas_time={meas_time_h}h gives {duty_cycle_percent:.3f}%")

if __name__ == '__main__':
    test_duty_cycle_24h_10us()
    test_reverse_engineer_duty_cycle()
