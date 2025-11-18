#!/usr/bin/env python3
"""
Test the overlap method behavior with different modes and compare duty cycle calculations.
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

from frame_overlap import Data

def test_overlap_modes():
    """Test overlap with superimpose vs extend modes"""
    print("=" * 70)
    print("Testing Overlap Modes: Superimpose vs Extend")
    print("=" * 70)

    # Load data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)

    print(f"\nOriginal data:")
    print(f"  Time range: {data.data['time'].min()} - {data.data['time'].max()} µs")
    print(f"  Points: {len(data.data)}")

    # Test 1: Superimpose mode (default)
    data1 = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                 flux=5e6, duration=0.5, freq=20)
    data1.convolute_response(200.0)
    data1.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)

    print(f"\n{'='*70}")
    print("TEST 1: mode='superimpose' (default)")
    print("="*70)
    data1.overlap(kernel=[0, 10, 12, 10], total_time=50, mode='superimpose')

    print(f"\nAfter overlap (superimpose):")
    print(f"  Time range: {data1.overlapped_data['time'].min()} - {data1.overlapped_data['time'].max()} µs")
    print(f"  Points: {len(data1.overlapped_data)}")
    print(f"  Expected: Should keep original time range (0-23990 µs)")
    print(f"  total_time=50 is IGNORED in superimpose mode")

    # Test 2: Extend mode
    data2 = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                 flux=5e6, duration=0.5, freq=20)
    data2.convolute_response(200.0)
    data2.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)

    print(f"\n{'='*70}")
    print("TEST 2: mode='extend'")
    print("="*70)
    data2.overlap(kernel=[0, 10, 12, 10], total_time=50, mode='extend')

    print(f"\nAfter overlap (extend):")
    print(f"  Time range: {data2.overlapped_data['time'].min()} - {data2.overlapped_data['time'].max()} µs")
    print(f"  Points: {len(data2.overlapped_data)}")
    print(f"  Expected: Should extend to 50 ms (50000 µs)")
    print(f"  total_time=50 is USED in extend mode")

    # Test 3: Using freq instead of total_time
    data3 = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                 flux=5e6, duration=0.5, freq=20)
    data3.convolute_response(200.0)
    data3.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)

    print(f"\n{'='*70}")
    print("TEST 3: mode='extend' with freq=20 Hz")
    print("="*70)
    data3.overlap(kernel=[0, 10, 12, 10], freq=20, mode='extend')

    print(f"\nAfter overlap (extend with freq):")
    print(f"  Time range: {data3.overlapped_data['time'].min()} - {data3.overlapped_data['time'].max()} µs")
    print(f"  Points: {len(data3.overlapped_data)}")
    print(f"  Expected: freq=20 Hz → total_time=1000/20=50 ms")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"To extend time window to 50 ms, you MUST use mode='extend':")
    print(f"  data.overlap(kernel=[0,10,12,10], total_time=50, mode='extend')")
    print(f"  OR")
    print(f"  data.overlap(kernel=[0,10,12,10], freq=20, mode='extend')")
    print()
    print(f"The default mode='superimpose' keeps the original time window!")

def test_duty_cycle_calculation():
    """Test duty cycle calculation and compare with Streamlit app"""
    print(f"\n\n{'='*70}")
    print("Testing Duty Cycle Calculation")
    print("="*70)

    # Test with exact Streamlit app parameters
    print(f"\nTest parameters (matching Streamlit app defaults):")
    print(f"  Original: flux=5e6, duration=0.5 hours, freq=20 Hz")
    print(f"  New: flux=1e6, measurement_time=8*60=480 minutes, freq=20 Hz")
    print(f"  Pulse duration: 200 µs")

    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)

    print(f"\n1. Convolve with pulse_duration=200 µs:")
    data.convolute_response(200.0)
    print(f"   pulse_duration stored: {data.pulse_duration} µs")

    print(f"\n2. Apply Poisson sampling:")
    data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)

    # Manual calculation to verify
    print(f"\n3. Manual verification of duty cycle:")
    flux_ratio = 1e6 / 5e6
    time_ratio = (8*60 / 60) / 0.5  # Convert minutes to hours, then divide by duration
    pulse_factor = 20 * (200 / 1e6)
    duty_cycle_manual = flux_ratio * time_ratio * pulse_factor

    print(f"   flux_ratio = {flux_ratio}")
    print(f"   time_ratio = ({8*60}/60) / 0.5 = {time_ratio}")
    print(f"   pulse_factor = 20 × (200/1e6) = {pulse_factor}")
    print(f"   duty_cycle = {flux_ratio} × {time_ratio} × {pulse_factor} = {duty_cycle_manual}")

    # Check the actual counts
    print(f"\n4. Result statistics:")
    print(f"   Poissoned signal mean: {data.poissoned_data['counts'].mean():.2f}")
    print(f"   Poissoned signal max: {data.poissoned_data['counts'].max():.2f}")
    print(f"   Poissoned signal min: {data.poissoned_data['counts'].min():.2f}")

    print(f"\n5. Compare with Streamlit app:")
    print(f"   If you see DIFFERENT statistics in the Streamlit app with the SAME")
    print(f"   parameters, there might be a difference in how parameters are interpreted.")
    print(f"   ")
    print(f"   Common issues:")
    print(f"   - measurement_time: Python API uses MINUTES, check if Streamlit uses HOURS")
    print(f"   - seed: Different seeds will give different random results")
    print(f"   - kernel order: kernel=[0,10,12,10] vs kernel=[0,25]")

if __name__ == '__main__':
    test_overlap_modes()
    test_duty_cycle_calculation()
