#!/usr/bin/env python3
"""
Test wavelength filtering functionality in the Streamlit app.
Verifies that the wavelength-to-TOF conversion works correctly and filters data properly.
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

from frame_overlap import Data

# Constants from streamlit_app.py
PLANCK_CONSTANT = 6.62607015e-34  # J·s
NEUTRON_MASS_KG = 1.67492749804e-27  # kg

def wavelength_to_tof(wavelength_angstrom, flight_path_length_m):
    """Convert neutron wavelength to time-of-flight (in microseconds)"""
    wavelength_m = wavelength_angstrom * 1e-10
    velocity = PLANCK_CONSTANT / (NEUTRON_MASS_KG * wavelength_m)
    tof_seconds = flight_path_length_m / velocity
    return tof_seconds * 1e6  # microseconds

def tof_to_wavelength(tof_us, flight_path_length_m):
    """Convert time-of-flight to wavelength (in Angstroms)"""
    tof_seconds = tof_us * 1e-6
    velocity = flight_path_length_m / tof_seconds
    wavelength_m = PLANCK_CONSTANT / (NEUTRON_MASS_KG * velocity)
    return wavelength_m * 1e10  # Angstroms

def test_wavelength_conversion():
    """Test the wavelength to TOF conversion"""
    print("=" * 60)
    print("Testing Wavelength ↔ TOF Conversion")
    print("=" * 60)

    L = 9.0  # Flight path in meters

    # Test known conversions
    test_cases = [
        (1.0, "1 Å"),
        (2.0, "2 Å"),
        (5.0, "5 Å"),
        (10.0, "10 Å"),
    ]

    print(f"\nFlight path: {L} m\n")
    print(f"{'Wavelength':>12} | {'TOF (µs)':>12} | {'TOF (ms)':>12} | {'Reverse λ':>12}")
    print("-" * 60)

    all_pass = True
    for wavelength_aa, label in test_cases:
        tof_us = wavelength_to_tof(wavelength_aa, L)
        tof_ms = tof_us / 1000

        # Reverse conversion
        wavelength_back = tof_to_wavelength(tof_us, L)

        # Check if conversion is reversible
        is_reversible = np.isclose(wavelength_aa, wavelength_back, rtol=1e-10)
        status = "✓" if is_reversible else "✗"

        print(f"{label:>12} | {tof_us:>12.2f} | {tof_ms:>12.4f} | {wavelength_back:>12.6f} {status}")

        if not is_reversible:
            all_pass = False

    print()
    if all_pass:
        print("✅ All conversion tests passed!")
    else:
        print("❌ Some conversion tests failed!")

    return all_pass

def test_data_filtering():
    """Test filtering data by wavelength range"""
    print("\n" + "=" * 60)
    print("Testing Data Filtering by Wavelength")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
               flux=5e6, duration=0.5, freq=20)

    original_signal_points = len(data.data)
    original_openbeam_points = len(data.op_data)

    print(f"   Signal data points: {original_signal_points}")
    print(f"   Openbeam data points: {original_openbeam_points}")

    # Define wavelength range (default from Streamlit app)
    lambda_min = 1.0  # Å
    lambda_max = 10.0  # Å
    L = 9.0  # m

    print(f"\n2. Converting wavelength range to TOF...")
    tof_min_us = wavelength_to_tof(lambda_min, L)
    tof_max_us = wavelength_to_tof(lambda_max, L)

    print(f"   Wavelength range: {lambda_min:.1f} - {lambda_max:.1f} Å")
    print(f"   TOF range: {tof_min_us:.2f} - {tof_max_us:.2f} µs")
    print(f"   TOF range: {tof_min_us/1000:.4f} - {tof_max_us/1000:.4f} ms")

    # Check original data range
    print(f"\n3. Original data ranges:")
    print(f"   Signal time: {data.data['time'].min():.2f} - {data.data['time'].max():.2f} µs")
    print(f"   Openbeam time: {data.op_data['time'].min():.2f} - {data.op_data['time'].max():.2f} µs")

    # Convert to wavelength to show what we have
    wl_signal_min = tof_to_wavelength(data.data['time'].min(), L)
    wl_signal_max = tof_to_wavelength(data.data['time'].max(), L)
    wl_openbeam_min = tof_to_wavelength(data.op_data['time'].min(), L)
    wl_openbeam_max = tof_to_wavelength(data.op_data['time'].max(), L)

    print(f"   Signal wavelength: {wl_signal_min:.2f} - {wl_signal_max:.2f} Å")
    print(f"   Openbeam wavelength: {wl_openbeam_min:.2f} - {wl_openbeam_max:.2f} Å")

    # Apply filtering
    print(f"\n4. Applying wavelength filter ({lambda_min:.1f}-{lambda_max:.1f} Å)...")

    # Filter signal data
    mask_signal = (data.data['time'] >= tof_min_us) & (data.data['time'] <= tof_max_us)
    data.data = data.data[mask_signal].copy()

    # Filter openbeam data
    mask_openbeam = (data.op_data['time'] >= tof_min_us) & (data.op_data['time'] <= tof_max_us)
    data.op_data = data.op_data[mask_openbeam].copy()

    filtered_signal_points = len(data.data)
    filtered_openbeam_points = len(data.op_data)

    print(f"   Filtered signal points: {filtered_signal_points} (kept {100*filtered_signal_points/original_signal_points:.1f}%)")
    print(f"   Filtered openbeam points: {filtered_openbeam_points} (kept {100*filtered_openbeam_points/original_openbeam_points:.1f}%)")

    # Verify filtered data is within range
    print(f"\n5. Verifying filtered data...")

    signal_in_range = ((data.data['time'] >= tof_min_us) & (data.data['time'] <= tof_max_us)).all()
    openbeam_in_range = ((data.op_data['time'] >= tof_min_us) & (data.op_data['time'] <= tof_max_us)).all()

    print(f"   All signal points in range: {signal_in_range} {'✓' if signal_in_range else '✗'}")
    print(f"   All openbeam points in range: {openbeam_in_range} {'✓' if openbeam_in_range else '✗'}")

    # Show filtered ranges
    if len(data.data) > 0:
        wl_filtered_min = tof_to_wavelength(data.data['time'].min(), L)
        wl_filtered_max = tof_to_wavelength(data.data['time'].max(), L)
        print(f"   Filtered wavelength range: {wl_filtered_min:.2f} - {wl_filtered_max:.2f} Å")

    # Test with different wavelength ranges
    print(f"\n6. Testing different wavelength ranges...")

    test_ranges = [
        (0.5, 2.0, "Narrow range (thermal neutrons)"),
        (1.0, 5.0, "Medium range"),
        (2.0, 15.0, "Wide range"),
    ]

    for wl_min, wl_max, description in test_ranges:
        # Reload data
        data_test = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                        flux=5e6, duration=0.5, freq=20)

        # Convert and filter
        tof_min = wavelength_to_tof(wl_min, L)
        tof_max = wavelength_to_tof(wl_max, L)

        mask = (data_test.data['time'] >= tof_min) & (data_test.data['time'] <= tof_max)
        filtered_count = mask.sum()
        percentage = 100 * filtered_count / len(data_test.data)

        print(f"   {description:30} [{wl_min:.1f}-{wl_max:.1f} Å]: {filtered_count:4d} points ({percentage:.1f}%)")

    print()
    if signal_in_range and openbeam_in_range and filtered_signal_points > 0:
        print("✅ Data filtering tests passed!")
        return True
    else:
        print("❌ Data filtering tests failed!")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Wavelength Filtering Test Suite")
    print("=" * 60)

    test1_pass = test_wavelength_conversion()
    test2_pass = test_data_filtering()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Wavelength Conversion: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Data Filtering:        {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print("=" * 60)

    if test1_pass and test2_pass:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
