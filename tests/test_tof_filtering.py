"""
Test TOF filtering in Data class initialization.
"""

import sys
sys.path.insert(0, 'src')

from frame_overlap import Data


def test_tof_filtering():
    """Test that tof_min and tof_max filter data during initialization."""
    print("\n" + "="*70)
    print("TEST: TOF Filtering in Data Initialization")
    print("="*70)

    # Load data without filtering
    data_unfiltered = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                           flux=5e6, duration=0.5, freq=20)

    print(f"\nWithout filtering:")
    print(f"  Signal: {len(data_unfiltered.data)} points")
    print(f"  Time range: {data_unfiltered.data['time'].min():.1f} - {data_unfiltered.data['time'].max():.1f} µs")
    print(f"  Openbeam: {len(data_unfiltered.op_data)} points")
    print(f"  Time range: {data_unfiltered.op_data['time'].min():.1f} - {data_unfiltered.op_data['time'].max():.1f} µs")

    # Load data with TOF filtering
    tof_min = 5000  # 5000 µs
    tof_max = 15000  # 15000 µs

    data_filtered = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                         flux=5e6, duration=0.5, freq=20,
                         tof_min=tof_min, tof_max=tof_max)

    print(f"\nWith filtering (tof_min={tof_min} µs, tof_max={tof_max} µs):")
    print(f"  Signal: {len(data_filtered.data)} points")
    print(f"  Time range: {data_filtered.data['time'].min():.1f} - {data_filtered.data['time'].max():.1f} µs")
    print(f"  Openbeam: {len(data_filtered.op_data)} points")
    print(f"  Time range: {data_filtered.op_data['time'].min():.1f} - {data_filtered.op_data['time'].max():.1f} µs")

    # Verify filtering worked
    assert len(data_filtered.data) < len(data_unfiltered.data), "Signal should be filtered"
    assert len(data_filtered.op_data) < len(data_unfiltered.op_data), "Openbeam should be filtered"
    assert data_filtered.data['time'].min() >= tof_min, "Min time should be >= tof_min"
    assert data_filtered.data['time'].max() <= tof_max, "Max time should be <= tof_max"
    assert data_filtered.op_data['time'].min() >= tof_min, "Openbeam min time should be >= tof_min"
    assert data_filtered.op_data['time'].max() <= tof_max, "Openbeam max time should be <= tof_max"

    print(f"\n✅ TOF filtering works correctly!")
    print(f"   Filtered {len(data_unfiltered.data) - len(data_filtered.data)} points from signal")
    print(f"   Filtered {len(data_unfiltered.op_data) - len(data_filtered.op_data)} points from openbeam")

    return True


if __name__ == "__main__":
    try:
        test_tof_filtering()
        print("\n" + "="*70)
        print("✅ TEST PASSED!")
        print("="*70)
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
