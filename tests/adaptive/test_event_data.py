"""
Tests for event-mode data structures.

Tests NeutronEvent, EventDataset, and ReconstructionResult classes.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from frame_overlap.adaptive.event_data import NeutronEvent, EventDataset, ReconstructionResult


class TestNeutronEvent:
    """Test NeutronEvent class."""

    def test_create_event_basic(self):
        """Test creating a basic neutron event."""
        event = NeutronEvent(
            detection_time=1000.0,
            trigger_time=950.0,
            pixel_id=42
        )

        assert event.detection_time == 1000.0
        assert event.trigger_time == 950.0
        assert event.pixel_id == 42
        assert event.previous_triggers is None
        assert event.wavelength is None

    def test_create_event_with_previous_triggers(self):
        """Test creating event with previous trigger timestamps."""
        previous = np.array([900.0, 850.0, 800.0, 750.0, 700.0])
        event = NeutronEvent(
            detection_time=1000.0,
            trigger_time=950.0,
            pixel_id=42,
            previous_triggers=previous
        )

        assert event.previous_triggers is not None
        assert len(event.previous_triggers) == 5
        np.testing.assert_array_equal(event.previous_triggers, previous)

    def test_create_event_with_wavelength(self):
        """Test creating event with wavelength."""
        event = NeutronEvent(
            detection_time=1000.0,
            trigger_time=950.0,
            pixel_id=42,
            wavelength=2.5
        )

        assert event.wavelength == 2.5

    def test_time_since_trigger(self):
        """Test time_since_trigger property."""
        event = NeutronEvent(
            detection_time=1000.0,
            trigger_time=950.0,
            pixel_id=42
        )

        assert event.time_since_trigger == 50.0

    def test_all_trigger_times(self):
        """Test all_trigger_times property."""
        previous = np.array([900.0, 850.0, 800.0])
        event = NeutronEvent(
            detection_time=1000.0,
            trigger_time=950.0,
            pixel_id=42,
            previous_triggers=previous
        )

        all_times = event.all_trigger_times
        expected = np.array([950.0, 900.0, 850.0, 800.0])
        np.testing.assert_array_equal(all_times, expected)

    def test_all_trigger_times_no_previous(self):
        """Test all_trigger_times when no previous triggers."""
        event = NeutronEvent(
            detection_time=1000.0,
            trigger_time=950.0,
            pixel_id=42
        )

        all_times = event.all_trigger_times
        expected = np.array([950.0])
        np.testing.assert_array_equal(all_times, expected)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        previous = np.array([900.0, 850.0])
        event = NeutronEvent(
            detection_time=1000.0,
            trigger_time=950.0,
            pixel_id=42,
            previous_triggers=previous,
            wavelength=2.5
        )

        d = event.to_dict()
        assert d['detection_time'] == 1000.0
        assert d['trigger_time'] == 950.0
        assert d['pixel_id'] == 42
        assert d['wavelength'] == 2.5
        np.testing.assert_array_equal(d['previous_triggers'], previous)

    def test_from_dict(self):
        """Test creating event from dictionary."""
        d = {
            'detection_time': 1000.0,
            'trigger_time': 950.0,
            'pixel_id': 42,
            'wavelength': 2.5,
            'previous_triggers': np.array([900.0, 850.0])
        }

        event = NeutronEvent.from_dict(d)
        assert event.detection_time == 1000.0
        assert event.trigger_time == 950.0
        assert event.pixel_id == 42
        assert event.wavelength == 2.5
        np.testing.assert_array_equal(event.previous_triggers, d['previous_triggers'])

    def test_repr(self):
        """Test string representation."""
        event = NeutronEvent(
            detection_time=1000.0,
            trigger_time=950.0,
            pixel_id=42
        )

        repr_str = repr(event)
        assert 'NeutronEvent' in repr_str
        assert '1000.0' in repr_str
        assert '950.0' in repr_str
        assert '42' in repr_str


class TestEventDataset:
    """Test EventDataset class."""

    def test_create_empty_dataset(self):
        """Test creating empty dataset."""
        dataset = EventDataset()
        assert len(dataset) == 0
        assert dataset.n_events == 0

    def test_add_events(self):
        """Test adding events to dataset."""
        dataset = EventDataset()

        event1 = NeutronEvent(1000.0, 950.0, 0)
        event2 = NeutronEvent(1100.0, 1050.0, 1)

        dataset.add_event(event1)
        dataset.add_event(event2)

        assert len(dataset) == 2
        assert dataset.n_events == 2

    def test_add_events_batch(self):
        """Test adding multiple events at once."""
        dataset = EventDataset()

        events = [
            NeutronEvent(1000.0, 950.0, 0),
            NeutronEvent(1100.0, 1050.0, 1),
            NeutronEvent(1200.0, 1150.0, 0)
        ]

        dataset.add_events(events)
        assert len(dataset) == 3

    def test_getitem(self):
        """Test indexing dataset."""
        dataset = EventDataset()
        event = NeutronEvent(1000.0, 950.0, 42)
        dataset.add_event(event)

        retrieved = dataset[0]
        assert retrieved.detection_time == 1000.0
        assert retrieved.trigger_time == 950.0
        assert retrieved.pixel_id == 42

    def test_iteration(self):
        """Test iterating over dataset."""
        dataset = EventDataset()
        events = [
            NeutronEvent(1000.0, 950.0, 0),
            NeutronEvent(1100.0, 1050.0, 1),
            NeutronEvent(1200.0, 1150.0, 0)
        ]
        dataset.add_events(events)

        count = 0
        for event in dataset:
            assert isinstance(event, NeutronEvent)
            count += 1
        assert count == 3

    def test_time_range(self):
        """Test time_range property."""
        dataset = EventDataset()
        events = [
            NeutronEvent(1000.0, 950.0, 0),
            NeutronEvent(1500.0, 1450.0, 1),
            NeutronEvent(1200.0, 1150.0, 0)
        ]
        dataset.add_events(events)

        tmin, tmax = dataset.time_range
        assert tmin == 1000.0
        assert tmax == 1500.0

    def test_time_range_empty(self):
        """Test time_range for empty dataset."""
        dataset = EventDataset()
        tmin, tmax = dataset.time_range
        assert tmin is None
        assert tmax is None

    def test_save_and_load_hdf5(self, tmp_path):
        """Test saving and loading dataset to/from HDF5."""
        # Create dataset
        dataset = EventDataset()
        events = [
            NeutronEvent(1000.0, 950.0, 0, wavelength=2.5),
            NeutronEvent(1100.0, 1050.0, 1, wavelength=3.0,
                        previous_triggers=np.array([1000.0, 950.0])),
            NeutronEvent(1200.0, 1150.0, 0, wavelength=2.8)
        ]
        dataset.add_events(events)

        # Save to file
        filepath = tmp_path / "test_events.h5"
        dataset.save_hdf5(str(filepath))

        # Load from file
        loaded = EventDataset.load_hdf5(str(filepath))

        # Verify
        assert len(loaded) == len(dataset)
        for i in range(len(dataset)):
            orig = dataset[i]
            load = loaded[i]
            assert load.detection_time == orig.detection_time
            assert load.trigger_time == orig.trigger_time
            assert load.pixel_id == orig.pixel_id
            assert load.wavelength == orig.wavelength

    def test_filter_by_time(self):
        """Test filtering events by time range."""
        dataset = EventDataset()
        events = [
            NeutronEvent(1000.0, 950.0, 0),
            NeutronEvent(1500.0, 1450.0, 1),
            NeutronEvent(1200.0, 1150.0, 0),
            NeutronEvent(2000.0, 1950.0, 1)
        ]
        dataset.add_events(events)

        filtered = dataset.filter_by_time(1100.0, 1600.0)
        assert len(filtered) == 2
        assert filtered[0].detection_time == 1500.0
        assert filtered[1].detection_time == 1200.0

    def test_filter_by_pixel(self):
        """Test filtering events by pixel ID."""
        dataset = EventDataset()
        events = [
            NeutronEvent(1000.0, 950.0, 0),
            NeutronEvent(1100.0, 1050.0, 1),
            NeutronEvent(1200.0, 1150.0, 0),
            NeutronEvent(1300.0, 1250.0, 2)
        ]
        dataset.add_events(events)

        filtered = dataset.filter_by_pixel(0)
        assert len(filtered) == 2
        assert all(e.pixel_id == 0 for e in filtered)


class TestReconstructionResult:
    """Test ReconstructionResult class."""

    def test_create_result(self):
        """Test creating reconstruction result."""
        wavelengths = np.linspace(1, 10, 100)
        intensities = np.ones(100) * 1000
        uncertainties = np.sqrt(intensities)

        result = ReconstructionResult(
            wavelengths=wavelengths,
            intensities=intensities,
            uncertainties=uncertainties,
            method='baseline'
        )

        assert len(result.wavelengths) == 100
        assert len(result.intensities) == 100
        assert len(result.uncertainties) == 100
        assert result.method == 'baseline'
        assert result.metadata == {}

    def test_create_result_with_metadata(self):
        """Test creating result with metadata."""
        wavelengths = np.linspace(1, 10, 100)
        intensities = np.ones(100) * 1000

        metadata = {
            'noise_power': 0.1,
            'n_iterations': 10,
            'chi2': 1.5
        }

        result = ReconstructionResult(
            wavelengths=wavelengths,
            intensities=intensities,
            method='wiener',
            metadata=metadata
        )

        assert result.metadata['noise_power'] == 0.1
        assert result.metadata['n_iterations'] == 10
        assert result.metadata['chi2'] == 1.5

    def test_get_spectrum(self):
        """Test getting spectrum as DataFrame."""
        wavelengths = np.array([1.0, 2.0, 3.0])
        intensities = np.array([100, 200, 150])
        uncertainties = np.array([10, 14, 12])

        result = ReconstructionResult(
            wavelengths=wavelengths,
            intensities=intensities,
            uncertainties=uncertainties,
            method='baseline'
        )

        df = result.get_spectrum()
        assert 'wavelength' in df.columns
        assert 'intensity' in df.columns
        assert 'uncertainty' in df.columns
        assert len(df) == 3

    def test_save_and_load_hdf5(self, tmp_path):
        """Test saving and loading result to/from HDF5."""
        wavelengths = np.linspace(1, 10, 100)
        intensities = np.random.poisson(1000, 100)
        uncertainties = np.sqrt(intensities)

        metadata = {'chi2': 1.5, 'method_params': {'noise_power': 0.1}}

        result = ReconstructionResult(
            wavelengths=wavelengths,
            intensities=intensities,
            uncertainties=uncertainties,
            method='wiener',
            metadata=metadata
        )

        # Save
        filepath = tmp_path / "test_result.h5"
        result.save_hdf5(str(filepath))

        # Load
        loaded = ReconstructionResult.load_hdf5(str(filepath))

        # Verify
        np.testing.assert_array_almost_equal(loaded.wavelengths, result.wavelengths)
        np.testing.assert_array_almost_equal(loaded.intensities, result.intensities)
        np.testing.assert_array_almost_equal(loaded.uncertainties, result.uncertainties)
        assert loaded.method == result.method
        assert loaded.metadata['chi2'] == result.metadata['chi2']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
