"""
Tests for GroupBy/parameter sweep functionality in Workflow class.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from frame_overlap import Workflow


@pytest.fixture
def data_files():
    """Provide paths to test data files."""
    base_path = Path(__file__).parent.parent / 'notebooks'
    return {
        'signal': str(base_path / 'iron_powder.csv'),
        'openbeam': str(base_path / 'openbeam.csv')
    }


class TestGroupByPulseDuration:
    """Test GroupBy with pulse_duration parameter."""

    def test_pulse_duration_sweep_basic(self, data_files):
        """Test basic pulse_duration sweep."""
        wf = Workflow(data_files['signal'], data_files['openbeam'],
                     flux=5e6, duration=0.5, freq=20)

        results = (wf
            .groupby('pulse_duration', low=100, high=300, num=3)
            .convolute()  # pulse_duration comes from sweep
            .poisson(flux=1e6, freq=60, measurement_time=30, seed=42)
            .overlap(kernel=[0, 25])
            .reconstruct(kind='wiener', noise_power=0.01)
            .analyze(xs='iron')
            .run(progress_bar=False))

        # Check results structure
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        assert 'pulse_duration' in results.columns
        assert 'redchi2' in results.columns or 'chi2' in results.columns

        # Check pulse_duration values
        assert results['pulse_duration'].min() >= 100
        assert results['pulse_duration'].max() <= 300

        # Check that we got valid results (not all NaN)
        assert not results['chi2'].isna().all(), "All chi2 values are NaN"

    def test_pulse_duration_sweep_with_step(self, data_files):
        """Test pulse_duration sweep using step parameter."""
        wf = Workflow(data_files['signal'], data_files['openbeam'],
                     flux=5e6, duration=0.5, freq=20)

        results = (wf
            .groupby('pulse_duration', low=100, high=200, step=50)
            .convolute()
            .poisson(flux=1e6, freq=60, measurement_time=10, seed=42)
            .overlap(kernel=[0, 25])
            .reconstruct(kind='wiener', noise_power=0.01)
            .analyze(xs='iron')
            .run(progress_bar=False))

        # Should have 3 values: 100, 150, 200
        assert len(results) == 3
        np.testing.assert_array_almost_equal(
            sorted(results['pulse_duration'].values),
            [100, 150, 200]
        )


class TestGroupByNoisePower:
    """Test GroupBy with noise_power parameter."""

    def test_noise_power_sweep(self, data_files):
        """Test noise_power sweep for Wiener filter."""
        wf = Workflow(data_files['signal'], data_files['openbeam'],
                     flux=5e6, duration=0.5, freq=20)

        results = (wf
            .convolute(pulse_duration=200)
            .poisson(flux=1e6, freq=60, measurement_time=30, seed=42)
            .overlap(kernel=[0, 25])
            .groupby('noise_power', low=0.001, high=0.1, num=5)
            .reconstruct(kind='wiener')  # noise_power from sweep
            .analyze(xs='iron')
            .run(progress_bar=False))

        assert len(results) == 5
        assert 'noise_power' in results.columns
        assert results['noise_power'].min() >= 0.001
        assert results['noise_power'].max() <= 0.1

        # Check that we got valid results
        assert not results['chi2'].isna().all(), "All chi2 values are NaN"


class TestGroupByIterations:
    """Test GroupBy with iterations parameter for Lucy-Richardson."""

    def test_iterations_sweep(self, data_files):
        """Test iterations sweep for Lucy-Richardson filter."""
        wf = Workflow(data_files['signal'], data_files['openbeam'],
                     flux=5e6, duration=0.5, freq=20)

        results = (wf
            .convolute(pulse_duration=200)
            .poisson(flux=1e6, freq=60, measurement_time=30, seed=42)
            .overlap(kernel=[0, 25])
            .groupby('iterations', low=5, high=25, step=10)
            .reconstruct(kind='lucy')  # iterations from sweep
            .analyze(xs='iron')
            .run(progress_bar=False))

        # Should have 3 values: 5, 15, 25
        assert len(results) == 3
        assert 'iterations' in results.columns
        assert not results['chi2'].isna().all(), "All chi2 values are NaN"


class TestGroupByWithMultipleFrames:
    """Test GroupBy works correctly with different numbers of overlapping frames."""

    def test_two_frames(self, data_files):
        """Test sweep with 2-frame overlap."""
        wf = Workflow(data_files['signal'], data_files['openbeam'],
                     flux=5e6, duration=0.5, freq=20)

        results = (wf
            .convolute(pulse_duration=200)
            .groupby('noise_power', low=0.01, high=0.05, num=3)
            .poisson(flux=1e6, freq=60, measurement_time=30, seed=42)
            .overlap(kernel=[0, 25])  # 2 frames
            .reconstruct(kind='wiener')
            .analyze(xs='iron')
            .run(progress_bar=False))

        assert len(results) == 3
        assert not results['chi2'].isna().all()

    def test_three_frames(self, data_files):
        """Test sweep with 3-frame overlap."""
        wf = Workflow(data_files['signal'], data_files['openbeam'],
                     flux=5e6, duration=0.5, freq=20)

        results = (wf
            .convolute(pulse_duration=200)
            .groupby('noise_power', low=0.01, high=0.05, num=3)
            .poisson(flux=1e6, freq=60, measurement_time=30, seed=42)
            .overlap(kernel=[0, 15, 30])  # 3 frames
            .reconstruct(kind='wiener')
            .analyze(xs='iron')
            .run(progress_bar=False))

        assert len(results) == 3
        assert not results['chi2'].isna().all()


class TestGroupByResultsContent:
    """Test that GroupBy results contain expected columns and values."""

    def test_results_columns(self, data_files):
        """Test that results contain all expected columns."""
        wf = Workflow(data_files['signal'], data_files['openbeam'],
                     flux=5e6, duration=0.5, freq=20)

        results = (wf
            .convolute(pulse_duration=200)
            .groupby('noise_power', low=0.01, high=0.05, num=3)
            .poisson(flux=1e6, freq=60, measurement_time=30, seed=42)
            .overlap(kernel=[0, 25])
            .reconstruct(kind='wiener')
            .analyze(xs='iron')
            .run(progress_bar=False))

        # Check required columns
        required_cols = ['noise_power', 'chi2', 'redchi2', 'aic', 'bic']
        for col in required_cols:
            assert col in results.columns, f"Missing column: {col}"

        # Check for fitted parameter columns
        assert any('param_' in col for col in results.columns), \
            "No fitted parameter columns found"

    def test_parameter_variations(self, data_files):
        """Test that swept parameter actually varies in results."""
        wf = Workflow(data_files['signal'], data_files['openbeam'],
                     flux=5e6, duration=0.5, freq=20)

        results = (wf
            .convolute(pulse_duration=200)
            .groupby('noise_power', low=0.01, high=0.1, num=5)
            .poisson(flux=1e6, freq=60, measurement_time=30, seed=42)
            .overlap(kernel=[0, 25])
            .reconstruct(kind='wiener')
            .analyze(xs='iron')
            .run(progress_bar=False))

        # Check that noise_power varies
        unique_values = results['noise_power'].nunique()
        assert unique_values == 5, f"Expected 5 unique values, got {unique_values}"

        # Check values are in expected range
        assert results['noise_power'].min() >= 0.01
        assert results['noise_power'].max() <= 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
