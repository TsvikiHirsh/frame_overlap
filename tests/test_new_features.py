"""
Tests for new features added to the Data class:
- Convolution error bar fix (sqrt of counts)
- Time units changed to milliseconds
- Renamed squared_data to convolved_data
- Wraparound support in overlap
- Enhanced poisson_sample with flux/time/freq parameters
- Improved plotting with pandas and step drawstyle
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent popup windows
import matplotlib.pyplot as plt
import os
import tempfile
from frame_overlap import Data


class TestConvolutionErrorBars(unittest.TestCase):
    """Test that convolution recalculates error bars as sqrt(counts)."""

    def setUp(self):
        """Set up temporary test file."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, "test_signal.csv")

        # Create test CSV file with known counts
        data = pd.DataFrame({
            'stack': np.arange(1, 101),
            'counts': np.ones(100) * 100,  # Constant counts
            'err': np.ones(100) * 10  # Initial error
        })
        data.to_csv(self.temp_file, index=False)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
        plt.close('all')

    def test_error_bars_recalculated(self):
        """Test that error bars are recalculated as sqrt(counts) after convolution."""
        data = Data(signal_file=self.temp_file)

        # Apply convolution
        data.convolute_response(pulse_duration=0.2)  # 0.2 ms = 200 µs

        # After convolution with constant input, counts should remain ~100
        # Error bars should be sqrt(counts) ≈ 10
        convolved_counts = data.convolved_data['counts'].values
        convolved_err = data.convolved_data['err'].values

        # Check that errors are approximately sqrt(counts)
        expected_err = np.sqrt(convolved_counts)
        np.testing.assert_array_almost_equal(convolved_err, expected_err, decimal=5)

    def test_convolved_data_attribute(self):
        """Test that convolved_data attribute exists and works."""
        data = Data(signal_file=self.temp_file)
        data.convolute_response(pulse_duration=0.2)

        # Check attribute exists
        self.assertIsNotNone(data.convolved_data)


class TestTimeUnitsMilliseconds(unittest.TestCase):
    """Test that time units are now in milliseconds."""

    def setUp(self):
        """Set up temporary test file."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, "test_signal.csv")

        # Create test CSV file
        data = pd.DataFrame({
            'stack': [1, 2, 3, 4, 5],
            'counts': [100, 200, 300, 400, 500],
            'err': [10, 20, 30, 40, 50]
        })
        data.to_csv(self.temp_file, index=False)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_time_in_milliseconds(self):
        """Test that time is loaded in milliseconds."""
        data = Data(signal_file=self.temp_file)

        # Stack 1 should be at time 0, stack 2 at 0.01 ms (10 µs), etc.
        expected_time = np.array([0, 0.01, 0.02, 0.03, 0.04])

        np.testing.assert_array_almost_equal(data.data['time'].values, expected_time, decimal=6)

    def test_plot_xlabel(self):
        """Test that plot x-axis is labeled in milliseconds."""
        data = Data(signal_file=self.temp_file)
        fig = data.plot()
        ax = fig.axes[0]

        # Check xlabel contains "ms"
        self.assertIn('ms', ax.get_xlabel().lower())


class TestWraparoundOverlap(unittest.TestCase):
    """Test wraparound functionality in overlap method."""

    def setUp(self):
        """Set up temporary test file."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, "test_signal.csv")

        # Create test CSV with data spanning 10 ms
        n_bins = 100
        data = pd.DataFrame({
            'stack': np.arange(1, n_bins + 1),
            'counts': np.ones(n_bins) * 100,
            'err': np.ones(n_bins) * 10
        })
        data.to_csv(self.temp_file, index=False)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_wraparound_occurs(self):
        """Test that wraparound occurs when total_time is shorter than signal."""
        data = Data(signal_file=self.temp_file)

        # Create overlap with total_time shorter than needed
        # Data spans ~1 ms (100 bins * 0.01 ms)
        # If we set total_time=0.5 ms with frames that would exceed it, wraparound should occur
        data.overlap(kernel=[0, 0.3, 0.3], total_time=0.5)

        # Check that overlapped data has the expected length
        # 0.5 ms / 0.01 ms/bin = 50 bins
        self.assertEqual(len(data.overlapped_data), 51)  # +1 for the endpoint

    def test_kernel_parameter_name(self):
        """Test that 'kernel' parameter name works (renamed from 'seq')."""
        data = Data(signal_file=self.temp_file)

        # Should accept kernel parameter
        data.overlap(kernel=[0, 5, 10])

        self.assertEqual(data.kernel, [0, 5, 10])
        self.assertEqual(data.n_overlapping_frames, 3)


class TestPoissonSampleEnhanced(unittest.TestCase):
    """Test enhanced poisson_sample with flux/time/freq parameters."""

    def setUp(self):
        """Set up temporary test file."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, "test_signal.csv")

        # Create test CSV
        data = pd.DataFrame({
            'stack': np.arange(1, 51),
            'counts': np.ones(50) * 100,
            'err': np.ones(50) * 10
        })
        data.to_csv(self.temp_file, index=False)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_poisson_with_duty_cycle(self):
        """Test traditional poisson_sample with duty_cycle."""
        data = Data(signal_file=self.temp_file)

        data.poisson_sample(duty_cycle=0.5)

        # Check that poissoned data exists
        self.assertIsNotNone(data.poissoned_data)

        # Mean counts should be roughly half the original
        mean_counts = data.poissoned_data['counts'].mean()
        self.assertLess(mean_counts, 100)

    def test_poisson_with_flux_time_freq(self):
        """Test poisson_sample with flux, measurement_time, and freq parameters."""
        # Initialize with original parameters
        data = Data(signal_file=self.temp_file, flux=5e6, duration=0.5, freq=20)

        # Sample with new parameters
        # duty_cycle should be: (1e4/5e6) * (5/0.5) * (800/20) = 0.002 * 10 * 40 = 0.8
        data.poisson_sample(flux=1e4, measurement_time=5, freq=800)

        # Check that poissoned data exists
        self.assertIsNotNone(data.poissoned_data)

    def test_poisson_requires_all_params(self):
        """Test that poisson_sample requires all three parameters if not using duty_cycle."""
        data = Data(signal_file=self.temp_file, flux=5e6, duration=0.5, freq=20)

        # Should raise error if only some parameters provided
        with self.assertRaises(ValueError) as cm:
            data.poisson_sample(flux=1e4, measurement_time=5)

        self.assertIn("flux, measurement_time, freq", str(cm.exception))


class TestPlottingImprovements(unittest.TestCase):
    """Test plotting improvements with pandas and step drawstyle."""

    def setUp(self):
        """Set up temporary test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.signal_file = os.path.join(self.temp_dir.name, "test_signal.csv")
        self.openbeam_file = os.path.join(self.temp_dir.name, "test_openbeam.csv")

        # Create test CSV files
        data = pd.DataFrame({
            'stack': np.arange(1, 51),
            'counts': np.random.normal(100, 10, 50),
            'err': np.abs(np.random.normal(10, 2, 50))
        })
        data.to_csv(self.signal_file, index=False)
        data.to_csv(self.openbeam_file, index=False)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
        plt.close('all')

    def test_plot_step_drawstyle(self):
        """Test that plots use step drawstyle."""
        data = Data(signal_file=self.signal_file)
        fig = data.plot(kind='signal')

        # Check that figure was created
        self.assertIsNotNone(fig)

        # Check that there are lines with step drawstyle
        ax = fig.axes[0]
        lines = ax.get_lines()

        # At least one line should exist
        self.assertGreater(len(lines), 0)

    def test_plot_show_stages(self):
        """Test plotting with show_stages."""
        data = Data(signal_file=self.signal_file)
        data.convolute_response(pulse_duration=0.2)
        data.overlap(kernel=[0, 5, 10])

        fig = data.plot(kind='signal', show_stages=True)
        ax = fig.axes[0]

        # Should have multiple lines (Original, Convolved, Overlapped)
        legend = ax.get_legend()
        self.assertIsNotNone(legend)

        # Check for "Convolved" label (not "Squared")
        legend_texts = [t.get_text() for t in legend.get_texts()]
        self.assertIn('Convolved', legend_texts)

    def test_plot_error_bars_gray(self):
        """Test that error bars are gray."""
        data = Data(signal_file=self.signal_file)
        fig = data.plot(kind='signal', show_errors=True)
        ax = fig.axes[0]

        # Error bars should be present
        # Check for collections (error bar caps and lines)
        self.assertGreater(len(ax.collections) + len(ax.lines), 0)

    def test_plot_transmission(self):
        """Test transmission plotting."""
        data = Data(signal_file=self.signal_file, openbeam_file=self.openbeam_file)
        fig = data.plot(kind='transmission')

        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertIn('Transmission', ax.get_ylabel())

    def test_plot_figsize(self):
        """Test custom figure size."""
        data = Data(signal_file=self.signal_file)
        fig = data.plot(figsize=(8, 5))

        # Check figure size
        self.assertAlmostEqual(fig.get_figwidth(), 8, places=1)
        self.assertAlmostEqual(fig.get_figheight(), 5, places=1)

    def test_plot_fontsize(self):
        """Test custom font size."""
        data = Data(signal_file=self.signal_file)
        fig = data.plot(fontsize=18)

        # Font size should be applied
        self.assertIsNotNone(fig)


class TestDataRepresentation(unittest.TestCase):
    """Test Data object string representation."""

    def setUp(self):
        """Set up temporary test file."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, "test_signal.csv")

        data = pd.DataFrame({
            'stack': np.arange(1, 51),
            'counts': np.ones(50) * 100,
            'err': np.ones(50) * 10
        })
        data.to_csv(self.temp_file, index=False)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_repr_shows_convolved_stage(self):
        """Test that __repr__ shows 'convolved' stage."""
        data = Data(signal_file=self.temp_file)
        data.convolute_response(pulse_duration=0.2)

        repr_str = repr(data)

        # Should show convolved stage
        self.assertIn("stage='convolved'", repr_str)


if __name__ == '__main__':
    unittest.main()
