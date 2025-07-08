import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
from scipy.stats import poisson
try:
    from frame_overlap.analysis import generate_kernel, wiener_deconvolution, apply_filter
    from frame_overlap.data import read_tof_data, prepare_full_frame
    from frame_overlap.optimization import chi2_analysis, deconvolution_model_wrapper, optimize_parameters
    from frame_overlap.visualization import plot_analysis
except ImportError as e:
    raise ImportError("Failed to import frame_overlap modules. Ensure the package is installed correctly and dependencies (numpy, pandas, scipy, matplotlib, lmfit) are available.") from e

class TestFrameOverlap(unittest.TestCase):
    def setUp(self):
        """Set up mock DataFrames and temporary files for tests."""
        try:
            import numpy
            import pandas
            import scipy
            import matplotlib
            import lmfit
        except ImportError as e:
            self.fail(f"Required dependency missing: {str(e)}. Please install numpy, pandas, scipy, matplotlib, and lmfit.")

        # Mock signal DataFrame
        signal_length = 1000
        counts = np.random.normal(100, 10, signal_length)
        self.t_signal_df = pd.DataFrame({
            'time': np.linspace(0, 1000, signal_length)
        })
        self.signal_df = pd.DataFrame({
            'time': self.t_signal_df['time'],
            'counts': counts,
            'errors': np.sqrt(counts),
            'stack': np.arange(1, signal_length + 1)
        })
        
        # Mock kernel DataFrame
        self.kernel_df = generate_kernel(n_pulses=3, window_size=1000, bin_width=10, pulse_duration=100)
        
        # Create temporary CSV file for ToF data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, "test_tof.csv")
        data = pd.DataFrame({
            'stack': [1, 2, 3, 4, 5],
            'counts': [100, 200, 300, 400, 500],
            'err': [10, 20, 30, 40, 50]
        })
        data.to_csv(self.temp_file, index=False)

    def tearDown(self):
        """Clean up temporary files and close plots."""
        self.temp_dir.cleanup()
        plt.close('all')

    # Tests for analysis.py
    def test_generate_kernel_valid(self):
        """Test generate_kernel with valid inputs."""
        kernel_df = generate_kernel(n_pulses=3, window_size=1000, bin_width=10, pulse_duration=100)
        self.assertEqual(len(kernel_df), 1000 // 10)
        self.assertAlmostEqual(kernel_df['kernel_value'].sum(), 3 * 10, places=5)  # 3 pulses, each 100/10 bins, height 1.0
        self.assertTrue(np.all(kernel_df['kernel_value'] >= 0))
        self.assertTrue(np.all(kernel_df['kernel_time'] >= 0))
        self.assertEqual(list(kernel_df.columns), ['kernel_time', 'kernel_value'])

    def test_generate_kernel_insufficient_space(self):
        """Test generate_kernel raises error for insufficient window size."""
        with self.assertRaises(ValueError) as cm:
            generate_kernel(n_pulses=10, window_size=100, bin_width=10, pulse_duration=20)
        self.assertIn("Total pulse space", str(cm.exception))

    def test_generate_kernel_invalid_params(self):
        """Test generate_kernel raises error for invalid parameters."""
        with self.assertRaises(ValueError) as cm:
            generate_kernel(n_pulses=0)
        self.assertIn("n_pulses must be a positive integer", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            generate_kernel(n_pulses=3, window_size=-100)
        self.assertIn("window_size, bin_width, pulse_duration must be positive", str(cm.exception))

    def test_wiener_deconvolution_valid(self):
        """Test wiener_deconvolution with valid inputs."""
        observed_df = pd.DataFrame({'counts': np.random.normal(0, 1, 1000)})
        kernel_df = pd.DataFrame({
            'kernel_value': np.ones(50) / 50,
            'kernel_time': np.linspace(0, 500, 50)
        })
        reconstructed_df = wiener_deconvolution(observed_df, kernel_df, noise_power=0.01)
        self.assertEqual(len(reconstructed_df), len(observed_df))
        self.assertTrue(np.all(np.isfinite(reconstructed_df['reconstructed'])))
        self.assertEqual(list(reconstructed_df.columns), ['reconstructed'])

    def test_wiener_deconvolution_invalid(self):
        """Test wiener_deconvolution raises error for invalid inputs."""
        observed_df = pd.DataFrame({'counts': np.random.normal(0, 1, 100)})
        kernel_df = pd.DataFrame({
            'kernel_value': np.ones(200),
            'kernel_time': np.linspace(0, 2000, 200)
        })
        with self.assertRaises(ValueError) as cm:
            wiener_deconvolution(observed_df, kernel_df)
        self.assertIn("Observed signal must be at least as long as the kernel", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            wiener_deconvolution(observed_df, kernel_df[:50], noise_power=0)
        self.assertIn("noise_power must be positive", str(cm.exception))

    def test_apply_filter_valid(self):
        """Test apply_filter with valid inputs."""
        signal_df = pd.DataFrame({'counts': np.random.normal(0, 1, 1000)})
        kernel_df = pd.DataFrame({
            'kernel_value': np.ones(50) / 50,
            'kernel_time': np.linspace(0, 500, 50)
        })
        observed_df, reconstructed_df = apply_filter(signal_df, kernel_df, filter_type='wiener', stats_fraction=0.2, noise_power=0.01)
        self.assertEqual(len(observed_df), len(signal_df))
        self.assertEqual(len(reconstructed_df), len(signal_df))
        self.assertTrue(np.all(observed_df['counts'] >= 1))
        self.assertTrue(np.all(np.isfinite(reconstructed_df['reconstructed'])))
        self.assertEqual(list(observed_df.columns), ['counts'])
        self.assertEqual(list(reconstructed_df.columns), ['reconstructed'])

    def test_apply_filter_invalid(self):
        """Test apply_filter raises error for invalid filter type."""
        signal_df = pd.DataFrame({'counts': np.random.normal(0, 1, 1000)})
        kernel_df = pd.DataFrame({
            'kernel_value': np.ones(50),
            'kernel_time': np.linspace(0, 500, 50)
        })
        with self.assertRaises(ValueError) as cm:
            apply_filter(signal_df, kernel_df, filter_type='invalid')
        self.assertIn("Filter type 'invalid' not supported", str(cm.exception))

    # Tests for data.py
    def test_read_tof_data_valid(self):
        """Test read_tof_data with a valid CSV file."""
        signal_df = read_tof_data(self.temp_file, threshold=100)
        self.assertEqual(len(signal_df), 4)  # Threshold filters out stack=1 (counts=100)
        self.assertEqual(list(signal_df.columns), ['time', 'counts', 'errors', 'stack'])
        np.testing.assert_array_equal(signal_df['time'], (np.array([2, 3, 4, 5]) - 1) * 10)
        np.testing.assert_array_equal(signal_df['counts'], [200, 300, 400, 500])
        np.testing.assert_array_equal(signal_df['errors'], [20, 30, 40, 50])
        np.testing.assert_array_equal(signal_df['stack'], [2, 3, 4, 5])

    def test_read_tof_data_file_not_found(self):
        """Test read_tof_data raises error for missing file."""
        with self.assertRaises(FileNotFoundError) as cm:
            read_tof_data('nonexistent.csv')
        self.assertIn("File 'nonexistent.csv' not found", str(cm.exception))

    def test_read_tof_data_invalid_threshold(self):
        """Test read_tof_data raises error for negative threshold."""
        with self.assertRaises(ValueError) as cm:
            read_tof_data(self.temp_file, threshold=-1)
        self.assertIn("Threshold must be non-negative", str(cm.exception))

    def test_prepare_full_frame_valid(self):
        """Test prepare_full_frame with valid inputs."""
        full_df = prepare_full_frame(self.signal_df, max_stack=1000)
        self.assertEqual(len(full_df), 1000)
        self.assertEqual(list(full_df.columns), ['stack', 'counts', 'errors'])
        np.testing.assert_array_almost_equal(full_df.loc[full_df['stack'].isin(self.signal_df['stack']), 'counts'],
                                            self.signal_df['counts'])
        np.testing.assert_array_almost_equal(full_df.loc[full_df['stack'].isin(self.signal_df['stack']), 'errors'],
                                            self.signal_df['errors'])
        self.assertTrue(np.all(full_df['stack'] == np.arange(1, 1001)))

    def test_prepare_full_frame_invalid(self):
        """Test prepare_full_frame raises error for invalid inputs."""
        invalid_df = self.signal_df.drop(columns=['counts'])
        with self.assertRaises(ValueError) as cm:
            prepare_full_frame(invalid_df, max_stack=1000)
        self.assertIn("signal_df must contain columns", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            prepare_full_frame(self.signal_df, max_stack=0)
        self.assertIn("max_stack must be positive", str(cm.exception))

    # Tests for optimization.py
    def test_chi2_analysis_valid(self):
        """Test chi2_analysis with valid inputs."""
        scaled_df = pd.DataFrame({'counts': self.signal_df['counts'] * 0.2})
        reconstructed_df = pd.DataFrame({'reconstructed': scaled_df['counts'] * 1.1})
        errors_df = pd.DataFrame({'errors': self.signal_df['errors']})
        chi2, chi2_per_dof = chi2_analysis(scaled_df, reconstructed_df, errors_df)
        self.assertGreater(chi2, 0)
        self.assertGreater(chi2_per_dof, 0)
        self.assertIsInstance(chi2, float)
        self.assertIsInstance(chi2_per_dof, float)

    def test_chi2_analysis_invalid(self):
        """Test chi2_analysis raises error for invalid inputs."""
        scaled_df = pd.DataFrame({'counts': self.signal_df['counts'] * 0.2})
        reconstructed_df = pd.DataFrame({'reconstructed': scaled_df['counts'] * 1.1})
        errors_df = pd.DataFrame({'errors': self.signal_df['errors']})
        with self.assertRaises(ValueError) as cm:
            chi2_analysis(scaled_df.iloc[:-1], reconstructed_df, errors_df)
        self.assertIn("All input arrays must have the same shape", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            chi2_analysis(scaled_df, reconstructed_df, pd.DataFrame({'errors': np.zeros_like(self.signal_df['errors'])}))
        self.assertIn("Errors must be positive", str(cm.exception))

    def test_deconvolution_model_valid(self):
        """Test deconvolution_model_wrapper with valid inputs."""
        x = np.arange(len(self.signal_df))
        reconstructed = deconvolution_model_wrapper(x, n_pulses=3, noise_power=0.01, pulse_duration=100, window_size=1000, stats_fraction=0.2, signal_df=self.signal_df)
        self.assertEqual(len(reconstructed), len(self.signal_df))
        self.assertTrue(np.all(np.isfinite(reconstructed)))

    def test_deconvolution_model_invalid(self):
        """Test deconvolution_model_wrapper raises error for invalid inputs."""
        x = np.arange(len(self.signal_df))
        with self.assertRaises(ValueError) as cm:
            deconvolution_model_wrapper(x, n_pulses=0, noise_power=0.01, pulse_duration=100, window_size=1000, stats_fraction=0.2, signal_df=self.signal_df)
        self.assertIn("n_pulses must be a positive integer", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            deconvolution_model_wrapper(x, n_pulses=3, noise_power=0, pulse_duration=100, window_size=1000, stats_fraction=0.2, signal_df=self.signal_df)
        self.assertIn("noise_power must be positive", str(cm.exception))

    def test_optimize_parameters_valid(self):
        """Test optimize_parameters with valid inputs."""
        result = optimize_parameters(self.t_signal_df, self.signal_df, initial_params={'n_pulses': 3, 'noise_power': 0.05, 'pulse_duration': 100})
        self.assertTrue(hasattr(result, 'best_values'))
        self.assertIn('n_pulses', result.best_values)
        self.assertIn('noise_power', result.best_values)
        self.assertIn('pulse_duration', result.best_values)

    def test_optimize_parameters_invalid(self):
        """Test optimize_parameters raises error for invalid inputs."""
        with self.assertRaises(ValueError) as cm:
            optimize_parameters(self.t_signal_df, self.signal_df, initial_params={'n_pulses': 0, 'noise_power': 0.05, 'pulse_duration': 100})
        self.assertIn("Initial n_pulses must be between 1 and 20", str(cm.exception))

    # Tests for visualization.py
    def test_plot_analysis_valid(self):
        """Test plot_analysis runs without errors."""
        scaled_df = pd.DataFrame({'counts': self.signal_df['counts'] * 0.2})
        observed_df, reconstructed_df = apply_filter(self.signal_df, self.kernel_df, filter_type='wiener', stats_fraction=0.2, noise_power=0.01)
        residuals_df = pd.DataFrame({'residuals': scaled_df['counts'] - reconstructed_df['reconstructed']})
        chi2, chi2_per_dof = chi2_analysis(scaled_df, reconstructed_df, pd.DataFrame({'errors': self.signal_df['errors']}))
        
        try:
            plot_analysis(self.t_signal_df, self.signal_df, scaled_df, self.kernel_df, observed_df, reconstructed_df, residuals_df, chi2_per_dof)
        except Exception as e:
            self.fail(f"plot_analysis raised {type(e).__name__}: {str(e)}")

    def test_plot_analysis_invalid(self):
        """Test plot_analysis raises error for invalid inputs."""
        scaled_df = pd.DataFrame({'counts': self.signal_df['counts'] * 0.2})
        observed_df, reconstructed_df = apply_filter(self.signal_df, self.kernel_df, filter_type='wiener', stats_fraction=0.2, noise_power=0.01)
        residuals_df = pd.DataFrame({'residuals': scaled_df['counts'] - reconstructed_df['reconstructed']})
        chi2, chi2_per_dof = chi2_analysis(scaled_df, reconstructed_df, pd.DataFrame({'errors': self.signal_df['errors']}))
        
        with self.assertRaises(ValueError) as cm:
            plot_analysis(self.t_signal_df, self.signal_df, scaled_df.iloc[:-1], self.kernel_df, observed_df, reconstructed_df, residuals_df, chi2_per_dof)
        self.assertIn("All signal-related DataFrames must have the same length", str(cm.exception))

if __name__ == '__main__':
    unittest.main()