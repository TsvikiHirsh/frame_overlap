import pytest
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
from frame_overlap.analysis import generate_kernel, wiener_deconvolution, apply_filter
from frame_overlap.data import read_tof_data, prepare_full_frame
from frame_overlap.optimization import chi2_analysis, deconvolution_model, optimize_parameters
from frame_overlap.visualization import plot_analysis
import os

# Fixtures for mock data
@pytest.fixture
def mock_tof_data(tmp_path):
    """Create a temporary CSV file with mock ToF data."""
    data = pd.DataFrame({
        'stack': [1, 2, 3, 4, 5],
        'counts': [100, 200, 300, 400, 500],
        'err': [10, 20, 30, 40, 50]
    })
    file_path = tmp_path / "test_tof.csv"
    data.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def signal_data():
    """Generate mock signal data."""
    t_signal = np.linspace(0, 1000, 100)
    signal = np.random.normal(100, 10, 100)
    errors = np.sqrt(signal)
    stacks = np.arange(1, 101)
    return t_signal, signal, errors, stacks

@pytest.fixture
def kernel_data():
    """Generate mock kernel data."""
    t_kernel, kernel = generate_kernel(n_pulses=3, window_size=1000, bin_width=10, pulse_duration=100)
    return t_kernel, kernel

# Tests for analysis.py
def test_generate_kernel_valid():
    """Test generate_kernel with valid inputs."""
    t_kernel, kernel = generate_kernel(n_pulses=3, window_size=1000, bin_width=10, pulse_duration=100)
    assert len(t_kernel) == len(kernel)
    assert np.sum(kernel) == 3 * 10  # 3 pulses, each 100/10 bins long, height 1.0
    assert np.all(kernel >= 0)
    assert np.all(t_kernel >= 0)

def test_generate_kernel_insufficient_space():
    """Test generate_kernel raises error for insufficient window size."""
    with pytest.raises(ValueError, match="Total pulse space"):
        generate_kernel(n_pulses=10, window_size=100, bin_width=10, pulse_duration=20)

def test_generate_kernel_invalid_params():
    """Test generate_kernel raises error for invalid parameters."""
    with pytest.raises(ValueError, match="n_pulses must be a positive integer"):
        generate_kernel(n_pulses=0)
    with pytest.raises(ValueError, match="window_size, bin_width, pulse_duration must be positive"):
        generate_kernel(n_pulses=3, window_size=-100)

def test_wiener_deconvolution_valid():
    """Test wiener_deconvolution with valid inputs."""
    observed = np.random.normal(0, 1, 1000)
    kernel = np.ones(50) / 50
    reconstructed = wiener_deconvolution(observed, kernel, noise_power=0.01)
    assert len(reconstructed) == len(observed)
    assert np.all(np.isfinite(reconstructed))

def test_wiener_deconvolution_invalid():
    """Test wiener_deconvolution raises error for invalid inputs."""
    observed = np.random.normal(0, 1, 100)
    kernel = np.ones(200)
    with pytest.raises(ValueError, match="Observed signal must be at least as long as the kernel"):
        wiener_deconvolution(observed, kernel)
    with pytest.raises(ValueError, match="noise_power must be positive"):
        wiener_deconvolution(observed, np.ones(50), noise_power=0)

def test_apply_filter_valid():
    """Test apply_filter with valid inputs."""
    signal = np.random.normal(0, 1, 1000)
    kernel = np.ones(50) / 50
    observed, reconstructed = apply_filter(signal, kernel, filter_type='wiener', stats_fraction=0.2, noise_power=0.01)
    assert len(observed) == len(signal)
    assert len(reconstructed) == len(signal)
    assert np.all(observed >= 1)  # Clipped to 1 in apply_filter
    assert np.all(np.isfinite(reconstructed))

def test_apply_filter_invalid():
    """Test apply_filter raises error for invalid filter type."""
    signal = np.random.normal(0, 1, 1000)
    kernel = np.ones(50)
    with pytest.raises(ValueError, match="Filter type 'invalid' not supported"):
        apply_filter(signal, kernel, filter_type='invalid')

# Tests for data.py
def test_read_tof_data_valid(mock_tof_data):
    """Test read_tof_data with a valid CSV file."""
    t_signal, signal, errors, stacks = read_tof_data(mock_tof_data, threshold=2)
    assert len(t_signal) == len(signal) == len(errors) == len(stacks) == 4  # Threshold filters out stack=1
    assert np.all(t_signal == (np.array([2, 3, 4, 5]) - 1) * 10)
    assert np.all(signal == [200, 300, 400, 500])
    assert np.all(errors == [20, 30, 40, 50])

def test_read_tof_data_file_not_found():
    """Test read_tof_data raises error for missing file."""
    with pytest.raises(FileNotFoundError, match="File 'nonexistent.csv' not found"):
        read_tof_data('nonexistent.csv')

def test_read_tof_data_invalid_threshold(mock_tof_data):
    """Test read_tof_data raises error for negative threshold."""
    with pytest.raises(ValueError, match="Threshold must be non-negative"):
        read_tof_data(mock_tof_data, threshold=-1)

def test_prepare_full_frame_valid(signal_data):
    """Test prepare_full_frame with valid inputs."""
    t_signal, signal, errors, stacks = signal_data
    all_stacks, full_signal, full_errors = prepare_full_frame(t_signal, signal, errors, stacks, max_stack=100)
    assert len(all_stacks) == 100
    assert len(full_signal) == len(full_errors) == 100
    assert np.all(full_signal[stacks - 1] == signal)
    assert np.all(full_errors[stacks - 1] == errors)
    assert np.sum(full_signal[50:]) == 0  # No stacks beyond 100

def test_prepare_full_frame_invalid(signal_data):
    """Test prepare_full_frame raises error for invalid inputs."""
    t_signal, signal, errors, stacks = signal_data
    with pytest.raises(ValueError, match="All input arrays must have the same length"):
        prepare_full_frame(t_signal, signal[:-1], errors, stacks)
    with pytest.raises(ValueError, match="max_stack must be positive"):
        prepare_full_frame(t_signal, signal, errors, stacks, max_stack=0)

# Tests for optimization.py
def test_chi2_analysis_valid(signal_data):
    """Test chi2_analysis with valid inputs."""
    t_signal, signal, errors, stacks = signal_data
    scaled = signal * 0.2
    reconstructed = scaled * 1.1
    chi2, chi2_per_dof = chi2_analysis(scaled, reconstructed, errors)
    assert chi2 > 0
    assert chi2_per_dof > 0
    assert isinstance(chi2, float)
    assert isinstance(chi2_per_dof, float)

def test_chi2_analysis_invalid(signal_data):
    """Test chi2_analysis raises error for invalid inputs."""
    t_signal, signal, errors, stacks = signal_data
    scaled = signal * 0.2
    reconstructed = scaled * 1.1
    with pytest.raises(ValueError, match="All input arrays must have the same shape"):
        chi2_analysis(scaled, reconstructed[:-1], errors)
    with pytest.raises(ValueError, match="Errors must be positive"):
        chi2_analysis(scaled, reconstructed, np.zeros_like(errors))

def test_deconvolution_model_valid(signal_data, kernel_data):
    """Test deconvolution_model with valid inputs."""
    t_signal, signal, _, _ = signal_data
    reconstructed = deconvolution_model(signal, n_pulses=3, noise_power=0.01, pulse_duration=100)
    assert len(reconstructed) == len(signal)
    assert np.all(np.isfinite(reconstructed))

def test_deconvolution_model_invalid(signal_data):
    """Test deconvolution_model raises error for invalid inputs."""
    t_signal, signal, _, _ = signal_data
    with pytest.raises(ValueError, match="n_pulses must be a positive integer"):
        deconvolution_model(signal, n_pulses=0, noise_power=0.01, pulse_duration=100)
    with pytest.raises(ValueError, match="noise_power must be positive"):
        deconvolution_model(signal, n_pulses=3, noise_power=0, pulse_duration=100)

def test_optimize_parameters_valid(signal_data):
    """Test optimize_parameters with valid inputs."""
    t_signal, signal, _, _ = signal_data
    result = optimize_parameters(t_signal, signal, initial_params={'n_pulses': 3, 'noise_power': 0.05, 'pulse_duration': 100})
    assert hasattr(result, 'best_values')
    assert 'n_pulses' in result.best_values
    assert 'noise_power' in result.best_values
    assert 'pulse_duration' in result.best_values

def test_optimize_parameters_invalid(signal_data):
    """Test optimize_parameters raises error for invalid inputs."""
    t_signal, signal, _, _ = signal_data
    with pytest.raises(ValueError, match="Initial n_pulses must be between 1 and 20"):
        optimize_parameters(t_signal, signal, initial_params={'n_pulses': 0, 'noise_power': 0.05, 'pulse_duration': 100})

# Tests for visualization.py
def test_plot_analysis_valid(signal_data, kernel_data):
    """Test plot_analysis runs without errors."""
    t_signal, signal, errors, stacks = signal_data
    t_kernel, kernel = kernel_data
    scaled = signal * 0.2
    observed = poisson.rvs(np.clip(scaled, 0, None))
    reconstructed = wiener_deconvolution(observed, kernel, noise_power=0.01)
    residuals = scaled - reconstructed
    chi2, chi2_per_dof = chi2_analysis(scaled, reconstructed, errors)
    
    # Run plot_analysis and ensure it completes without errors
    plot_analysis(t_signal, signal, scaled, t_kernel, kernel, observed, reconstructed, residuals, chi2_per_dof)
    plt.close('all')  # Clean up to avoid memory leaks

def test_plot_analysis_invalid(signal_data, kernel_data):
    """Test plot_analysis raises error for invalid inputs."""
    t_signal, signal, errors, stacks = signal_data
    t_kernel, kernel = kernel_data
    scaled = signal * 0.2
    observed = poisson.rvs(np.clip(scaled, 0, None))
    reconstructed = wiener_deconvolution(observed, kernel, noise_power=0.01)
    residuals = scaled - reconstructed
    chi2, chi2_per_dof = chi2_analysis(scaled, reconstructed, errors)
    
    with pytest.raises(ValueError, match="All signal-related arrays must have the same length"):
        plot_analysis(t_signal, signal, scaled[:-1], t_kernel, kernel, observed, reconstructed, residuals, chi2_per_dof)
    plt.close('all')