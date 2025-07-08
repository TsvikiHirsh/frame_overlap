import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import poisson

def generate_kernel(n_pulses, window_size=5000, bin_width=10, pulse_duration=200):
    """
    Generate a kernel DataFrame for convolution with specified number of pulses.

    Parameters
    ----------
    n_pulses : int
        Number of pulses in the kernel.
    window_size : int, optional
        Total size of the kernel window (default: 5000).
    bin_width : int, optional
        Width of each bin in time units (default: 10).
    pulse_duration : int, optional
        Duration of each pulse in time units (default: 200).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'kernel_time' and 'kernel_value' representing the kernel.

    Raises
    ------
    ValueError
        If n_pulses is not a positive integer or if window_size, bin_width, or pulse_duration are not positive.
    """
    n_pulses = int(n_pulses)
    if n_pulses < 1:
        raise ValueError("n_pulses must be a positive integer")
    if window_size <= 0 or bin_width <= 0 or pulse_duration <= 0:
        raise ValueError("window_size, bin_width, pulse_duration must be positive")
    
    total_pulse_space = n_pulses * pulse_duration
    if total_pulse_space > window_size:
        raise ValueError(f"Total pulse space ({total_pulse_space}) exceeds window_size ({window_size})")
    
    n_bins = window_size // bin_width
    kernel = np.zeros(n_bins)
    
    pulse_width = pulse_duration // bin_width
    for i in range(n_pulses):
        start_idx = i * pulse_width
        kernel[start_idx:start_idx + pulse_width] = 1.0
    
    kernel_time = np.arange(0, window_size, bin_width)
    
    return pd.DataFrame({
        'kernel_time': kernel_time,
        'kernel_value': kernel
    })

def wiener_deconvolution(observed_df, kernel_df, noise_power=0.01):
    """
    Apply Wiener deconvolution to the observed signal using the provided kernel.

    Parameters
    ----------
    observed_df : pandas.DataFrame
        DataFrame with column 'counts' containing the observed signal.
    kernel_df : pandas.DataFrame
        DataFrame with column 'kernel_value' containing the kernel.
    noise_power : float, optional
        Noise power for Wiener filter (default: 0.01).

    Returns
    -------
    pandas.DataFrame
        DataFrame with column 'reconstructed' containing the deconvolved signal.

    Raises
    ------
    ValueError
        If inputs are invalid or observed signal is shorter than kernel.
    """
    if 'counts' not in observed_df.columns or 'kernel_value' not in kernel_df.columns:
        raise ValueError("observed_df must have 'counts' column and kernel_df must have 'kernel_value' column")
    if noise_power <= 0:
        raise ValueError("noise_power must be positive")
    
    observed = observed_df['counts'].to_numpy()
    kernel = kernel_df['kernel_value'].to_numpy()
    
    if len(observed) < len(kernel):
        raise ValueError("Observed signal must be at least as long as the kernel")
    
    f_observed = np.fft.fft(observed)
    f_kernel = np.fft.fft(kernel, n=len(observed))
    kernel_power = np.abs(f_kernel)**2
    wiener_filter = np.conj(f_kernel) / (kernel_power + noise_power)
    f_reconstructed = f_observed * wiener_filter
    reconstructed = np.fft.ifft(f_reconstructed).real
    
    return pd.DataFrame({
        'reconstructed': reconstructed[:len(observed)]
    })

def apply_filter(signal_df, kernel_df, filter_type='wiener', stats_fraction=0.2, noise_power=0.01):
    """
    Apply a filter (e.g., Wiener deconvolution) to the signal.

    Parameters
    ----------
    signal_df : pandas.DataFrame
        DataFrame with column 'counts' containing the input signal.
    kernel_df : pandas.DataFrame
        DataFrame with column 'kernel_value' containing the kernel.
    filter_type : str, optional
        Type of filter to apply (default: 'wiener').
    stats_fraction : float, optional
        Scaling factor for Poisson noise (default: 0.2).
    noise_power : float, optional
        Noise power for Wiener filter (default: 0.01).

    Returns
    -------
    tuple
        - pandas.DataFrame: DataFrame with column 'observed' containing the observed signal with Poisson noise.
        - pandas.DataFrame: DataFrame with column 'reconstructed' containing the filtered signal.

    Raises
    ------
    ValueError
        If filter_type is unsupported or inputs are invalid.
    """
    if filter_type != 'wiener':
        raise ValueError(f"Filter type '{filter_type}' not supported")
    
    if 'counts' not in signal_df.columns or 'kernel_value' not in kernel_df.columns:
        raise ValueError("signal_df must have 'counts' column and kernel_df must have 'kernel_value' column")
    
    signal = signal_df['counts'].to_numpy()
    kernel = kernel_df['kernel_value'].to_numpy()
    
    convolved = signal.convolve(signal, kernel, mode='full')[:len(signal)]
    scaled = convolved * stats_fraction
    observed = poisson.rvs(np.clip(scaled, 0, None))
    observed = np.clip(observed, 1, None)
    
    observed_df = pd.DataFrame({'observed': observed})
    reconstructed_df = wiener_deconvolution(observed_df, kernel_df, noise_power=noise_power)
    
    return observed_df, reconstructed_df