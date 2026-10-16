import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import poisson

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
    # Ensure all parameters are integers (important for lmfit compatibility)
    n_pulses = int(n_pulses)
    window_size = int(window_size)
    bin_width = int(bin_width)
    pulse_duration = int(pulse_duration)
    
    if n_pulses < 1:
        raise ValueError("n_pulses must be a positive integer")
    if window_size <= 0 or bin_width <= 0 or pulse_duration <= 0:
        raise ValueError("window_size, bin_width, pulse_duration must be positive")
    
    # Calculate number of bins
    n_bins = window_size // bin_width
    
    # For frame overlap, we want the kernel to represent the pulse shape
    # Create a more realistic pulse shape (e.g., exponential decay)
    kernel = np.zeros(n_bins)
    
    pulse_width_bins = pulse_duration // bin_width
    
    for i in range(n_pulses):
        # Calculate start position for this pulse
        if n_pulses == 1:
            start_idx = 0  # Single pulse starts at beginning
        else:
            # Space pulses evenly across the window
            pulse_spacing = n_bins // (n_pulses + 1)
            start_idx = (i + 1) * pulse_spacing
        
        # Add bounds checking to prevent index out of bounds
        if start_idx < n_bins:
            end_idx = min(start_idx + pulse_width_bins, n_bins)
            
            # Create exponential decay pulse shape for more realistic modeling
            pulse_length = end_idx - start_idx
            if pulse_length > 0:
                pulse_shape = np.exp(-np.linspace(0, 3, pulse_length))  # Exponential decay
                kernel[start_idx:end_idx] = pulse_shape
    
    # Normalize kernel so it sums to 1
    if np.sum(kernel) > 0:
        kernel = kernel / np.sum(kernel)
    
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
        - pandas.DataFrame: DataFrame with column 'counts' containing the observed signal with Poisson noise.
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
    
    signal_array = signal_df['counts'].to_numpy()
    kernel = kernel_df['kernel_value'].to_numpy()
    
    # Pad or truncate kernel to match signal length for proper convolution
    signal_length = len(signal_array)
    if len(kernel) > signal_length:
        kernel = kernel[:signal_length]
    elif len(kernel) < signal_length:
        # Pad kernel with zeros to match signal length
        kernel = np.pad(kernel, (0, signal_length - len(kernel)), mode='constant', constant_values=0)
    
    # Apply convolution - use 'same' mode to maintain signal length
    convolved = signal.convolve(signal_array, kernel, mode='same')
    scaled = convolved * stats_fraction
    observed = poisson.rvs(np.clip(scaled, 0, None))
    observed = np.clip(observed, 1, None)
    
    # Create observed_df with 'counts' column (not 'observed')
    observed_df = pd.DataFrame({'counts': observed})
    
    # Create a kernel_df that matches the signal length for deconvolution
    kernel_matched_df = pd.DataFrame({'kernel_value': kernel})
    reconstructed_df = wiener_deconvolution(observed_df, kernel_matched_df, noise_power=noise_power)
    
    return observed_df, reconstructed_df

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
        - pandas.DataFrame: DataFrame with column 'counts' containing the observed signal with Poisson noise.
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
    
    signal_array = signal_df['counts'].to_numpy()
    kernel = kernel_df['kernel_value'].to_numpy()
    
    # Use 'same' mode to maintain signal length (this was the key difference)
    convolved = signal.convolve(signal_array, kernel, mode='same')
    scaled = convolved * stats_fraction
    observed = poisson.rvs(np.clip(scaled, 0, None))
    observed = np.clip(observed, 1, None)
    
    # Create observed_df with 'counts' column (not 'observed')
    observed_df = pd.DataFrame({'counts': observed})
    reconstructed_df = wiener_deconvolution(observed_df, kernel_df, noise_power=noise_power)
    
    return observed_df, reconstructed_df