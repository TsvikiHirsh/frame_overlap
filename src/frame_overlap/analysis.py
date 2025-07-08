import numpy as np
from scipy import signal
from scipy.stats import poisson

def generate_kernel(n_pulses, window_size=5000, bin_width=10, pulse_duration=200, pulse_height=1.0):
    """
    Generate a kernel with non-overlapping rectangular pulses for signal convolution.

    Parameters
    ----------
    n_pulses : int
        Number of pulses in the kernel.
    window_size : int, optional
        Total size of the kernel window in microseconds (default: 5000).
    bin_width : float, optional
        Width of each time bin in microseconds (default: 10).
    pulse_duration : float, optional
        Duration of each pulse in microseconds (default: 200).
    pulse_height : float, optional
        Amplitude of each pulse (default: 1.0).

    Returns
    -------
    tuple
        A tuple containing:
        - t_kernel : array-like, time array for the kernel
        - kernel : array-like, kernel array with rectangular pulses

    Raises
    ------
    ValueError
        If parameters are invalid (e.g., negative values, insufficient window size).
    TypeError
        If inputs are not of expected numeric types.

    Examples
    --------
    >>> import numpy as np
    >>> t, k = generate_kernel(n_pulses=3, window_size=1000, bin_width=10, pulse_duration=100)
    >>> t.shape == k.shape
    True
    >>> np.sum(k) > 0
    True
    """
    # Input validation
    if not isinstance(n_pulses, int) or n_pulses < 1:
        raise ValueError("n_pulses must be a positive integer")
    if window_size <= 0 or bin_width <= 0 or pulse_duration <= 0 or pulse_height < 0:
        raise ValueError("window_size, bin_width, pulse_duration must be positive; pulse_height must be non-negative")
    
    t_kernel = np.arange(0, window_size, bin_width, dtype=float)
    kernel = np.zeros_like(t_kernel, dtype=float)
    
    pulse_length = int(pulse_duration / bin_width)
    total_pulse_space = n_pulses * pulse_length
    
    if total_pulse_space > len(t_kernel):
        raise ValueError(f"Total pulse space ({total_pulse_space * bin_width} µs) exceeds window size ({window_size} µs)")
    
    available_indices = list(range(len(t_kernel) - pulse_length + 1))
    start_indices = []
    
    for _ in range(n_pulses):
        if not available_indices:
            raise ValueError("Not enough space for non-overlapping pulses")
        start_idx = np.random.choice(available_indices)
        start_indices.append(start_idx)
        overlap_range = range(max(0, start_idx - pulse_length + 1), min(len(t_kernel) - pulse_length + 1, start_idx + pulse_length))
        available_indices = [idx for idx in available_indices if idx not in overlap_range]
    
    start_indices.sort()
    for start_idx in start_indices:
        kernel[start_idx:start_idx + pulse_length] = pulse_height
    
    return t_kernel, kernel

def wiener_deconvolution(observed, kernel, noise_power=0.01):
    """
    Perform Wiener deconvolution to reconstruct the original signal.

    Parameters
    ----------
    observed : array-like
        Observed signal after convolution and noise.
    kernel : array-like
        Kernel used in the convolution process.
    noise_power : float, optional
        Noise power for Wiener filter (default: 0.01).

    Returns
    -------
    array-like
        Reconstructed signal after deconvolution.

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., negative noise_power, mismatched shapes).
    TypeError
        If inputs are not array-like or contain non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> observed = np.random.normal(0, 1, 1000)
    >>> kernel = np.ones(50) / 50
    >>> recon = wiener_deconvolution(observed, kernel, noise_power=0.01)
    >>> recon.shape
    (1000,)
    """
    observed = np.asarray(observed, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    
    if noise_power <= 0:
        raise ValueError("noise_power must be positive")
    if len(observed) < len(kernel):
        raise ValueError("Observed signal must be at least as long as the kernel")
    if np.any(np.isnan(observed)) or np.any(np.isnan(kernel)):
        raise ValueError("Input arrays must not contain NaN values")

    kernel_padded = np.pad(kernel, (0, len(observed) - len(kernel)), 'constant')
    H = np.fft.fft(kernel_padded)
    Y = np.fft.fft(observed)
    H_conj = np.conj(H)
    G = H_conj / (np.abs(H)**2 + noise_power)
    X_est = Y * G
    x_est = np.real(np.fft.ifft(X_est))
    return x_est

def apply_filter(signal, kernel, filter_type='wiener', stats_fraction=0.2, noise_power=0.01):
    """
    Apply a filter to a signal with Poisson sampling.

    Parameters
    ----------
    signal : array-like
        Input signal to be filtered.
    kernel : array-like
        Kernel for convolution.
    filter_type : str, optional
        Type of filter to apply (default: 'wiener').
    stats_fraction : float, optional
        Scaling factor for Poisson noise (default: 0.2).
    noise_power : float, optional
        Noise power for Wiener filter (default: 0.01).

    Returns
    -------
    tuple
        A tuple containing:
        - observed_poisson : array-like, observed signal with Poisson noise
        - reconstructed : array-like, reconstructed signal after filtering

    Raises
    ------
    ValueError
        If filter_type is unsupported or parameters are invalid.
    TypeError
        If inputs are not array-like or contain non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.random.normal(0, 1, 1000)
    >>> kernel = np.ones(50) / 50
    >>> obs, recon = apply_filter(signal, kernel, filter_type='wiener')
    >>> obs.shape == recon.shape
    True
    """
    signal = np.asarray(signal, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    
    if stats_fraction <= 0 or stats_fraction > 1:
        raise ValueError("stats_fraction must be between 0 and 1")
    if noise_power <= 0:
        raise ValueError("noise_power must be positive")
    if np.any(np.isnan(signal)) or np.any(np.isnan(kernel)):
        raise ValueError("Input arrays must not contain NaN values")

    observed = np.convolve(signal, kernel, mode='full')[:len(signal)]
    scaled_observed = observed * stats_fraction
    observed_poisson = poisson.rvs(np.clip(scaled_observed, 0, None))
    observed_poisson = np.clip(observed_poisson, 1, None)
    
    if filter_type.lower() == 'wiener':
        reconstructed = wiener_deconvolution(observed_poisson, kernel, noise_power=noise_power)
    else:
        raise ValueError(f"Filter type '{filter_type}' not supported. Use 'wiener'.")
    
    return observed_poisson, reconstructed