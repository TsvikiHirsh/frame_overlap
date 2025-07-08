import matplotlib.pyplot as plt
import numpy as np

def plot_analysis(t_signal, original_signal, scaled_original, t_kernel, kernel, observed_poisson, reconstructed, residuals, chi2_per_dof):
    """
    Plot comprehensive analysis of signal processing results.

    Parameters
    ----------
    t_signal : array-like
        Time array for the signal in microseconds.
    original_signal : array-like
        Original unscaled signal counts.
    scaled_original : array-like
        Scaled original signal counts.
    t_kernel : array-like
        Time array for the kernel in microseconds.
    kernel : array-like
        Kernel amplitude values.
    observed_poisson : array-like
        Observed signal after convolution and Poisson sampling.
    reconstructed : array-like
        Reconstructed signal after deconvolution.
    residuals : array-like
        Residuals (scaled_original - reconstructed).
    chi2_per_dof : float
        Reduced chi-squared statistic.

    Returns
    -------
    None
        Displays a matplotlib figure with six subplots.

    Raises
    ------
    ValueError
        If input arrays have inconsistent shapes or contain invalid values.
    TypeError
        If inputs are not array-like or contain non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1000, 100)
    >>> sig = np.random.normal(100, 10, 100)
    >>> scaled = sig * 0.2
    >>> t_k = np.linspace(0, 1000, 100)
    >>> k = np.ones(100) / 100
    >>> obs = np.random.poisson(scaled)
    >>> recon = obs * 1.1
    >>> res = scaled - recon
    >>> plot_analysis(t, sig, scaled, t_k, k, obs, recon, res, chi2_per_dof=1.234)
    """
    # Convert inputs to numpy arrays and validate
    arrays = {
        't_signal': np.asarray(t_signal, dtype=float),
        'original_signal': np.asarray(original_signal, dtype=float),
        'scaled_original': np.asarray(scaled_original, dtype=float),
        't_kernel': np.asarray(t_kernel, dtype=float),
        'kernel': np.asarray(kernel, dtype=float),
        'observed_poisson': np.asarray(observed_poisson, dtype=float),
        'reconstructed': np.asarray(reconstructed, dtype=float),
        'residuals': np.asarray(residuals, dtype=float)
    }
    
    # Check for consistent shapes
    signal_length = len(arrays['t_signal'])
    if not all(len(arr) == signal_length for arr in [arrays['original_signal'], arrays['scaled_original'], 
                                                    arrays['observed_poisson'], arrays['reconstructed'], arrays['residuals']]):
        raise ValueError("All signal-related arrays must have the same length")
    if len(arrays['t_kernel']) != len(arrays['kernel']):
        raise ValueError("t_kernel and kernel must have the same length")
    if np.any([np.any(np.isnan(arr)) for arr in arrays.values()]):
        raise ValueError("Input arrays must not contain NaN values")
    if chi2_per_dof < 0:
        raise ValueError("chi2_per_dof must be non-negative")

    plt.figure(figsize=(12, 14))
    
    plt.subplot(6, 1, 1)
    plt.plot(arrays['t_signal'], arrays['original_signal'])
    plt.title('Original Neutron Signal (Unscaled)')
    plt.xlabel('Time (µs)')
    plt.ylabel('Counts')
    
    plt.subplot(6, 1, 2)
    plt.plot(arrays['t_signal'], arrays['scaled_original'])
    plt.title('Scaled Original Signal')
    plt.xlabel('Time (µs)')
    plt.ylabel('Counts')
    
    plt.subplot(6, 1, 3)
    plt.plot(arrays['t_kernel'], arrays['kernel'])
    plt.title('Random Kernel')
    plt.xlabel('Time (µs)')
    plt.ylabel('Amplitude')
    
    plt.subplot(6, 1, 4)
    plt.plot(arrays['t_signal'], arrays['observed_poisson'])
    plt.title('Observed Signal (Convolution + Poisson Sampling)')
    plt.xlabel('Time (µs)')
    plt.ylabel('Counts')
    
    plt.subplot(6, 1, 5)
    plt.plot(arrays['t_signal'], arrays['scaled_original'], 'b-', alpha=0.7, label='Scaled Original')
    plt.plot(arrays['t_signal'], arrays['reconstructed'], 'r--', alpha=0.7, label='Reconstructed')
    plt.title(f'Scaled Original vs Reconstructed Signal (χ²/dof = {chi2_per_dof:.3f})')
    plt.xlabel('Time (µs)')
    plt.ylabel('Counts')
    plt.legend()
    
    plt.subplot(6, 1, 6)
    plt.plot(arrays['t_signal'], arrays['residuals'], 'k-', label='Residual (Scaled Original - Reconstructed)')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Residuals')
    plt.xlabel('Time (µs)')
    plt.ylabel('Counts')
    plt.legend()
    
    plt.tight_layout()
    plt.show()