import numpy as np
import pandas as pd
from lmfit import Model
from scipy.stats import poisson

def chi2_analysis(scaled_original, reconstructed, errors):
    """
    Calculate chi-squared and reduced chi-squared statistics for model fit evaluation.

    Parameters
    ----------
    scaled_original : array-like
        The scaled original signal data.
    reconstructed : array-like
        The reconstructed signal data from the model.
    errors : array-like
        The uncertainties associated with the scaled original signal.

    Returns
    -------
    tuple
        A tuple containing:
        - chi2 : float, the chi-squared statistic
        - chi2_per_dof : float, the reduced chi-squared (chi2 per degree of freedom)

    Raises
    ------
    ValueError
        If inputs have inconsistent shapes or contain invalid values (e.g., zero or negative errors).
    TypeError
        If inputs are not array-like or contain non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> scaled = np.array([1.0, 2.0, 3.0])
    >>> recon = np.array([1.1, 2.1, 2.9])
    >>> errs = np.array([0.1, 0.1, 0.1])
    >>> chi2, chi2_dof = chi2_analysis(scaled, recon, errs)
    >>> print(f"Chi2: {chi2:.2f}, Reduced Chi2: {chi2_dof:.2f}")
    """
    # Input validation
    scaled_original = np.asarray(scaled_original, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    errors = np.asarray(errors, dtype=float)
    
    if scaled_original.shape != reconstructed.shape or scaled_original.shape != errors.shape:
        raise ValueError("All input arrays must have the same shape")
    if np.any(errors <= 0):
        raise ValueError("Errors must be positive")
    if np.any(np.isnan(scaled_original)) or np.any(np.isnan(reconstructed)) or np.any(np.isnan(errors)):
        raise ValueError("Input arrays must not contain NaN values")

    residuals = scaled_original - reconstructed
    chi2 = np.sum((residuals / errors)**2)
    chi2_per_dof = chi2 / len(scaled_original)
    return chi2, chi2_per_dof

def deconvolution_model(x, n_pulses, noise_power, pulse_duration, window_size=5000, stats_fraction=0.2):
    """
    Apply Wiener deconvolution to model a signal with specified parameters.

    Parameters
    ----------
    x : array-like
        Input signal to be convolved.
    n_pulses : int
        Number of pulses in the kernel.
    noise_power : float
        Noise power for Wiener deconvolution.
    pulse_duration : float
        Duration of each pulse in the kernel.
    window_size : int, optional
        Size of the kernel window (default: 5000).
    stats_fraction : float, optional
        Scaling factor for Poisson noise (default: 0.2).

    Returns
    -------
    array-like
        Reconstructed signal after Wiener deconvolution.

    Raises
    ------
    ValueError
        If input parameters are invalid (e.g., negative values, non-positive n_pulses).
    ImportError
        If required analysis module components are not available.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.normal(0, 1, 1000)
    >>> recon = deconvolution_model(x, n_pulses=5, noise_power=0.05, pulse_duration=200)
    >>> recon.shape
    (1000,)
    """
    try:
        from .analysis import generate_kernel, wiener_deconvolution, signal
    except ImportError as e:
        raise ImportError("Required analysis module components not found") from e

    # Input validation
    x = np.asarray(x, dtype=float)
    n_pulses = int(n_pulses)  # Cast to integer to handle lmfit float inputs
    if n_pulses < 1:
        raise ValueError("n_pulses must be a positive integer")
    if noise_power <= 0:
        raise ValueError("noise_power must be positive")
    if pulse_duration <= 0:
        raise ValueError("pulse_duration must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stats_fraction <= 0 or stats_fraction > 1:
        raise ValueError("stats_fraction must be between 0 and 1")

    kernel = generate_kernel(n_pulses, window_size=window_size, pulse_duration=pulse_duration)[1]
    observed = signal.convolve(x, kernel, mode='full')[:len(x)]
    scaled_observed = observed * stats_fraction
    observed_poisson = poisson.rvs(np.clip(scaled_observed, 0, None))
    reconstructed = wiener_deconvolution(observed_poisson, kernel, noise_power=noise_power)
    return reconstructed[:len(x)]

def optimize_parameters(t_signal, original_signal, initial_params=None):
    """
    Optimize deconvolution parameters using lmfit's least-squares minimization.

    Parameters
    ----------
    t_signal : array-like
        Time array or independent variable for the signal.
    original_signal : array-like
        Original signal to fit the model to.
    initial_params : dict, optional
        Initial parameter guesses with keys 'n_pulses', 'noise_power', 'pulse_duration'.
        Defaults to {'n_pulses': 5, 'noise_power': 0.05, 'pulse_duration': 200}.

    Returns
    -------
    lmfit.ModelResult
        Result object containing optimized parameters and fit statistics.

    Raises
    ------
    ValueError
        If input signals are invalid or initial parameters are out of bounds.
    TypeError
        If inputs are not array-like or contain non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1000, 1000)
    >>> signal = np.random.normal(0, 1, 1000)
    >>> result = optimize_parameters(t, signal)
    >>> print(result.best_values)
    {'n_pulses': ..., 'noise_power': ..., 'pulse_duration': ..., 'window_size': 5000, 'stats_fraction': 0.2}
    """
    # Input validation
    t_signal = np.asarray(t_signal, dtype=float)
    original_signal = np.asarray(original_signal, dtype=float)
    if t_signal.shape != original_signal.shape:
        raise ValueError("t_signal and original_signal must have the same shape")
    if np.any(np.isnan(t_signal)) or np.any(np.isnan(original_signal)):
        raise ValueError("Input signals must not contain NaN values")

    if initial_params is None:
        initial_params = {'n_pulses': 5, 'noise_power': 0.05, 'pulse_duration': 200}
    
    # Validate initial parameters
    if not (1 <= initial_params['n_pulses'] <= 20):
        raise ValueError("Initial n_pulses must be between 1 and 20")
    if not (0.001 <= initial_params['noise_power'] <= 1.0):
        raise ValueError("Initial noise_power must be between 0.001 and 1.0")
    if not (10 <= initial_params['pulse_duration'] <= 1000):
        raise ValueError("Initial pulse_duration must be between 10 and 1000")

    model = Model(deconvolution_model)
    model.set_param_hint('n_pulses', value=initial_params['n_pulses'], min=1, max=20, vary=True)
    model.set_param_hint('noise_power', value=initial_params['noise_power'], min=0.001, max=1.0, vary=True)
    model.set_param_hint('pulse_duration', value=initial_params['pulse_duration'], min=10, max=1000, vary=True)
    model.set_param_hint('window_size', value=5000, vary=False)
    model.set_param_hint('stats_fraction', value=0.2, vary=False)
    
    result = model.fit(original_signal * 0.2, x=original_signal, method='leastsq')
    return result