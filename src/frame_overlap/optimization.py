import numpy as np
import pandas as pd
from lmfit import Model
from scipy.stats import poisson

def chi2_analysis(scaled_original_df, reconstructed_df, errors_df):
    """
    Calculate chi-squared and reduced chi-squared statistics for model fit evaluation.

    Parameters
    ----------
    scaled_original_df : pandas.DataFrame
        DataFrame with column 'counts' containing the scaled original signal.
    reconstructed_df : pandas.DataFrame
        DataFrame with column 'reconstructed' containing the reconstructed signal.
    errors_df : pandas.DataFrame
        DataFrame with column 'errors' containing the uncertainties.

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
    """
    if 'counts' not in scaled_original_df.columns or 'reconstructed' not in reconstructed_df.columns or 'errors' not in errors_df.columns:
        raise ValueError("Input DataFrames must have 'counts', 'reconstructed', and 'errors' columns respectively")
    
    scaled_original = scaled_original_df['counts'].to_numpy()
    reconstructed = reconstructed_df['reconstructed'].to_numpy()
    errors = errors_df['errors'].to_numpy()
    
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

def deconvolution_model(signal_df, n_pulses, noise_power, pulse_duration, window_size=5000, stats_fraction=0.2):
    """
    Apply Wiener deconvolution to model a signal with specified parameters.

    Parameters
    ----------
    signal_df : pandas.DataFrame
        DataFrame with column 'counts' containing the input signal.
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
    pandas.DataFrame
        DataFrame with column 'reconstructed' containing the reconstructed signal.

    Raises
    ------
    ValueError
        If input parameters are invalid (e.g., negative values, non-positive n_pulses).
    """
    from .analysis import generate_kernel, wiener_deconvolution

    if 'counts' not in signal_df.columns:
        raise ValueError("signal_df must have 'counts' column")
    
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

    kernel_df = generate_kernel(n_pulses, window_size=window_size, pulse_duration=pulse_duration)
    observed = signal_df['counts'].to_numpy()
    scaled_observed = observed * stats_fraction
    observed_poisson = poisson.rvs(np.clip(scaled_observed, 0, None))
    observed_df = pd.DataFrame({'counts': observed_poisson})
    reconstructed_df = wiener_deconvolution(observed_df, kernel_df, noise_power=noise_power)
    return reconstructed_df

def optimize_parameters(t_signal_df, signal_df, initial_params=None):
    """
    Optimize deconvolution parameters using lmfit's least-squares minimization.

    Parameters
    ----------
    t_signal_df : pandas.DataFrame
        DataFrame with column 'time' containing the time array.
    signal_df : pandas.DataFrame
        DataFrame with column 'counts' containing the original signal.
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
    """
    if 'time' not in t_signal_df.columns or 'counts' not in signal_df.columns:
        raise ValueError("t_signal_df must have 'time' column and signal_df must have 'counts' column")
    
    t_signal = t_signal_df['time'].to_numpy()
    signal = signal_df['counts'].to_numpy()
    
    if t_signal.shape != signal.shape:
        raise ValueError("t_signal and signal must have the same shape")
    if np.any(np.isnan(t_signal)) or np.any(np.isnan(signal)):
        raise ValueError("Input signals must not contain NaN values")

    if initial_params is None:
        initial_params = {'n_pulses': 5, 'noise_power': 0.05, 'pulse_duration': 200}
    
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
    
    scaled_signal_df = pd.DataFrame({'counts': signal * 0.2})
    result = model.fit(scaled_signal_df['counts'], signal_df=signal_df, method='leastsq')
    return result