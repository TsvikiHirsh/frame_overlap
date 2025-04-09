import numpy as np
import pandas as pd
from lmfit import Model

def chi2_analysis(scaled_original, reconstructed, errors):
    """Calculate chi-squared and reduced chi-squared."""
    residuals = scaled_original - reconstructed
    chi2 = np.sum((residuals / errors)**2)
    chi2_per_dof = chi2 / len(scaled_original)
    return chi2, chi2_per_dof

def deconvolution_model(x, n_pulses, noise_power, pulse_duration, window_size=5000, stats_fraction=0.2):
    """Model for optimization: applies Wiener deconvolution."""
    from .analysis import generate_kernel, wiener_deconvolution, signal
    kernel = generate_kernel(int(n_pulses), window_size=window_size, pulse_duration=pulse_duration)[1]
    observed = signal.convolve(x, kernel, mode='full')[:len(x)]
    scaled_observed = observed * stats_fraction
    observed_poisson = poisson.rvs(scaled_observed)
    observed_poisson = np.clip(observed_poisson, 0, None)
    reconstructed = wiener_deconvolution(observed_poisson, kernel, noise_power=noise_power)
    return reconstructed[:len(x)]

def optimize_parameters(t_signal, original_signal, initial_params=None):
    """Optimize deconvolution parameters using lmfit."""
    if initial_params is None:
        initial_params = {'n_pulses': 5, 'noise_power': 0.05, 'pulse_duration': 200}
    
    model = Model(deconvolution_model)
    model.set_param_hint('n_pulses', value=initial_params['n_pulses'], min=1, max=20, vary=True)
    model.set_param_hint('noise_power', value=initial_params['noise_power'], min=0.001, max=1.0, vary=True)
    model.set_param_hint('pulse_duration', value=initial_params['pulse_duration'], min=10, max=1000, vary=True)
    model.set_param_hint('window_size', value=5000, vary=False)
    model.set_param_hint('stats_fraction', value=0.2, vary=False)
    
    result = model.fit(original_signal * 0.2, x=original_signal, method='leastsq')  # Fit to scaled original
    return result