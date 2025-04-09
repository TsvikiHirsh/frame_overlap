import numpy as np
from scipy import signal
from scipy.stats import poisson

def generate_kernel(n_pulses, window_size=5000, bin_width=10, pulse_duration=200, pulse_height=1.0):
    """Generate a kernel with non-overlapping rectangular pulses."""
    t_kernel = np.arange(0, window_size, bin_width)
    kernel = np.zeros_like(t_kernel, dtype=float)
    
    pulse_length = int(pulse_duration / bin_width)
    total_pulse_space = n_pulses * pulse_length
    
    if total_pulse_space > len(t_kernel):
        raise ValueError(f"Total pulse space ({total_pulse_space * bin_width} µs) exceeds window size ({window_size} µs).")
    
    available_indices = list(range(len(t_kernel) - pulse_length + 1))
    start_indices = []
    
    for _ in range(n_pulses):
        if not available_indices:
            raise ValueError("Not enough space for non-overlapping pulses.")
        start_idx = np.random.choice(available_indices)
        start_indices.append(start_idx)
        overlap_range = range(max(0, start_idx - pulse_length + 1), min(len(t_kernel) - pulse_length + 1, start_idx + pulse_length))
        available_indices = [idx for idx in available_indices if idx not in overlap_range]
    
    start_indices.sort()
    for start_idx in start_indices:
        kernel[start_idx:start_idx + pulse_length] = pulse_height
    
    return t_kernel, kernel

def wiener_deconvolution(observed, kernel, noise_power=0.01):
    """Perform Wiener deconvolution."""
    kernel_padded = np.pad(kernel, (0, len(observed) - len(kernel)), 'constant')
    H = np.fft.fft(kernel_padded)
    Y = np.fft.fft(observed)
    H_conj = np.conj(H)
    G = H_conj / (np.abs(H)**2 + noise_power)
    X_est = Y * G
    x_est = np.real(np.fft.ifft(X_est))
    return x_est

def apply_filter(signal, kernel, filter_type='wiener', stats_fraction=0.2, noise_power=0.01):
    """Apply a filter (e.g., Wiener) with Poisson sampling."""
    observed = signal.convolve(signal, kernel, mode='full')[:len(signal)]
    scaled_observed = observed * stats_fraction
    observed_poisson = poisson.rvs(scaled_observed)
    observed_poisson = np.clip(observed_poisson, 0, None)
    
    if filter_type == 'wiener':
        reconstructed = wiener_deconvolution(observed_poisson, kernel, noise_power=noise_power)
    else:
        raise ValueError(f"Filter type '{filter_type}' not supported. Use 'wiener'.")
    
    return observed_poisson, reconstructed