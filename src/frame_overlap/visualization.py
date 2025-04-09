import matplotlib.pyplot as plt

def plot_analysis(t_signal, original_signal, scaled_original, t_kernel, kernel, observed_poisson, reconstructed, residuals, chi2_per_dof):
    """Plot the full analysis."""
    plt.figure(figsize=(12, 14))
    
    plt.subplot(6, 1, 1)
    plt.plot(t_signal, original_signal)
    plt.title('Original Neutron Signal (Unscaled)')
    plt.xlabel('Time (µs)')
    plt.ylabel('Counts')
    
    plt.subplot(6, 1, 2)
    plt.plot(t_signal, scaled_original)
    plt.title('Scaled Original Signal')
    plt.xlabel('Time (µs)')
    plt.ylabel('Counts')
    
    plt.subplot(6, 1, 3)
    plt.plot(t_kernel, kernel)
    plt.title('Random Kernel')
    plt.xlabel('Time (µs)')
    plt.ylabel('Amplitude')
    
    plt.subplot(6, 1, 4)
    plt.plot(t_signal, observed_poisson)
    plt.title('Observed Signal (Convolution + Poisson Sampling)')
    plt.xlabel('Time (µs)')
    plt.ylabel('Counts')
    
    plt.subplot(6, 1, 5)
    plt.plot(t_signal, scaled_original, 'b-', alpha=0.7, label='Scaled Original')
    plt.plot(t_signal, reconstructed, 'r--', alpha=0.7, label='Reconstructed')
    plt.title(f'Scaled Original vs Reconstructed Signal (χ²/dof = {chi2_per_dof:.3f})')
    plt.xlabel('Time (µs)')
    plt.ylabel('Counts')
    plt.legend()
    
    plt.subplot(6, 1, 6)
    plt.plot(t_signal, residuals, 'k-', label='Residual (Scaled Original - Reconstructed)')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Residuals')
    plt.xlabel('Time (µs)')
    plt.ylabel('Counts')
    plt.legend()
    
    plt.tight_layout()
    plt.show()