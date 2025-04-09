# Frame Overlap

A Python package for analyzing frame overlap in neutron Time-of-Flight (ToF) data using deconvolution techniques, parameter optimization, and visualization.

## Features
- **Data Utilities**: Read and prepare ToF data from CSV files.
- **Analysis**: Wiener deconvolution with non-overlapping rectangular pulse kernels and Poisson sampling.
- **Optimization**: Chi-squared comparison and parameter optimization using `lmfit`.
- **Visualization**: Comprehensive plotting of signals, kernels, and residuals.

## Installation
Install via pip:
```bash
pip install git+https://github.com/TsvikiHirsh/frame_overlap.git
```

Or clone and install locally:
```bash
git clone https://github.com/TsvikiHirsh/frame_overlap.git
cd frame_overlap
pip install .
```

## Requirements
- Python >= 3.7
- NumPy >= 1.21
- Pandas >= 1.3
- SciPy >= 1.7
- Matplotlib >= 3.4
- lmfit >= 1.0

## Usage
```python
from frame_overlap import read_tof_data, generate_kernel, wiener_deconvolution, plot_analysis, optimize_parameters

# Read data
t_signal, signal, errors, stacks = read_tof_data('tof_data.csv', threshold=30)

# Generate kernel
t_kernel, kernel = generate_kernel(n_pulses=5, pulse_duration=200)

# Apply Wiener deconvolution
observed_poisson, reconstructed = apply_filter(signal, kernel, stats_fraction=0.2, noise_power=0.05)

# Plot results
scaled_original = signal * 0.2
chi2, chi2_per_dof = chi2_analysis(scaled_original, reconstructed, np.sqrt(observed_poisson))
plot_analysis(t_signal, signal, scaled_original, t_kernel, kernel, observed_poisson, reconstructed, scaled_original - reconstructed, chi2_per_dof)

# Optimize parameters
result = optimize_parameters(t_signal, signal)
print(result.fit_report())
```

## License
MIT License

## Contributing
Pull requests are welcome! Please open an issue first to discuss changes.