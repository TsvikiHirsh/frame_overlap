# Frame Overlap

A Python package for analyzing frame overlap in neutron Time-of-Flight (ToF) data using deconvolution techniques, reconstruction, and material analysis. Version 0.2.0 introduces a modern object-oriented API for intuitive data processing workflows.

## Features

### Core Functionality
- **Data Processing**: Load signal and openbeam data, apply convolution, create frame overlap sequences, and perform Poisson sampling
- **Reconstruction**: Multiple deconvolution methods (Wiener, Richardson-Lucy, Tikhonov) with quality statistics
- **Analysis**: Fit reconstructed data to extract material parameters (thickness, composition) using various response functions
- **Parametric Scans**: Explore sensitivity to different processing parameters with built-in visualization

### Key Capabilities
- Frame overlap simulation with customizable time sequences
- Instrument response convolution (square pulse)
- Wiener, Richardson-Lucy, and Tikhonov deconvolution filters
- Material composition fitting with default Fe-alpha and Cellulose cross sections
- Statistical analysis (χ², R², RMSE) for reconstruction quality
- Comprehensive plotting methods for Jupyter notebooks
- Groupby operations for parametric studies

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

## Quick Start

### New Object-Oriented API (Recommended)

```python
from frame_overlap import Data, Reconstruct, Analysis

# 1. Load and process data
data = Data(signal_file='iron_powder.csv',
            openbeam_file='openbeam.csv',
            flux=1e6,
            duration=3600)

# Apply instrument response convolution
data.convolute_response(pulse_duration=200)

# Create frame overlap with custom time sequence
# seq=[0, 12, 10, 25] means frames at t=0, 12, 22, 47 ms
data.overlap(seq=[0, 12, 10, 25])

# Apply Poisson sampling
data.poisson_sample(duty_cycle=0.8)

# Plot the processed data
data.plot()

# 2. Reconstruct original signal
recon = Reconstruct(data)
recon.filter(kind='wiener', noise_power=0.01)

# View reconstruction quality
print(recon.get_statistics())
recon.plot_comparison()
recon.plot_residuals()

# 3. Fit to extract material parameters
analysis = Analysis(recon)
analysis.set_cross_section(['Fe_alpha', 'Cellulose'], [0.96, 0.04])
analysis.fit(response='square')

# View fit results
print(analysis.get_fit_report())
analysis.plot_fit()
analysis.plot_materials()
```

### Parametric Scans

```python
from frame_overlap import Data, ParametricScan

# Set up template data
data = Data(signal_file='iron_powder.csv')

# Create parametric scan
scan = ParametricScan(data)
scan.add_parameter('pulse_duration', [100, 200, 300, 400])
scan.add_parameter('n_frames', [2, 3, 4, 5])
scan.add_parameter('noise_power', [0.001, 0.01, 0.1])

# Run scan (explores all combinations)
scan.run()

# Visualize results
scan.plot_parameter_sensitivity('pulse_duration', 'fit_reduced_chi_square')
scan.plot_heatmap('pulse_duration', 'n_frames', 'fit_thickness')
scan.plot_summary()

# Get results as DataFrame
results = scan.get_results()
print(results[['pulse_duration', 'n_frames', 'fit_thickness', 'fit_reduced_chi_square']])
```

### Method Chaining

The new API supports fluent method chaining:

```python
from frame_overlap import Data, Reconstruct, Analysis

# Process, reconstruct, and analyze in a single chain
analysis = (Data('signal.csv')
            .convolute_response(pulse_duration=200)
            .overlap(seq=[0, 12, 10, 25])
            .poisson_sample(duty_cycle=0.8)
            |> (lambda data: Reconstruct(data).filter(kind='wiener'))
            |> (lambda recon: Analysis(recon).fit(response='square')))

print(analysis.get_fit_report())
```

### Legacy Functional API

The original functional API remains available for backward compatibility:

```python
from frame_overlap import (read_tof_data, generate_kernel, apply_filter,
                           chi2_analysis, plot_analysis, optimize_parameters)

# Read data
t_signal, signal, errors, stacks = read_tof_data('tof_data.csv', threshold=30)

# Generate kernel
t_kernel, kernel = generate_kernel(n_pulses=5, pulse_duration=200)

# Apply Wiener deconvolution
observed_poisson, reconstructed = apply_filter(signal, kernel,
                                               stats_fraction=0.2,
                                               noise_power=0.05)

# Analyze results
scaled_original = signal * 0.2
chi2, chi2_per_dof = chi2_analysis(scaled_original, reconstructed,
                                   np.sqrt(observed_poisson))
plot_analysis(t_signal, signal, scaled_original, t_kernel, kernel,
             observed_poisson, reconstructed,
             scaled_original - reconstructed, chi2_per_dof)

# Optimize parameters
result = optimize_parameters(t_signal, signal)
print(result.fit_report())
```

## API Overview

### Main Classes

- **`Data`**: Load and process ToF data with convolution, overlap, and sampling
- **`Reconstruct`**: Apply deconvolution filters and calculate quality metrics
- **`Analysis`**: Fit reconstructed data to extract material parameters
- **`ParametricScan`**: Perform parametric studies over processing parameters
- **`CrossSection`**: Define material composition for fitting

### Key Methods

#### Data
- `load_signal_data(file_path)`: Load signal data from CSV
- `load_openbeam_data(file_path)`: Load openbeam reference data
- `convolute_response(pulse_duration)`: Convolve with square instrument response
- `overlap(seq, total_time)`: Create frame overlap with time sequence
- `poisson_sample(duty_cycle)`: Apply Poisson counting statistics
- `plot()`: Visualize data
- `plot_comparison()`: Compare signal with openbeam

#### Reconstruct
- `filter(kind, noise_power, **kwargs)`: Apply deconvolution filter
  - Supported: 'wiener', 'lucy' (Richardson-Lucy), 'tikhonov'
- `get_statistics()`: Get reconstruction quality metrics
- `plot_reconstruction()`: Plot reconstructed signal
- `plot_comparison()`: Compare with reference
- `plot_residuals()`: Plot residuals

#### Analysis
- `set_cross_section(materials, fractions)`: Define material composition
- `fit(response, **kwargs)`: Fit data with response function
  - Supported: 'square', 'square-jorgensen'
- `get_fit_results()`: Get fit parameters and statistics
- `get_fit_report()`: Get formatted fit report
- `plot_fit()`: Plot fitted model
- `plot_residuals()`: Plot fit residuals
- `plot_materials()`: Plot material composition

#### ParametricScan
- `add_parameter(name, values)`: Add parameter to scan
- `run()`: Execute parametric scan
- `get_results()`: Get results as DataFrame
- `plot_parameter_sensitivity()`: Plot metric vs parameter
- `plot_heatmap()`: 2D heatmap of two parameters
- `plot_correlation_matrix()`: Correlation between parameters
- `plot_summary()`: Summary statistics

## Data Format

Input CSV files should have the following columns:
- `stack`: Stack number (integer, starting from 1)
- `counts`: Neutron counts (float)
- `err`: Uncertainty in counts (float, must be positive)

Example:
```
stack,counts,err
1,100,10
2,250,15
3,420,20
...
```

Time is automatically calculated as: `time (µs) = (stack - 1) * 10`

## Examples

See `notebooks/frame_overlap_tutorial.ipynb` for comprehensive examples including:
- Basic data loading and processing
- Frame overlap creation and reconstruction
- Material parameter fitting
- Parametric sensitivity studies
- Comparison of different filter methods

## License
MIT License

## Contributing
Pull requests are welcome! Please open an issue first to discuss changes.

## Version History

### v0.2.0 (Current)
- Added object-oriented API with Data, Reconstruct, and Analysis classes
- Implemented frame overlap functionality
- Added Richardson-Lucy and Tikhonov deconvolution
- Introduced ParametricScan for sensitivity studies
- Added material composition fitting
- Comprehensive plotting methods for Jupyter notebooks
- Maintained backward compatibility with v0.1.0 functional API

### v0.1.0
- Initial release with functional API
- Wiener deconvolution
- Parameter optimization with lmfit
- Basic visualization