# Changelog

All notable changes to frame_overlap will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-31

### Added - Major Features

- **Workflow Class**: Complete fluent API for method chaining entire pipeline from data loading through analysis
  - Chain data processing, reconstruction, and analysis in one expression
  - Automatic state tracking and smart method routing
  - Access to all intermediate results (`wf.data`, `wf.recon`, `wf.analysis`, `wf.result`)

- **Parameter Sweeps**: Automatic parameter space exploration with progress tracking
  - `groupby(param_name, low, high, num/step)` for range-based sweeps
  - `sweep(param_name, values)` for custom value lists
  - Returns pandas DataFrame with chi², AIC, BIC, and all fitted parameters
  - Automatic tqdm progress bars (Jupyter and terminal compatible)
  - Smart error handling - continues even if individual runs fail

- **Analysis Class**: Simplified nbragg integration with clean API
  - Predefined cross-sections: 'iron', 'iron_square_response', 'iron_with_cellulose'
  - Set 'iron' (Fe_sg229_Iron-alpha.ncmat) as default model
  - Easy access to nbragg model and parameters
  - Automatic conversion with `to_nbragg()` method

### Added - Data Processing

- **Automatic Flux Scaling**: Poisson sampling now automatically scales flux by pulse_duration
  - Formula: `duty_cycle = (flux_new / flux_orig) × freq × pulse_duration`
  - No need for manual calculation
  - Works for both pulsed and continuous sources

- **Time Range Filtering**: Added `tmin`/`tmax` parameters to Reconstruct
  - Calculate chi² only on specific time ranges
  - Show full data but focus statistics on relevant region
  - Vertical indicators (green/orange) show filtered range on plots

- **Enhanced Plotting**: Two-subplot layout for reconstructions
  - Top: Data comparison with error bands
  - Bottom: Residuals in σ units
  - Automatic stage detection (shows most processed stage)
  - Fixed plot priorities: overlapped → poissoned → convolved → original

### Added - Documentation

- **Sphinx Documentation**: Complete documentation site with ReadTheDocs theme
  - Installation guide, Quick Start, Workflow Guide
  - Full API reference for all classes
  - Contribution guidelines and changelog
  - Deployed to GitHub Pages

- **Example Notebooks**: 4 comprehensive Jupyter notebooks
  - `example_1_basic_workflow.ipynb`: Complete pipeline demonstration
  - `example_2_parameter_optimization.ipynb`: Parameter sweeps and optimization
  - `example_3_multi_frame_overlap.ipynb`: Working with 3-4 overlapping frames
  - `example_4_reconstruction_methods.ipynb`: Comparing Wiener, Lucy, Tikhonov

- **Updated Tutorial**: Comprehensive tutorial covering new Workflow API
  - Correct processing order explanation
  - Method chaining examples
  - Parameter sweep examples
  - Updated to use Analysis class (renamed from Model)

### Changed

- **Breaking**: Minimum Python version increased to 3.9+ (was 3.7+)
- **Breaking**: Renamed `Model` class to `Analysis` for clarity
- **Processing Order**: Emphasized correct order: Convolute → Poisson → Overlap
  - Convolute MUST come before Poisson to store pulse_duration
  - Updated all examples and documentation

- **Dependencies**: Added scikit-image and tqdm as required dependencies
  - lmfit now optional (install with `pip install frame-overlap[lmfit]`)
  - Added optional dependency groups: `nbragg`, `lmfit`, `dev`, `all`

- **Plot Behavior**: Fixed stage priority in all plot methods
  - Now correctly shows most processed stage when not using `show_stages=True`
  - Transmission plots now show overlapped data (not poissoned) after overlap

### Fixed

- **Duty Cycle Calculation**: Corrected formula for pulsed sources
  - Removed measurement_time from calculation when pulse_duration is set
  - Correct formula: `duty_cycle = (flux_new / flux_orig) × freq × pulse_duration`
  - Added clear distinction between pulsed and continuous source formulas

- **Plot Stage Order**: Fixed incorrect stage ordering in `show_stages=True`
  - Was showing: Original → Convolved → Overlapped → Poissoned
  - Now shows: Original → Convolved → Poissoned → Overlapped

- **nbragg Integration**: Fixed `to_nbragg()` to pass both signal and openbeam
  - Now reconstructs both signal and openbeam DataFrames independently
  - Passes both to `nbragg.Data.from_counts()` as expected

- **Reconstruct.__repr__**: Fixed format string issue with chi²/dof display

- **Workflow Parameter Sweep**: Fixed data reloading to preserve flux/duration/freq
  - Sweep iterations now correctly pass original parameters when creating new Data objects

### Documentation

- Full documentation site at https://tsvikihirsh.github.io/frame_overlap/
- Simplified README focused on Workflow API (removed legacy code examples)
- Direct links to documentation, examples, and API reference
- GitHub Actions workflow for automatic documentation deployment

## [0.1.0] - 2024-XX-XX

### Added

- Initial release with functional API
- Data loading from CSV files (signal and openbeam)
- Wiener deconvolution for frame overlap reconstruction
- Parameter optimization with lmfit
- Basic visualization and plotting
- Frame overlap generation with customizable kernels
- Chi-squared analysis and quality metrics
- Support for Poisson sampling

[0.2.0]: https://github.com/TsvikiHirsh/frame_overlap/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/TsvikiHirsh/frame_overlap/releases/tag/v0.1.0
