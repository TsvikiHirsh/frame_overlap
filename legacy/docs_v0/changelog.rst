Changelog
=========

Version 0.2.0 (2025-01-XX)
--------------------------

**New Features**

- ✨ **Workflow class**: Complete fluent API for method chaining
- 📊 **Parameter sweeps**: ``groupby()`` and ``sweep()`` for parameter exploration
- 🎯 **Progress tracking**: Automatic tqdm progress bars in Jupyter and terminal
- 📈 **Rich results**: DataFrame output with chi², AIC, BIC, and all fitted parameters
- 🔧 **Automatic flux scaling**: Poisson sampling automatically uses pulse_duration
- 📏 **Time range filtering**: ``tmin``/``tmax`` parameters for chi² calculation
- 📍 **Visual indicators**: Vertical lines showing tmin/tmax on plots
- 🔗 **nbragg integration**: ``to_nbragg()`` for wavelength conversion
- 🎛️ **Noise optimization**: ``optimize_noise()`` with lmfit

**Improvements**

- Set 'iron' as default cross-section model
- Fixed plot priorities to show most processed stage
- Improved error handling in parameter sweeps
- Enhanced documentation with examples and API reference

**Bug Fixes**

- Fixed transmission plot showing wrong stage after overlap
- Fixed Reconstruct.__repr__ format string issue
- Fixed Workflow parameter sweep data reloading
- Fixed stage ordering in plot methods

Version 0.1.0 (2024-XX-XX)
--------------------------

**Initial Release**

- Basic Data class for loading and processing ToF data
- Reconstruct class with Wiener, Lucy-Richardson, and Tikhonov filters
- Frame overlap simulation and reconstruction
- Integration with nbragg for material analysis
