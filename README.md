# frame_overlap

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tsvikihirsh.github.io/frame_overlap/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frame-overlap.streamlit.app)

A Python package for analyzing neutron Time-of-Flight (ToF) frame overlap data using deconvolution techniques, with advanced **adaptive Bragg edge measurement optimization**.

**📚 [Full Documentation](https://tsvikihirsh.github.io/frame_overlap/)** | **📓 [Examples](notebooks/)** | **🚀 [Quick Start](#quick-start)** | **🎮 [Try the Interactive App](https://frame-overlap.streamlit.app)**

## Features

### Frame Overlap Analysis
✨ **Fluent API** - Chain complete pipeline in one expression
📊 **Parameter Sweeps** - Automatic optimization with progress tracking
🔧 **Multi-Frame** - Support 2+ overlapping frames
📈 **Material Analysis** - Integrated nbragg fitting
🎯 **Smart Scaling** - Automatic flux scaling by pulse duration

### Adaptive Bragg Edge Measurement (NEW!)
⚛️ **Bayesian Optimization** - Adaptive chopper pattern design
🎯 **2-5x Speedup** - Reach target precision faster
🧠 **Information-Theoretic** - Focus on high-value measurements
🔄 **Real-Time Adaptation** - Update patterns based on data
📊 **Performance Metrics** - Compare strategies quantitatively

## Installation

```bash
pip install git+https://github.com/TsvikiHirsh/frame_overlap.git
```

## Quick Start

### TL;DR - Complete Pipeline

```python
from frame_overlap import Workflow

# One chain from data to analysis
wf = Workflow('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)

result = (wf
    .convolute(pulse_duration=200)
    .poisson(flux=1e6, freq=60, measurement_time=30)
    .overlap(kernel=[0, 25])
    .reconstruct(kind='wiener', noise_power=0.01)
    .analyze(xs='iron'))

wf.plot()  # Visualize results
```

### Parameter Optimization

```python
# Find optimal noise_power
results = (Workflow('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)
    .convolute(pulse_duration=200)
    .poisson(flux=1e6, freq=60, measurement_time=30)
    .overlap(kernel=[0, 25])
    .groupby('noise_power', low=0.01, high=0.1, num=20)  # Sweep parameter
    .reconstruct(kind='wiener')
    .analyze(xs='iron')
    .run())  # Shows progress bar!

# Find best
best = results.loc[results['redchi2'].idxmin()]
print(f"Optimal noise_power: {best['noise_power']:.4f}")
```

### Adaptive Bragg Edge Measurement

```python
from frame_overlap import (
    BraggEdgeSample,
    BraggEdgeMeasurementSystem,
    AdaptiveEdgeOptimizer,
    MeasurementTarget
)

# Create iron sample with strain
sample = BraggEdgeSample.create_iron_sample(strain=0.001)

# Set up measurement system
system = BraggEdgeMeasurementSystem(
    flight_path=10.0,  # meters
    wavelength_range=(3.0, 5.0)  # Angstrom
)

# Define measurement target
target = MeasurementTarget(
    material='Fe',
    expected_edge=4.05,  # Angstrom
    precision_required=0.005,  # 5 milliAngstrom
    max_measurement_time=300.0  # seconds
)

# Run adaptive optimization
optimizer = AdaptiveEdgeOptimizer(system, target, strategy='bayesian')
adaptive_result, uniform_result = optimizer.simulate_comparison(
    sample, flux=1e6, measurement_time_per_pattern=10.0
)

# Compare results
speedup = uniform_result.measurement_time / adaptive_result.measurement_time
print(f"Speedup: {speedup:.2f}x faster")
print(f"Measured strain: {adaptive_result.strain*1e6:.1f} microstrain")
```

#### Interactive Demo App

Try the adaptive Bragg edge measurement demo:

```bash
streamlit run streamlit_adaptive_bragg.py
```

## Processing Pipeline

```
Data → Convolute → Poisson → Overlap → Reconstruct → Analysis
        ↓           ↓          ↓          ↓           ↓
    Instrument  Add noise  Frame ops  Recover     Material
     response   (+flux              signal      fitting
                scaling)
```

**Key**: Convolute **before** Poisson! The `pulse_duration` defines flux scaling.

## Documentation

📚 **[Full Documentation](https://tsvikihirsh.github.io/frame_overlap/)**

- [Installation Guide](https://tsvikihirsh.github.io/frame_overlap/installation.html)
- [Quick Start](https://tsvikihirsh.github.io/frame_overlap/quickstart.html)
- [Workflow Guide](https://tsvikihirsh.github.io/frame_overlap/workflow_guide.html)
- [API Reference](https://tsvikihirsh.github.io/frame_overlap/api/workflow.html)

## Examples

📓 **[Example Notebooks](notebooks/)**

### Frame Overlap Analysis
- [Basic Workflow](notebooks/example_1_basic_workflow.ipynb) - Complete pipeline
- [Parameter Optimization](notebooks/example_2_parameter_optimization.ipynb) - Find optimal parameters
- [Multi-Frame Overlap](notebooks/example_3_multi_frame_overlap.ipynb) - 3-4 frame setups
- [Reconstruction Methods](notebooks/example_4_reconstruction_methods.ipynb) - Compare algorithms

### Adaptive Bragg Edge Measurement
- [Adaptive Optimization](notebooks/example_adaptive_bragg_edge.ipynb) - Complete guide to adaptive measurements

## Main Classes

### Frame Overlap Analysis
**`Workflow`** - Complete pipeline with method chaining and parameter sweeps
**`Data`** - Load and process ToF data
**`Reconstruct`** - Deconvolution (Wiener, Lucy-Richardson, Tikhonov)
**`Analysis`** - Material fitting with nbragg

### Adaptive Bragg Edge Measurement
**`BraggEdgeMeasurementSystem`** - Complete measurement system with TOF calibration
**`AdaptiveEdgeOptimizer`** - Bayesian optimization for adaptive measurements
**`BraggEdgeSample`** - Sample model with multiple Bragg edges
**`PatternLibrary`** - Collection of chopper pattern generation strategies
**`PerformanceEvaluator`** - Tools for evaluating measurement strategies

See [API Reference](https://tsvikihirsh.github.io/frame_overlap/api/workflow.html) for details.

## Requirements

- Python 3.9+
- numpy, pandas, matplotlib, scipy, scikit-image, tqdm

Optional: nbragg (for analysis), lmfit (for optimization), jupyter (for notebooks)

## Citation

If you use this package, please cite:

```bibtex
@software{frame_overlap,
  title = {frame_overlap: Neutron ToF Frame Overlap Analysis},
  author = {Hirsh, Tsviki and contributors},
  year = {2025},
  url = {https://github.com/TsvikiHirsh/frame_overlap}
}
```

## Deployment

The interactive Streamlit app can be deployed to Streamlit Cloud. See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Deployment instructions
- nbragg installation troubleshooting
- Graceful fallback behavior when dependencies are unavailable

**Live App**: https://frame-overlap.streamlit.app

## License

MIT License - see [LICENSE](LICENSE) file

## Links

- **Documentation**: https://tsvikihirsh.github.io/frame_overlap/
- **Repository**: https://github.com/TsvikiHirsh/frame_overlap
- **Issues**: https://github.com/TsvikiHirsh/frame_overlap/issues
