# frame_overlap

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tsvikihirsh.github.io/frame_overlap/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frame-overlap.streamlit.app)

A Python package for analyzing neutron Time-of-Flight (ToF) frame overlap data using deconvolution techniques.

**📚 [Full Documentation](https://tsvikihirsh.github.io/frame_overlap/)** | **📓 [Examples](notebooks/)** | **🚀 [Quick Start](#quick-start)** | **🎮 [Try the Interactive App](https://frame-overlap.streamlit.app)**

## Features

✨ **Fluent API** - Chain complete pipeline in one expression
📊 **Parameter Sweeps** - Automatic optimization with progress tracking
🔧 **Multi-Frame** - Support 2+ overlapping frames
📈 **Material Analysis** - Integrated nbragg fitting
🎯 **Smart Scaling** - Automatic flux scaling by pulse duration

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

- [Basic Workflow](notebooks/example_1_basic_workflow.ipynb) - Complete pipeline
- [Parameter Optimization](notebooks/example_2_parameter_optimization.ipynb) - Find optimal parameters
- [Multi-Frame Overlap](notebooks/example_3_multi_frame_overlap.ipynb) - 3-4 frame setups
- [Reconstruction Methods](notebooks/example_4_reconstruction_methods.ipynb) - Compare algorithms

## Main Classes

**`Workflow`** - Complete pipeline with method chaining and parameter sweeps
**`Data`** - Load and process ToF data
**`Reconstruct`** - Deconvolution (Wiener, Lucy-Richardson, Tikhonov)
**`Analysis`** - Material fitting with nbragg

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
