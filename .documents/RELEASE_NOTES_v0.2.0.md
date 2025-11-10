# frame_overlap v0.2.0 - Major Feature Release

We're excited to announce frame_overlap v0.2.0, a major update that introduces a modern fluent API, parameter sweeps, and comprehensive documentation!

## 🎉 Highlights

### Workflow Class - Complete Pipeline in One Chain

The new `Workflow` class lets you chain the entire pipeline in a single fluent expression:

```python
from frame_overlap import Workflow

wf = Workflow('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)

result = (wf
    .convolute(pulse_duration=200)
    .poisson(flux=1e6, freq=60, measurement_time=30)
    .overlap(kernel=[0, 25])
    .reconstruct(kind='wiener', noise_power=0.01)
    .analyze(xs='iron'))

wf.plot()  # Visualize results
```

### Parameter Sweeps with Progress Bars

Find optimal parameters automatically:

```python
results = (Workflow('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)
    .convolute(pulse_duration=200)
    .poisson(flux=1e6, freq=60, measurement_time=30)
    .overlap(kernel=[0, 25])
    .groupby('noise_power', low=0.01, high=0.1, num=20)  # Sweep!
    .reconstruct(kind='wiener')
    .analyze(xs='iron')
    .run())  # Shows progress bar

best = results.loc[results['redchi2'].idxmin()]
```

### Complete Documentation Site

📚 **New documentation**: https://tsvikihirsh.github.io/frame_overlap/

- Installation guide and quick start
- Complete Workflow guide with best practices
- Full API reference
- 4 comprehensive example notebooks

## ✨ New Features

**Workflow Class**
- Complete fluent API for method chaining
- Automatic state tracking
- Parameter sweeps with `groupby()` and `sweep()`
- Progress bars with tqdm (Jupyter and terminal)
- Returns pandas DataFrame with all metrics

**Analysis Class**
- Simplified nbragg integration
- Predefined cross-sections: 'iron', 'iron_square_response', 'iron_with_cellulose'
- Easy model parameter access

**Smart Processing**
- Automatic flux scaling by pulse_duration in Poisson sampling
- Time range filtering with `tmin`/`tmax` parameters
- Enhanced plotting with residuals in σ units
- Vertical indicators showing filtered ranges

**Documentation**
- Sphinx documentation with ReadTheDocs theme
- 4 example notebooks (basic, optimization, multi-frame, methods)
- GitHub Actions for automatic deployment
- Simplified README focused on modern API

## 🔧 Improvements

- **Correct Processing Order**: Emphasized Convolute → Poisson → Overlap
  - Pulse duration from convolution defines flux scaling
  - All examples and docs updated

- **Fixed Duty Cycle**: Corrected formula for pulsed sources
  - `duty_cycle = (flux_new / flux_orig) × freq × pulse_duration`

- **Better Plots**: Fixed stage ordering and priorities
  - Shows most processed stage by default
  - Correct order with `show_stages=True`

- **Enhanced nbragg**: Fixed `to_nbragg()` to pass both signal and openbeam

## 📦 Installation

```bash
pip install git+https://github.com/TsvikiHirsh/frame_overlap.git@v0.2.0
```

With optional dependencies:

```bash
# With nbragg
pip install "git+https://github.com/TsvikiHirsh/frame_overlap.git@v0.2.0#egg=frame_overlap[nbragg]"

# With all extras
pip install "git+https://github.com/TsvikiHirsh/frame_overlap.git@v0.2.0#egg=frame_overlap[all]"
```

## 📖 Quick Start

See the [Quick Start Guide](https://tsvikihirsh.github.io/frame_overlap/quickstart.html) for a 5-minute introduction.

Check out the [example notebooks](https://github.com/TsvikiHirsh/frame_overlap/tree/main/notebooks) for hands-on tutorials.

## 🔄 Upgrading from v0.1.0

**Breaking Changes:**
- Minimum Python version: 3.9+ (was 3.7+)
- `Model` class renamed to `Analysis`
- lmfit is now optional (install with `[lmfit]`)

**Migration:**
```python
# Old (v0.1.0)
from frame_overlap import Model
model = Model(...)

# New (v0.2.0)
from frame_overlap import Analysis
analysis = Analysis(...)
```

The functional API remains unchanged for backward compatibility.

## 📝 Full Changelog

See [CHANGELOG.md](CHANGELOG.md) for complete details.

## 🙏 Acknowledgments

Thanks to all contributors and users who provided feedback!

## 📧 Contact

- **Documentation**: https://tsvikihirsh.github.io/frame_overlap/
- **Repository**: https://github.com/TsvikiHirsh/frame_overlap
- **Issues**: https://github.com/TsvikiHirsh/frame_overlap/issues
