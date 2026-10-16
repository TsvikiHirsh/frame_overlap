# frame_overlap

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tsvikihirsh.github.io/frame_overlap/)

A Python package for analyzing neutron Time-of-Flight (ToF) frame overlap data using deconvolution techniques.

## Features

✨ **Modern API**: Fluent method chaining for complete pipeline processing  
📊 **Parameter Sweeps**: Automatic parameter optimization with progress tracking  
🔧 **Flexible Processing**: Support for 2+ overlapping frames with multiple reconstruction methods  
📈 **Rich Analysis**: Integration with nbragg for material analysis  
🎯 **Smart Handling**: Automatic flux scaling and error propagation

## Installation

\`\`\`bash
pip install frame-overlap
\`\`\`

## Quick Start

\`\`\`python
from frame_overlap import Workflow

# Complete analysis pipeline
wf = Workflow('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)

result = (wf
    .convolute(pulse_duration=200)
    .poisson(flux=1e6, freq=60, measurement_time=30)
    .overlap(kernel=[0, 25])
    .reconstruct(kind='wiener', noise_power=0.01)
    .analyze(xs='iron'))

wf.plot()
\`\`\`

## Documentation

📚 **Full documentation**: https://tsvikihirsh.github.io/frame_overlap/

## License

MIT License
