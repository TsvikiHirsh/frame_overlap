Frame Overlap Documentation
============================

**frame_overlap** is a Python package for analyzing neutron Time-of-Flight (ToF) frame overlap data using deconvolution techniques.

.. image:: https://img.shields.io/badge/python-3.9%2B-blue
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green
   :alt: License

Features
--------

✨ **Modern API**: Fluent method chaining for complete pipeline processing

📊 **Parameter Sweeps**: Automatic parameter optimization with progress tracking

🔧 **Flexible Processing**: Support for 2+ overlapping frames with multiple reconstruction methods

📈 **Rich Analysis**: Integration with nbragg for material analysis

🎯 **Smart Handling**: Automatic flux scaling and error propagation

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install frame-overlap

Basic usage:

.. code-block:: python

   from frame_overlap import Workflow

   # Complete pipeline in one chain
   wf = Workflow('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)
   result = (wf.convolute(pulse_duration=200)
              .poisson(flux=1e6, freq=60, measurement_time=30)
              .overlap(kernel=[0, 25])
              .reconstruct(kind='wiener', noise_power=0.01)
              .analyze(xs='iron'))

   # Parameter optimization
   results = (wf.groupby('noise_power', low=0.01, high=0.1, num=20)
               .reconstruct('wiener').analyze(xs='iron').run())

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   workflow_guide
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/data
   api/reconstruct
   api/workflow
   api/analysis

.. toctree::
   :maxdepth: 1
   :caption: Examples

   notebooks/example_1_basic_workflow
   notebooks/example_2_parameter_optimization
   notebooks/example_3_multi_frame_overlap
   notebooks/example_4_reconstruction_methods

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
