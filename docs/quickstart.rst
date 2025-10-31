Quick Start Guide
=================

This guide will help you get started with frame_overlap in 5 minutes.

Basic Concepts
--------------

Frame overlap occurs when neutron pulses in Time-of-Flight measurements overlap before reaching the detector. This package provides tools to:

1. **Simulate** frame overlap conditions
2. **Reconstruct** the original signal using deconvolution
3. **Analyze** material properties using nbragg

Processing Pipeline
-------------------

The correct processing order is:

.. code-block:: text

   Data → Convolute → Poisson → Overlap → Reconstruct → Analysis
           ↓           ↓          ↓          ↓           ↓
       Instrument  Add noise  Frame ops  Recover     Fit with
        response   (+flux              signal      nbragg
                   scaling)

Minimal Example
---------------

.. code-block:: python

   from frame_overlap import Workflow

   # Complete analysis in one chain
   wf = Workflow('signal.csv', 'openbeam.csv',
                 flux=5e6, duration=0.5, freq=20)

   result = (wf
       .convolute(pulse_duration=200)
       .poisson(flux=1e6, freq=60, measurement_time=30)
       .overlap(kernel=[0, 25])
       .reconstruct(kind='wiener', noise_power=0.01)
       .analyze(xs='iron'))

   # Plot results
   wf.plot()

Step-by-Step Example
---------------------

For more control, use the individual classes:

.. code-block:: python

   from frame_overlap import Data, Reconstruct, Analysis

   # 1. Load data
   data = Data('signal.csv', 'openbeam.csv',
               flux=5e6, duration=0.5, freq=20)

   # 2. Apply convolution (stores pulse_duration)
   data.convolute_response(pulse_duration=200)

   # 3. Poisson sampling (automatically uses pulse_duration)
   data.poisson_sample(flux=1e6, freq=60, measurement_time=30)

   # 4. Create frame overlap
   data.overlap(kernel=[0, 25])

   # 5. Reconstruct signal
   recon = Reconstruct(data, tmin=10, tmax=40)
   recon.filter(kind='wiener', noise_power=0.01)

   # 6. Analyze with nbragg
   analysis = Analysis(xs='iron')
   result = analysis.fit(recon)

   # 7. Visualize
   recon.plot()
   analysis.plot()

Parameter Optimization
----------------------

Find optimal processing parameters:

.. code-block:: python

   # Optimize noise_power
   results = (Workflow('signal.csv', 'openbeam.csv',
                       flux=5e6, duration=0.5, freq=20)
       .convolute(pulse_duration=200)
       .poisson(flux=1e6, freq=60, measurement_time=30)
       .overlap(kernel=[0, 25])
       .groupby('noise_power', low=0.01, high=0.1, num=20)
       .reconstruct(kind='wiener')
       .analyze(xs='iron')
       .run())

   # Find best parameters
   best = results.loc[results['redchi2'].idxmin()]
   print(f"Optimal noise_power: {best['noise_power']:.4f}")
   print(f"χ²/dof: {best['redchi2']:.2f}")

   # Visualize
   results.plot(x='noise_power', y='redchi2')

Common Parameters
-----------------

**Data Loading**

- ``flux``: Original flux in n/cm²/s
- ``duration``: Measurement duration in hours
- ``freq``: Pulse frequency in Hz

**Convolution**

- ``pulse_duration``: Pulse width in microseconds
- ``bin_width``: Time bin width (default: 10 µs)

**Poisson Sampling**

- ``flux``: New flux condition
- ``freq``: New frequency
- ``measurement_time``: New measurement duration (minutes)
- ``seed``: Random seed for reproducibility

**Frame Overlap**

- ``kernel``: List of frame start times in ms, e.g., ``[0, 25]``
- ``total_time``: Total time span in ms

**Reconstruction**

- ``kind``: Method - ``'wiener'``, ``'lucy'``, or ``'tikhonov'``
- ``noise_power``: Regularization parameter for Wiener/Tikhonov
- ``iterations``: Number of iterations for Lucy-Richardson
- ``tmin``/``tmax``: Time range for chi² calculation (ms)

**Analysis**

- ``xs``: Cross-section - ``'iron'``, ``'iron_with_cellulose'``, etc.
- ``vary_background``: Fit background (default: True)
- ``vary_response``: Fit response function
- ``L``: Flight path length in meters (default: 9.0)
- ``tstep``: Time step in seconds (default: 10e-6)

Next Steps
----------

- See :doc:`workflow_guide` for detailed workflow documentation
- Check out :doc:`notebooks/example_1_basic_workflow` for a complete example
- Explore :doc:`api/workflow` for API reference
