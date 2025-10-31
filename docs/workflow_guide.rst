Workflow Guide
==============

The ``Workflow`` class provides a high-level fluent API for chaining the complete processing pipeline.

Three Usage Patterns
---------------------

1. Step-by-Step (Full Control)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use individual classes for maximum flexibility:

.. code-block:: python

   from frame_overlap import Data, Reconstruct, Analysis

   data = Data('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)
   data.convolute_response(200)
   data.poisson_sample(flux=1e6, freq=60, measurement_time=30)
   data.overlap([0, 25])

   recon = Reconstruct(data, tmin=10, tmax=40)
   recon.filter(kind='wiener', noise_power=0.01)

   analysis = Analysis(xs='iron')
   result = analysis.fit(recon)

2. Method Chaining (Data class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chain Data methods for concise processing:

.. code-block:: python

   data = (Data('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)
           .convolute_response(200)
           .poisson_sample(flux=1e6, freq=60, measurement_time=30)
           .overlap([0, 25]))

   recon = Reconstruct(data).filter(kind='wiener', noise_power=0.01)

3. Workflow (Complete Pipeline)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chain everything including analysis:

.. code-block:: python

   from frame_overlap import Workflow

   wf = Workflow('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)
   result = (wf.convolute(200)
              .poisson(1e6, 60, 30)
              .overlap([0, 25])
              .reconstruct('wiener', noise_power=0.01)
              .analyze(xs='iron'))

Workflow Methods
----------------

Initialization
^^^^^^^^^^^^^^

.. code-block:: python

   wf = Workflow(signal_path, openbeam_path,
                 flux=5e6, duration=0.5, freq=20)

Parameters:
- ``signal_path``: Path to signal CSV file
- ``openbeam_path``: Path to openbeam CSV file
- ``flux``: Original flux in n/cm²/s
- ``duration``: Original measurement duration in hours
- ``freq``: Original pulse frequency in Hz

Processing Methods
^^^^^^^^^^^^^^^^^^

**convolute(pulse_duration, bin_width=10)**

Apply instrument response function.

.. code-block:: python

   wf.convolute(pulse_duration=200, bin_width=10)

**poisson(flux, freq, measurement_time, seed=None)**

Apply Poisson sampling with flux scaling.

.. code-block:: python

   wf.poisson(flux=1e6, freq=60, measurement_time=30, seed=42)

**overlap(kernel, total_time=None, poisson_seed=None)**

Create overlapping frames.

.. code-block:: python

   # 2-frame overlap
   wf.overlap(kernel=[0, 25])

   # 3-frame overlap
   wf.overlap(kernel=[0, 15, 30], total_time=45)

   # With optional second Poisson
   wf.overlap(kernel=[0, 25], poisson_seed=42)

**reconstruct(kind='wiener', tmin=None, tmax=None, **kwargs)**

Reconstruct signal using deconvolution.

.. code-block:: python

   # Wiener filter
   wf.reconstruct(kind='wiener', noise_power=0.01)

   # Lucy-Richardson
   wf.reconstruct(kind='lucy', iterations=20)

   # With time range filtering
   wf.reconstruct(kind='wiener', noise_power=0.01, tmin=10, tmax=40)

**analyze(xs='iron', vary_background=True, **kwargs)**

Fit with nbragg.

.. code-block:: python

   wf.analyze(xs='iron', vary_background=True, vary_response=True)

Parameter Sweeps
----------------

The ``Workflow`` class supports automatic parameter exploration.

Using groupby()
^^^^^^^^^^^^^^^

.. code-block:: python

   # Using num (number of points)
   wf.groupby('noise_power', low=0.01, high=0.1, num=20)

   # Using step (step size)
   wf.groupby('pulse_duration', low=100, high=300, step=50)

Using sweep()
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np

   # Custom values
   wf.sweep('noise_power', [0.01, 0.02, 0.05, 0.1])

   # From array
   wf.sweep('pulse_duration', np.linspace(100, 300, 20))

Running Sweeps
^^^^^^^^^^^^^^

.. code-block:: python

   results = (Workflow('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)
       .convolute(pulse_duration=200)
       .poisson(flux=1e6, freq=60, measurement_time=30)
       .overlap(kernel=[0, 25])
       .groupby('noise_power', low=0.01, high=0.1, num=20)
       .reconstruct(kind='wiener')
       .analyze(xs='iron')
       .run())  # Shows progress bar!

   # Results is a pandas DataFrame
   print(results.head())

   # Find optimal
   best = results.loc[results['redchi2'].idxmin()]
   print(f"Best noise_power: {best['noise_power']:.4f}")

Results DataFrame
^^^^^^^^^^^^^^^^^

The sweep returns a DataFrame with:

- Swept parameter values
- ``chi2``, ``redchi2``: Chi-squared statistics
- ``aic``, ``bic``: Information criteria
- ``param_*``: All fitted parameters
- ``param_*_err``: Parameter uncertainties

.. code-block:: python

   # Available columns
   print(results.columns)

   # Plot sweep results
   results.plot(x='noise_power', y='redchi2', marker='o')

   # Filter successful runs
   good_results = results.dropna(subset=['redchi2'])

   # Compare metrics
   fig, axes = plt.subplots(1, 3, figsize=(15, 4))
   good_results.plot(x='noise_power', y='redchi2', ax=axes[0])
   good_results.plot(x='noise_power', y='aic', ax=axes[1])
   good_results.plot(x='noise_power', y='bic', ax=axes[2])

Accessing Results
-----------------

After running a workflow, access intermediate and final results:

.. code-block:: python

   # Workflow state
   print(wf)  # Shows: Workflow(convolved → poissoned → overlapped → reconstructed → analyzed)

   # Access objects
   data = wf.data          # Data object
   recon = wf.recon        # Reconstruct object
   analysis = wf.analysis  # Analysis object
   result = wf.result      # lmfit.ModelResult

   # Plot current state
   wf.plot()  # Automatically plots appropriate visualization

   # Access specific stages
   wf.data.plot(kind='signal', show_stages=True)
   wf.recon.plot(kind='transmission')
   wf.analysis.plot()

Advanced Examples
-----------------

Multi-Frame with Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Optimize 3-frame overlap
   results = (Workflow('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)
       .convolute(pulse_duration=200)
       .poisson(flux=1e6, freq=60, measurement_time=30)
       .overlap(kernel=[0, 15, 30], total_time=45)
       .groupby('noise_power', low=0.05, high=0.3, num=15)
       .reconstruct(kind='wiener')
       .analyze(xs='iron')
       .run())

   best = results.loc[results['redchi2'].idxmin()]

Comparing Methods
^^^^^^^^^^^^^^^^^

.. code-block:: python

   methods = []
   for kind in ['wiener', 'lucy', 'tikhonov']:
       wf = Workflow('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)
       (wf.convolute(200).poisson(1e6, 60, 30).overlap([0, 25])
          .reconstruct(kind=kind, noise_power=0.01))
       methods.append((kind, wf.recon.statistics['chi2_per_dof']))

   for name, chi2 in methods:
       print(f"{name}: χ²/dof = {chi2:.2f}")

Best Practices
--------------

1. **Always specify flux, duration, freq** when creating Workflow
2. **Use progress_bar=False** in automated scripts
3. **Filter failed runs** with ``dropna()`` after sweeps
4. **Save results** to CSV for later analysis
5. **Use tmin/tmax** to focus chi² on relevant time ranges
6. **Set seed** for reproducible Poisson sampling
7. **Compare metrics**: Use χ²/dof, AIC, and BIC together

Error Handling
--------------

Sweeps continue even if individual runs fail:

.. code-block:: python

   results = wf.groupby('noise_power', low=0.001, high=0.1, num=20).run()

   # Check for errors
   errors = results[results['chi2'].isna()]
   print(f"Failed runs: {len(errors)}/{len(results)}")
   print(errors[['noise_power', 'error']])

   # Use only successful runs
   good_results = results.dropna(subset=['chi2'])

Performance Tips
----------------

1. Use ``num`` wisely - more points = longer runtime
2. For quick tests, use ``num=5`` or ``num=10``
3. Disable progress bar in scripts: ``run(progress_bar=False)``
4. Consider using ``step`` for fine control over parameter ranges
5. Use ``haiku`` model in sweeps for faster fitting (if available)

Next Steps
----------

- See :doc:`notebooks/example_2_parameter_optimization` for detailed examples
- Check :doc:`api/workflow` for complete API reference
- Explore :doc:`notebooks/example_3_multi_frame_overlap` for multi-frame workflows
