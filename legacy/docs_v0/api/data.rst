Data API
========

.. currentmodule:: frame_overlap

The ``Data`` class handles loading and processing of neutron Time-of-Flight data.

Data Class
----------

.. autoclass:: Data
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Methods
-------

Loading Data
^^^^^^^^^^^^

.. automethod:: Data.load_signal_data
.. automethod:: Data.load_openbeam_data

Processing
^^^^^^^^^^

.. automethod:: Data.convolute_response
.. automethod:: Data.poisson_sample
.. automethod:: Data.overlap

Visualization
^^^^^^^^^^^^^

.. automethod:: Data.plot

Attributes
----------

.. attribute:: Data.data

   Original signal DataFrame (time, counts, err)

.. attribute:: Data.convolved_data

   Signal after convolution

.. attribute:: Data.poissoned_data

   Signal after Poisson sampling

.. attribute:: Data.overlapped_data

   Signal after frame overlap

.. attribute:: Data.op_data

   Original openbeam DataFrame

.. attribute:: Data.op_convolved_data

   Openbeam after convolution

.. attribute:: Data.op_poissoned_data

   Openbeam after Poisson sampling

.. attribute:: Data.op_overlapped_data

   Openbeam after frame overlap

.. attribute:: Data.pulse_duration

   Pulse duration in microseconds (set by convolute_response)

.. attribute:: Data.kernel

   Frame overlap kernel (start times in ms)

.. attribute:: Data.flux

   Neutron flux in n/cm²/s

.. attribute:: Data.duration

   Measurement duration in hours

.. attribute:: Data.freq

   Pulse frequency in Hz
