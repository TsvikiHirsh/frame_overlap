"""
frame_overlap: Bragg-edge transmission with programmable, overlapping pulse
patterns, on top of nbragg.

The workflow follows nbragg, with the pulse pattern as one more ingredient::

    import nbragg
    import frame_overlap as fo

    xs = nbragg.CrossSection(iron="Fe_sg229_Iron-alpha.ncmat")             # the sample, as in nbragg
    pattern = fo.Pattern([0, 11.97e-3, 22.33e-3, 35.48e-3])                 # pulse start times [s]
    model = fo.TransmissionModel(xs, pattern=pattern, vary_basic=True, vary_strain=True)
    data = fo.Data.from_counts("sample.csv", "openbeam.csv", pattern=pattern, time=8 * 3600)
    result = model.fit(data)                                                # Rietveld fit of the counts
    result.plot()

Planning a measurement takes one call::

    plan = fo.plan(xs, thickness=2.0, target="strain", n_pulses=[1, 2, 4, 8])
    print(plan.summary(strain=1000))      # pulse times, gains, hours for 1000 microstrain

Main objects: Pattern, TransmissionModel, Data, FitResult; planning: plan,
Plan, Instrument, camera_count_variance; building blocks for advanced use:
TimeGrid, OpenBeam.
"""
from .grid import TimeGrid, time_to_wavelength, wavelength_to_time
from .pattern import Pattern
from .openbeam import OpenBeam
from .data import Data
from .model import TransmissionModel
from .fitting import FitResult
from .plan import plan, Plan, Instrument, camera_count_variance

__version__ = "1.0.0.dev0"
__all__ = ["Pattern", "TransmissionModel", "Data", "FitResult", "plan", "Plan", "Instrument",
           "camera_count_variance", "TimeGrid", "OpenBeam", "time_to_wavelength", "wavelength_to_time"]
