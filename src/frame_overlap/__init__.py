"""
frame_overlap: Bragg-edge transmission with programmable pulse patterns and
frame overlap, on top of nbragg.

Planning a measurement takes one call::

    import nbragg
    import frame_overlap as fo

    xs = nbragg.CrossSection(materials={"iron": {"mat": "Fe_sg229_Iron-alpha.ncmat"}})
    plan = fo.plan(xs, thickness=2.0, target="strain", n_pulses=[1, 2, 4, 8])
    print(plan.summary(strain=1000))      # pulse times, gains, hours for 1000 microstrain
    plan.time_for(bin_size=1.0, strain=1000)

A full session, step by step::

    # the material, as in nbragg
    xs = nbragg.CrossSection(materials={"iron": {"mat": "Fe_sg229_Iron-alpha.ncmat"}})
    model = nbragg.TransmissionModel(xs, response="jorgensen_inv", tof_length=9.0, vary_response=False)
    model.params["thickness"].set(value=2.0)
    model.params["α0"].set(value=3000.0)   # moderator response
    model.params["β0"].set(value=50.0)
    sample = fo.Sample(model, free=["thickness", "strain"])

    # the instrument and the measurement design
    grid = fo.TimeGrid(flight_path=9.0, frequency=20.0, bin_width=10e-6)
    beam = fo.OpenBeam.maxwellian(grid, flux=5e6, area=0.03)
    exp = fo.Experiment(sample, beam, fo.Pattern.times([0, 12e-3, 22.3e-3, 35.5e-3]), time=8 * 3600)

    # what the design can reach, and how a measurement is analysed
    exp.precision()                       # Cramer-Rao bound of each parameter
    exp.gain()                            # time-reduction factor against one pulse per period
    data = exp.simulate(seed=1)
    result = fo.RietveldFit(window=(1, 5)).fit(data, sample)
    print(result.summary())

Main objects: TimeGrid, OpenBeam, Pattern, Sample, Experiment, Measurement,
RietveldFit, WienerFit; planning: plan, Plan, Instrument,
camera_count_variance; design tools: precision, gain, survey,
optimise_pattern, monte_carlo, pattern_bias.
"""
from .grid import TimeGrid, time_to_wavelength, wavelength_to_time
from .beam import OpenBeam
from .pattern import Pattern
from .sample import Sample
from .experiment import Experiment, Measurement
from .fitting import RietveldFit, WienerFit, FitResult, Window
from .design import fisher, precision, gain, survey, optimise_pattern, monte_carlo, pattern_bias
from .plan import plan, Plan, Instrument, camera_count_variance

__version__ = "1.0.0.dev0"
__all__ = ["TimeGrid", "time_to_wavelength", "wavelength_to_time", "OpenBeam", "Pattern", "Sample",
           "Experiment", "Measurement", "RietveldFit", "WienerFit", "FitResult", "Window", "fisher",
           "precision", "gain", "survey", "optimise_pattern", "monte_carlo", "pattern_bias", "plan", "Plan",
           "Instrument", "camera_count_variance"]
