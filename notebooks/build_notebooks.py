"""
Build and execute the tutorial notebooks (run from the repository root):

    python notebooks/build_notebooks.py

Each notebook is written from the cells below and executed in place, so that
the repository holds them with their outputs. All examples use simulated data.
"""
import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = '''import numpy as np
import matplotlib.pyplot as plt
import nbragg
import frame_overlap as fo

# the material, as in nbragg: 2 cm of alpha-iron
xs = nbragg.CrossSection(materials={"iron": {"mat": "Fe_sg229_Iron-alpha.ncmat"}})
model = nbragg.TransmissionModel(xs, response="jorgensen_inv", tof_length=9.0, vary_response=False)
model.params["thickness"].set(value=2.0)     # cm
model.params["α0"].set(value=3000.0)         # moderator response (rise), close to a water moderator
model.params["β0"].set(value=50.0)           # moderator response (decay)
sample = fo.Sample(model, free=["thickness", "strain"])

# the instrument: 9 m flight path, 50 ms period, 10 us bins, a thermal moderator
grid = fo.TimeGrid(flight_path=9.0, frequency=20.0, bin_width=10e-6)
beam = fo.OpenBeam.maxwellian(grid, flux=5e6, area=0.03)
hours = 8'''

NB = {}

NB["01_quickstart"] = [
    md("""# Quick start: from a material to a fitted measurement

This notebook simulates one Bragg-edge transmission measurement with four
overlapping pulses per period and analyses it in two ways:

* the **Rietveld fit of the overlapped spectrum**: the nbragg model is passed
  through the known pulse pattern and fitted to the counts, and
* **Wiener deconvolution** as in frame-overlap Bragg-edge imaging (FOBI),
  followed by a fit of the recovered transmission.

The material is described exactly as in nbragg."""),
    code(SETUP),
    md("""## The measurement design

A pulse pattern says when the chopper opens within the period. Here: one
200 µs pulse per period, and four pulses at times chosen so that the intense
thermal part of each spectrum avoids the Bragg edges of the others."""),
    code('''one = fo.Pattern.single(200e-6)
four = fo.Pattern.times([0.0, 11.97e-3, 22.33e-3, 35.48e-3], width=200e-6)
exp1 = fo.Experiment(sample, beam, one, time=hours * 3600)
exp4 = fo.Experiment(sample, beam, four, time=hours * 3600)
print(exp4)
print(exp1.precision())    # Cramer-Rao bound: the best any unbiased analysis can do
print(exp4.precision())
print("time-reduction factor of the four pulses:", exp4.gain())'''),
    md("""The time-reduction factor *R* is the factor by which the measurement with
one pulse per period would have to be longer to reach the same precision."""),
    code('''data = exp4.simulate(seed=1)
fig, (a0, a1) = plt.subplots(2, 1, figsize=(7, 4), sharex=True, gridspec_kw=dict(height_ratios=[1, 4]))
four.plot(a0, period=grid.period)
data.plot(a1)
a0.set_xlabel("")
plt.show()'''),
    md("## Rietveld fit of the overlapped spectrum"),
    md("""The fit starts away from the truth (here 1.9 cm and 300 microstrain);
the greys show the contribution of each pulse to the fitted counts."""),
    code('''rietveld = fo.RietveldFit(window=(1.0, 5.0)).fit(data, sample, theta0=[1.9, 3e-4])
print(rietveld.fit_report())
fig, ax = plt.subplots(figsize=(7, 3))
rietveld.plot(ax)
plt.show()'''),
    md("""The fit also recovers the transmission spectrum: the counts of each bin
are divided between the pulses in proportion to their fitted contributions,
as Rietveld programs divide the counts of overlapping reflections."""),
    code('''lam, T, dT = rietveld.transmission()
sel = (lam > 1) & (lam < 5)
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(lam[sel], T[sel], color="0.5", lw=0.4, label="recovered from the overlapped counts")
ax.plot(lam[sel], sample.transmission(grid)[sel], color="#0072B2", lw=1, label="nbragg model (truth)")
ax.set_xlabel("Wavelength (Å)"); ax.set_ylabel("Transmission"); ax.legend()
plt.show()'''),
    md("## Wiener deconvolution (FOBI)"),
    code('''wiener = fo.WienerFit(noise_power=1e-3).fit(data, sample, theta0=[1.9, 3e-4])
print(wiener.fit_report())
fig, ax = plt.subplots(figsize=(7, 3))
wiener.plot(ax)
ax.set_ylim(-0.05, 0.4)
plt.show()'''),
    md("""## Comparison

Both analyses use the same counts. The Rietveld fit reaches the Cramér–Rao
bound; the deconvolution followed by a fit with independent bins does not."""),
    code('''import pandas as pd
bound = exp4.precision().set_index("parameter").sigma
pd.DataFrame({"Cramer-Rao bound": bound,
              "Rietveld fit": pd.Series(rietveld.errors),
              "Wiener deconvolution": pd.Series(wiener.errors)})'''),
    md("""## Check with a Monte Carlo

The errors above come from the curvature of the likelihood. A Monte Carlo of
simulated measurements, each fitted from displaced starting values, checks
that the scatter of the results matches them and that the fit is unbiased.
Use more draws (and ``workers``) for a real check."""),
    code('''est, mc = fo.monte_carlo(exp4, fo.RietveldFit(), n=24, seed=0, workers=4)
mc.round(6)'''),
]

NB["02_choosing_a_pulse_pattern"] = [
    md("""# Choosing a pulse pattern

More pulses per period deliver more neutrons, but the gain in measurement
time depends on how the spectra of the pulses are kept apart or analysed
together. This notebook compares, for N pulses of 200 µs per 50 ms period:

* one longer pulse of N × 200 µs,
* N equally spaced pulses, with and without an ideal bandwidth chopper,
* N pulses at random times and a maximum-length (pseudorandom) sequence,
* N pulses at times optimised for the edge position (strain).

All numbers are Cramér–Rao bounds of the Rietveld fit, which reaches them."""),
    code(SETUP),
    md("""The reference is one 200 µs pulse per period without a bandwidth
chopper. ``gain`` gives the time-reduction factor of a design against it, and
``survey`` does the same for many designs at once."""),
    code('''one = fo.Experiment(sample, beam, fo.Pattern.single(200e-6), time=hours * 3600)
print(one.precision())
fo.survey(one, {"one pulse": fo.Pattern.single(200e-6),
                "two pulses, 0 and 25 ms": fo.Pattern.times([0, 25e-3]),
                "POLDI-like sequence": fo.Pattern.poldi(grid=grid)}).round(3)'''),
    md("""A bandwidth chopper passes only part of each spectrum, so that the
spectra of N equally spaced pulses do not overlap. ``with_band_for(n)`` gives
the open beam of such a chopper: the band 0.97–5.25 Å while it fits between
the pulses, otherwise the widest band that still reaches 4.55 Å, just beyond
the (110) edge of iron."""),
    code('''rows = []
for n in [1, 2, 4, 8]:
    designs = {"one longer pulse": fo.Pattern([0.0], n * 200e-6),
               "equal, no chopper": fo.Pattern.equal(n, grid=grid),
               "random times": fo.Pattern.random(n, grid=grid, seed=n)}
    if n in (2, 4, 8):
        designs["maximum-length sequence"] = fo.Pattern.maximum_length({2: 2, 4: 3, 8: 4}[n], grid=grid)
    for name, p in designs.items():
        rows.append(dict(N=n, design=name, **one.with_pattern(p).gain(reference=one)))
    chopped = one.with_beam(beam.with_band_for(n)).with_pattern(fo.Pattern.equal(n, grid=grid))
    rows.append(dict(N=n, design="equal, bandwidth chopper", **chopped.gain(reference=one)))
    if n > 1:
        best = fo.optimise_pattern(one, n, target="strain", n_restarts=2, n_steps=400, seed=n)
        rows.append(dict(N=n, design="optimised for strain", **one.with_pattern(best).gain(reference=one)))
import pandas as pd
table = pd.DataFrame(rows)
table.pivot(index="N", columns="design", values="strain").round(2)'''),
    code('''fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
for ax, par in zip(axes, ["strain", "thickness"]):
    for name, g in table.groupby("design"):
        ax.plot(g.N, g[par], "o-", label=name)
    ax.plot([1, 8], [1, 8], "k:", lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xlabel("pulses per period"); ax.set_title(par)
axes[0].set_ylabel("time-reduction factor R")
axes[1].legend(fontsize=8)
plt.show()'''),
    md("""With a bandwidth chopper the pulses do not overlap and R grows with N as
long as the band needed for the edges fits between pulses; beyond that the
narrower band keeps gaining for the strain only if the edge shape is known.
Without a chopper the spectra overlap, and optimised pulse times keep the
thermal peak of each pulse away from the edges of the others. Pseudorandom
sequences are designed for an inversion with the least noise, which the
Rietveld fit does not need. A longer pulse helps the thickness but not the
edge position.

The optimiser is a stochastic search: more restarts and steps can find
better patterns. ``optimise_pattern(..., target="thickness")`` optimises for
the thickness instead."""),
]

NB["03_fitting_your_measurement"] = [
    md("""# Fitting your own measurement

To analyse measured data you need three things besides the material model:

1. the **counts** of the sample measurement, histogrammed on the time grid of
   one period and summed over all periods,
2. the **open beam of one pulse**, from a separate calibration with one short
   pulse per period, with much better statistics than the sample measurement
   (for example summed over the whole detector), and
3. the **pulse pattern**, as programmed in the chopper.

Here the 'measured' data are simulated and written to CSV files, then read
back as you would read your own."""),
    code(SETUP),
    code('''import pandas as pd
pattern = fo.Pattern.times([0.0, 11.97e-3, 22.33e-3, 35.48e-3], width=200e-6)
truth = fo.Experiment(sample.with_values(thickness=1.95), beam, pattern, time=hours * 3600)
sim = truth.simulate(seed=7)
pd.DataFrame(dict(time_s=grid.t, counts=sim.counts)).to_csv("my_sample_counts.csv", index=False)
# calibration: one 10 us pulse per period for 8 h, summed over a detector area 100 times larger
calib = fo.Experiment(sample.with_values(thickness=0.0), beam.scaled(100), fo.Pattern.single(10e-6), time=8 * 3600)
pd.DataFrame(dict(time_s=grid.t, counts=calib.simulate(seed=1).counts)).to_csv(
    "my_open_beam_one_pulse.csv", index=False)
print("written two CSV files")'''),
    md("## Read the data and fit"),
    code('''counts = pd.read_csv("my_sample_counts.csv").counts.values
ob = pd.read_csv("my_open_beam_one_pulse.csv").counts.values
open_beam = fo.OpenBeam.from_counts(grid, ob, pulse_width=10e-6, time=8 * 3600).scaled(1 / 100)
data = fo.Measurement(grid, counts, pattern, open_beam, time=hours * 3600)
result = fo.RietveldFit(window=(1.0, 5.0)).fit(data, sample)     # starts from the sample's 2.0 cm
print(result.fit_report())
result.to_dataframe()'''),
    md("""## Check the fit

The deviance per bin should be close to one, and the residuals should look
like noise. The true thickness of the simulation was 1.95 cm."""),
    code('''fig, ax = plt.subplots(figsize=(7, 2.5))
ax.plot(grid.t * 1e3, result.residuals(), lw=0.3, color="k")
ax.set_xlabel("Time in the period (ms)"); ax.set_ylabel("deviance residual")
plt.show()
print("deviance per bin:", round(result.deviance / result.n_used, 3))'''),
    md("""## A background

A constant count background (room background, detector noise) can be given
to the Measurement if it is known (``background=``, counts per bin and
second), or fitted. Here the simulated data have none, and the fitted value
is consistent with zero."""),
    code('''with_bg = fo.RietveldFit(window=(1.0, 5.0), fit_background=True).fit(data, sample)
print(with_bg.fit_report())'''),
    md("""The fit treats the open beam as exact. Its own noise adds to the
uncertainty of the result, so check its effect, for example by refitting with
the open beam of half the calibration data, and keep its statistics well
above those of the sample measurement."""),
]

NB["04_what_must_be_known"] = [
    md("""# What must be known

The Rietveld fit takes the pulse pattern, the open beam and the edge shape
(the nbragg response) as known. This notebook estimates how errors in these
inputs bias the result, and what freeing them costs in precision."""),
    code(SETUP),
    code('''pattern = fo.Pattern.times([0.0, 11.97e-3, 22.33e-3, 35.48e-3], width=200e-6)
exp = fo.Experiment(sample, beam, pattern, time=hours * 3600)
cases = {
    "one pulse 1 us late": fo.Pattern(pattern.starts + np.r_[0, 1e-6, 0, 0], 200e-6),
    "all pulses 1 us late (time zero)": fo.Pattern(pattern.starts + 1e-6, 200e-6),
    "one pulse 1 % stronger, same total": fo.Pattern(pattern.starts, 200e-6, intensity=[1.01, 1 - 0.01 / 3, 1 - 0.01 / 3, 1 - 0.01 / 3]),
    "all pulses 1 % stronger (normalisation)": fo.Pattern(pattern.starts, 200e-6, intensity=1.01),
    "pulses 1 % wider, same centre and neutrons": fo.Pattern((pattern.starts - 1e-6) % grid.period, 202e-6, intensity=200 / 202),
    "pulses 1 % longer, same start": fo.Pattern(pattern.starts, 202e-6),
}
import pandas as pd
pd.DataFrame({k: fo.pattern_bias(exp, p) for k, p in cases.items()}).T.round(2)'''),
    md("""The table gives the bias in units of the standard deviation of each
parameter. Pulse times matter for the strain (a digital chopper sets them to
far better than a microsecond), and a common delay is a time-zero error of any
time-of-flight measurement, which cancels in strain measured against a
reference sample. A pulse that ends later moves its centre and acts like a
delay of half the change; a pulse that is wider about the same centre hardly
matters. The normalisation matters for the thickness.

## The cost of freeing what is assumed known"""),
    code('''model_r = nbragg.TransmissionModel(xs, response="jorgensen_inv", tof_length=9.0, vary_response=True)
model_r.params["thickness"].set(value=2.0)
model_r.params["α0"].set(value=3000.0)
model_r.params["β0"].set(value=50.0)
rows = {}
for name, free in {"thickness, strain": ["thickness", "strain"],
                   "+ normalisation": ["thickness", "strain", "norm"],
                   "+ moderator response": ["thickness", "strain", "α0", "β0"]}.items():
    s = fo.Sample(model_r, free=free)
    rows[name] = fo.Experiment(s, beam, pattern, time=hours * 3600).precision().set_index("parameter").sigma
pd.DataFrame(rows).T'''),
    md("""Freeing the normalisation costs most of the thickness precision, and
freeing the moderator response costs most of the strain precision: both should
be calibrated in separate measurements.

The background, a common time offset and the relative pulse intensities can
also be treated as unknown, in the bound and in ``RietveldFit``
(``fit_background``, ``fit_t0``, ``fit_intensities``):"""),
    code('''pd.DataFrame({"known": exp.precision().set_index("parameter").sigma,
              "+ background": exp.precision(background=True).set_index("parameter").sigma,
              "+ time offset": exp.precision(t0=True).set_index("parameter").sigma,
              "+ pulse intensities": exp.precision(intensities=True).set_index("parameter").sigma}).T'''),
]


NB["05_planning_a_measurement"] = [
    md("""# Planning a measurement

Give the sample as an nbragg cross-section. `fo.plan` optimises the pulse times
for the quantity you want to measure and tells you the precision and the
measurement time for any bin size. The instrument here is the SARAF-II
time-of-flight station of the article (9 m, 20 Hz, 5e6 n/cm²/s, the event-mode
camera); describe another one with `fo.Instrument(...)`."""),
    code('''import nbragg
import frame_overlap as fo

xs = nbragg.CrossSection(materials={"iron": {"mat": "Fe_sg229_Iron-alpha.ncmat"}})
plan = fo.plan(xs, thickness=2.0, target="strain", n_pulses=[1, 2, 4, 8, 12],
               instrument=fo.Instrument.saraf_ii())
plan.candidates          # every number of pulses tried, with its gain R'''),
    code('''print(plan.summary(strain=1000))'''),
    code('''plan.time_for(bin_size=[0.5, 1, 2, 5], strain=1000) / 3600     # hours for 1000 microstrain'''),
    code('''fig = plan.plot(strain=1000)
fig.savefig("../docs/plan_iron.png", dpi=150, bbox_inches="tight")'''),
    md("""The numbers are Cramér–Rao bounds of the Rietveld fit. They hold when a
spectrum has more than about 5e4 independent counts; check smaller bins with
`fo.monte_carlo(plan.experiment(time, bin_size), fo.RietveldFit())`. For another
instrument give its flight path, frequency, flux, detector efficiency and
counting statistics to `fo.Instrument`, or a measured open beam with
`open_beam=fo.OpenBeam.from_counts(...)`."""),
]

def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = os.path.join(HERE, f"{name}.ipynb")
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": HERE}})
    nbf.write(nb, path)
    print("written", path)


if __name__ == "__main__":
    import sys
    for name, cells in NB.items():
        if len(sys.argv) > 1 and name not in sys.argv[1:]:
            continue
        build(name, cells)
