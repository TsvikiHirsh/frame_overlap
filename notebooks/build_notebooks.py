"""
Build and execute the tutorial notebooks (run from the repository root):

    python notebooks/build_notebooks.py [name ...]

Each notebook is written from the cells below and executed in place, so that
the repository holds them with their outputs. All examples use simulated data.
"""
import os
from datetime import date
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
AUTHOR = "Tsviki Y. Hirsh"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

NB = {}

NB["01_quickstart"] = [
    md("""# Quick start

frame_overlap works like nbragg, with the pulse pattern as one more ingredient.
Here: 2 cm of iron, four pulses per 50 ms period at 9 m (times optimised for
the Bragg-edge position), a projected 8 h measurement, and its fit."""),
    code('''import nbragg
import frame_overlap as fo

xs = nbragg.CrossSection(iron="Fe_sg229_Iron-alpha.ncmat")         # the sample, as in nbragg
pattern = fo.Pattern([0, 11.97e-3, 22.33e-3, 35.48e-3])             # pulse start times [s]
model = fo.TransmissionModel(xs, pattern=pattern, vary_basic=True, vary_strain=True)
model.params["thickness"].set(value=2.0)                            # cm
model'''),
    md("""The expected counts: each pulse produces its own spectrum (fills), and the
detector records their sum. The beam here is the SARAF-II projection of the
article: 5e6 n/cm²/s on 72 mm² with 4.5 % detection efficiency."""),
    code('''beam = dict(flux=5e6, area=0.72, efficiency=0.045)
model.plot(**beam);'''),
    md("""A projected measurement of 8 h (Poisson counts with the 2.3-fold variance that
the event-mode camera records), and its Rietveld fit: the model is convolved
with the pulse pattern and fitted to the counts."""),
    code('''data = model.simulate(time=8 * 3600, count_variance=2.3, seed=1, **beam)
result = model.fit(data)
print(result.fit_report())'''),
    code('''ax = result.plot()
ax.figure.savefig("../docs/fit_iron.png", dpi=150, bbox_inches="tight")'''),
    md("""The plot shows the transmission recovered from the overlapped counts (the
counts of each bin divided between the pulses) with the fitted model, and the
residuals in units of their standard deviation.

For comparison, Wiener deconvolution as in frame-overlap Bragg-edge imaging
(FOBI): it spreads the noise of all pulses over the recovered spectrum and
loses most of the edge information."""),
    code('''wiener = model.fit(data, method="wiener")
print(wiener.fit_report())
wiener.plot();'''),
    md("""What this design can reach, from the Cramér–Rao bound of the Rietveld fit, and
how much shorter it is than one pulse per period:"""),
    code('''print(model.precision(time=8 * 3600, count_variance=2.3, **beam)[["parameter", "sigma_display", "unit_display"]])
model.gain(**beam)'''),
]

NB["02_planning"] = [
    md("""# Planning a measurement

Give the sample as an nbragg cross-section and say what you want to measure.
`fo.plan` optimises the pulse times and tells you the precision and the
measurement time for any bin size. The instrument here is the SARAF-II
time-of-flight station of the article (9 m, 20 Hz, 5e6 n/cm²/s, the event-mode
camera); describe another one with `fo.Instrument(...)`."""),
    code('''import nbragg
import frame_overlap as fo

xs = nbragg.CrossSection(iron="Fe_sg229_Iron-alpha.ncmat")
plan = fo.plan(xs, thickness=2.0, target="strain", n_pulses=[1, 2, 4, 8, 12],
               instrument=fo.Instrument.saraf_ii())
plan.candidates          # every number of pulses tried, with its gain R'''),
    code('''print(plan.summary(strain=1000))'''),
    code('''plan.time_for(bin_size=[0.5, 1, 2, 5], strain=1000) / 3600     # hours for 1000 microstrain'''),
    code('''fig = plan.plot(strain=1000)
fig.savefig("../docs/plan_iron.png", dpi=150, bbox_inches="tight")'''),
    md("""The plan's model carries the chosen pattern. Other patterns with the same
model: equally spaced pulses put the spectrum of one pulse on the edges of the
next, and random or pseudorandom times gain less than optimised ones."""),
    code('''model = plan.model
patterns = {"4 equal": fo.Pattern.equal(4), "8 equal": fo.Pattern.equal(8),
            "8 random": fo.Pattern.random(8, seed=1), "8 POLDI": fo.Pattern.poldi(),
            "optimised": plan.pattern}
{name: round(model.gain(pattern=p)["strain"], 2) for name, p in patterns.items()}'''),
    md("""The numbers are Cramér–Rao bounds of the Rietveld fit. They hold when a
spectrum has more than about 5e4 independent counts; check smaller bins with
`plan.monte_carlo(n=100, time=..., bin_size=...)`."""),
]

NB["03_your_data"] = [
    md("""# Fitting your own measurement

Measured counts are read as in `nbragg.Data.from_counts`: the sample counts of
one period (summed over all periods) and the open beam measured with one short
pulse per period. Here we first write such files from a simulation."""),
    code('''import numpy as np
import pandas as pd
import nbragg
import frame_overlap as fo

xs = nbragg.CrossSection(iron="Fe_sg229_Iron-alpha.ncmat")
pattern = fo.Pattern([0, 11.97e-3, 22.33e-3, 35.48e-3])
truth = fo.TransmissionModel(xs, pattern=pattern, vary_basic=True, vary_strain=True, strain=5e-4)
truth.params["thickness"].set(value=1.8)
sim = truth.simulate(time=8 * 3600, flux=5e6, area=0.72, efficiency=0.045, seed=7)
sim.to_csv("my_sample_counts.csv")                                       # tof, counts, err
open_counts = sim.openbeam.rate * 200e-6 * (16 * 3600) * 20              # one 200 us pulse per period, 16 h
pd.DataFrame(dict(tof=np.arange(len(open_counts)), counts=np.random.default_rng(8).poisson(open_counts))
             ).to_csv("my_open_beam_one_pulse.csv", index=False)'''),
    md("""Read the files, with the pulse pattern, the measurement times, the flight path
and the time bin:"""),
    code('''data = fo.Data.from_counts("my_sample_counts.csv", "my_open_beam_one_pulse.csv", pattern=pattern,
                           time=8 * 3600, openbeam_time=16 * 3600, L=9.0, tstep=10e-6)
data.plot();'''),
    code('''model = fo.TransmissionModel(xs, pattern=pattern, vary_basic=True, vary_strain=True)
result = model.fit(data)
print(result.fit_report())
result.plot();'''),
    code('''result.params["strain"]            # lmfit Parameters, as in nbragg'''),
    code('''result.transmission().query("1 < wavelength < 5").head()     # the recovered spectrum'''),
    md("""## What must be known

The fit takes the pulse times, their intensities, the open beam and the
moderator response as known. A delay of one pulse by 1 µs, to first order, in
units of the standard deviation:"""),
    code('''late = fo.Pattern(np.array(pattern.starts) + np.r_[0, 1e-6, 0, 0])
model.pattern_bias(late, flux=5e6, area=0.72, efficiency=0.045)'''),
    md("""A common time offset, a constant background and the relative pulse
intensities can be fitted as well, at a cost in precision:"""),
    code('''model.fit(data, fit_t0=True, fit_intensities=True).to_dataframe()'''),
]


def byline(cell):
    """The author and the date under the title of the first cell."""
    title, _, rest = cell.source.partition("\n")
    when = f"{date.today().day} {date.today():%B %Y}"
    return md(f"{title}\n\n*{AUTHOR}, {when}*\n{rest}")


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [byline(cells[0])] + cells[1:]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = os.path.join(HERE, f"{name}.ipynb")
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": HERE}})
    nbf.write(nb, path)
    print("written", path)


if __name__ == "__main__":
    import sys
    os.makedirs(os.path.join(HERE, "..", "docs"), exist_ok=True)
    for name, cells in NB.items():
        if len(sys.argv) > 1 and name not in sys.argv[1:]:
            continue
        build(name, cells)
