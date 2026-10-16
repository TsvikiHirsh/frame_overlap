# frame_overlap

**Bragg-edge transmission with programmable pulse patterns and frame overlap, on top of [nbragg](https://github.com/TsvikiHirsh/nbragg).**

`frame_overlap` answers two questions for time-of-flight Bragg-edge imaging at a source whose
pulses can be shaped, such as a continuous-wave accelerator with a digital chopper:

1. **Which pulse pattern should I use?** One pulse per period, a longer pulse, equally spaced
   pulses with a bandwidth chopper, or overlapping pulses at chosen times. `frame_overlap`
   computes the precision every design can reach (the Cramér–Rao bound of the counts) and
   searches for good pulse times.
2. **How do I analyse overlapped spectra?** With a *Rietveld fit of the overlapped spectrum*: the
   nbragg transmission model is passed through the known pulse pattern and fitted to the counts
   with the Poisson likelihood. It works for any pattern and reaches the information limit of the
   data. Wiener deconvolution as in frame-overlap Bragg-edge imaging (FOBI) is included for
   comparison.

You describe the material exactly as in nbragg; `frame_overlap` adds the instrument, the pulse
pattern and the measurement time.

## Installation

```bash
pip install git+https://github.com/TsvikiHirsh/frame_overlap.git
```

Requires Python ≥ 3.9 and nbragg ≥ 1.0 (with NCrystal).

## Quick start

```python
import nbragg
import frame_overlap as fo

# 1. the material, as in nbragg
xs = nbragg.CrossSection(materials={"iron": {"mat": "Fe_sg229_Iron-alpha.ncmat"}})
model = nbragg.TransmissionModel(xs, response="jorgensen_inv", tof_length=9.0, vary_response=False)
model.params["thickness"].set(value=2.0)                 # cm
model.params["α0"].set(value=3000.0)                     # moderator response, close to
model.params["β0"].set(value=50.0)                       # a water moderator
sample = fo.Sample(model, free=["thickness", "strain"])  # what to determine

# 2. the instrument and the measurement design
grid = fo.TimeGrid(flight_path=9.0, frequency=20.0, bin_width=10e-6)
beam = fo.OpenBeam.maxwellian(grid, flux=5e6, area=0.03)          # n/cm²/s, cm²
pattern = fo.Pattern.times([0, 12.0e-3, 22.3e-3, 35.5e-3], width=200e-6)
exp = fo.Experiment(sample, beam, pattern, time=8 * 3600)

# 3. what this design can reach
print(exp.precision())     # Cramér–Rao bound of each parameter (strain in microstrain)
print(exp.gain())          # time-reduction factor against one 200 µs pulse per period

# 4. simulate a measurement and analyse it
data = exp.simulate(seed=1)
result = fo.RietveldFit(window=(1.0, 5.0)).fit(data, sample)
print(result.summary())
result.plot()
```

## Concepts

| Object | What it is |
|---|---|
| `TimeGrid` | time bins of one period (flight path, repetition rate, bin width); slow neutrons of one period arrive in the next |
| `OpenBeam` | open-beam rate of one pulse: from a measured spectrum (`from_counts`, `from_reference`) or a thermal moderator model (`maxwellian`); `with_band` adds an ideal bandwidth chopper |
| `Pattern` | the pulse pattern ("kernel"): `single`, `equal`, `times`, `random`, `maximum_length`, `quadratic_residue`, `poldi`, or `optimise_pattern` |
| `Sample` | an nbragg `TransmissionModel` with free parameters; `"strain"` moves all Bragg edges by Δλ/λ smoothly (nbragg passes lattice parameters with four decimals) |
| `Experiment` | sample + open beam + pattern + time; `expected_counts`, `simulate`, `precision`, `gain` |
| `Measurement` | counts of a measurement with its pattern and open beam (simulated or your own data) |
| `RietveldFit` | forward fit of the counts through the pattern (Poisson likelihood); nbragg inside an analysis window, a free transmission outside it |
| `WienerFit` | FOBI-style deconvolution of sample and open beam with the same filter, then a fit of their ratio |
| `precision`, `gain`, `survey`, `optimise_pattern`, `monte_carlo`, `pattern_bias` | design tools |

The expected counts of a design are μ(t) = n_periods · [c ⊛ (o T)](t): the open time c of the
pattern in each bin, convolved over the period with the open beam o of one pulse times the
sample transmission T.

## Tutorials

Jupyter notebooks in [`notebooks/`](notebooks/), all on simulated data:

1. [`01_quickstart.ipynb`](notebooks/01_quickstart.ipynb): one measurement from model to fit, Rietveld fit against Wiener deconvolution.
2. [`02_choosing_a_pulse_pattern.ipynb`](notebooks/02_choosing_a_pulse_pattern.ipynb): precision and gain of pulse patterns, bandwidth chopper, optimisation.
3. [`03_fitting_your_measurement.ipynb`](notebooks/03_fitting_your_measurement.ipynb): load measured counts, open beam and pattern, fit, check the result.
4. [`04_what_must_be_known.ipynb`](notebooks/04_what_must_be_known.ipynb): how well the pulse pattern, the normalisation and the edge shape must be known.

## Validation

`pytest -q` runs tests with known answers: noiseless round trips of the Rietveld fit for one,
two and four pulses, the doubling of information by two separate pulses, the Wiener ratio for
a pattern with spectral zeros, and the bias from an error in the assumed pattern. The method and
its validation against Monte Carlo are described in the article below.

## Citation

T. Y. Hirsh et al., *Projected precision of Bragg-edge imaging at the SARAF-II time-of-flight
facility with programmable pulse patterns* (in preparation). Please also cite nbragg and NCrystal.

## History

Version 1.0 replaces the earlier deconvolution-centred package; the old code, notebooks and
apps are kept in [`legacy/`](legacy/) and are no longer installed.

## License

MIT
