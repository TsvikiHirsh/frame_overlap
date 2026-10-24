"""
Planning a measurement: from an nbragg cross-section to an optimised pulse
pattern, the precision it reaches and the measurement time it needs.

A typical session::

    import nbragg
    import frame_overlap as fo

    xs = nbragg.CrossSection(materials={"iron": {"mat": "Fe_sg229_Iron-alpha.ncmat"}})
    plan = fo.plan(xs, thickness=2.0, target="strain", n_pulses=[1, 2, 4, 8])
    print(plan.summary(strain=1000))            # pulse times, gains, times for 1000 microstrain
    plan.precision(time=8 * 3600, bin_size=1.0)  # one standard deviation in 8 h, 1 mm bins
    plan.time_for(bin_size=1.0, strain=1000)     # seconds for 1000 microstrain in 1 mm bins

The precision is the Cramer-Rao bound of the Rietveld fit of the overlapped
counts. It scales as sqrt(F / (time * area)), with F the variance of the
recorded counts relative to Poisson counts, so one bound per design (for 8 h
and a bin of 1 mm) gives every time and bin size, as in the article. (The weak
prior on the transmission outside the analysis window does not scale; exact
bounds for much fewer counts are a few per cent smaller for the thickness.) In the article the Monte Carlo fits reached the bound above
about 6e3 (one pulse per period) to 5e4 (eight pulses) independent counts per
spectrum; below that the plan warns.
"""
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .grid import TimeGrid
from .beam import OpenBeam
from .pattern import Pattern
from .sample import Sample
from .experiment import Experiment
from .design import precision as _precision, optimise_pattern

# Counting statistics of the event-mode camera of the SARAF-II projections
# (scintillator, image intensifier, Timepix3 camera, event clustering): variance
# of the recorded counts relative to Poisson counts against the side of a square
# bin [mm], measured from the reference images (the article, Section 3.1).
EVENT_CAMERA_F = ((0.109, 1.10), (0.219, 1.34), (0.438, 1.71), (0.875, 2.05), (1.75, 2.28), (3.5, 2.30))

# Independent counts per spectrum above which the Monte Carlo fits of the
# article stayed within 10 % of the bound.
MIN_COUNTS = {1: 6e3, "overlap": 5e4}

_REF_TIME, _REF_BIN = 8 * 3600.0, 1.0       # reference exposure of a plan: 8 h, a bin of 1 mm


def camera_count_variance(bin_size, table=EVENT_CAMERA_F):
    """
    Variance of the recorded counts relative to Poisson counts for square bins
    of side ``bin_size`` [mm], interpolated in the logarithm of the bin size
    and constant beyond the table (default: the camera of the article).
    """
    b = np.log(np.asarray(bin_size, float))
    x, y = np.log([p[0] for p in table]), [p[1] for p in table]
    return np.interp(b, x, y)


@dataclass(frozen=True)
class Instrument:
    """
    A time-of-flight imaging instrument at a source whose pulses a digital
    chopper sets.

    Parameters
    ----------
    flight_path : float
        Source-to-detector distance [m].
    frequency : float
        Repetition rate of the pulse pattern [Hz].
    bin_width : float
        Time bin [s]; the period must be a whole number of bins.
    flux : float
        Continuous (unchopped) neutron flux at the sample [n/cm^2/s].
    efficiency : float
        Fraction of the neutrons that the detector records.
    temperature : float
        Temperature [K] of the Maxwellian spectrum of the moderator, used
        unless ``open_beam`` is given.
    response : (float, float)
        Moderator response of nbragg (``jorgensen_inv``): alpha0 and beta0.
        Used when the sample is given as a CrossSection.
    pulse_width : float
        Width of each pulse [s].
    window : (float, float)
        Analysis window of the Rietveld fit [Angstrom].
    count_variance : float or callable
        Variance of the recorded counts relative to Poisson counts (1 for a
        counting detector), or a function of the bin size [mm], such as
        ``camera_count_variance``.
    open_beam : OpenBeam, optional
        A measured open beam of one pulse on the grid of this instrument,
        for a detector area of ``open_beam_area`` [cm^2]. It replaces the
        Maxwellian spectrum, flux and efficiency.
    open_beam_area : float
        Detector area [cm^2] of ``open_beam``.
    """
    flight_path: float = 9.0
    frequency: float = 20.0
    bin_width: float = 10e-6
    flux: float = 5e6
    efficiency: float = 1.0
    temperature: float = 300.0
    response: tuple = (3000.0, 50.0)
    pulse_width: float = 200e-6
    window: tuple = (1.0, 5.0)
    count_variance: object = 1.0
    open_beam: object = None
    open_beam_area: float = 1.0

    @classmethod
    def saraf_ii(cls, **changes):
        """
        The SARAF-II time-of-flight station of the article: 9 m, 20 Hz,
        5e6 n/cm^2/s, 200 us pulses, and the event-mode camera (4.5 %
        detection efficiency, counting statistics ``camera_count_variance``).
        Keyword arguments change single values.
        """
        args = dict(flight_path=9.0, frequency=20.0, bin_width=10e-6, flux=5e6, efficiency=0.045,
                    pulse_width=200e-6, count_variance=camera_count_variance)
        args.update(changes)
        return cls(**args)

    @property
    def grid(self):
        return TimeGrid(self.flight_path, self.frequency, self.bin_width)

    def beam(self, bin_size):
        """Open beam of one pulse for a square bin of side ``bin_size`` [mm]."""
        area = (float(bin_size) / 10.0) ** 2
        if self.open_beam is not None:
            if self.open_beam.grid != self.grid:
                raise ValueError("open_beam must be on the time grid of the instrument")
            return self.open_beam.scaled(area / self.open_beam_area)
        return OpenBeam.maxwellian(self.grid, flux=self.flux, temperature=self.temperature, area=area,
                                   efficiency=self.efficiency)

    def F(self, bin_size):
        """Variance of the recorded counts relative to Poisson counts for bins of ``bin_size`` [mm]."""
        f = self.count_variance
        return np.asarray(f(bin_size), float) if callable(f) else np.full(np.shape(bin_size), float(f))


def _display(name):
    """Scale and unit in which a parameter is shown and given."""
    if name == "strain":
        return 1e6, "microstrain"
    if name == "thickness":
        return 1e4, "um"
    return 1.0, ""


def _make_sample(material, thickness, instrument, free):
    if isinstance(material, Sample):
        sample = material if thickness is None else material.with_values(thickness=thickness)
        if tuple(sample.free) != tuple(free):
            sample = Sample(sample.model, params=sample.params, free=free)
        return sample
    import nbragg
    if isinstance(material, nbragg.CrossSection):
        if thickness is None:
            raise ValueError("give the thickness of the sample [cm]")
        model = nbragg.TransmissionModel(material, response="jorgensen_inv", tof_length=instrument.flight_path,
                                         vary_response=False)
        model.params["α0"].set(value=instrument.response[0])
        model.params["β0"].set(value=instrument.response[1])
    else:                                            # an nbragg TransmissionModel
        model = material
    if thickness is not None:
        model.params["thickness"].set(value=thickness)
    return Sample(model, free=free)


def plan(material, thickness=None, *, target="strain", n_pulses=8, instrument=None, free=None,
         n_restarts=4, n_steps=800, seed=0, verbose=False):
    """
    Plan a Bragg-edge imaging measurement with overlapping pulses.

    The pulse times are optimised for ``target`` with the Rietveld fit (they
    minimise its Cramer-Rao bound). The returned ``Plan`` gives the pulse
    pattern, its time-reduction factor against one pulse per period, the
    precision for any measurement time and bin size, and the time needed
    for a chosen precision.

    Parameters
    ----------
    material : nbragg.CrossSection, nbragg.TransmissionModel or Sample
        The sample. A CrossSection gets the moderator response and the flight
        path of the instrument; a TransmissionModel or Sample is used as
        given (with ``thickness``, if given).
    thickness : float
        Sample thickness [cm]; required for a CrossSection.
    target : str
        The quantity to optimise for: ``"strain"`` (the Bragg-edge
        position), ``"thickness"``, or another nbragg parameter (e.g. a
        phase weight).
    n_pulses : int or sequence of int
        Pulses per period. For a sequence, each number is optimised and the
        best for ``target`` is chosen; ``Plan.candidates`` lists them all.
    instrument : Instrument
        Default: ``Instrument.saraf_ii()``.
    free : sequence of str
        Parameters the fit determines (default: thickness and strain, and the
        target). Every free parameter costs precision in the others.
    n_restarts, n_steps, seed, verbose
        Settings of ``optimise_pattern`` (a stochastic search; more restarts
        and steps can find better times). About ten seconds per number of
        pulses with the defaults.
    """
    instrument = instrument or Instrument.saraf_ii()
    free = list(free) if free is not None else ["thickness", "strain"]
    if target not in free:
        free.append(target)
    sample = _make_sample(material, thickness, instrument, free)
    sample.check_window(instrument.window)
    exp = Experiment(sample, instrument.beam(_REF_BIN), Pattern.single(instrument.pulse_width), _REF_TIME)
    counts = sorted({int(n) for n in np.atleast_1d(n_pulses)})
    if counts[0] < 1:
        raise ValueError("n_pulses must be at least 1")
    one = exp.precision(window=instrument.window).set_index("parameter").sigma
    rows, patterns = [], {}
    for n in counts:
        if n == 1:
            pat = Pattern.single(instrument.pulse_width)
        else:
            pat = optimise_pattern(exp, n, target=target, width=instrument.pulse_width, n_restarts=n_restarts,
                                   n_steps=n_steps, window=instrument.window, seed=seed + n)
        s = exp.with_pattern(pat).precision(window=instrument.window).set_index("parameter").sigma
        row = dict(n_pulses=n, times_ms=" ".join(f"{t:.2f}" for t in pat.times_ms))
        row.update({f"R_{p}": float((one[p] / s[p]) ** 2) for p in sample.free})
        rows.append(row)
        patterns[n] = pat
        if verbose:
            print(f"{n} pulses: R({target}) = {row[f'R_{target}']:.2f}", flush=True)
    candidates = pd.DataFrame(rows)
    best = int(candidates.loc[candidates[f"R_{target}"].idxmax(), "n_pulses"])
    return Plan(instrument, sample, patterns[best], target, candidates, exp)


class Plan:
    """
    A planned measurement: the optimised pulse pattern and what it reaches
    (made by ``frame_overlap.plan``).

    Attributes
    ----------
    pattern : Pattern
        The pulse pattern chosen for the target.
    candidates : pandas.DataFrame
        Every number of pulses tried, with its pulse times [ms] and the
        time-reduction factor of each free parameter.
    gain : dict
        Time-reduction factor of each free parameter against one pulse per
        period: one pulse per period needs R times longer for the same
        precision.
    """

    def __init__(self, instrument, sample, pattern, target, candidates, exp):
        self.instrument, self.sample, self.pattern, self.target = instrument, sample, pattern, target
        self.candidates = candidates
        self._exp = exp.with_pattern(pattern)
        self._one = exp.with_pattern(Pattern.single(instrument.pulse_width))
        w = instrument.window
        self._sigma = self._exp.precision(window=w).set_index("parameter").sigma      # reference exposure, F = 1
        self._sigma_one = self._one.precision(window=w).set_index("parameter").sigma
        self._counts = float(self._exp.expected_counts().sum())
        self.gain = {p: float((self._sigma_one[p] / self._sigma[p]) ** 2) for p in sample.free}

    # ------------------------------------------------------------ scaling
    def _scale(self, time, bin_size):
        """Factor on the reference bound for a time [s] and square bins [mm]."""
        t, b = np.asarray(time, float), np.asarray(bin_size, float)
        return np.sqrt(self.instrument.F(b) * _REF_TIME / t) * (_REF_BIN / b)

    def counts(self, time=8 * 3600, bin_size=1.0):
        """Independent counts per spectrum (recorded counts / F) for a time [s] and bin size [mm]."""
        t, b = np.asarray(time, float), np.asarray(bin_size, float)
        return self._counts * (t / _REF_TIME) * (b / _REF_BIN) ** 2 / self.instrument.F(b)

    def _check_counts(self, time, bin_size):
        need = MIN_COUNTS[1] if self.pattern.n_pulses == 1 else MIN_COUNTS["overlap"]
        n = np.min(self.counts(time, bin_size))
        if n < need:
            warnings.warn(f"only {n:.3g} independent counts per spectrum (below {need:.0e}): the Rietveld fit may "
                          "scatter more than the bound; check with frame_overlap.monte_carlo", stacklevel=3)

    # ------------------------------------------------------------ results
    def precision(self, time=8 * 3600, bin_size=1.0):
        """
        One standard deviation of each free parameter for a measurement of
        ``time`` [s] in square bins of side ``bin_size`` [mm], with this
        pattern and with one pulse per period (strain in microstrain,
        thickness in micrometres).
        """
        self._check_counts(time, bin_size)
        k = float(self._scale(time, bin_size))
        rows = []
        for p in self.sample.free:
            f, unit = _display(p)
            rows.append(dict(parameter=p, sigma=self._sigma[p] * k * f, one_pulse=self._sigma_one[p] * k * f,
                             unit=unit, R=self.gain[p]))
        return pd.DataFrame(rows)

    def time_for(self, bin_size=1.0, **targets):
        """
        Measurement time [s] for which every given parameter reaches the given
        standard deviation in square bins of side ``bin_size`` [mm] (a number
        or an array). Targets are given in the units of ``precision``:
        ``strain`` in microstrain, ``thickness`` in micrometres, e.g.
        ``plan.time_for(bin_size=1.0, strain=1000)``.
        """
        if not targets:
            raise TypeError(f"give a target precision, e.g. time_for(bin_size=1.0, {self.target}=...)")
        b = np.asarray(bin_size, float)
        t = self._time(self._sigma, b, targets)
        self._check_counts(t, b)
        return float(t) if t.ndim == 0 else t

    def _time(self, sigma, b, targets):
        """Time [s] for the targets with the reference bounds ``sigma``."""
        t = np.zeros(b.shape)
        for p, goal in targets.items():
            if p not in self.sample.free:
                raise KeyError(f"{p!r} is not a free parameter; choose from {self.sample.free}")
            f, _ = _display(p)
            s = sigma[p] * f * np.sqrt(self.instrument.F(b)) * (_REF_BIN / b)    # at the reference time
            t = np.maximum(t, _REF_TIME * (s / float(goal)) ** 2)
        return t

    def summary(self, time=8 * 3600, bin_sizes=(0.5, 1.0, 2.0, 5.0), **targets):
        """Text summary: instrument, pulse times, gains, precision in ``time``
        for some bin sizes, and the times for the given targets."""
        ins = self.instrument
        lines = [f"Plan for {self.target}: {self.pattern.n_pulses} pulses of {ins.pulse_width * 1e6:g} us per "
                 f"{1e3 / ins.frequency:g} ms at {ins.flight_path:g} m",
                 "pulse times (ms): " + ", ".join(f"{t:.2f}" for t in self.pattern.times_ms),
                 "time-reduction factor against one pulse per period: "
                 + ", ".join(f"{p} {r:.2f}" for p, r in self.gain.items())]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for b in bin_sizes:
                df = self.precision(time, b)
                lines.append(f"{time / 3600:g} h, {b:g} mm bins: " + ", ".join(
                    f"{r.parameter} {r.sigma:.3g} {r.unit} (one pulse {r.one_pulse:.3g})" for r in df.itertuples()))
            if targets:
                goal = ", ".join(f"{p} {v:g} {_display(p)[1]}" for p, v in targets.items())
                for b in bin_sizes:
                    lines.append(f"time for {goal}, {b:g} mm bins: {self.time_for(b, **targets) / 3600:.3g} h")
        lines.append("Cramer-Rao bound of the Rietveld fit; check small bins with frame_overlap.monte_carlo.")
        return "\n".join(lines)

    def experiment(self, time=8 * 3600, bin_size=1.0):
        """The Experiment of this plan for one bin (to simulate and fit a measurement)."""
        return Experiment(self.sample, self.instrument.beam(bin_size), self.pattern, time,
                          count_variance=float(self.instrument.F(bin_size)))

    def plot(self, time=8 * 3600, bin_sizes=None, **targets):
        """
        (a) The pulse pattern: the expected counts and the spectrum of each
        pulse. (b) With targets (e.g. ``strain=1000``): the measurement time
        they need against the bin size; without: the standard deviation of
        the target in ``time``. Both with this pattern and with one pulse per
        period.
        """
        import matplotlib.pyplot as plt
        b = np.geomspace(0.25, 10, 60) if bin_sizes is None else np.asarray(bin_sizes, float)
        g, exp = self.instrument.grid, self._exp
        T = self.sample.transmission(g)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
        t_ms = g.t * 1e3
        for k, (t0, w) in enumerate(zip(self.pattern.starts, self.pattern.widths)):
            part = exp.n_periods * g.convolve(exp.beam.rate * T, Pattern([t0], w).array(g))
            a1.fill_between(t_ms, part, color=("#6baed6", "#9ecae1")[k % 2], lw=0, alpha=0.55)
        a1.plot(t_ms, exp.expected_counts(), color="k", lw=0.7, label="all pulses")
        top = a1.get_ylim()[1]
        a1.plot(self.pattern.starts * 1e3, np.full(self.pattern.n_pulses, 0.97 * top), "v", color="k", ms=5,
                label="pulse times")
        a1.set_ylim(0, top)
        a1.set_xlim(0, g.period * 1e3)
        a1.set_xlabel("Time in the period (ms)")
        a1.set_ylabel("Expected counts per bin")
        a1.set_title(f"(a) {self.pattern.n_pulses} pulses optimised for {self.target}", fontsize=10, loc="left")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if targets:
                t_opt = self._time(self._sigma, b, targets) / 3600
                t_one = self._time(self._sigma_one, b, targets) / 3600
                goal = ", ".join(f"{v:g} {_display(p)[1]}" for p, v in targets.items())
                a2.loglog(b, t_opt, color="C0", lw=2, label=f"{self.pattern.n_pulses} pulses, optimised")
                a2.loglog(b, t_one, color="0.4", ls="--", lw=1.5, label="one pulse per period")
                a2.axhline(time / 3600, color="0.5", lw=0.8, ls=":")
                a2.text(b[0], time / 3600 * 1.15, f"{time / 3600:g} h", color="0.4", fontsize=8)
                a2.set_ylabel("Measurement time (h)")
                a2.set_title(f"(b) time for {goal}", fontsize=10, loc="left")
            else:
                f, unit = _display(self.target)
                k = self._scale(time, b)
                a2.loglog(b, self._sigma[self.target] * f * k, color="C0", lw=2,
                          label=f"{self.pattern.n_pulses} pulses, optimised")
                a2.loglog(b, self._sigma_one[self.target] * f * k, color="0.4", ls="--", lw=1.5,
                          label="one pulse per period")
                a2.set_ylabel(f"Standard deviation of {self.target} ({unit})")
                a2.set_title(f"(b) precision in {time / 3600:g} h", fontsize=10, loc="left")
        a2.set_xlabel("Bin size (mm)")
        plain = plt.FuncFormatter(lambda v, _: f"{v:g}")
        a2.set_xticks([x for x in (0.1, 0.25, 0.5, 1, 2, 5, 10, 20) if b.min() <= x <= b.max()])
        a2.xaxis.set_major_formatter(plain)
        a2.xaxis.set_minor_formatter(plt.NullFormatter())
        a2.yaxis.set_major_formatter(plain)
        a2.grid(True, which="both", color="0.9", lw=0.5)
        a2.legend(frameon=False)
        fig.tight_layout()
        return fig

    def __repr__(self):
        return (f"Plan({self.target}: {self.pattern.n_pulses} pulses, "
                + ", ".join(f"R_{p} = {r:.2f}" for p, r in self.gain.items()) + ")")
