"""A measurement design (sample, open beam, pulse pattern, time) and its data."""
import numpy as np
from .pattern import Pattern


def _check_time(time):
    if not time > 0:
        raise ValueError("the measurement time must be positive (seconds)")
    return float(time)


class Measurement:
    """
    Counts of one measurement, summed over all periods, on a TimeGrid.

    Parameters
    ----------
    grid : TimeGrid
    counts : array
        Sample counts per time bin (overlapped if the pattern has several pulses).
    pattern : Pattern
        The pulse pattern used (assumed known).
    beam : OpenBeam
        The open beam of one pulse (known from a separate measurement).
    time : float
        Measurement time [s].
    background : float
        Known constant count background, in counts per bin and second of
        measurement (0 if none; fits can also determine it).
    truth : dict, optional
        True parameter values, for simulated data.
    """

    def __init__(self, grid, counts, pattern, beam, time, background=0.0, truth=None):
        counts = np.asarray(counts, float)
        if counts.shape != (grid.n_bins,):
            raise ValueError(f"counts must have {grid.n_bins} bins (one period of {grid!r}), got {counts.shape}")
        if np.any(counts < 0) or np.any(~np.isfinite(counts)):
            raise ValueError("counts must be finite and non-negative")
        if beam.grid != grid:
            raise ValueError(f"the open beam is on {beam.grid!r}, the counts on {grid!r}")
        pattern.check(grid)
        self.grid, self.pattern, self.beam, self.time = grid, pattern, beam, _check_time(time)
        self.counts = counts
        self.background = float(background)
        self.truth = truth

    @property
    def n_periods(self):
        return self.time * self.grid.frequency

    def open_counts(self):
        """Expected open-beam counts with the same pattern and time."""
        return self.beam.counts(self.pattern, self.time)

    def plot(self, ax=None, **kw):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        ax.plot(self.grid.t * 1e3, self.counts, lw=kw.pop("lw", 0.4), color=kw.pop("color", "k"), **kw)
        ax.set_xlabel("Time in the period (ms)")
        ax.set_ylabel(f"Counts per {self.grid.bin_width * 1e6:g} µs")
        ax.set_xlim(0, self.grid.period * 1e3)
        return ax

    def __repr__(self):
        return (f"Measurement({self.counts.sum():.3g} counts in {self.time / 3600:g} h, "
                f"{self.pattern.n_pulses} pulse(s) per period)")


class Experiment:
    """
    A measurement design: a sample, the open beam of one pulse, a pulse
    pattern and the measurement time.

    The expected sample counts are

        mu(t) = n_periods * [c (*) (o T)](t) + background * time,

    with c(t) the open time of the pattern in each bin, o the open-beam rate
    of one pulse, T the transmission of the sample and (*) a circular
    convolution over the period (slow neutrons of one period arrive in the
    next).

    Parameters
    ----------
    sample : Sample
    beam : OpenBeam
    pattern : Pattern
    time : float
        Measurement time [s] (e.g. ``8 * 3600`` for 8 h).
    background : float
        Constant count background per time bin and second of measurement.
        It is passed on to simulated measurements, so fits know it.
    """

    def __init__(self, sample, beam, pattern, time, background=0.0):
        pattern.check(beam.grid)
        if beam.total_rate() <= 0:
            raise ValueError("the open beam is empty")
        self.sample, self.beam, self.pattern, self.time = sample, beam, pattern, _check_time(time)
        self.background = float(background)
        self.grid = beam.grid

    @property
    def n_periods(self):
        return self.time * self.grid.frequency

    def expected_counts(self, theta=None):
        """Expected sample counts per bin for the whole measurement."""
        T = self.sample.transmission(self.grid, theta)
        mu = self.n_periods * self.grid.convolve(self.beam.rate * T, self.pattern.array(self.grid))
        return np.clip(mu, 0, None) + self.background * self.time

    def simulate(self, seed=None):
        """One Poisson draw of the sample counts (a Measurement)."""
        rng = np.random.default_rng(seed)
        y = rng.poisson(self.expected_counts()).astype(float)
        truth = dict(zip(self.sample.free, (float(v) for v in self.sample.theta)))
        return Measurement(self.grid, y, self.pattern, self.beam, self.time, self.background, truth=truth)

    def _variant(self, **kw):
        args = dict(sample=self.sample, beam=self.beam, pattern=self.pattern, time=self.time,
                    background=self.background)
        args.update(kw)
        e = Experiment(**args)
        if "_dT" in self.__dict__ and args["sample"] is self.sample:     # derivatives do not depend on these
            e.__dict__["_dT"] = self.__dict__["_dT"]
        return e

    def with_pattern(self, pattern):
        """The same design with another pulse pattern."""
        return self._variant(pattern=pattern)

    def with_time(self, time):
        """The same design with another measurement time."""
        return self._variant(time=time)

    def with_beam(self, beam):
        """The same design with another open beam (e.g. a bandwidth chopper)."""
        return self._variant(beam=beam)

    # convenience access to the design tools
    def precision(self, **kw):
        """Cramér–Rao bound of each free parameter (see frame_overlap.precision)."""
        from .design import precision
        return precision(self, **kw)

    def gain(self, reference=None, **kw):
        """Time-reduction factor against a reference design (see frame_overlap.gain)."""
        from .design import gain
        return gain(self, reference=reference, **kw)

    def plot(self, ax=None):
        """Expected sample counts and the pulse pattern."""
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        ax.plot(self.grid.t * 1e3, self.expected_counts(), color="k", lw=0.6)
        for t0 in self.pattern.starts:
            ax.axvline(t0 * 1e3, color="0.6", lw=0.6, ls=":")
        ax.set_xlabel("Time in the period (ms)")
        ax.set_ylabel(f"Expected counts per {self.grid.bin_width * 1e6:g} µs")
        ax.set_xlim(0, self.grid.period * 1e3)
        return ax

    def __repr__(self):
        return f"Experiment({self.pattern!r}, {self.time / 3600:g} h, {self.sample!r})"
