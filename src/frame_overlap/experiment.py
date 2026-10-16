"""A measurement design (sample, open beam, pulse pattern, time) and its data."""
import numpy as np
from .pattern import Pattern


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
    truth : dict, optional
        True parameter values, for simulated data.
    """

    def __init__(self, grid, counts, pattern, beam, time, truth=None):
        self.grid, self.pattern, self.beam, self.time = grid, pattern, beam, float(time)
        self.counts = np.asarray(counts, float)
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

        mu(t) = n_periods * [c (*) (o T)](t),

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
        Measurement time [s].
    background : float
        Constant count background per time bin and second of measurement.
    """

    def __init__(self, sample, beam, pattern, time, background=0.0):
        self.sample, self.beam, self.pattern, self.time = sample, beam, pattern, float(time)
        self.background = float(background)
        self.grid = beam.grid

    @property
    def n_periods(self):
        return self.time * self.grid.frequency

    def expected_counts(self, theta=None):
        T = self.sample.transmission(self.grid, theta)
        mu = self.n_periods * self.grid.convolve(self.beam.rate * T, self.pattern.array(self.grid))
        return mu + self.background * self.time

    def simulate(self, seed=None):
        """One Poisson draw of the sample counts (a Measurement)."""
        rng = np.random.default_rng(seed)
        y = rng.poisson(np.clip(self.expected_counts(), 0, None)).astype(float)
        truth = dict(zip(self.sample.free, self.sample.theta))
        return Measurement(self.grid, y, self.pattern, self.beam, self.time, truth=truth)

    def with_pattern(self, pattern):
        e = Experiment(self.sample, self.beam, pattern, self.time, self.background)
        if "_dT" in self.__dict__:          # derivatives do not depend on the pattern
            e.__dict__["_dT"] = self.__dict__["_dT"]
        return e

    def with_time(self, time):
        e = Experiment(self.sample, self.beam, self.pattern, time, self.background)
        if "_dT" in self.__dict__:
            e.__dict__["_dT"] = self.__dict__["_dT"]
        return e

    def with_beam(self, beam):
        return Experiment(self.sample, beam, self.pattern, self.time, self.background)

    # convenience access to the design tools
    def precision(self, **kw):
        from .design import precision
        return precision(self, **kw)

    def gain(self, reference=None, **kw):
        from .design import gain
        return gain(self, reference=reference, **kw)

    def __repr__(self):
        return (f"Experiment({self.pattern!r}, {self.time / 3600:g} h, {self.sample!r})")
