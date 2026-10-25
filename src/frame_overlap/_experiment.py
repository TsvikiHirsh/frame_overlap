"""A measurement design (sample, open beam, pulse pattern, time), used internally by the design tools."""
import numpy as np
from .data import Data, _check_time


class Experiment:
    """
    A measurement design: a sample, the open beam of one pulse, a pulse
    pattern and the measurement time. The expected sample counts are

        mu(t) = n_periods * [c (*) (o T)](t) + background * time,

    with c(t) the open time of the pattern in each bin, o the open-beam rate
    of one pulse, T the transmission of the sample and (*) a circular
    convolution over the period (slow neutrons of one period arrive in the
    next). ``count_variance`` is the variance of the recorded counts relative
    to Poisson counts: the Cramer-Rao bounds grow by its square root, and
    simulated counts get that variance.
    """

    def __init__(self, sample, beam, pattern, time, background=0.0, count_variance=1.0):
        pattern.check(beam.grid)
        if beam.total_rate() <= 0:
            raise ValueError("the open beam is empty")
        self.sample, self.beam, self.pattern, self.time = sample, beam, pattern, _check_time(time)
        self.background = float(background)
        if count_variance < 1:
            raise ValueError("count_variance must be at least 1 (1 = Poisson counts)")
        self.count_variance = float(count_variance)
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
        """One draw of the sample counts (Poisson, or count_variance times the Poisson variance)."""
        rng = np.random.default_rng(seed)
        mu = self.expected_counts()
        if self.count_variance == 1.0:
            y = rng.poisson(mu).astype(float)
        else:   # mean mu, variance count_variance * mu
            y = self.count_variance * rng.poisson(mu / self.count_variance).astype(float)
        truth = dict(zip(self.sample.free, (float(v) for v in self.sample.theta)))
        return Data(self.grid, y, self.pattern, self.beam, self.time, self.background, self.count_variance,
                    truth=truth)

    def _variant(self, **kw):
        args = dict(sample=self.sample, beam=self.beam, pattern=self.pattern, time=self.time,
                    background=self.background, count_variance=self.count_variance)
        args.update(kw)
        e = Experiment(**args)
        if "_dT" in self.__dict__ and args["sample"] is self.sample:     # derivatives do not depend on these
            e.__dict__["_dT"] = self.__dict__["_dT"]
        return e

    def with_pattern(self, pattern):
        return self._variant(pattern=pattern)

    def with_time(self, time):
        return self._variant(time=time)

    def with_beam(self, beam):
        return self._variant(beam=beam)

    def precision(self, **kw):
        from .design import precision
        return precision(self, **kw)

    def gain(self, reference=None, **kw):
        from .design import gain
        return gain(self, reference=reference, **kw)

    def __repr__(self):
        return f"Experiment({self.pattern!r}, {self.time / 3600:g} h, {self.sample!r})"
