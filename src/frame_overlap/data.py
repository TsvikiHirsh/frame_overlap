"""Overlapped counts of one measurement, with its pulse pattern and open beam."""
import numpy as np
import pandas as pd
from .grid import TimeGrid
from .openbeam import OpenBeam


def _check_time(time):
    if time is None or not time > 0:
        raise ValueError("the measurement time must be positive (seconds)")
    return float(time)


def _read_counts(source):
    """Counts per time bin from a CSV path, a DataFrame or an array, with nbragg's
    column rules (tof, counts, err; common aliases accepted)."""
    if isinstance(source, (str, pd.DataFrame)):
        try:
            import nbragg
            table = nbragg.Data._read_counts(source)
        except (ImportError, AttributeError):             # without nbragg: a 'counts' column or the second one
            table = pd.read_csv(source) if isinstance(source, str) else source
            if "counts" not in table:
                table = table.rename(columns={table.columns[min(1, table.shape[1] - 1)]: "counts"})
        return np.asarray(table["counts"], float)
    return np.asarray(source, float)


class Data:
    """
    Counts of one measurement on the time bins of one period, summed over all
    periods, with the pulse pattern that produced them and the open beam of
    one pulse. The counterpart of ``nbragg.Data`` for overlapped spectra.

    Build it with ``Data.from_counts`` from measured counts, or with
    ``TransmissionModel.simulate`` for a projected measurement.

    Attributes
    ----------
    counts : array
        Sample counts per time bin (overlapped if the pattern has several pulses).
    pattern : Pattern
        The pulse pattern used (taken as known).
    openbeam : OpenBeam
        The open beam of one pulse (taken as known).
    time : float
        Measurement time [s].
    grid : TimeGrid
        Flight path, period and time bins.
    background : float
        Known constant count background, in counts per bin and second.
    count_variance : float
        Variance of the recorded counts relative to Poisson counts (1 for a
        counting detector; about 2.3 for the event-mode camera of the SARAF-II
        projections in bins of 1 mm and more). Fit uncertainties are scaled by
        its square root.
    truth : dict or None
        True parameter values of simulated data.
    """

    def __init__(self, grid, counts, pattern, openbeam, time, background=0.0, count_variance=1.0, truth=None):
        counts = np.asarray(counts, float)
        if counts.shape != (grid.n_bins,):
            raise ValueError(f"counts must have {grid.n_bins} bins (one period of {grid!r}), got {counts.shape}")
        if np.any(counts < 0) or np.any(~np.isfinite(counts)):
            raise ValueError("counts must be finite and non-negative")
        if openbeam.grid != grid:
            raise ValueError(f"the open beam is on {openbeam.grid!r}, the counts on {grid!r}")
        if count_variance < 1:
            raise ValueError("count_variance must be at least 1 (1 = Poisson counts)")
        pattern.check(grid)
        self.grid, self.pattern, self.openbeam, self.time = grid, pattern, openbeam, _check_time(time)
        self.counts = counts
        self.background = float(background)
        self.count_variance = float(count_variance)
        self.truth = truth

    @classmethod
    def from_counts(cls, signal, openbeam, pattern, time, openbeam_time=None, openbeam_pulse=200e-6, L=9.0,
                    tstep=10e-6, background=0.0, count_variance=1.0, tail_from=None):
        """
        Data from measured counts, as ``nbragg.Data.from_counts``.

        Parameters
        ----------
        signal : str, DataFrame or array
            Sample counts per time bin over one period (summed over all
            periods), from the start of the period. CSV files and DataFrames
            follow nbragg's columns (tof, counts, err).
        openbeam : str, DataFrame or array
            Open beam measured with one short pulse per period (starting at
            its pulse), with much better statistics than the sample, for
            example summed over a large detector area. It may be shorter than
            the period; its tail is then extended by an exponential fit.
        pattern : Pattern
            The pulse pattern of the sample measurement.
        time : float
            Sample measurement time [s].
        openbeam_time : float
            Open-beam measurement time [s] (default: ``time``).
        openbeam_pulse : float
            Width of the open-beam pulse [s].
        L : float
            Flight path [m].
        tstep : float
            Time bin [s]. The period is the number of sample bins times tstep.
        background, count_variance : see the class.
        """
        y = _read_counts(signal)
        o = _read_counts(openbeam)
        grid = TimeGrid(L, 1.0 / (len(y) * tstep), tstep)
        ob = OpenBeam.from_counts(grid, o, pulse_width=openbeam_pulse, time=openbeam_time or time,
                                  tail_from=tail_from)
        return cls(grid, y, pattern, ob, time, background=background, count_variance=count_variance)

    # ------------------------------------------------------------ properties
    @property
    def beam(self):                      # the name used by the fitting code
        return self.openbeam

    @property
    def n_periods(self):
        return self.time * self.grid.frequency

    @property
    def table(self):
        """Time [s], wavelength of a neutron from the first pulse [A] and counts."""
        return pd.DataFrame(dict(tof=self.grid.t, wavelength=self.grid.wavelength, counts=self.counts))

    def open_counts(self):
        """Expected open-beam counts with the same pattern and time."""
        return self.openbeam.counts(self.pattern, self.time)

    def to_csv(self, path):
        """Write the counts in nbragg's format (tof in bins, counts, err)."""
        pd.DataFrame(dict(tof=np.arange(self.grid.n_bins), counts=self.counts,
                          err=np.sqrt(self.count_variance * np.clip(self.counts, 0, None)))).to_csv(path, index=False)

    def plot(self, ax=None, **kw):
        """The counts over the period, with the pulse times."""
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        ax.plot(self.grid.t * 1e3, self.counts, lw=kw.pop("lw", 0.4), color=kw.pop("color", "k"), **kw)
        top = 1.12 * max(self.counts.max(), 1)
        ax.plot(self.pattern.starts * 1e3, np.full(self.pattern.n_pulses, top / 1.05), "v", color="k", ms=4)
        ax.set_ylim(0, top)
        ax.set_xlabel("Time in the period (ms)")
        ax.set_ylabel(f"Counts per {self.grid.bin_width * 1e6:g} µs")
        ax.set_xlim(0, self.grid.period * 1e3)
        return ax

    def __repr__(self):
        return (f"Data({self.counts.sum():.3g} counts in {self.time / 3600:g} h, {self.pattern.n_pulses} pulse(s) per "
                f"{self.grid.period * 1e3:g} ms at {self.grid.flight_path:g} m)")
