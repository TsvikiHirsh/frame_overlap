"""Open beam of one pulse: the neutrons that reach the detector without sample."""
import warnings
import numpy as np
from .grid import TimeGrid

K_B = 8.617333e-5      # eV / K
MN_C2 = 939.5654e6     # neutron rest energy [eV]
HC = 12398.42          # eV * Angstrom
_trapz = getattr(np, "trapezoid", None) or np.trapz


class OpenBeam:
    """
    Open-beam rate of one pulse on a TimeGrid.

    ``rate[k]`` is the number of neutrons detected in time bin k after the
    start of a pulse, per second that the chopper is open. For a pulse of
    width w (seconds) repeated over a measurement of n periods, the expected
    open-beam counts are n * w * rate (convolved with the pulse shape).
    The rate includes the detector area and efficiency.

    Build it from a measured spectrum (``from_counts``, ``from_reference``) or
    from a model of a thermal moderator (``maxwellian``), and restrict it to a
    wavelength band with an ideal bandwidth chopper (``with_band``,
    ``with_band_for``).
    """

    def __init__(self, grid: TimeGrid, rate):
        rate = np.asarray(rate, float)
        if rate.shape != (grid.n_bins,):
            raise ValueError(f"rate must have {grid.n_bins} bins (one period of {grid!r}), got {rate.shape}")
        if np.any(~np.isfinite(rate)) or np.any(rate < 0):
            raise ValueError("the open-beam rate must be finite and non-negative")
        self.grid, self.rate = grid, rate

    # -------------------------------------------------------------- builders
    @classmethod
    def maxwellian(cls, grid, flux=5e6, temperature=300.0, area=1.0, efficiency=1.0,
                   epithermal_fraction=0.05, lambda_min=0.2):
        """
        Thermal moderator: a Maxwellian in wavelength with a 1/E epithermal
        tail, for a continuous beam of ``flux`` [n/cm^2/s] (all wavelengths
        above ``lambda_min``) times area [cm^2] and detector efficiency. The
        moderator pulse shape is part of the nbragg response. Neutrons slower
        than the period (wavelengths beyond ``grid.wavelength_max``) are not
        included; a warning is given if they are more than 1 % of the beam.
        """
        lam_t = HC / np.sqrt(2 * MN_C2 * K_B * temperature)          # Angstrom, about 1.78 at 300 K
        lam_all = np.linspace(lambda_min, max(60.0, 4 * grid.wavelength_max), 20000)

        def phi(lam):
            maxw = 2 * lam_t ** 4 / lam ** 5 * np.exp(-(lam_t / lam) ** 2)
            epi = 1.0 / lam * (1 - np.exp(-(lam_t / (1.5 * lam)) ** 6))
            return maxw, epi

        m_all, e_all = phi(lam_all)
        nm, ne = _trapz(m_all, lam_all), _trapz(e_all, lam_all)
        lam = np.clip(grid.wavelength, 1e-3, None)
        m, e = phi(lam)
        p = (1 - epithermal_fraction) * m / nm + epithermal_fraction * e / ne        # per Angstrom
        p[lam < lambda_min] = 0.0
        dlam = float(grid.wavelength_at(grid.bin_width))                               # Angstrom per bin
        beyond = 1.0 - float(np.sum(p) * dlam)
        if beyond > 0.01:
            warnings.warn(f"{beyond:.1%} of the beam is slower than one period ({grid.wavelength_max:.1f} A) "
                          "and is not included; use a longer period or a bandwidth chopper", stacklevel=2)
        return cls(grid, flux * area * efficiency * p * dlam)

    @classmethod
    def from_counts(cls, grid, counts, pulse_width, time, tail_from=None):
        """
        Open beam measured with one pulse of ``pulse_width`` seconds per
        period for ``time`` seconds (counts per bin, starting at the pulse).
        Shorter spectra are extended to the period by an exponential fit of
        their tail (from ``tail_from`` seconds, default 2/3 of the data).

        The fits take the open beam as known, so measure it with much better
        statistics than the sample (for example summed over a large detector
        area and scaled with ``scaled``), and with a short pulse: the counts
        contain the shape of the calibration pulse, which would otherwise be
        applied twice.
        """
        if pulse_width <= 0 or time <= 0:
            raise ValueError("pulse_width and time must be positive (seconds)")
        counts = _extend(grid, counts, tail_from)
        return cls(grid, counts / (pulse_width * time * grid.frequency))

    @classmethod
    def from_reference(cls, grid, counts, reference_flux, reference_time, flux, tail_from=None):
        """
        Spectral shape from a reference measurement at another source (counts
        per bin of an instantaneous pulse, summed over ``reference_time``
        seconds at a time-averaged ``reference_flux``), scaled to a continuous
        beam of ``flux`` with the same area and efficiency.
        """
        counts = _extend(grid, counts, tail_from)
        return cls(grid, counts * flux / (reference_flux * reference_time))

    # -------------------------------------------------------------- variants
    def with_band(self, wavelength_min, wavelength_max):
        """Ideal bandwidth chopper: pass only neutrons of each pulse with
        wavelength_min <= lambda < wavelength_max [Angstrom]."""
        if not wavelength_min < wavelength_max:
            raise ValueError("wavelength_min must be smaller than wavelength_max")
        lam = self.grid.wavelength
        rate = self.rate * ((lam >= wavelength_min) & (lam < wavelength_max))
        if rate.sum() == 0:
            raise ValueError(f"no neutrons in the band {wavelength_min}-{wavelength_max} A on this grid")
        return OpenBeam(self.grid, rate)

    def with_band_for(self, n_pulses, pulse_width=200e-6, wavelength_min=0.97, wavelength_max=5.25,
                      keep_up_to=4.55, gap=50e-6):
        """
        Bandwidth chopper for n equally spaced pulses: the band
        [wavelength_min, wavelength_max] if it fits between pulses, otherwise
        the widest band that fits and still reaches ``keep_up_to`` (for
        example just beyond the last Bragg edge of interest).
        """
        g = self.grid
        width = g.period / n_pulses - pulse_width - gap
        if width <= 0:
            raise ValueError("the pulses do not leave room for a band")
        lo_t, hi_t = float(g.time_at(wavelength_min)), float(g.time_at(wavelength_max))
        if lo_t + width >= hi_t:
            return self.with_band(wavelength_min, wavelength_max)
        hi_t = max(lo_t + width, float(g.time_at(keep_up_to)))
        return self.with_band(float(g.wavelength_at(max(lo_t, hi_t - width))), float(g.wavelength_at(hi_t)))

    def scaled(self, factor):
        """The same spectrum times a factor (e.g. another area or efficiency)."""
        return OpenBeam(self.grid, self.rate * factor)

    # -------------------------------------------------------------- use
    def counts(self, pattern, time):
        """Expected open-beam counts of a measurement of ``time`` seconds with
        this pulse pattern (circular over the period)."""
        n = time * self.grid.frequency
        return n * self.grid.convolve(self.rate, pattern.array(self.grid))

    def total_rate(self):
        """Detected neutrons per second of open chopper."""
        return float(self.rate.sum())

    def plot(self, ax=None, **kw):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        ax.semilogy(self.grid.wavelength, np.clip(self.rate, 1e-30, None), **kw)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Rate per bin (n per second open)")
        return ax

    def __repr__(self):
        return f"OpenBeam({self.total_rate():.3g} n per second of open chopper, {self.grid!r})"


def _extend(grid, counts, tail_from=None):
    counts = np.asarray(counts, float)
    n = len(counts)
    if n > grid.n_bins:
        raise ValueError(f"spectrum of {n} bins is longer than the period ({grid.n_bins} bins)")
    if np.any(counts < 0) or np.any(~np.isfinite(counts)):
        raise ValueError("counts must be finite and non-negative")
    if n == grid.n_bins:
        return counts.copy()
    t = grid.t
    k0 = int((tail_from / grid.bin_width) if tail_from is not None else 2 * n // 3)
    b, a = np.polyfit(t[k0:n], np.log(np.clip(counts[k0:n], 1e-12, None)), 1)
    out = np.empty(grid.n_bins)
    out[:n] = counts
    if b < 0:
        out[n:] = np.exp(a + b * t[n:])
    else:
        warnings.warn("the tail of the spectrum does not fall; it is extended with its last value", stacklevel=3)
        out[n:] = counts[n - 1]
    return out
