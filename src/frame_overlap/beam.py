"""Open beam of one pulse: the neutrons that reach the detector without sample."""
import numpy as np
from .grid import TimeGrid, H_OVER_M

K_B = 8.617333e-5      # eV / K
MN_C2 = 939.5654e6     # neutron rest energy [eV]
HC = 12398.42          # eV * Angstrom


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
    wavelength band with an ideal bandwidth chopper (``with_band``).
    """

    def __init__(self, grid: TimeGrid, rate):
        rate = np.asarray(rate, float)
        if rate.shape != (grid.n_bins,):
            raise ValueError(f"rate must have {grid.n_bins} bins")
        self.grid, self.rate = grid, rate

    # -------------------------------------------------------------- builders
    @classmethod
    def maxwellian(cls, grid, flux=5e6, temperature=300.0, area=1.0, efficiency=1.0,
                   epithermal_fraction=0.05, lambda_min=0.2):
        """
        Thermal moderator: a Maxwellian in wavelength with a 1/E epithermal
        tail, normalised to ``flux`` [n/cm^2/s] (continuous beam, all
        wavelengths above ``lambda_min``) times area [cm^2] and detector
        efficiency. The moderator pulse shape is part of the nbragg response.
        """
        lam = np.clip(grid.wavelength, 1e-3, None)
        lam_t = HC / np.sqrt(2 * MN_C2 * K_B * temperature)          # Angstrom, about 1.78 at 300 K
        maxw = 2 * lam_t ** 4 / lam ** 5 * np.exp(-(lam_t / lam) ** 2)
        # 1/E in energy is 1/lambda in wavelength, cut smoothly at the thermal peak
        epi = 1.0 / lam * (1 - np.exp(-(lam_t / (1.5 * lam)) ** 6))
        phi = (1 - epithermal_fraction) * maxw / np.trapezoid(maxw, lam) + \
            epithermal_fraction * epi / np.trapezoid(epi, lam)
        phi[lam < lambda_min] = 0.0
        dlam = np.gradient(lam)
        rate = flux * area * efficiency * phi * dlam / np.sum(phi * dlam)
        return cls(grid, rate)

    @classmethod
    def from_counts(cls, grid, counts, pulse_width, n_periods, tail_from=None):
        """
        Open beam measured with one pulse of ``pulse_width`` seconds per
        period over ``n_periods`` periods (counts per bin, starting at the
        pulse). Shorter spectra are extended to the period by an exponential
        fit of their tail (from ``tail_from`` seconds, default 2/3 of the data).

        The fits take the open beam as known, so measure it with much better
        statistics than the sample (for example summed over a large detector
        area and scaled with ``scaled``), and with a short pulse: the counts
        contain the shape of the calibration pulse, which would otherwise be
        applied twice.
        """
        counts = _extend(grid, counts, tail_from)
        return cls(grid, counts / (pulse_width * n_periods))

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
        lam = self.grid.wavelength
        return OpenBeam(self.grid, self.rate * ((lam >= wavelength_min) & (lam < wavelength_max)))

    def scaled(self, factor):
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

    def __repr__(self):
        return f"OpenBeam({self.total_rate():.3g} n per second of open chopper, {self.grid!r})"


def _extend(grid, counts, tail_from=None):
    counts = np.asarray(counts, float)
    n = len(counts)
    if n > grid.n_bins:
        raise ValueError("spectrum longer than the period")
    if n == grid.n_bins:
        return counts.copy()
    t = grid.t
    k0 = int((tail_from / grid.bin_width) if tail_from is not None else 2 * n // 3)
    b, a = np.polyfit(t[k0:n], np.log(np.clip(counts[k0:n], 1e-12, None)), 1)
    out = np.empty(grid.n_bins)
    out[:n] = counts
    out[n:] = np.exp(a + b * t[n:])
    return out
