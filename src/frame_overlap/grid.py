"""Time-of-flight grid of one period of a pulsed or chopped beam."""
import numpy as np

H_OVER_M = 3956.0339  # h / m_n in Angstrom * m / s


def time_to_wavelength(t, flight_path):
    """Wavelength [Angstrom] of a neutron that arrives t seconds after its pulse."""
    return H_OVER_M * np.asarray(t, float) / float(flight_path)


def wavelength_to_time(wavelength, flight_path):
    """Arrival time [s] after the pulse of a neutron of the given wavelength [Angstrom]."""
    return np.asarray(wavelength, float) * float(flight_path) / H_OVER_M


class TimeGrid:
    """
    Time bins of one period. Everything in frame_overlap lives on such a grid:
    the open beam of one pulse, the pulse pattern and the measured counts.

    Parameters
    ----------
    flight_path : float
        Source-to-detector distance [m].
    frequency : float
        Repetition rate of the period [Hz]. The pulse pattern repeats with it,
        and slow neutrons of one period arrive in the next (all convolutions
        are circular over the period).
    bin_width : float
        Width of a time bin [s].

    Examples
    --------
    >>> grid = TimeGrid(flight_path=9.0, frequency=20.0, bin_width=10e-6)
    >>> grid.n_bins, round(grid.wavelength_at(12.5e-3), 2)
    (5000, 5.49)
    """

    def __init__(self, flight_path=9.0, frequency=20.0, bin_width=10e-6):
        self.flight_path = float(flight_path)
        self.frequency = float(frequency)
        self.bin_width = float(bin_width)
        self.period = 1.0 / self.frequency
        self.n_bins = int(round(self.period / self.bin_width))
        if abs(self.n_bins * self.bin_width - self.period) > 1e-9 * self.period:
            raise ValueError("the period must be a whole number of bins")
        self.t = (np.arange(self.n_bins) + 0.5) * self.bin_width
        self.wavelength = time_to_wavelength(self.t, self.flight_path)

    def wavelength_at(self, t):
        return time_to_wavelength(t, self.flight_path)

    def time_at(self, wavelength):
        return wavelength_to_time(wavelength, self.flight_path)

    def convolve(self, x, c):
        """Circular convolution over the period (along the last axis of x)."""
        n = self.n_bins
        return np.fft.irfft(np.fft.rfft(x, axis=-1) * np.fft.rfft(c), n=n, axis=-1)

    def __repr__(self):
        return (f"TimeGrid(flight_path={self.flight_path} m, frequency={self.frequency} Hz, "
                f"bin_width={self.bin_width * 1e6:g} us, n_bins={self.n_bins})")
