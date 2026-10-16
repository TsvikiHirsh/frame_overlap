"""Pulse patterns (the 'kernel' of frame overlap): when the chopper opens."""
import numpy as np

POLDI_SLITS_DEG = (0, 9.363, 21.475, 37.039, 50.417, 56.664, 67.422, 75.406)


class Pattern:
    """
    Pulses of one period: start times [s], widths [s] and relative
    intensities (1 = the full beam while open).

    Build patterns with the constructors ``single``, ``equal``, ``times``,
    ``random``, ``maximum_length``, ``quadratic_residue`` and ``poldi``, or
    optimise one for a measurement with ``frame_overlap.optimise_pattern``.

    Examples
    --------
    >>> Pattern.equal(4, width=200e-6, period=50e-3).times_ms
    [0.0, 12.5, 25.0, 37.5]
    """

    def __init__(self, starts, width=200e-6, intensity=1.0, name=None):
        starts = np.atleast_1d(np.asarray(starts, float))
        self.starts = starts
        self.widths = np.broadcast_to(np.asarray(width, float), starts.shape).copy()
        self.intensities = np.broadcast_to(np.asarray(intensity, float), starts.shape).copy()
        self.name = name

    # -------------------------------------------------------------- builders
    @classmethod
    def single(cls, width=200e-6):
        return cls([0.0], width, name=f"one pulse of {width * 1e6:g} us")

    @classmethod
    def times(cls, starts, width=200e-6, intensity=1.0):
        return cls(starts, width, intensity, name=f"{len(np.atleast_1d(starts))} pulses")

    @classmethod
    def equal(cls, n, width=200e-6, period=50e-3):
        """n equally spaced pulses (n times the repetition rate of the period)."""
        return cls(np.arange(n) * period / n, width, name=f"{n} equally spaced pulses")

    @classmethod
    def random(cls, n, width=200e-6, period=50e-3, seed=None, bin_width=10e-6):
        """n pulses at random times, the first at 0, none overlapping."""
        rng = np.random.default_rng(seed)
        free = period - n * width
        if free < 0:
            raise ValueError("the pulses do not fit into the period")
        v = np.sort(np.round(rng.uniform(0, free, n - 1) / bin_width) * bin_width)
        return cls(np.r_[0.0, np.arange(1, n) * width + v], width, name=f"{n} random pulses")

    @classmethod
    def maximum_length(cls, order, width=200e-6, period=50e-3, taps=None):
        """Maximum-length sequence of 2^order - 1 slots: a pulse at the start of
        every slot that holds a one (2^(order-1) pulses). Its Fourier transform
        is flat apart from zero frequency, as for correlation choppers."""
        primitive = {2: 1, 3: 1, 4: 1, 5: 2, 6: 1, 7: 1}
        k = taps or primitive[order]
        reg = [1] + [0] * (order - 1)
        bits = []
        for _ in range(2 ** order - 1):
            bits.append(reg[-1])
            reg = [reg[order - 1] ^ reg[k - 1]] + reg[:-1]
        slots = np.where(bits)[0]
        return cls(slots * period / (2 ** order - 1), width, name=f"maximum-length sequence, {len(slots)} pulses")

    @classmethod
    def quadratic_residue(cls, prime, width=200e-6, period=50e-3, complement=False):
        """Quadratic-residue sequence of ``prime`` slots (prime = 3 mod 4):
        a flat-spectrum sequence with (prime - 1)/2 pulses, or (prime + 1)/2
        for the complement."""
        if prime % 4 != 3:
            raise ValueError("prime must be 3 mod 4")
        q = {x * x % prime for x in range(1, prime)}
        slots = sorted(set(range(prime)) - q) if complement else sorted(q)
        return cls(np.array(slots) * period / prime, width, name=f"quadratic-residue sequence, {len(slots)} pulses")

    @classmethod
    def poldi(cls, width=200e-6, period=50e-3):
        """The eight slit positions of one quarter of the POLDI chopper (PSI),
        mapped onto the period."""
        return cls(np.array(POLDI_SLITS_DEG) / 90 * period, width, name="POLDI-like sequence")

    # -------------------------------------------------------------- arrays
    def array(self, grid):
        """Open time [s] in each bin of the grid (boxes with exact fractional
        edges, wrapped around the period), weighted by the intensities."""
        n, dt = grid.n_bins, grid.bin_width
        edges = np.arange(n + 1) * dt
        c = np.zeros(n)
        for t0, w, a in zip(self.starts % grid.period, self.widths, self.intensities):
            for lo, hi in _split(t0, t0 + w, grid.period):
                c += a * np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None)
        return c

    def delta(self, grid):
        """Unit deltas at the start bins of the pulses (the kernel of the
        Wiener deconvolution in FOBI)."""
        c = np.zeros(grid.n_bins)
        for t0, a in zip(self.starts, self.intensities):
            c[int(round((t0 % grid.period) / grid.bin_width)) % grid.n_bins] += a
        return c

    # -------------------------------------------------------------- info
    @property
    def n_pulses(self):
        return len(self.starts)

    @property
    def times_ms(self):
        return [round(float(x) * 1e3, 4) for x in self.starts]

    def duty_cycle(self, period):
        return float(np.sum(self.widths * self.intensities) / period)

    def overlaps(self, period):
        """True if two pulses of the pattern overlap in time."""
        x = np.sort(self.starts % period)
        ends = x + self.widths[np.argsort(self.starts % period)]
        return bool(np.any(np.r_[x[1:], x[0] + period] < ends - 1e-12))

    def plot(self, ax=None, period=50e-3, **kw):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        for t0, w, a in zip(self.starts, self.widths, self.intensities):
            ax.axvspan(t0 * 1e3, (t0 + w) * 1e3, ymax=a, color=kw.get("color", "k"))
        ax.set_xlim(0, period * 1e3)
        ax.set_yticks([])
        ax.set_xlabel("Time (ms)")
        return ax

    def __repr__(self):
        name = self.name or "Pattern"
        return f"Pattern({name}: starts {self.times_ms} ms, widths {np.unique(self.widths * 1e6).tolist()} us)"


def _split(lo, hi, period):
    """Split an interval on the circle [0, period) into one or two pieces."""
    if hi <= period:
        return [(lo, hi)]
    return [(lo, period), (0.0, hi - period)]
