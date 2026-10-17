"""Pulse patterns (the 'kernel' of frame overlap): when the chopper opens."""
import numpy as np

POLDI_SLITS_DEG = (0, 9.363, 21.475, 37.039, 50.417, 56.664, 67.422, 75.406)
# primitive trinomials x^n + x^k + 1 for maximum-length sequences
_PRIMITIVE = {2: 1, 3: 1, 4: 1, 5: 2, 6: 1, 7: 1, 9: 4, 10: 3, 11: 2}


def _period(period, grid):
    if grid is not None:
        return grid.period
    if period is None:
        raise TypeError("give the period (seconds) or a TimeGrid: e.g. Pattern.equal(4, period=grid.period)")
    return float(period)


class Pattern:
    """
    Pulses of one period: start times [s], widths [s] and relative
    intensities (1 = the full beam while open).

    Build patterns with the constructors ``single``, ``times``, ``equal``,
    ``random``, ``maximum_length``, ``quadratic_residue`` and ``poldi``, or
    optimise one for a measurement with ``frame_overlap.optimise_pattern``.
    Constructors that place pulses within the period need the period (or a
    TimeGrid).

    Examples
    --------
    >>> Pattern.equal(4, width=200e-6, period=50e-3).times_ms
    [0.0, 12.5, 25.0, 37.5]
    """

    def __init__(self, starts, width=200e-6, intensity=1.0, name=None, period=None):
        starts = np.atleast_1d(np.asarray(starts, float))
        if starts.size == 0:
            raise ValueError("a pattern needs at least one pulse")
        self.starts = starts
        self.widths = np.broadcast_to(np.asarray(width, float), starts.shape).copy()
        self.intensities = np.broadcast_to(np.asarray(intensity, float), starts.shape).copy()
        if np.any(self.widths <= 0) or np.any(self.intensities < 0) or np.any(starts < 0):
            raise ValueError("pulse widths must be positive, intensities and start times non-negative")
        self.name = name
        self.period = period

    # -------------------------------------------------------------- builders
    @classmethod
    def single(cls, width=200e-6):
        """One pulse per period, starting at t = 0."""
        return cls([0.0], width, name=f"one pulse of {width * 1e6:g} us")

    @classmethod
    def times(cls, starts, width=200e-6, intensity=1.0):
        """Pulses at the given start times [s] within the period."""
        return cls(starts, width, intensity, name=f"{len(np.atleast_1d(starts))} pulses")

    @classmethod
    def equal(cls, n, width=200e-6, period=None, grid=None):
        """n equally spaced pulses (n times the repetition rate of the period)."""
        P = _period(period, grid)
        return cls(np.arange(n) * P / n, width, name=f"{n} equally spaced pulses", period=P)

    @classmethod
    def random(cls, n, width=200e-6, period=None, seed=None, bin_width=10e-6, grid=None):
        """n pulses at random times on the bin grid, the first at 0, none overlapping."""
        P = _period(period, grid)
        if grid is not None:
            bin_width = grid.bin_width
        rng = np.random.default_rng(seed)
        free = P - n * width
        if free < 0:
            raise ValueError("the pulses do not fit into the period")
        v = np.sort(np.round(rng.uniform(0, free, n - 1) / bin_width) * bin_width)
        return cls(np.r_[0.0, np.arange(1, n) * width + v], width, name=f"{n} random pulses", period=P)

    @classmethod
    def maximum_length(cls, order, width=200e-6, period=None, taps=None, grid=None):
        """Maximum-length sequence of 2^order - 1 slots: a pulse at the start of
        every slot that holds a one (2^(order-1) pulses). Its Fourier transform
        is flat apart from zero frequency, as for correlation choppers."""
        P = _period(period, grid)
        if taps is None and order not in _PRIMITIVE:
            raise ValueError(f"no built-in primitive polynomial for order {order}; use one of "
                             f"{sorted(_PRIMITIVE)} or give taps=k for x^order + x^k + 1")
        k = taps or _PRIMITIVE[order]
        reg = [1] + [0] * (order - 1)
        bits = []
        for _ in range(2 ** order - 1):
            bits.append(reg[-1])
            reg = [reg[order - 1] ^ reg[k - 1]] + reg[:-1]
        slots = np.where(bits)[0]
        if len(slots) != 2 ** (order - 1):
            raise ValueError("taps do not give a primitive polynomial")
        return cls(slots * P / (2 ** order - 1), width, name=f"maximum-length sequence, {len(slots)} pulses", period=P)

    @classmethod
    def quadratic_residue(cls, prime, width=200e-6, period=None, complement=False, grid=None):
        """Quadratic-residue sequence of ``prime`` slots (prime = 3 mod 4):
        a flat-spectrum sequence with (prime - 1)/2 pulses, or (prime + 1)/2
        for the complement."""
        P = _period(period, grid)
        if prime % 4 != 3 or any(prime % q == 0 for q in range(2, int(prime ** 0.5) + 1)):
            raise ValueError("prime must be a prime number equal to 3 mod 4")
        q = {x * x % prime for x in range(1, prime)}
        slots = sorted(set(range(prime)) - q) if complement else sorted(q)
        return cls(np.array(slots) * P / prime, width, name=f"quadratic-residue sequence, {len(slots)} pulses", period=P)

    @classmethod
    def poldi(cls, width=200e-6, period=None, grid=None):
        """The eight slit positions of one quarter of the POLDI chopper (PSI),
        mapped onto the period."""
        P = _period(period, grid)
        return cls(np.array(POLDI_SLITS_DEG) / 90 * P, width, name="POLDI-like sequence", period=P)

    # -------------------------------------------------------------- checks
    def check(self, grid):
        """Raise a ValueError if the pattern does not fit the grid."""
        if self.period is not None and abs(self.period - grid.period) > 1e-12:
            raise ValueError(f"the pattern was built for a period of {self.period * 1e3:g} ms, "
                             f"the grid has {grid.period * 1e3:g} ms")
        if np.any(self.starts >= grid.period):
            raise ValueError(f"pulse start times must lie within the period [0, {grid.period:g} s); "
                             f"got up to {self.starts.max():g} s (times are in seconds)")
        if np.any(self.widths >= grid.period):
            raise ValueError("a pulse is longer than the period")
        if self.overlaps(grid.period):
            raise ValueError("pulses of the pattern overlap in time; merge them into one longer pulse")

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
        """Deltas at the pulse starts (the kernel of the Wiener deconvolution in
        FOBI), split linearly between the two nearest bin edges for starts off
        the grid."""
        c = np.zeros(grid.n_bins)
        for t0, a in zip(self.starts % grid.period, self.intensities):
            x = t0 / grid.bin_width
            k = int(np.floor(x))
            f = x - k
            c[k % grid.n_bins] += a * (1 - f)
            c[(k + 1) % grid.n_bins] += a * f
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
        order = np.argsort(self.starts % period)
        x = (self.starts % period)[order]
        ends = x + self.widths[order]
        return bool(np.any(np.r_[x[1:], x[0] + period] < ends - 1e-12))

    def plot(self, ax=None, period=None, **kw):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        for t0, w, a in zip(self.starts, self.widths, self.intensities):
            ax.axvspan(t0 * 1e3, (t0 + w) * 1e3, ymax=min(a, 1.0), color=kw.get("color", "k"))
        P = period or self.period
        if P:
            ax.set_xlim(0, P * 1e3)
        ax.set_yticks([])
        ax.set_xlabel("Time in the period (ms)")
        return ax

    def __repr__(self):
        name = self.name or "Pattern"
        return f"Pattern({name}: starts {self.times_ms} ms, widths {np.unique(self.widths * 1e6).tolist()} us)"


def _split(lo, hi, period):
    """Split an interval on the circle [0, period) into one or two pieces."""
    if hi <= period:
        return [(lo, hi)]
    return [(lo, period), (0.0, hi - period)]
