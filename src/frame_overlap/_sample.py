"""A sample described by an nbragg TransmissionModel, with some parameters free."""
import warnings
from collections import OrderedDict
import numpy as np


class Sample:
    """
    Transmission of a sample from an nbragg ``TransmissionModel``, on the
    time bins of a ``TimeGrid``, as a function of a few free parameters.

    Parameters
    ----------
    model : nbragg.TransmissionModel
        The material model, built in nbragg as usual (cross-section,
        response, background, flight path). Give it a moderator response:
        ``response="jorgensen_inv"`` together with ``vary_response=False``
        (or True), and set its parameters ``α0`` and ``β0``; without
        ``vary_response`` nbragg applies no response at all.
    params : lmfit.Parameters, optional
        Values of all model parameters (default: ``model.params``). They are
        copied: they are the true values of a simulation and the starting
        values of a fit. Change them later with ``with_values`` or through
        ``sample.params`` (not through ``model.params``).
    free : sequence of str
        The parameters to determine. Names of nbragg parameters (for example
        ``"thickness"`` in cm, ``"norm"``, a phase weight such as ``"p1"``)
        and ``"strain"``: a uniform strain of the wavelength axis that moves
        every Bragg edge by d(lambda)/lambda = strain, as a change of the
        lattice parameter does. (nbragg passes lattice parameters to NCrystal
        with four decimals, 35 microstrain for iron, so ``strain`` is the
        smooth way to fit edge positions.)
    substeps : int
        Wavelengths per time bin at which nbragg is evaluated inside
        ``fine_range``; the transmission is averaged over each bin, as the
        detector integrates over its bins (point sampling would make Bragg
        edges jump from bin to bin).
    fine_range : (float, float)
        Wavelength range [Angstrom] with the Bragg edges of interest. Strain
        acts only inside it, so analysis windows must lie inside it.
    margin : float, optional
        Extra wavelength range [Angstrom] evaluated around an analysis window,
        so that the response convolution is complete inside the window.
        Default: from the response width (at least 0.3 A).

    Examples
    --------
    >>> import nbragg, frame_overlap as fo                      # doctest: +SKIP
    >>> xs = nbragg.CrossSection(materials={"iron": {"mat": "Fe_sg229_Iron-alpha.ncmat"}})
    >>> model = nbragg.TransmissionModel(xs, response="jorgensen_inv", tof_length=9.0, vary_response=False)
    >>> model.params["thickness"].set(value=2.0)
    >>> model.params["α0"].set(value=3000.0); model.params["β0"].set(value=50.0)
    >>> sample = fo.Sample(model, free=["thickness", "strain"])
    """

    CACHE = 16

    def __init__(self, model, params=None, free=("thickness", "strain"), substeps=16,
                 fine_range=(0.5, 6.0), steps=None, margin=None, strain=0.0):
        self.model = model
        self.params = (params if params is not None else model.params).copy()
        self.free = list(free)
        if len(set(self.free)) != len(self.free):
            raise ValueError("free parameters must be unique")
        for n in self.free:
            if n != "strain" and n not in self.params:
                raise KeyError(f"{n!r} is not a parameter of the model; choose from {list(self.params)} or 'strain'")
        self.substeps = int(substeps)
        self.fine_range = tuple(float(x) for x in fine_range)
        self.steps = dict(steps or {})
        self.strain = float(strain)
        self._margin = margin
        self._cache = OrderedDict()
        self._layouts = {}
        self.n_eval = 0
        resp = getattr(model, "response", None)
        if resp is None:
            warnings.warn("the nbragg model has no moderator response (pass vary_response=False or True when "
                          "building it): Bragg edges are perfectly sharp and precisions too optimistic", stacklevel=2)
        elif getattr(resp, "kind", "jorgensen_inv") != "jorgensen_inv":
            warnings.warn(f"nbragg response {resp.kind!r} depends on the wavelength sampling; use 'jorgensen_inv' "
                          "with frame_overlap's sub-sampling", stacklevel=2)

    # ------------------------------------------------------------------ values
    @property
    def theta(self):
        """Current values of the free parameters."""
        return np.array([self.strain if n == "strain" else self.params[n].value for n in self.free], float)

    @property
    def margin(self):
        if self._margin is not None:
            return float(self._margin)
        p = self.params
        if "β0" in p and p["β0"].value > 0:
            sig = max((p[k].value for k in ("σ0", "σ1", "σ2") if k in p), default=0.0)
            return max(0.3, 15.0 / p["β0"].value + 8.0 * abs(sig))
        return 0.3

    def bounds(self):
        """(lower, upper) bounds of the free parameters from the lmfit parameters."""
        lo, hi = [], []
        for n in self.free:
            if n == "strain":
                lo.append(-0.1)
                hi.append(0.1)
            else:
                lo.append(self.params[n].min if np.isfinite(self.params[n].min) else -np.inf)
                hi.append(self.params[n].max if np.isfinite(self.params[n].max) else np.inf)
        return np.array(lo, float), np.array(hi, float)

    def step(self, name, value):
        if name in self.steps:
            return self.steps[name]
        if name == "strain":
            return 1e-4
        return max(1e-3 * abs(value), 1e-6)

    def with_values(self, **values):
        """A copy with other parameter values (nbragg names, or strain=...)."""
        s = self._copy()
        for k, v in values.items():
            if k == "strain":
                s.strain = float(v)
            elif k in s.params:
                s.params[k].set(value=v)
            else:
                raise KeyError(f"{k!r} is not a parameter of the model")
        return s

    def _copy(self):
        s = Sample.__new__(Sample)
        s.__dict__.update(self.__dict__)
        s.params = self.params.copy()
        s.free = list(self.free)
        s._cache = OrderedDict()
        s._layouts = {}
        return s

    def check_window(self, window):
        """Raise if an analysis window lies (partly) outside fine_range while
        the strain is free (the strain acts only inside fine_range)."""
        if "strain" in self.free and (window[0] < self.fine_range[0] or window[1] > self.fine_range[1]):
            raise ValueError(f"the analysis window {tuple(window)} A must lie inside fine_range {self.fine_range} A "
                             "when the strain is free; build the Sample with a wider fine_range")

    def _nbragg_params(self, theta):
        p = self.params.copy()
        for n, v in zip(self.free, theta):
            if n != "strain":
                p[n].set(value=float(v))
        return p

    def _strain(self, theta):
        return float(theta[self.free.index("strain")]) if "strain" in self.free else self.strain

    def _fingerprint(self, p):
        return tuple((k, round(float(v.value), 12)) for k, v in p.items())

    # ------------------------------------------------------------ evaluation
    def _layout(self, grid, wl_range):
        key = (grid.key, wl_range)
        if key in self._layouts:
            return self._layouts[key]
        lam = grid.wavelength
        fine = (lam >= self.fine_range[0]) & (lam <= self.fine_range[1])
        use = np.ones(grid.n_bins, bool) if wl_range is None else (lam >= wl_range[0]) & (lam <= wl_range[1])
        fine &= use
        kb = np.where(fine)[0]
        sub = (np.arange(self.substeps) + 0.5) / self.substeps
        wf = grid.wavelength_at(((kb[:, None] + sub[None, :]) * grid.bin_width).ravel())
        coarse = np.where(use & ~fine)[0]
        below, above = coarse[lam[coarse] < self.fine_range[0]], coarse[lam[coarse] > self.fine_range[1]]
        lay = dict(kb=kb, wf=wf, below=below, above=above)
        self._layouts[key] = lay
        return lay

    def _evaluate(self, grid, p, wl_range):
        key = (grid.key, wl_range, self._fingerprint(p))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        lay = self._layout(grid, wl_range)
        T = np.zeros(grid.n_bins)
        for part in (lay["below"], lay["above"]):          # contiguous pieces: nbragg convolves along the array
            if len(part):
                T[part] = np.asarray(self.model.eval(params=p, wl=grid.wavelength[part]), float)
        Tf = np.asarray(self.model.eval(params=p, wl=lay["wf"]), float) if len(lay["kb"]) else np.zeros(0)
        self.n_eval += 1
        self._cache[key] = (T, Tf)
        if len(self._cache) > self.CACHE:
            self._cache.popitem(last=False)
        return T, Tf

    def transmission(self, grid, theta=None, wl_range=None):
        """
        Transmission of each time bin of the grid for a pulse at t = 0 (the
        pulse pattern is applied to the counts, not here). With
        ``wl_range = (lo, hi)`` only bins in that wavelength range are
        computed (others are 0).
        """
        theta = self.theta if theta is None else np.asarray(theta, float)
        if wl_range is not None:
            wl_range = (round(float(wl_range[0]), 9), round(float(wl_range[1]), 9))
        lay = self._layout(grid, wl_range)
        T, Tf = self._evaluate(grid, self._nbragg_params(theta), wl_range)
        T = T.copy()
        if len(lay["kb"]):
            s = 1.0 + self._strain(theta)
            Ts = Tf if s == 1.0 else np.interp(lay["wf"] / s, lay["wf"], Tf)
            T[lay["kb"]] = Ts.reshape(len(lay["kb"]), self.substeps).mean(axis=1)
        return T

    def derivatives(self, grid, theta=None, wl_range=None, central=True):
        """Derivatives of the transmission with respect to the free parameters,
        shape (n_free, n_bins), by finite differences (one-sided at bounds)."""
        theta = self.theta if theta is None else np.asarray(theta, float)
        lo, hi = self.bounds()
        T0 = self.transmission(grid, theta, wl_range)
        out = []
        for j, n in enumerate(self.free):
            h = self.step(n, theta[j])
            up, down = theta[j] + h <= hi[j], theta[j] - h >= lo[j]
            tp, tm = theta.copy(), theta.copy()
            tp[j] += h
            tm[j] -= h
            if central and up and down:
                out.append((self.transmission(grid, tp, wl_range) - self.transmission(grid, tm, wl_range)) / (2 * h))
            elif up:
                out.append((self.transmission(grid, tp, wl_range) - T0) / h)
            else:
                out.append((T0 - self.transmission(grid, tm, wl_range)) / h)
        return np.array(out)

    def __repr__(self):
        vals = ", ".join(f"{n}={v:.6g}" for n, v in zip(self.free, self.theta))
        return f"Sample(free: {vals})"
