"""A sample described by an nbragg TransmissionModel, with some parameters free."""
import numpy as np


class Sample:
    """
    Transmission of a sample from an nbragg ``TransmissionModel``, on the
    time bins of a ``TimeGrid``, as a function of a few free parameters.

    Parameters
    ----------
    model : nbragg.TransmissionModel
        The material model, built in nbragg as usual (cross-section,
        response, background, flight path).
    params : lmfit.Parameters, optional
        Values of all model parameters (default: ``model.params``). They are
        the true values of a simulation and the starting values of a fit.
    free : sequence of str
        The parameters to determine. Names of nbragg parameters (for example
        ``"thickness"``, ``"norm"``, a phase weight such as ``"p1"``) and
        ``"strain"``: a uniform strain of the wavelength axis that moves every
        Bragg edge by d(lambda)/lambda = strain, as a change of the lattice
        parameter does. (nbragg passes lattice parameters to NCrystal with
        four decimals, 35 microstrain for iron, so ``strain`` is the smooth
        way to fit edge positions.)
    substeps : int
        Wavelengths per time bin at which nbragg is evaluated inside
        ``fine_range``; the transmission is averaged over each bin, as the
        detector integrates over its bins (point sampling would make Bragg
        edges jump from bin to bin). Use nbragg's ``jorgensen_inv`` response,
        whose width does not depend on the sampling.
    fine_range : (float, float)
        Wavelength range [Angstrom] with Bragg edges of interest.

    Examples
    --------
    >>> import nbragg, frame_overlap as fo                      # doctest: +SKIP
    >>> xs = nbragg.CrossSection(materials={"iron": {"mat": "Fe_sg229_Iron-alpha.ncmat"}})
    >>> model = nbragg.TransmissionModel(xs, response="jorgensen_inv", tof_length=9.0)
    >>> model.params["thickness"].set(value=2.0)
    >>> sample = fo.Sample(model, free=["thickness", "strain"])
    """

    def __init__(self, model, params=None, free=("thickness", "strain"), substeps=16,
                 fine_range=(0.5, 6.0), steps=None):
        self.model = model
        self.params = (params if params is not None else model.params).copy()
        self.free = list(free)
        for n in self.free:
            if n != "strain" and n not in self.params:
                raise KeyError(f"{n!r} is not a parameter of the model; choose from {list(self.params)}")
        self.substeps = int(substeps)
        self.fine_range = tuple(fine_range)
        self.steps = dict(steps or {})
        self._ref = (None, None)
        self.n_eval = 0

    # ------------------------------------------------------------------ values
    @property
    def theta(self):
        """Current values of the free parameters (strain 0 for the model as given)."""
        return np.array([0.0 if n == "strain" else self.params[n].value for n in self.free], float)

    def step(self, name, value):
        if name in self.steps:
            return self.steps[name]
        if name == "strain":
            return 1e-4
        return max(1e-3 * abs(value), 1e-6)

    def with_values(self, **values):
        """A copy with other parameter values (nbragg names, or strain=...)."""
        s = Sample(self.model, self.params, self.free, self.substeps, self.fine_range, self.steps)
        for k, v in values.items():
            s.params[k].set(value=v)
        return s

    def _nbragg_params(self, theta):
        p = self.params.copy()
        for n, v in zip(self.free, theta):
            if n != "strain":
                p[n].set(value=float(v))
        return p

    def _strain(self, theta):
        return float(theta[self.free.index("strain")]) if "strain" in self.free else 0.0

    # ------------------------------------------------------------ evaluation
    def _layout(self, grid, wl_range):
        key = (id(grid), wl_range)
        lay = getattr(self, "_layouts", {}).get(key)
        if lay is not None:
            return lay
        lam = grid.wavelength
        fine = (lam >= self.fine_range[0]) & (lam <= self.fine_range[1])
        if wl_range is not None:
            use = (lam >= wl_range[0]) & (lam <= wl_range[1])
            fine &= use
        else:
            use = np.ones(grid.n_bins, bool)
        kb = np.where(fine)[0]
        sub = (np.arange(self.substeps) + 0.5) / self.substeps
        wf = grid.wavelength_at(((kb[:, None] + sub[None, :]) * grid.bin_width).ravel())
        lay = dict(use=use, kb=kb, wf=wf, coarse=np.where(use & ~fine)[0])
        self.__dict__.setdefault("_layouts", {})[key] = lay
        return lay

    def transmission(self, grid, theta=None, wl_range=None):
        """
        Transmission of each time bin of the grid for a pulse at t = 0 (the
        pulse pattern is applied to the counts, not here). With
        ``wl_range = (lo, hi)`` only bins in that wavelength range are
        computed (others are 0).
        """
        theta = self.theta if theta is None else np.asarray(theta, float)
        lay = self._layout(grid, wl_range)
        p = self._nbragg_params(theta)
        key = (id(grid), wl_range) + tuple(round(float(p[n].value), 12) for n in self.free if n != "strain")
        if self._ref[0] != key:
            T = np.zeros(grid.n_bins)
            if len(lay["coarse"]):
                T[lay["coarse"]] = np.asarray(self.model.eval(params=p, wl=grid.wavelength[lay["coarse"]]), float)
            Tf = np.asarray(self.model.eval(params=p, wl=lay["wf"]), float) if len(lay["kb"]) else np.zeros(0)
            self._ref = (key, (T, Tf))
            self.n_eval += 1
        T, Tf = self._ref[1]
        T = T.copy()
        if len(lay["kb"]):
            s = 1.0 + self._strain(theta)
            Ts = Tf if s == 1.0 else np.interp(lay["wf"] / s, lay["wf"], Tf)
            T[lay["kb"]] = Ts.reshape(len(lay["kb"]), self.substeps).mean(axis=1)
        return T

    def derivatives(self, grid, theta=None, wl_range=None, central=True):
        """Derivatives of the transmission with respect to the free parameters,
        shape (n_free, n_bins), by finite differences."""
        theta = self.theta if theta is None else np.asarray(theta, float)
        T0 = None if central else self.transmission(grid, theta, wl_range)
        out = []
        for j, n in enumerate(self.free):
            h = self.step(n, theta[j])
            tp = theta.copy()
            tp[j] += h
            if central:
                tm = theta.copy()
                tm[j] -= h
                out.append((self.transmission(grid, tp, wl_range) - self.transmission(grid, tm, wl_range)) / (2 * h))
            else:
                out.append((self.transmission(grid, tp, wl_range) - T0) / h)
        return np.array(out)

    def __repr__(self):
        vals = ", ".join(f"{n}={v:.6g}" for n, v in zip(self.free, self.theta))
        return f"Sample(free: {vals})"
