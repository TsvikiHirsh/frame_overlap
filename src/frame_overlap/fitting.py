"""
Analysis of overlapped counts: the Rietveld fit (forward model through the
pulse pattern) and Wiener deconvolution as in frame-overlap Bragg-edge
imaging (FOBI).
"""
import numpy as np
import pandas as pd


# ------------------------------------------------------------------- window

class Window:
    """
    Split of the time after a pulse into the analysis window
    [wavelength_min, wavelength_max], where the transmission is the nbragg
    model, and the rest, where it is a free piecewise-linear function of time
    (nuisance values at knots, held constant beyond the last knot). The
    neutrons outside the window are still modelled where they overlap other
    pulses, but the physics parameters take no information from them, and
    model errors there cannot bias them.
    """

    def __init__(self, grid, wavelength_min=1.0, wavelength_max=5.0, knots=None, last=None, tail=None):
        t = grid.t
        lam = grid.wavelength
        self.wavelength_min, self.wavelength_max = wavelength_min, wavelength_max
        self.inside = (lam >= wavelength_min) & (lam <= wavelength_max)
        t1, t2 = t[self.inside][0], t[self.inside][-1]
        if knots is None:
            # dense near the window edges (the pulse width smears counts from
            # just outside the window into it), coarser far from it
            dt = grid.bin_width
            lo = np.r_[t[:6], np.geomspace(t[6], 0.6 * t1, 8), np.arange(0.6 * t1, t1 - dt, 100e-6)]
            lo = np.r_[lo[lo < t1 - dt - 0.5 * dt], t1 - dt]
            t_end = last if last is not None else min(grid.period - dt, 2.2 * t2)
            near = np.arange(t2 + dt, min(t2 + 2e-3, t_end), 200e-6)
            far = np.geomspace(near[-1] + 200e-6, t_end, 10) if t_end > near[-1] + 200e-6 else []
            knots = np.r_[lo, near, far]
        self.knots = np.unique(np.asarray(knots, float))
        n_lo = int(np.sum(self.knots < t1))
        B = np.zeros((len(self.knots), grid.n_bins))
        for side, idx in ((t < t1, range(0, n_lo)), (t > t2, range(n_lo, len(self.knots)))):
            kn = self.knots[list(idx)]
            for j, k in enumerate(idx):
                e = np.zeros(len(kn))
                e[j] = 1.0
                B[k, side] = np.interp(t[side], kn, e)
        if tail is not None:
            # beyond the last knot: the model's smooth shape (absorption) scaled
            # by the last free value, instead of a constant
            far = t > self.knots[-1]
            T_last = np.interp(self.knots[-1], t, tail)
            B[-1, far] = np.asarray(tail, float)[far] / max(T_last, 1e-12)
        self.B = B

    @property
    def range(self):
        return (self.wavelength_min - 0.25, self.wavelength_max + 0.25)

    def centre(self, sample, grid, theta):
        """Model transmission at the knots (the prior centre and start of the
        nuisance values)."""
        T = sample.transmission(grid, theta)
        return np.interp(self.knots, grid.t, T)

    def T(self, T_in, beta):
        return np.where(self.inside, T_in, 0.0) + beta @ self.B


# ------------------------------------------------------------------- result

class FitResult:
    """Result of a fit: values, standard errors (inverse Fisher matrix at the
    solution), deviance, model counts, and what the method produced."""

    def __init__(self, method, names, theta, sigma, deviance, success, nfev, data, sample, **extra):
        self.method, self.names = method, list(names)
        self.theta, self.sigma = np.asarray(theta, float), np.asarray(sigma, float)
        self.deviance, self.success, self.nfev = float(deviance), bool(success), int(nfev)
        self.data, self.sample = data, sample
        self.extra = extra

    @property
    def values(self):
        return dict(zip(self.names, self.theta))

    @property
    def errors(self):
        return dict(zip(self.names, self.sigma))

    def to_dataframe(self):
        rows = []
        for n, v, s in zip(self.names, self.theta, self.sigma):
            row = dict(parameter=n, value=v, error=s)
            if self.data is not None and self.data.truth and n in self.data.truth:
                row["truth"] = self.data.truth[n]
            if n == "strain":
                row.update(value_microstrain=v * 1e6, error_microstrain=s * 1e6)
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self):
        df = self.to_dataframe()
        stat = "deviance" if "model_counts" in self.extra else "chi2"
        head = f"{self.method}: {stat} per bin {self.deviance / self.data.grid.n_bins:.3f}, " \
               f"{'converged' if self.success else 'NOT converged'} after {self.nfev} evaluations"
        return head + "\n" + df.to_string(index=False)

    def plot(self, ax=None):
        """Counts and fitted counts (Rietveld fit), or the recovered
        transmission and the fitted model (Wiener)."""
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        g = self.data.grid
        if "model_counts" in self.extra:
            ax.plot(g.t * 1e3, self.data.counts, color="k", lw=0.35, label="counts")
            ax.plot(g.t * 1e3, self.extra["model_counts"], color="#0072B2", lw=0.8, label=self.method)
            ax.set_xlabel("Time in the period (ms)")
            ax.set_ylabel(f"Counts per {g.bin_width * 1e6:g} µs")
        else:
            sel = self.extra["selection"]
            ax.plot(g.wavelength[sel], self.extra["transmission"][sel], color="#D55E00", lw=0.5, label="recovered")
            ax.plot(g.wavelength[sel], self.extra["model_transmission"][sel], color="#555555", lw=1.0, label="fit")
            ax.set_xlabel("Wavelength (Å)")
            ax.set_ylabel("Transmission")
        ax.legend()
        return ax

    def __repr__(self):
        vals = ", ".join(f"{n}={v:.6g}±{s:.2g}" for n, v, s in zip(self.names, self.theta, self.sigma))
        return f"FitResult({self.method}: {vals})"


def _dev_resid(y, mu):
    mu = np.clip(mu, 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    dev = np.clip(2.0 * (mu - y + term), 0, None)
    r = np.sign(y - mu) * np.sqrt(dev)
    with np.errstate(divide="ignore", invalid="ignore"):
        drdmu = np.where(np.abs(r) > 1e-9, (1.0 - y / mu) / r, -1.0 / np.sqrt(mu))
    return r, drdmu


# ------------------------------------------------------------------- Rietveld

class RietveldFit:
    """
    Rietveld fit of the (overlapped) counts: the nbragg model is passed
    through the known pulse pattern,

        mu_theta(t) = n_periods * [c (*) (o T_theta)](t),

    and compared with the counts by the Poisson likelihood. Works for any
    pattern, including one pulse per period and patterns whose Fourier
    transform has zeros. Within ``window`` the transmission is the nbragg
    model; outside it a free piecewise-linear function with a weak prior
    (``prior_sigma``) centred on the model at the starting values.

    Parameters
    ----------
    window : (float, float)
        Analysis window [Angstrom].
    prior_sigma : float
        Width of the prior on the free transmission outside the window.
    max_nfev : int
        Maximum number of function evaluations of the solver.
    bounds : dict, optional
        {name: (lo, hi)} for the free parameters (default: thickness >= 0).
    """

    def __init__(self, window=(1.0, 5.0), prior_sigma=1.0, max_nfev=200, bounds=None, knots=None, tol=1e-10):
        self.window, self.prior_sigma, self.max_nfev = tuple(window), float(prior_sigma), int(max_nfev)
        self.tol = float(tol)
        self.bounds = dict(bounds or {})
        self.knots = knots

    def window_for(self, grid, sample=None, theta=None):
        """The analysis window; with a sample, the far tail beyond the last knot
        follows the model's shape at theta (scaled by the last free value)."""
        tail = None if sample is None else sample.transmission(grid, theta)
        return Window(grid, *self.window, knots=self.knots, tail=tail)

    def fit(self, data, sample, theta0=None):
        from scipy.optimize import least_squares
        grid = data.grid
        th0 = sample.theta if theta0 is None else np.asarray(theta0, float)
        win = self.window_for(grid, sample, th0)
        c = data.pattern.array(grid)
        Cf = np.fft.rfft(c)
        conv = lambda x: np.fft.irfft(np.fft.rfft(x, axis=-1) * Cf, n=grid.n_bins, axis=-1)
        ob = data.n_periods * data.beam.rate
        y = data.counts
        nth = len(th0)
        beta0 = win.centre(sample, grid, th0)
        nb = len(beta0)
        rng = win.range
        sp = self.prior_sigma

        def T_of(x):
            return win.T(sample.transmission(grid, x[:nth], rng), x[nth:])

        def fun(x):
            r = _dev_resid(y, conv(ob * T_of(x)))[0]
            return np.r_[r, (x[nth:] - beta0) / sp]

        def jac(x):
            _, drdmu = _dev_resid(y, conv(ob * T_of(x)))
            d = np.vstack([sample.derivatives(grid, x[:nth], rng, central=False) * win.inside, win.B])
            J = (drdmu[None, :] * conv(ob * d)).T
            prior = np.hstack([np.zeros((nb, nth)), np.eye(nb) / sp])
            return np.vstack([J, prior])

        lo = [self.bounds.get(n, (0.0, np.inf) if n == "thickness" else (-np.inf, np.inf))[0] for n in sample.free]
        hi = [self.bounds.get(n, (0.0, np.inf) if n == "thickness" else (-np.inf, np.inf))[1] for n in sample.free]
        lb = np.r_[lo, np.full(nb, -0.5)]
        ub = np.r_[hi, np.full(nb, 1.5)]
        x0 = np.clip(np.r_[th0, beta0], lb + 1e-12, ub - 1e-12)
        # tight tolerances: the cost includes the misfit of the free transmission
        # outside the window, which does not depend on theta, so relative
        # tolerances would stop the solver early
        r = least_squares(fun, x0, jac=jac, method="trf", x_scale="jac", bounds=(lb, ub), max_nfev=self.max_nfev,
                          ftol=self.tol, xtol=self.tol, gtol=self.tol)
        mu = conv(ob * T_of(r.x))
        d = np.vstack([sample.derivatives(grid, r.x[:nth], rng) * win.inside, win.B])
        J = conv(ob * d)
        F = (J / np.clip(mu, 1e-12, None)) @ J.T
        F[nth:, nth:] += np.eye(nb) / sp ** 2
        sig = np.sqrt(np.diag(np.linalg.inv(F))[:nth])
        parts = [data.n_periods * grid.convolve(data.beam.rate * T_of(r.x), data.pattern.__class__(
            [s], w, a).array(grid)) for s, w, a in zip(data.pattern.starts, data.pattern.widths, data.pattern.intensities)]
        return FitResult("Rietveld fit", sample.free, r.x[:nth], sig, np.sum(r.fun[:grid.n_bins] ** 2),
                         r.status > 0, r.nfev, data, sample, model_counts=mu, contributions=parts,
                         nuisance=r.x[nth:], window=win)


# ------------------------------------------------------------------- Wiener

def wiener_filter(delta, noise_power):
    G = np.fft.fft(delta)
    return np.conj(G) / (np.abs(G) ** 2 + noise_power)


def wiener(y, delta, noise_power):
    """Wiener-deconvolved spectrum and its per-bin variance (Poisson y)."""
    H = wiener_filter(delta, noise_power)
    x = np.real(np.fft.ifft(np.fft.fft(y) * H))
    g = np.real(np.fft.ifft(H))
    var = np.real(np.fft.ifft(np.fft.fft(np.clip(y, 0, None)) * np.fft.fft(g ** 2)))
    return x, np.clip(var, 1e-12, None)


class WienerFit:
    """
    Wiener deconvolution as in frame-overlap Bragg-edge imaging (FOBI, Busi
    et al. 2020) followed by a fit of the nbragg model to the recovered
    transmission. The sample counts and the open beam measured with the same
    pattern are deconvolved with the same filter, the transmission is their
    ratio, and it is fitted in ``window`` with independent bins.

    Parameters
    ----------
    noise_power : float or "optimal"
        Constant regularisation of the filter (as in FOBI), or the optimal
        filter eps(f) = noise power / signal power estimated from the model
        at the starting values.
    """

    def __init__(self, noise_power=1e-3, window=(1.0, 5.0), max_nfev=200, tol=1e-10):
        self.noise_power, self.window, self.max_nfev = noise_power, tuple(window), int(max_nfev)
        self.tol = float(tol)

    def fit(self, data, sample, theta0=None):
        from scipy.ndimage import uniform_filter1d
        from scipy.optimize import least_squares
        grid = data.grid
        pat = data.pattern
        box = pat.__class__([0.0], pat.widths[0]).array(grid)
        th0 = sample.theta if theta0 is None else np.asarray(theta0, float)
        rate = data.beam.rate
        n = data.n_periods
        y = data.counts
        O = data.open_counts()
        delta = pat.delta(grid)
        npow = self.noise_power
        if isinstance(npow, str):
            s1 = n * grid.convolve(rate * sample.transmission(grid, th0), box)
            npow = y.sum() / np.clip(np.abs(np.fft.fft(s1)) ** 2, 1e-30, None)
        if pat.n_pulses == 1:
            shat, var, ohat = y, np.clip(uniform_filter1d(y, 21, mode="wrap"), 1, None), O
        else:
            shat, _ = wiener(y, delta, npow)
            _, var = wiener(uniform_filter1d(y, 21, mode="wrap"), delta, npow)
            ohat, _ = wiener(O, delta, npow)
        lam = grid.wavelength
        sel = (lam >= self.window[0]) & (lam <= self.window[1]) & (ohat > 0)
        sd = np.sqrt(var)
        rng = (self.window[0] - 0.25, self.window[1] + 0.25)
        ob1 = grid.convolve(rate, box)
        Tbox = lambda T: grid.convolve(rate * T, box) / np.clip(ob1, 1e-30, None)

        def fun(th):
            return ((shat - ohat * Tbox(sample.transmission(grid, th, rng))) / sd)[sel]

        def jac(th, central=False):
            d = sample.derivatives(grid, th, rng, central=central)
            return -((ohat * grid.convolve(rate * d, box) / np.clip(ob1, 1e-30, None))[:, sel] / sd[sel]).T

        lo = [0.0 if nm == "thickness" else -np.inf for nm in sample.free]
        r = least_squares(fun, th0, jac=jac, method="trf", x_scale="jac", bounds=(lo, np.inf), max_nfev=self.max_nfev,
                          ftol=self.tol, xtol=self.tol, gtol=self.tol)
        J = jac(r.x, central=True)
        sig = np.sqrt(np.diag(np.linalg.inv(J.T @ J)))
        That = shat / np.where(np.abs(ohat) > 0, ohat, np.nan)
        return FitResult("Wiener deconvolution", sample.free, r.x, sig, np.sum(r.fun ** 2), r.status > 0, r.nfev,
                         data, sample, transmission=That,
                         model_transmission=Tbox(sample.transmission(grid, r.x)), selection=sel)
