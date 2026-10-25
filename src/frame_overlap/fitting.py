"""
Analysis of overlapped counts: the Rietveld fit (forward model through the
pulse pattern) and Wiener deconvolution as in frame-overlap Bragg-edge
imaging (FOBI).
"""
import warnings
import numpy as np
import pandas as pd


# ------------------------------------------------------------------- window

class Window:
    """
    Split of the time after a pulse into the analysis window
    [wavelength_min, wavelength_max], where the transmission is the nbragg
    model, and the rest, where it is a free piecewise-linear function of time
    (nuisance values at knots). Beyond the last knot it follows the model's
    smooth shape (``tail``), scaled by the last free value, or is constant.
    The neutrons outside the window are still modelled where they overlap
    other pulses, but the physics parameters take no information from them,
    and model errors there cannot bias them.
    """

    def __init__(self, grid, wavelength_min=1.0, wavelength_max=5.0, knots=None, last=None, tail=None):
        if not wavelength_min < wavelength_max:
            raise ValueError("the window needs wavelength_min < wavelength_max")
        if wavelength_min < grid.wavelength[0] or wavelength_max > grid.wavelength[-1]:
            raise ValueError(f"the window {wavelength_min}-{wavelength_max} A is not inside the wavelengths of the "
                             f"grid ({grid.wavelength[0]:.3f}-{grid.wavelength[-1]:.2f} A)")
        t = grid.t
        lam = grid.wavelength
        self.wavelength_min, self.wavelength_max = float(wavelength_min), float(wavelength_max)
        self.inside = (lam >= wavelength_min) & (lam <= wavelength_max)
        if self.inside.sum() < 10:
            raise ValueError("the window holds fewer than 10 time bins")
        t1, t2 = t[self.inside][0], t[self.inside][-1]
        dt = grid.bin_width
        if knots is None:
            # dense near the window edges (the pulse width smears counts from
            # just outside the window into it), coarser far from it
            lo = np.r_[t[:6], np.geomspace(t[6], max(0.6 * t1, t[7]), 8), np.arange(0.6 * t1, t1 - dt, 100e-6)]
            lo = np.r_[lo[lo < t1 - 1.5 * dt], t1 - dt] if t1 - dt > t[0] else np.array([])
            t_end = last if last is not None else min(grid.period - dt, 2.2 * t2)
            near = np.arange(t2 + dt, min(t2 + 2e-3, t_end), 200e-6) if t_end > t2 + dt else np.array([t2 + dt])
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
        if tail is not None and len(self.knots) > n_lo:
            far = t > self.knots[-1]
            T_last = np.interp(self.knots[-1], t, tail)
            B[-1, far] = np.asarray(tail, float)[far] / max(T_last, 1e-12)
        self.B = B

    def range_for(self, sample):
        """Wavelength range evaluated by nbragg: the window plus the sample's margin."""
        m = sample.margin
        return (self.wavelength_min - m, self.wavelength_max + m)

    def centre(self, sample, grid, theta):
        """Model transmission at the knots (the prior centre and start of the
        nuisance values)."""
        T = sample.transmission(grid, theta)
        return np.interp(self.knots, grid.t, T)

    def T(self, T_in, beta):
        return np.where(self.inside, T_in, 0.0) + beta @ self.B


# ------------------------------------------------------------------- result

class FitResult:
    """
    Result of ``TransmissionModel.fit``, with the attributes of an lmfit
    ModelResult as in nbragg: ``params`` (lmfit Parameters with the fitted
    values and standard errors, the strain included), ``fit_report()``,
    ``plot()``, ``chisqr`` (the deviance of the counts for the Rietveld fit,
    chi^2 of the recovered spectrum for Wiener deconvolution), ``redchi`` (per
    bin), ``best_fit`` (expected counts), ``residual``, ``success`` and
    ``nfev``. ``transmission()`` returns the transmission recovered from the
    counts. Standard errors include the counting statistics of the data
    (``count_variance``).
    """

    def __init__(self, method, names, theta, sigma, covariance, statistic, n_used, success, nfev, data, sample,
                 **extra):
        self.method, self.names = method, list(names)
        self.theta, self.sigma = np.asarray(theta, float), np.asarray(sigma, float)
        self.covariance = covariance
        self.statistic, self.n_used = float(statistic), int(n_used)
        self.success, self.nfev = bool(success), int(nfev)
        self.data, self.sample = data, sample
        self.extra = extra

    # ---------------------------------------------------------------- values
    @property
    def values(self):
        return {n: float(v) for n, v in zip(self.names, self.theta)}

    @property
    def errors(self):
        return {n: float(s) for n, s in zip(self.names, self.sigma)}

    @property
    def chisqr(self):
        return self.statistic

    deviance = chisqr

    @property
    def redchi(self):
        return self.statistic / max(self.n_used, 1)

    @property
    def best_fit(self):
        """Expected counts of the fit (Rietveld), or the fitted transmission (Wiener)."""
        return self.extra.get("model_counts", self.extra.get("model_transmission"))

    @property
    def residual(self):
        return self.residuals()

    @property
    def params(self):
        """The model's lmfit Parameters with the fitted values and standard errors (strain included)."""
        import lmfit
        p = self.sample.params.copy()
        if "strain" not in p:
            p.add("strain", value=float(self.sample.strain), vary="strain" in self.names)
        for n, v, s in zip(self.names, self.theta, self.sigma):
            if n in p:
                p[n].set(value=float(v))
                p[n].stderr = float(s)
            else:                                          # background, t0, pulse intensities
                p.add(n, value=float(v))
                p[n].stderr = float(s)
        return p

    def to_dataframe(self):
        """Values and standard errors of the determined quantities (strain also in microstrain)."""
        rows = []
        for n, v, s in zip(self.names, self.theta, self.sigma):
            row = dict(parameter=n, value=float(v), error=float(s))
            if self.data is not None and self.data.truth and n in self.data.truth:
                row["truth"] = self.data.truth[n]
            if n == "strain":
                row.update(value_microstrain=v * 1e6, error_microstrain=s * 1e6)
            rows.append(row)
        return pd.DataFrame(rows)

    def fit_report(self):
        stat = "deviance" if self.method.startswith("Rietveld") else "chi-square"
        F = getattr(self.data, "count_variance", 1.0)
        lines = [f"[[{self.method}]]",
                 f"    {'converged' if self.success else 'NOT converged'} after {self.nfev} evaluations",
                 f"    {stat} per bin = {self.redchi:.3f} ({self.n_used} bins)"
                 + (f"; expected {F:g} for count_variance {F:g}" if F != 1.0 else ""),
                 "[[Variables]]"]
        for n, v, s in zip(self.names, self.theta, self.sigma):
            tr = ""
            if self.data is not None and self.data.truth and n in self.data.truth:
                tr = f"   (truth {self.data.truth[n]:.6g})"
            unit = f"  = {v * 1e6:.1f} +/- {s * 1e6:.1f} microstrain" if n == "strain" else ""
            lines.append(f"    {n:12s} {v:12.6g} +/- {s:.3g}{unit}{tr}")
        if "note" in self.extra:
            lines.append(f"    note: {self.extra['note']}")
        return "\n".join(lines)

    summary = fit_report

    def __str__(self):
        return self.fit_report()

    def residuals(self):
        """Deviance residuals of the counts (Rietveld fit), or normalised
        residuals of the recovered transmission in the window (Wiener)."""
        if "model_counts" in self.extra:
            return _dev_resid(self.data.counts, self.extra["model_counts"])[0]
        return self.extra["residuals"]

    def _model_transmission(self):
        """The fitted transmission of one pulse, averaged over the pulse as the
        recovered spectrum is (single-pulse open beam as weight)."""
        g = self.data.grid
        box = self.data.pattern.__class__([0.0], self.data.pattern.widths[0]).array(g)
        rate = self.data.openbeam.rate
        T = self.extra.get("fitted_transmission")
        if T is None:
            return self.extra["model_transmission"]
        return g.convolve(rate * T, box) / np.clip(g.convolve(rate, box), 1e-30, None)

    def transmission(self):
        """
        Transmission spectrum recovered from the data, for a pulse at t = 0,
        as a DataFrame (wavelength, trans, err, model): for the Rietveld fit
        the counts of each bin are divided between the pulses in proportion
        to their fitted contributions and moved back to their pulse (as
        Rietveld programs divide the counts of overlapping reflections); for
        Wiener deconvolution it is the ratio of the deconvolved sample and
        open beam. ``model`` is the fitted transmission averaged in the same way.
        """
        g = self.data.grid
        F = getattr(self.data, "count_variance", 1.0)
        if "transmission" in self.extra:
            T = self.extra["transmission"]
            err = self.extra.get("transmission_sigma", np.full_like(T, np.nan))
        else:
            y = self.data.counts - self.data.background * self.data.time
            parts = self.extra["contributions"]
            mu = np.clip(sum(parts), 1e-12, None)
            num, var, ob = np.zeros(g.n_bins), np.zeros(g.n_bins), np.zeros(g.n_bins)
            n = self.data.n_periods
            pat = self.data.pattern
            for (s, w, a), part in zip(zip(pat.starts, pat.widths, pat.intensities), parts):
                share = part / mu
                k = int(round(s / g.bin_width)) % g.n_bins
                num += np.roll(y * share, -k)
                var += np.roll(np.clip(mu, 0, None) * share ** 2, -k)
                ob += n * a * g.convolve(self.data.openbeam.rate, pat.__class__([0.0], w).array(g))
            ob = np.clip(ob, 1e-12, None)
            T, err = num / ob, np.sqrt(F * var) / ob
        return pd.DataFrame(dict(wavelength=g.wavelength, trans=T, err=err, model=self._model_transmission()))

    def plot(self, kind="transmission", wlmin=None, wlmax=None, ax=None):
        """
        nbragg-style plot of the fit. ``kind="transmission"`` (default): the
        transmission recovered from the counts with the fitted model (grey)
        and, below, the residuals in units of the standard deviation of each
        bin with chi^2 per bin. ``kind="counts"``: the counts, the fitted
        expected counts and the contribution of each pulse.
        Returns the matplotlib axes.
        """
        import matplotlib.pyplot as plt
        colour = "#0072B2" if self.method.startswith("Rietveld") else "#D55E00"
        g = self.data.grid
        if kind == "counts":
            ax = ax or plt.figure(figsize=(8, 3.2)).gca()
            ax.plot(g.t * 1e3, self.data.counts, color="k", lw=0.35, label="counts")
            if "contributions" in self.extra and len(self.extra["contributions"]) > 1:
                for k, part in enumerate(self.extra["contributions"]):
                    ax.fill_between(g.t * 1e3, 0, part, color=("#9ecae1", "#c6dbef")[k % 2], lw=0, alpha=0.8)
            if "model_counts" in self.extra:
                ax.plot(g.t * 1e3, self.extra["model_counts"], color=colour, lw=0.8, label=self.method)
            ax.set_xlabel("Time in the period (ms)")
            ax.set_ylabel(f"Counts per {g.bin_width * 1e6:g} µs")
            ax.set_xlim(0, g.period * 1e3)
            ax.legend(frameon=False)
            return ax
        win = self.extra.get("window_range", (1.0, 5.0))
        lo, hi = wlmin or win[0], wlmax or win[1]
        tab = self.transmission()
        sel = (tab.wavelength >= lo) & (tab.wavelength <= hi) & np.isfinite(tab.trans)
        if ax is None:
            fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True, height_ratios=[3, 1],
                                          gridspec_kw=dict(hspace=0.06))
        else:
            axr = None
        t = tab[sel]
        ax.plot(t.wavelength, t.trans, color=colour, lw=0.5, label="data", zorder=2)
        ax.plot(t.wavelength, t.model, color="#9a9a9a", lw=2.0, label="best fit", zorder=1)
        ax.set_ylabel("Transmission")
        ax.set_title(self.method, loc="left", fontsize=10)
        ax.legend(frameon=False, loc="upper left")
        if axr is not None:
            r = ((t.trans - t.model) / t.err).replace([np.inf, -np.inf], np.nan)
            axr.plot(t.wavelength, r, color=colour, lw=0.4)
            axr.axhline(0, color="#555555", lw=0.5)
            axr.set_ylabel(r"Resid. ($\sigma$)")
            axr.set_xlabel("Wavelength (Å)")
            axr.text(0.99, 0.95, f"$\\chi^2$ per bin {np.nanmean(r ** 2):.2f}", transform=axr.transAxes,
                     ha="right", va="top", fontsize=8)
            lim = max(4.0, np.nanpercentile(np.abs(r), 99.5) * 1.1) if np.isfinite(r).any() else 4.0
            axr.set_ylim(-lim, lim)
        else:
            ax.set_xlabel("Wavelength (Å)")
        ax.set_xlim(lo, hi)
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


def _inverse(F):
    try:
        return np.linalg.inv(F)
    except np.linalg.LinAlgError:
        warnings.warn("the Fisher matrix is singular: some quantities are not determined by the data", stacklevel=3)
        return np.linalg.pinv(F)


# ------------------------------------------------------------------- Rietveld

class RietveldFit:
    """
    Rietveld fit of the (overlapped) counts: the nbragg model is passed
    through the known pulse pattern,

        mu(t) = n_periods * [c (*) (o T_theta)](t) + background * time,

    and compared with the counts by the Poisson likelihood. Works for any
    pattern, including one pulse per period and patterns whose Fourier
    transform has zeros. Within ``window`` the transmission is the nbragg
    model; outside it a free piecewise-linear function with a weak prior
    (``prior_sigma``) centred on the model at the starting values.

    Parameters
    ----------
    window : (float, float)
        Analysis window [Angstrom]. It must lie inside the sample's
        ``fine_range`` when the strain is free.
    prior_sigma : float
        Width of the prior on the free transmission outside the window.
    fit_background : bool
        Also determine a constant count background (counts per bin and second).
    fit_t0 : bool
        Also determine a common time offset of the pulses (seconds).
    fit_intensities : bool
        Also determine the relative intensities of pulses 2..N (pulse 1 fixed).
    max_nfev, tol : solver settings (bounded trust-region least squares).
    """

    def __init__(self, window=(1.0, 5.0), prior_sigma=1.0, fit_background=False, fit_t0=False,
                 fit_intensities=False, max_nfev=200, knots=None, tol=1e-10):
        self.window, self.prior_sigma = tuple(window), float(prior_sigma)
        self.fit_background, self.fit_t0, self.fit_intensities = fit_background, fit_t0, fit_intensities
        self.max_nfev, self.knots, self.tol = int(max_nfev), knots, float(tol)

    def window_for(self, grid, sample=None, theta=None):
        """The analysis window; with a sample, the far tail beyond the last knot
        follows the model's shape at theta (scaled by the last free value)."""
        tail = None if sample is None else sample.transmission(grid, theta)
        return Window(grid, *self.window, knots=self.knots, tail=tail)

    def fit(self, data, sample, theta0=None):
        """Fit a Measurement with a Sample; start from ``theta0`` (default:
        the sample's values). Returns a FitResult."""
        from scipy.optimize import least_squares
        sample.check_window(self.window)
        grid = data.grid
        th0 = sample.theta if theta0 is None else np.asarray(theta0, float)
        nth = len(th0)
        win = self.window_for(grid, sample, th0)
        rng = win.range_for(sample)
        pat = data.pattern
        n = data.n_periods
        y = data.counts
        o = data.beam.rate
        bg_known = data.background * data.time
        beta0 = win.centre(sample, grid, th0)
        nb = len(beta0)
        sp = self.prior_sigma

        names_x = []
        x_extra0 = []
        if self.fit_background:
            names_x.append("background")
            x_extra0.append(0.0)
        if self.fit_t0:
            names_x.append("t0")
            x_extra0.append(0.0)
        if self.fit_intensities and pat.n_pulses > 1:
            names_x += [f"intensity_{k + 1}" for k in range(1, pat.n_pulses)]
            x_extra0 += list(pat.intensities[1:])
        ne = len(names_x)

        def split(x):
            return x[:nth], x[nth:nth + ne], x[nth + ne:]

        def unpack(ex):
            """(background, t0, intensities) from the extra parameters."""
            i, bg, t0, a = 0, 0.0, 0.0, pat.intensities.copy()
            if self.fit_background:
                bg, i = ex[i], i + 1
            if self.fit_t0:
                t0, i = ex[i], i + 1
            if self.fit_intensities and pat.n_pulses > 1:
                a[1:] = ex[i:]
            return bg, t0, a

        def pattern_array(t0, a):
            if t0 == 0.0 and np.array_equal(a, pat.intensities):
                return pat.array(grid)
            return pat.__class__((pat.starts + t0) % grid.period, pat.widths, a).array(grid)

        def mu_of(x):
            th, ex, be = split(x)
            bg, t0, a = unpack(ex)
            c = pattern_array(t0, a)
            T = win.T(sample.transmission(grid, th, rng), be)
            return n * grid.convolve(o * T, c) + bg_known + bg * data.time, T, c

        def dmu(x):
            """mu and d mu / d x, shape (len(x), n_bins)."""
            th, ex, be = split(x)
            mu, T, c = mu_of(x)
            d = np.vstack([sample.derivatives(grid, th, rng, central=False) * win.inside, win.B])
            Jm = n * grid.convolve(o * d, c)
            rows = [Jm[:nth]]
            for j in range(ne):
                h = {"background": 1.0, "t0": 1e-7}.get(names_x[j], 1e-3)
                xp = x.copy()
                xp[nth + j] += h
                rows.append(((mu_of(xp)[0] - mu) / h)[None, :])
            rows.append(Jm[nth:])
            return mu, np.vstack(rows)

        def fun(x):
            return np.r_[_dev_resid(y, mu_of(x)[0])[0], (x[nth + ne:] - beta0) / sp]

        def jac(x):
            mu, D = dmu(x)
            _, drdmu = _dev_resid(y, mu)
            prior = np.hstack([np.zeros((nb, nth + ne)), np.eye(nb) / sp])
            return np.vstack([(drdmu[None, :] * D).T, prior])

        lo_t, hi_t = sample.bounds()
        lo_e = [-np.inf if nm in ("background", "t0") else 0.0 for nm in names_x]
        lb = np.r_[lo_t, lo_e, np.full(nb, -0.5)]
        ub = np.r_[hi_t, np.full(ne, np.inf), np.full(nb, 1.5)]
        x0 = np.clip(np.r_[th0, x_extra0, beta0], lb + 1e-12, ub - 1e-12)
        r = least_squares(fun, x0, jac=jac, method="trf", x_scale="jac", bounds=(lb, ub),
                          max_nfev=self.max_nfev, ftol=self.tol, xtol=self.tol, gtol=self.tol)
        mu, D = dmu(r.x)
        cv = getattr(data, "count_variance", 1.0)                  # recorded counts: variance cv * mu
        F = (D / np.clip(cv * mu, 1e-12, None)[None, :]) @ D.T
        F[nth + ne:, nth + ne:] += np.eye(nb) / sp ** 2
        C = _inverse(F)
        k = nth + ne
        sig = np.sqrt(np.clip(np.diag(C)[:k], 0, None))
        th, ex, be = split(r.x)
        _, t0, a = unpack(ex)
        T = win.T(sample.transmission(grid, th, rng), be)
        parts = [n * grid.convolve(o * T, pat.__class__([(s + t0) % grid.period], w, ak).array(grid))
                 for s, w, ak in zip(pat.starts, pat.widths, a)]
        used = (mu > 1e-3) | (y > 0)
        return FitResult("Rietveld fit", sample.free + names_x, r.x[:k], sig, C[:k, :k],
                         np.sum(r.fun[:grid.n_bins] ** 2), used.sum(), r.status > 0, r.nfev, data, sample,
                         model_counts=mu, contributions=parts, nuisance=be, window=win, fitted_transmission=T,
                         window_range=self.window)


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
    ratio, and it is fitted in ``window`` with independent bins, as in
    practice. The reported errors account for the correlations that the
    deconvolution creates between bins (a sandwich estimate); the fit
    itself ignores them, which is what loses information.

    Parameters
    ----------
    noise_power : float > 0 or "optimal"
        Constant regularisation of the filter (as in FOBI), or the optimal
        filter eps(f) = noise power / signal power estimated from the model
        at the starting values.
    """

    def __init__(self, noise_power=1e-3, window=(1.0, 5.0), max_nfev=200, tol=1e-10):
        if isinstance(noise_power, str):
            if noise_power != "optimal":
                raise ValueError('noise_power must be a positive number or "optimal"')
        elif not noise_power > 0:
            raise ValueError("noise_power must be positive (or \"optimal\")")
        self.noise_power, self.window, self.max_nfev, self.tol = noise_power, tuple(window), int(max_nfev), float(tol)

    def fit(self, data, sample, theta0=None):
        """Fit a Measurement with a Sample. Returns a FitResult."""
        from scipy.ndimage import uniform_filter1d
        from scipy.optimize import least_squares
        sample.check_window(self.window)
        grid = data.grid
        pat = data.pattern
        if np.ptp(pat.widths) > 0:
            raise NotImplementedError("WienerFit needs pulses of equal width")
        box = pat.__class__([0.0], pat.widths[0]).array(grid)
        th0 = sample.theta if theta0 is None else np.asarray(theta0, float)
        rate = data.beam.rate
        n = data.n_periods
        y = data.counts - data.background * data.time
        y_sm = np.clip(uniform_filter1d(data.counts, 21, mode="wrap"), 1, None)
        O = data.open_counts()
        delta = pat.delta(grid)
        npow = self.noise_power
        if isinstance(npow, str):
            s1 = n * grid.convolve(rate * sample.transmission(grid, th0), box)
            npow = data.counts.sum() / np.clip(np.abs(np.fft.fft(s1)) ** 2, 1e-30, None)
        single = pat.n_pulses == 1
        if single:
            shat, var, ohat = y, y_sm, O
            Hf = None
        else:
            Hf = wiener_filter(delta, npow)
            shat, _ = wiener(y, delta, npow)
            _, var = wiener(y_sm, delta, npow)
            ohat, _ = wiener(O, delta, npow)
        lam = grid.wavelength
        sel = (lam >= self.window[0]) & (lam <= self.window[1]) & (ohat > 0)
        sd = np.sqrt(var)
        rng = (self.window[0] - sample.margin, self.window[1] + sample.margin)
        ob1 = grid.convolve(rate, box)
        Tbox = lambda T: grid.convolve(rate * T, box) / np.clip(ob1, 1e-30, None)

        def fun(th):
            return ((shat - ohat * Tbox(sample.transmission(grid, th, rng))) / sd)[sel]

        def jac(th, central=False):
            d = sample.derivatives(grid, th, rng, central=central)
            return -((ohat * grid.convolve(rate * d, box) / np.clip(ob1, 1e-30, None))[:, sel] / sd[sel]).T

        lo, hi = sample.bounds()
        r = least_squares(fun, np.clip(th0, lo + 1e-12, hi - 1e-12), jac=jac, method="trf", x_scale="jac",
                          bounds=(lo, hi), max_nfev=self.max_nfev, ftol=self.tol, xtol=self.tol, gtol=self.tol)
        A = jac(r.x, central=True)                                   # d residual / d theta on sel
        AtA_inv = _inverse(A.T @ A)
        if single:
            meat = A.T @ A
        else:
            # covariance of the residuals: D H diag(y) H^T D, D = 1 / sd on sel
            B = np.zeros((A.shape[1], grid.n_bins))
            B[:, sel] = (A / sd[sel][:, None]).T
            HtB = np.real(np.fft.ifft(np.fft.fft(B, axis=1) * np.conj(Hf)[None, :], axis=1))
            CB = np.real(np.fft.ifft(np.fft.fft(HtB * y_sm[None, :], axis=1) * Hf[None, :], axis=1))
            meat = (B[:, sel] / 1.0) @ CB[:, sel].T
        cv = getattr(data, "count_variance", 1.0)                  # recorded counts: variance cv * mu
        C = cv * (AtA_inv @ meat @ AtA_inv)
        sig = np.sqrt(np.clip(np.diag(C), 0, None))
        That = shat / np.where(np.abs(ohat) > 0, ohat, np.nan)
        res = np.full(grid.n_bins, np.nan)
        res[sel] = r.fun
        return FitResult("Wiener deconvolution", sample.free, r.x, sig, C, np.sum(r.fun ** 2), sel.sum(),
                         r.status > 0, r.nfev, data, sample, transmission=That,
                         transmission_sigma=np.sqrt(cv) * sd / np.where(np.abs(ohat) > 0, np.abs(ohat), np.nan),
                         model_transmission=Tbox(sample.transmission(grid, r.x)), selection=sel, residuals=res,
                         window_range=self.window,
                         note="errors include the correlations created by the deconvolution (sandwich estimate)")
