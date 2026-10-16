"""
Measurement design: the precision a design can reach (Cramer-Rao bound of
the Rietveld fit), the time-reduction factor of a pulse pattern, pattern
surveys and optimisation, Monte Carlo validation, and the bias from errors in
the assumed pulse pattern.
"""
import numpy as np
import pandas as pd
from .fitting import Window
from .pattern import Pattern


def _setup(exp, window, knots=None):
    """Analysis window and derivatives of the transmission at the true values,
    cached on the experiment (shared by its variants with other patterns)."""
    grid = exp.grid
    key = (id(exp.sample), id(grid), tuple(window), None if knots is None else tuple(knots))
    cache = exp.__dict__.setdefault("_dT", {})
    if key not in cache:
        rng = (window[0] - 0.25, window[1] + 0.25)
        win = Window(grid, *window, knots=knots, tail=exp.sample.transmission(grid))
        cache[key] = (win, exp.sample.derivatives(grid, wl_range=rng))
    return cache[key]


def fisher(exp, window=(1.0, 5.0), prior_sigma=1.0, extra=None, knots=None):
    """
    Fisher matrix of the Poisson counts of a design for the free parameters
    and the free transmission outside the window (with its weak prior).
    ``extra``: further derivative rows of the transmission (e.g. of a
    broadening) to be treated as free.
    """
    win, dT = _setup(exp, window, knots)
    grid = exp.grid
    c = exp.pattern.array(grid)
    n = exp.n_periods
    mu = exp.expected_counts()
    rows = [dT * win.inside, win.B]
    if extra is not None:
        rows.append(np.atleast_2d(extra) * win.inside)
    J = n * grid.convolve(exp.beam.rate * np.vstack(rows), c)
    F = (J / np.clip(mu, 1e-12, None)) @ J.T
    k = dT.shape[0]
    nb = win.B.shape[0]
    F[k:k + nb, k:k + nb] += np.eye(nb) / prior_sigma ** 2
    return F


def precision(exp, window=(1.0, 5.0), prior_sigma=1.0):
    """
    Cramer-Rao bound (one standard deviation) of each free parameter for a
    design, as a DataFrame; strain also in microstrain.
    """
    F = fisher(exp, window, prior_sigma)
    k = len(exp.sample.free)
    sig = np.sqrt(np.diag(np.linalg.inv(F))[:k])
    df = pd.DataFrame(dict(parameter=exp.sample.free, value=exp.sample.theta, sigma=sig))
    df["sigma_microstrain"] = np.where(df.parameter == "strain", df.sigma * 1e6, np.nan)
    return df


def gain(exp, reference=None, window=(1.0, 5.0)):
    """
    Time-reduction factor R = (sigma_reference / sigma)^2 of each free
    parameter: the factor by which the reference design (default: one 200 us
    pulse per period, same time) would have to measure longer to reach the
    precision of this design.
    """
    ref = exp.with_pattern(reference or Pattern.single(200e-6))
    s = precision(exp, window).sigma.values
    s1 = precision(ref, window).sigma.values
    return dict(zip(exp.sample.free, (s1 / s) ** 2))


def survey(exp, patterns, reference=None, window=(1.0, 5.0)):
    """Precision and time-reduction factor for several patterns (dict of
    name: Pattern) at the same time and beam."""
    ref = precision(exp.with_pattern(reference or Pattern.single(200e-6)), window).sigma.values
    rows = []
    for name, p in patterns.items():
        s = precision(exp.with_pattern(p), window).sigma.values
        row = dict(pattern=name, n_pulses=p.n_pulses)
        for n, v, v1 in zip(exp.sample.free, s, ref):
            row[f"sigma_{n}"] = v
            row[f"R_{n}"] = (v1 / v) ** 2
        rows.append(row)
    return pd.DataFrame(rows)


def optimise_pattern(exp, n_pulses, target="strain", width=200e-6, n_restarts=8, n_steps=2000,
                     window=(1.0, 5.0), seed=0, verbose=False):
    """
    Stochastic search for the pulse times of ``n_pulses`` pulses that
    minimise the Cramer-Rao bound of ``target``: single-pulse moves with a
    shrinking step, pulses never overlapping. The result is a good pattern,
    not a proven optimum (more restarts and steps can do better).
    """
    rng = np.random.default_rng(seed)
    grid = exp.grid
    period, dt = grid.period, grid.bin_width
    j = exp.sample.free.index(target)
    win, dT = _setup(exp, window)

    def sigma(starts):
        F = fisher(exp.with_pattern(Pattern(starts, width)), window)
        return np.sqrt(np.linalg.inv(F)[j, j])

    def valid(x):
        x = np.sort(x % period)
        return np.all(np.diff(np.r_[x, x[0] + period]) >= width - 1e-12)

    best_all, s_all = None, np.inf
    for _ in range(n_restarts):
        x = Pattern.random(n_pulses, width, period, seed=rng.integers(1 << 31), bin_width=dt).starts
        s = sigma(x)
        step = period / 4
        for it in range(n_steps):
            cand = x.copy()
            i = rng.integers(1, n_pulses) if n_pulses > 1 else 0
            cand[i] = np.round(((cand[i] + rng.normal(0, step)) % period) / dt) * dt
            if not valid(cand):
                continue
            sc = sigma(cand)
            if sc < s:
                x, s = np.sort(cand), sc
            if it % 300 == 299:
                step = max(step / 2, 2 * dt)
        if verbose:
            print(f"restart: sigma({target}) = {s:.4g}")
        if s < s_all:
            best_all, s_all = x, s
    return Pattern(best_all, width, name=f"{n_pulses} pulses optimised for {target}")


def _mc_one(args):
    exp, fitter, seed, start_sd = args
    data = exp.simulate(seed)
    th0 = exp.sample.theta
    if start_sd is not None:
        th0 = th0 + np.asarray(start_sd) * np.random.default_rng(seed + 7919).standard_normal(len(th0))
    try:
        r = fitter.fit(data, exp.sample, theta0=th0)
        return r.theta, r.success
    except Exception:
        return np.full(len(th0), np.nan), False


def monte_carlo(exp, fitter, n=100, seed=0, start_sd=None, workers=1):
    """
    Fit ``n`` simulated measurements of a design with ``fitter`` (a
    RietveldFit or WienerFit). Fits start from the true values displaced by
    ``start_sd`` standard deviations (default: one Cramer-Rao standard
    deviation). Returns (estimates DataFrame, summary DataFrame with mean
    bias, standard deviation and the bound).
    """
    crb = precision(exp, getattr(fitter, "window", (1.0, 5.0))).sigma.values
    start_sd = crb if start_sd is None else start_sd
    args = [(exp, fitter, seed + i, start_sd) for i in range(n)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            out = pool.map(_mc_one, args)
    else:
        out = [_mc_one(a) for a in args]
    est = pd.DataFrame([o[0] for o in out], columns=exp.sample.free)
    est["success"] = [o[1] for o in out]
    ok = est[est.success].drop(columns="success")
    truth = exp.sample.theta
    summary = pd.DataFrame(dict(parameter=exp.sample.free, truth=truth, mean=ok.mean().values,
                                bias=ok.mean().values - truth, std=ok.std(ddof=1).values, bound=crb))
    summary["std/bound"] = summary["std"] / summary["bound"]
    summary["n_ok"] = len(ok)
    return est, summary


def pattern_bias(exp, true_pattern, window=(1.0, 5.0)):
    """
    Bias of the Rietveld fit, to first order, when the real pulse pattern is
    ``true_pattern`` but the fit assumes ``exp.pattern``: F^-1 J^T W dmu, in
    units of the Cramer-Rao standard deviation of each parameter.
    """
    win, dT = _setup(exp, window)
    grid = exp.grid
    c = exp.pattern.array(grid)
    n = exp.n_periods
    mu = exp.expected_counts()
    J = n * grid.convolve(exp.beam.rate * np.vstack([dT * win.inside, win.B]), c)
    F = fisher(exp, window)
    Finv = np.linalg.inv(F)
    dmu = exp.with_pattern(true_pattern).expected_counts() - mu
    k = len(exp.sample.free)
    b = (Finv @ ((J / np.clip(mu, 1e-12, None)) @ dmu))[:k]
    return dict(zip(exp.sample.free, b / np.sqrt(np.diag(Finv)[:k])))
