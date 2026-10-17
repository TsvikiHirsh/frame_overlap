"""
Measurement design: the precision a design can reach (Cramer-Rao bound of
the Rietveld fit), the time-reduction factor of a pulse pattern, pattern
surveys and optimisation, Monte Carlo validation, and the bias from errors in
the assumed pulse pattern.
"""
import warnings
import numpy as np
import pandas as pd
from .fitting import Window, _inverse
from .pattern import Pattern

UNITS = {"thickness": "cm", "strain": "1", "norm": "1", "background": "counts/bin/s", "t0": "s"}


def _setup(exp, window, knots=None):
    """Analysis window and derivatives of the transmission at the true values,
    cached on the experiment by value (shared by its variants with other
    patterns, times or beams)."""
    exp.sample.check_window(window)
    grid = exp.grid
    s = exp.sample
    key = (grid.key, tuple(window), None if knots is None else tuple(knots),
           s._fingerprint(s._nbragg_params(s.theta)), s.strain, tuple(s.free))
    cache = exp.__dict__.setdefault("_dT", {})
    if key not in cache:
        win = Window(grid, *window, knots=knots, tail=s.transmission(grid))
        cache[key] = (win, s.derivatives(grid, wl_range=win.range_for(s)))
    return cache[key]


def _extra_rows(exp, background, t0, intensities):
    """d mu / d (background, t0, relative intensities of pulses 2..N)."""
    g, p = exp.grid, exp.pattern
    rows, names = [], []
    if background:
        rows.append(np.full(g.n_bins, exp.time))
        names.append("background")
    if t0:
        h = 1e-7
        a = exp.with_pattern(Pattern((p.starts + h) % g.period, p.widths, p.intensities)).expected_counts()
        b = exp.with_pattern(Pattern((p.starts - h) % g.period, p.widths, p.intensities)).expected_counts()
        rows.append((a - b) / (2 * h))
        names.append("t0")
    if intensities and p.n_pulses > 1:
        T = exp.sample.transmission(g)
        for k in range(1, p.n_pulses):
            rows.append(exp.n_periods * g.convolve(exp.beam.rate * T, Pattern([p.starts[k]], p.widths[k]).array(g)))
            names.append(f"intensity_{k + 1}")
    return rows, names


def fisher(exp, window=(1.0, 5.0), prior_sigma=1.0, extra=None, knots=None,
           background=False, t0=False, intensities=False):
    """
    Fisher matrix of the Poisson counts of a design for the free parameters,
    optional extra quantities (a constant background, a common time offset,
    the relative pulse intensities), and the free transmission outside the
    window (with its weak prior). ``extra``: further derivative rows of the
    transmission (e.g. of a broadening) treated as free.
    Returns (F, names of the leading rows).
    """
    win, dT = _setup(exp, window, knots)
    grid = exp.grid
    c = exp.pattern.array(grid)
    n = exp.n_periods
    mu = exp.expected_counts()
    trows = [dT * win.inside]
    if extra is not None:
        trows.append(np.atleast_2d(extra) * win.inside)
    J_t = n * grid.convolve(exp.beam.rate * np.vstack(trows), c)
    erows, enames = _extra_rows(exp, background, t0, intensities)
    J_b = n * grid.convolve(exp.beam.rate * win.B, c)
    J = np.vstack([J_t] + ([np.array(erows)] if erows else []) + [J_b])
    F = (J / np.clip(mu, 1e-12, None)) @ J.T
    nb = win.B.shape[0]
    F[-nb:, -nb:] += np.eye(nb) / prior_sigma ** 2
    names = list(exp.sample.free) + [f"extra_{i}" for i in range(J_t.shape[0] - dT.shape[0])] + enames
    return F, names


def precision(exp, window=(1.0, 5.0), prior_sigma=1.0, background=False, t0=False, intensities=False):
    """
    Cramer-Rao bound (one standard deviation) of each free parameter for a
    design, as a DataFrame with units; the strain also in microstrain.
    ``background``, ``t0``, ``intensities``: treat these as unknown too.
    """
    F, names = fisher(exp, window, prior_sigma, background=background, t0=t0, intensities=intensities)
    k = len(names)
    sig = np.sqrt(np.diag(_inverse(F))[:k])
    vals = list(exp.sample.theta) + [np.nan] * (k - len(exp.sample.free))
    df = pd.DataFrame(dict(parameter=names, value=vals, sigma=sig,
                           unit=[UNITS.get(n, "") for n in names]))
    df["sigma_microstrain"] = np.where(df.parameter == "strain", df.sigma * 1e6, np.nan)
    return df


def gain(exp, reference=None, window=(1.0, 5.0)):
    """
    Time-reduction factor R = (sigma_reference / sigma)^2 of each free
    parameter: the factor by which the reference design would have to
    measure longer to reach the precision of this design. ``reference`` is a
    Pattern (same beam, sample and time), an Experiment, or None for one
    200 us pulse per period with the same beam.
    """
    if reference is None:
        reference = Pattern.single(200e-6)
    ref = reference if hasattr(reference, "sample") else exp.with_pattern(reference)
    s = precision(exp, window).set_index("parameter").sigma
    s1 = precision(ref, window).set_index("parameter").sigma
    return {n: float((s1[n] / s[n]) ** 2) for n in exp.sample.free}


def survey(exp, patterns, reference=None, window=(1.0, 5.0)):
    """Precision and time-reduction factor of several designs: a dict of
    name: Pattern (same beam and time) or name: Experiment."""
    if reference is None:
        reference = Pattern.single(200e-6)
    ref = reference if hasattr(reference, "sample") else exp.with_pattern(reference)
    s1 = precision(ref, window).set_index("parameter").sigma
    rows = []
    for name, p in patterns.items():
        e = p if hasattr(p, "sample") else exp.with_pattern(p)
        s = precision(e, window).set_index("parameter").sigma
        row = dict(design=name, n_pulses=e.pattern.n_pulses)
        for n in exp.sample.free:
            row[f"sigma_{n}"] = float(s[n])
            row[f"R_{n}"] = float((s1[n] / s[n]) ** 2)
        rows.append(row)
    return pd.DataFrame(rows)


def optimise_pattern(exp, n_pulses, target="strain", width=200e-6, n_restarts=4, n_steps=800,
                     window=(1.0, 5.0), seed=0, verbose=False):
    """
    Stochastic search for the start times of ``n_pulses`` pulses of
    ``width`` that minimise the Cramer-Rao bound of ``target``: single-pulse
    moves on the time grid with a shrinking step, pulses never overlapping.
    The result is a good pattern, not a proven optimum; more restarts and
    steps can do better (the defaults take about a minute).
    """
    if target not in exp.sample.free:
        raise ValueError(f"target {target!r} is not a free parameter; choose from {exp.sample.free}")
    rng = np.random.default_rng(seed)
    grid = exp.grid
    period, dt = grid.period, grid.bin_width
    j = exp.sample.free.index(target)
    _setup(exp, window)

    def sigma(starts):
        F, _ = fisher(exp.with_pattern(Pattern(starts, width)), window)
        return np.sqrt(_inverse(F)[j, j])

    def valid(x):
        x = np.sort(x % period)
        return np.all(np.diff(np.r_[x, x[0] + period]) >= width - 1e-12)

    best_all, s_all = None, np.inf
    for restart in range(n_restarts):
        x = Pattern.random(n_pulses, width, period, seed=int(rng.integers(1 << 31)), bin_width=dt).starts
        s = sigma(x)
        step = period / 4
        for it in range(n_steps):
            if n_pulses == 1:
                break
            cand = x.copy()
            i = rng.integers(1, n_pulses)
            cand[i] = np.round(((cand[i] + rng.normal(0, step)) % period) / dt) * dt
            if not valid(cand):
                continue
            sc = sigma(cand)
            if sc < s:
                x, s = np.sort(cand), sc
            if it % 200 == 199:
                step = max(step / 2, 2 * dt)
        if verbose:
            print(f"restart {restart + 1}/{n_restarts}: sigma({target}) = {s:.4g}")
        if s < s_all:
            best_all, s_all = x, s
    return Pattern(best_all, width, name=f"{n_pulses} pulses optimised for {target}", period=period)


_MC_JOB = {}          # the job of a parallel Monte Carlo, inherited by forked workers


def _mc_draw(i):
    exp, fitter, seed, start_sd = _MC_JOB["job"]
    data = exp.simulate(seed + i)
    th0 = exp.sample.theta
    if start_sd is not None:
        th0 = th0 + np.asarray(start_sd) * np.random.default_rng(seed + i + 7919).standard_normal(len(th0))
    try:
        r = fitter.fit(data, exp.sample, theta0=th0)
        return r.theta[:len(th0)], r.success, ""
    except Exception as err:                       # recorded, not hidden
        return np.full(len(th0), np.nan), False, f"{type(err).__name__}: {err}"


def monte_carlo(exp, fitter, n=100, seed=0, start_sd="bound", workers=1):
    """
    Fit ``n`` simulated measurements of a design with ``fitter`` (a
    RietveldFit or WienerFit). Fits start from the true values displaced by
    ``start_sd`` (default: one Cramer-Rao standard deviation; None: from the
    truth). ``workers > 1`` uses processes (fork, Linux and macOS).
    Returns (estimates, summary): the fitted values of every draw with a
    success flag and any error message, and per parameter the mean bias, the
    standard deviation, the bound and their ratio.
    """
    crb = precision(exp, getattr(fitter, "window", (1.0, 5.0))).set_index("parameter").sigma[exp.sample.free].values
    sd = crb if isinstance(start_sd, str) and start_sd == "bound" else start_sd
    _MC_JOB["job"] = (exp, fitter, seed, sd)
    try:
        if workers > 1:
            import multiprocessing as mp
            try:
                ctx = mp.get_context("fork")         # workers inherit the job (nbragg models do not pickle)
            except ValueError:
                warnings.warn("process forking is not available on this system; running serially", stacklevel=2)
                ctx = None
            if ctx is not None:
                with ctx.Pool(workers) as pool:
                    out = pool.map(_mc_draw, range(n))
            else:
                out = [_mc_draw(i) for i in range(n)]
        else:
            out = [_mc_draw(i) for i in range(n)]
    finally:
        _MC_JOB.pop("job", None)
    est = pd.DataFrame([o[0] for o in out], columns=exp.sample.free)
    est["success"] = [o[1] for o in out]
    est["error"] = [o[2] for o in out]
    ok = est[est.success].drop(columns=["success", "error"])
    truth = exp.sample.theta
    summary = pd.DataFrame(dict(parameter=exp.sample.free, truth=truth, mean=ok.mean().values,
                                bias=ok.mean().values - truth, std=ok.std(ddof=1).values, bound=crb))
    summary["std/bound"] = summary["std"] / summary["bound"]
    summary["bias/std"] = summary["bias"] / summary["std"]
    summary["n_ok"] = len(ok)
    return est, summary


def pattern_bias(exp, true_pattern, window=(1.0, 5.0)):
    """
    Bias of the Rietveld fit, to first order, when the real pulse pattern is
    ``true_pattern`` but the fit assumes ``exp.pattern``: F^-1 J^T W dmu, in
    units of the Cramer-Rao standard deviation of each free parameter.
    """
    win, dT = _setup(exp, window)
    grid = exp.grid
    c = exp.pattern.array(grid)
    n = exp.n_periods
    mu = exp.expected_counts()
    J = n * grid.convolve(exp.beam.rate * np.vstack([dT * win.inside, win.B]), c)
    F, _ = fisher(exp, window)
    Finv = _inverse(F)
    dmu = exp.with_pattern(true_pattern).expected_counts() - mu
    k = len(exp.sample.free)
    b = (Finv @ ((J / np.clip(mu, 1e-12, None)) @ dmu))[:k]
    return {nm: float(v) for nm, v in zip(exp.sample.free, b / np.sqrt(np.diag(Finv)[:k]))}
