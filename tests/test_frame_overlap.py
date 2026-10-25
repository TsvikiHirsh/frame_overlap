"""Tests with known answers for frame_overlap (run: pytest -q tests)."""
import os
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

import frame_overlap as fo

P = 50e-3          # period of the 20 Hz test grid
OPT4 = [0.0, 11.97e-3, 22.33e-3, 35.48e-3]          # four pulses optimised for the edge position (the article)


@pytest.fixture(scope="module")
def grid():
    return fo.TimeGrid(9.0, 20.0, 10e-6)


# --------------------------------------------------------------- no nbragg

def test_grid(grid):
    assert grid.n_bins == 5000
    assert np.isclose(grid.wavelength_at(grid.time_at(4.05)), 4.05)
    assert np.isclose(grid.wavelength_at(12.5e-3), 5.494, atol=1e-3)
    assert grid == fo.TimeGrid(9.0, 20.0, 10e-6) and hash(grid) == hash(fo.TimeGrid(9.0, 20.0, 10e-6))
    assert np.isclose(grid.wavelength_max, 21.98, atol=0.01)
    with pytest.raises(ValueError):
        fo.TimeGrid(9.0, 20.0, 7e-6)                     # not a whole number of bins


def test_pattern_arrays(grid):
    p = fo.Pattern.equal(4)                              # default period 50 ms
    assert p.times_ms == [0.0, 12.5, 25.0, 37.5]
    assert np.isclose(p.array(grid).sum(), 4 * 200e-6)
    wrap = fo.Pattern([49.9e-3], 200e-6)                 # wraps around the period
    a = wrap.array(grid)
    assert np.isclose(a.sum(), 200e-6) and a[0] > 0 and a[-1] > 0
    frac = fo.Pattern([5e-6], 10e-6).array(grid)         # half bins
    assert np.allclose(frac[:2], [5e-6, 5e-6])
    d = fo.Pattern([12.5e-6]).delta(grid)                # delta between bin edges
    assert np.isclose(d.sum(), 1.0) and np.isclose(d[1], 0.75) and np.isclose(d[2], 0.25)
    assert not p.overlaps(P) and fo.Pattern([0, 100e-6], 200e-6).overlaps(P)


def test_pattern_checks(grid):
    with pytest.raises(ValueError):
        fo.Pattern([60.0]).check(grid)                   # milliseconds instead of seconds
    with pytest.raises(ValueError):
        fo.Pattern([0, 100e-6], 200e-6).check(grid)      # overlapping pulses
    with pytest.raises(ValueError):
        fo.Pattern.equal(2, period=0.1).check(grid)      # built for another period
    with pytest.raises(ValueError):
        fo.Pattern([0.0], width=-1e-6)
    assert fo.Pattern.equal(4, grid=grid).period == P


def test_flat_sequences(grid):
    for order in (2, 3, 4, 5, 7):
        p = fo.Pattern.maximum_length(order)
        assert p.n_pulses == 2 ** (order - 1)
        p.check(grid)
    assert fo.Pattern.quadratic_residue(11).n_pulses == 5
    assert fo.Pattern.quadratic_residue(11, complement=True).n_pulses == 6
    with pytest.raises(ValueError):
        fo.Pattern.quadratic_residue(13)                 # 13 = 1 mod 4
    po = fo.Pattern.poldi()
    assert po.n_pulses == 8 and po.starts.max() < P


def test_open_beam(grid):
    b = fo.OpenBeam.maxwellian(grid, flux=1e6, area=2.0, efficiency=0.5)
    assert np.isclose(b.total_rate(), 1e6, rtol=2e-3)
    band = b.with_band(1.0, 5.0)
    lam = grid.wavelength
    assert np.all(band.rate[(lam < 1.0) | (lam >= 5.0)] == 0)
    counts = b.counts(fo.Pattern.single(200e-6), 3600)
    assert np.isclose(counts.sum(), b.total_rate() * 200e-6 * 20 * 3600, rtol=1e-6)
    short = fo.OpenBeam.from_counts(grid, np.exp(-np.arange(2400) / 500.0), pulse_width=1.0, time=P)
    assert short.rate.shape == (5000,) and np.isclose(short.rate[2400], np.exp(-2400 / 500), rtol=0.05)
    with pytest.raises(ValueError):
        fo.OpenBeam(grid, -np.ones(grid.n_bins))
    with pytest.raises(ValueError):
        b.with_band(5.0, 1.0)


def test_band_for(grid):
    b = fo.OpenBeam.maxwellian(grid, flux=1e6)
    lam = grid.wavelength
    for n, lo, hi in ((4, 0.97, 5.25), (5, 0.97, 5.25)):
        on = lam[b.with_band_for(n).rate > 0]
        assert np.isclose(on.min(), lo, atol=0.01) and np.isclose(on.max(), hi, atol=0.01)
    on = lam[b.with_band_for(8).rate > 0]                # narrower band that still reaches 4.55 A
    assert np.isclose(on.max(), 4.55, atol=0.01) and on.min() > 1.5


def test_data(grid, tmp_path):
    b = fo.OpenBeam.maxwellian(grid, flux=1e6)
    with pytest.raises(ValueError):
        fo.Data(grid, np.ones(100), fo.Pattern.single(), b, 3600)
    with pytest.raises(ValueError):
        fo.Data(grid, np.ones(grid.n_bins), fo.Pattern.single(), b, 0)
    d = fo.Data(grid, np.full(grid.n_bins, 3.0), fo.Pattern.equal(2), b, 3600)
    assert np.isclose(d.n_periods, 72000) and list(d.table.columns) == ["tof", "wavelength", "counts"]
    # from files in nbragg's format (tof, counts, err), the open beam measured with one pulse per period
    d.to_csv(tmp_path / "sample.csv")
    ob = b.rate * 200e-6 * 7200 * 20
    pd.DataFrame(dict(tof=np.arange(grid.n_bins), counts=ob)).to_csv(tmp_path / "ob.csv", index=False)
    d2 = fo.Data.from_counts(str(tmp_path / "sample.csv"), str(tmp_path / "ob.csv"), pattern=fo.Pattern.equal(2),
                             time=3600, openbeam_time=7200)
    assert d2.grid == grid and np.allclose(d2.counts, 3.0)
    big = b.rate > 1e-6 * b.rate.max()
    assert np.allclose(d2.openbeam.rate[big], b.rate[big])


# ------------------------------------------------------------------ nbragg

nbragg = pytest.importorskip("nbragg")
BEAM = dict(flux=5e6, area=0.03)


@pytest.fixture(scope="module")
def xs():
    return nbragg.CrossSection(iron="Fe_sg229_Iron-alpha.ncmat")


@pytest.fixture
def model(xs):
    m = fo.TransmissionModel(xs, pattern=fo.Pattern(OPT4), vary_basic=True, vary_strain=True)
    m.params["thickness"].set(value=2.0)
    return m


def noiseless(model, time=8 * 3600, **beam):
    beam = beam or BEAM
    return fo.Data(model.grid, model.expected_counts(time, **beam), model.pattern, model.openbeam(**beam), time)


def test_model_parameters(xs, model):
    assert model.free == ["thickness", "strain"]
    assert not model.params["norm"].vary                   # the open beam is known
    assert model.params["α0"].value == 3000.0 and not model.params["α0"].vary
    free_norm = fo.TransmissionModel(xs, vary_basic=True, vary_norm=True)
    assert free_norm.free == ["thickness", "norm"]
    assert hasattr(model, "show_available_params")         # nbragg's methods pass through
    nb = nbragg.TransmissionModel(xs, response="jorgensen_inv", vary_response=False)
    wrapped = fo.TransmissionModel(nb, vary_strain=True)
    wrapped.params["thickness"].set(value=3.0)
    assert nb.params["thickness"].value == 1.0             # a wrapped nbragg model stays as it is
    with pytest.raises(ValueError):
        fo.TransmissionModel(xs, pattern=fo.Pattern.equal(2, period=0.1))
    with pytest.raises(ValueError):
        fo.TransmissionModel(xs, vary_basic=False).simulate()   # nothing to determine


def test_strain_moves_edges(model):
    s = model._sample()
    g = model.grid
    T0 = s.transmission(g)
    T1 = s.transmission(g, s.theta + [0, 1e-3])
    lam = g.wavelength
    k = np.argmax(np.abs(np.diff(T0 * (lam > 3.8) * (lam < 4.3))))       # (110) edge
    k1 = np.argmax(np.abs(np.diff(T1 * (lam > 3.8) * (lam < 4.3))))
    assert np.isclose(lam[k1] / lam[k], 1.001, atol=2e-3)


@pytest.mark.parametrize("starts", [[0.0], [0.0, 25e-3], OPT4])
def test_rietveld_round_trip(model, starts):
    """Noiseless counts, start one standard deviation away: the fit returns the
    truth, up to the approximation of the transmission outside the window by
    straight lines between knots (a few hundredths of a standard deviation for
    overlapping pulses); its errors are the bound."""
    model.pattern = fo.Pattern(starts)
    data = noiseless(model)
    sig = model.precision(**BEAM).sigma.values
    r = model.fit(data, start=np.array([2.0, 0.0]) + sig)
    assert r.success
    assert np.all(np.abs(r.theta - [2.0, 0.0]) < 0.1 * sig)
    assert np.allclose(r.params["thickness"].stderr, sig[0], rtol=0.05)


def test_fit_result(model):
    model.pattern = fo.Pattern([0.0, 25e-3])
    data = model.simulate(seed=4, **BEAM)
    r = model.fit(data)
    assert list(r.to_dataframe().parameter) == ["thickness", "strain"]
    assert r.params["strain"].stderr > 0 and r.params["thickness"].stderr > 0
    assert "thickness" in r.fit_report() and r.fit_report() == str(r)
    res = r.residuals()
    assert res.shape == (model.grid.n_bins,) and 0.8 < np.mean(res[data.counts > 0] ** 2) < 1.2
    assert 0.8 < r.redchi < 1.2 and r.best_fit.shape == data.counts.shape
    tab = r.transmission()
    assert list(tab.columns) == ["wavelength", "trans", "err", "model"]
    inwin = (tab.wavelength > 1.05) & (tab.wavelength < 4.95)
    assert 0.8 < np.mean(((tab.trans - tab.model) / tab.err)[inwin] ** 2) < 1.2
    assert r.plot() is not None and r.plot(kind="counts") is not None
    w = model.fit(data, method="wiener")
    assert w.method == "Wiener deconvolution" and w.plot() is not None
    with pytest.raises(ValueError):
        model.fit(data, method="magic")


def test_fitted_background(model):
    """A constant background, unknown to the fit, is found by fit_background."""
    exp = model._experiment(8 * 3600, background=0.02, **BEAM)
    data = fo.Data(model.grid, exp.expected_counts(), model.pattern, exp.beam, exp.time)    # background not given
    r = model.fit(data, fit_background=True)
    sig = model.precision(background=True, **BEAM).sigma.values
    assert np.isclose(r.values["background"], 0.02, rtol=0.02)
    assert np.all(np.abs(r.theta[:2] - [2.0, 0.0]) < 0.1 * sig[:2])


def test_two_separate_pulses_double_the_information(model):
    model.pattern = fo.Pattern([0.0, 25e-3])
    R = model.gain(**BEAM)
    assert np.isclose(R["thickness"], 2.0, rtol=0.02) and np.isclose(R["strain"], 2.0, rtol=0.02)


def test_count_variance(model):
    """Recorded counts with F times the Poisson variance: bounds and fit errors
    x sqrt(F), gains unchanged, simulated counts with the right mean and variance."""
    s0 = model.precision(**BEAM).set_index("parameter").sigma
    s1 = model.precision(count_variance=2.3, **BEAM).set_index("parameter").sigma
    assert np.allclose(s1 / s0, np.sqrt(2.3), rtol=0.01)
    g0, g1 = model.gain(**BEAM), model.gain(count_variance=2.3, **BEAM)
    assert all(np.isclose(g0[n], g1[n], rtol=0.01) for n in g0)
    data = noiseless(model)
    data.count_variance = 2.3
    assert np.isclose(model.fit(data).params["strain"].stderr, s1["strain"], rtol=0.05)
    mu = model.expected_counts(**BEAM)
    ys = np.array([model.simulate(seed=k, count_variance=2.3, **BEAM).counts for k in range(40)])
    hi = mu > 200
    assert np.isclose(ys.mean(axis=0)[hi].sum(), mu[hi].sum(), rtol=0.01)
    assert np.isclose((ys.var(axis=0)[hi] / mu[hi]).mean(), 2.3, rtol=0.1)
    with pytest.raises(ValueError):
        model.simulate(count_variance=0.5, **BEAM)


def test_bandwidth_chopper_gain(model):
    """Four equally spaced pulses with a bandwidth chopper: R = 4."""
    ob = model.openbeam(**BEAM)
    s4 = model.precision(pattern=fo.Pattern.equal(4), openbeam=ob.with_band_for(4)).sigma.values
    s1 = model.precision(pattern=fo.Pattern.single(), openbeam=ob.with_band_for(1)).sigma.values
    assert np.allclose((s1 / s4) ** 2, 4.0, rtol=0.03)


def test_wiener_ratio_half_period(model):
    """With the open beam deconvolved by the same filter, two pulses half a
    period apart (a kernel with zeros) recover the transmission, up to the
    small overlap of the far tail of one pulse with the other (a real, small
    bias of the method, here below half a standard deviation)."""
    model.pattern = fo.Pattern([0.0, 25e-3])
    r = model.fit(noiseless(model), method="wiener", noise_power=1e-3, start=[2.01, 2e-4])
    sig = model.precision(**BEAM).sigma.values
    assert np.all(np.abs(r.theta - [2.0, 0.0]) < 0.5 * sig)


def test_pattern_bias(model):
    assert all(abs(v) < 1e-6 for v in model.pattern_bias(model.pattern, **BEAM).values())
    late = model.pattern_bias(fo.Pattern(np.array(OPT4) + 1e-6), **BEAM)
    assert late["strain"] > 0.5                         # a common delay looks like a longer wavelength


def test_optimise_two_pulses(model):
    """Two pulses: the best second pulse is where the spectra do not overlap (R = 2)."""
    best = model.optimise_pattern(2, target="strain", n_restarts=2, n_steps=300, seed=1, **BEAM)
    assert model.gain(pattern=best, **BEAM)["strain"] > 1.95 and not best.overlaps(P)
    with pytest.raises(ValueError):
        model.optimise_pattern(2, target="norm")


def test_monte_carlo(model):
    est, summary = model.monte_carlo(n=6, seed=3, **BEAM)
    assert summary.n_ok.iloc[0] == 6 and (est.error == "").all()
    assert np.all(np.abs(summary.bias / summary.bound) < 3)
    par, _ = model.monte_carlo(n=6, seed=3, workers=2, **BEAM)        # same draws in parallel
    assert np.allclose(par[model.free].values, est[model.free].values)


def test_plan(xs, model):
    """plan: an optimised pattern, its gain, time and precision that invert each
    other, a model to simulate and fit with, and no change to a given model."""
    ins = fo.Instrument(efficiency=0.2, count_variance=2.0)
    p = fo.plan(xs, thickness=2.0, target="strain", n_pulses=[1, 2], instrument=ins, n_restarts=1, n_steps=150,
                seed=1)
    assert p.pattern.n_pulses == 2 and p.gain["strain"] > 1.9 and not p.pattern.overlaps(P)
    assert list(p.candidates.n_pulses) == [1, 2] and p.model.pattern is p.pattern
    t = p.time_for(bin_size=1.0, strain=500)
    assert np.isclose(p.precision(time=t, bin_size=1.0).set_index("parameter").sigma["strain"], 500)
    s1 = p.precision(8 * 3600, 1.0).set_index("parameter").sigma
    s2 = p.precision(8 * 3600, 2.0).set_index("parameter").sigma
    assert np.allclose(s1 / s2, 2.0)                      # one over the bin side at constant F
    with pytest.warns(UserWarning):
        p.precision(time=60, bin_size=0.1)                # too few counts for the bound
    r = p.model.fit(p.simulate(time=8 * 3600, bin_size=2.0, seed=2))
    assert r.success and abs(r.values["thickness"] - 2.0) < 5 * r.errors["thickness"]
    before = [(n, q.vary) for n, q in model.params.items()]
    fo.plan(model, target="strain", n_pulses=[1], instrument=fo.Instrument(efficiency=0.2))
    assert [(n, q.vary) for n, q in model.params.items()] == before and model.pattern.n_pulses == 4


def test_camera_count_variance():
    F = fo.camera_count_variance([0.109, 1.0, 3.5, 20.0])
    assert np.allclose(F[[0, 2, 3]], [1.10, 2.30, 2.30]) and 2.05 < F[1] < 2.28
    assert np.isclose(float(fo.Instrument.saraf_ii().F(1.0)), F[1])
