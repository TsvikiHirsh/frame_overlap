"""Tests with known answers for frame_overlap (run: pytest -q tests)."""
import numpy as np
import pytest

import frame_overlap as fo

P = 50e-3          # period of the 20 Hz test grid


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
    p = fo.Pattern.equal(4, 200e-6, period=P)
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
    with pytest.raises(TypeError):
        fo.Pattern.equal(4)                              # the period is needed
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
        p = fo.Pattern.maximum_length(order, period=P)
        assert p.n_pulses == 2 ** (order - 1)
        p.check(grid)
    assert fo.Pattern.quadratic_residue(11, period=P).n_pulses == 5
    assert fo.Pattern.quadratic_residue(11, period=P, complement=True).n_pulses == 6
    with pytest.raises(ValueError):
        fo.Pattern.quadratic_residue(13, period=P)       # 13 = 1 mod 4
    po = fo.Pattern.poldi(grid=grid)
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


def test_measurement_checks(grid):
    b = fo.OpenBeam.maxwellian(grid, flux=1e6)
    with pytest.raises(ValueError):
        fo.Measurement(grid, np.ones(100), fo.Pattern.single(), b, 3600)
    with pytest.raises(ValueError):
        fo.Measurement(grid, np.ones(grid.n_bins), fo.Pattern.single(), b, 0)
    m = fo.Measurement(grid, np.ones(grid.n_bins), fo.Pattern.single(), b, 3600)
    assert np.isclose(m.n_periods, 72000)


# ------------------------------------------------------------------ nbragg

nbragg = pytest.importorskip("nbragg")


@pytest.fixture(scope="module")
def setup(grid):
    xs = nbragg.CrossSection(materials={"iron": {"mat": "Fe_sg229_Iron-alpha.ncmat", "weight": 1.0}})
    model = nbragg.TransmissionModel(xs, response="jorgensen_inv", tof_length=9.0, vary_response=False)
    model.params["thickness"].set(value=2.0)
    model.params["α0"].set(value=3000.0)
    model.params["β0"].set(value=50.0)
    sample = fo.Sample(model, free=["thickness", "strain"])
    beam = fo.OpenBeam.maxwellian(grid, flux=5e6, area=0.03)
    return sample, grid, beam


def test_sample_checks(setup):
    sample, grid, _ = setup
    with pytest.raises(KeyError):
        fo.Sample(sample.model, free=["thickness", "lattice"])
    with pytest.raises(ValueError):
        sample.check_window((0.2, 5.0))                  # outside fine_range with the strain free
    s2 = sample.with_values(strain=1e-3, thickness=1.0)
    assert np.allclose(s2.theta, [1.0, 1e-3]) and np.allclose(sample.theta, [2.0, 0.0])


def test_strain_moves_edges(setup):
    sample, grid, _ = setup
    T0 = sample.transmission(grid)
    T1 = sample.transmission(grid, sample.theta + [0, 1e-3])
    lam = grid.wavelength
    k = np.argmax(np.abs(np.diff(T0 * (lam > 3.8) * (lam < 4.3))))       # (110) edge
    k1 = np.argmax(np.abs(np.diff(T1 * (lam > 3.8) * (lam < 4.3))))
    assert np.isclose(lam[k1] / lam[k], 1.001, atol=2e-3)


@pytest.mark.parametrize("starts", [[0.0], [0.0, 25e-3], [0.0, 12e-3, 22.3e-3, 35.5e-3]])
def test_rietveld_round_trip(setup, starts):
    """Noiseless counts, start one standard deviation away: the fit returns the
    truth, up to the approximation of the transmission outside the window by
    straight lines between knots (a few hundredths of a standard deviation for
    overlapping pulses)."""
    sample, grid, beam = setup
    exp = fo.Experiment(sample, beam, fo.Pattern.times(starts), time=8 * 3600)
    data = fo.Measurement(grid, exp.expected_counts(), exp.pattern, beam, exp.time)
    sig = exp.precision().sigma.values
    r = fo.RietveldFit().fit(data, sample, theta0=sample.theta + sig)
    assert r.success
    assert np.all(np.abs(r.theta - sample.theta) < 0.1 * sig)
    assert np.allclose(r.errors["thickness"], sig[0], rtol=0.05)          # covariance = bound


def test_fit_result(setup):
    sample, grid, beam = setup
    exp = fo.Experiment(sample, beam, fo.Pattern.times([0.0, 25e-3]), time=8 * 3600)
    r = fo.RietveldFit().fit(exp.simulate(seed=4), sample)
    df = r.to_dataframe()
    assert list(df.parameter) == ["thickness", "strain"]
    assert "thickness" in r.fit_report() and r.fit_report() == r.summary()
    res = r.residuals()
    assert res.shape == (grid.n_bins,) and 0.8 < np.mean(res[r.data.counts > 0] ** 2) < 1.2


def test_fitted_background(setup):
    """A constant background, unknown to the fit, is found by fit_background."""
    sample, grid, beam = setup
    exp = fo.Experiment(sample, beam, fo.Pattern.times([0.0, 12e-3, 22.3e-3, 35.5e-3]), time=8 * 3600,
                        background=0.02)
    mu = exp.expected_counts()
    data = fo.Measurement(grid, mu, exp.pattern, beam, exp.time)        # background not given
    r = fo.RietveldFit(fit_background=True).fit(data, sample)
    sig = exp.precision(background=True).sigma.values
    assert np.isclose(r.values["background"], 0.02, rtol=0.02)
    assert np.all(np.abs(r.theta[:2] - sample.theta) < 0.1 * sig[:2])


def test_two_separate_pulses_double_the_information(setup):
    sample, grid, beam = setup
    exp = fo.Experiment(sample, beam, fo.Pattern.times([0.0, 25e-3]), time=8 * 3600)
    R = exp.gain()
    assert np.isclose(R["thickness"], 2.0, rtol=0.02) and np.isclose(R["strain"], 2.0, rtol=0.02)


def test_count_variance(setup):
    """Recorded counts with F times the Poisson variance: bounds x sqrt(F), gains unchanged,
    simulated counts with the right mean and variance."""
    sample, grid, beam = setup
    pat = fo.Pattern.times([0.0, 12e-3, 22.3e-3, 35.5e-3])
    exp = fo.Experiment(sample, beam, pat, time=8 * 3600)
    cam = fo.Experiment(sample, beam, pat, time=8 * 3600, count_variance=2.3)
    s0 = exp.precision().set_index("parameter").sigma
    s1 = cam.precision().set_index("parameter").sigma
    assert np.allclose(s1 / s0, np.sqrt(2.3), rtol=0.01)
    assert cam.with_pattern(fo.Pattern.single()).count_variance == 2.3
    g0, g1 = exp.gain(), cam.gain()
    assert all(np.isclose(g0[n], g1[n], rtol=0.01) for n in g0)
    mu = cam.expected_counts()
    ys = np.array([cam.simulate(seed=k).counts for k in range(40)])
    hi = mu > 200
    assert np.isclose(ys.mean(axis=0)[hi].sum(), mu[hi].sum(), rtol=0.01)
    assert np.isclose((ys.var(axis=0)[hi] / mu[hi]).mean(), 2.3, rtol=0.1)
    with pytest.raises(ValueError):
        fo.Experiment(sample, beam, pat, time=3600, count_variance=0.5)


def test_bandwidth_chopper_gain(setup):
    """Four equally spaced pulses with a bandwidth chopper: R = 4."""
    sample, grid, beam = setup
    exp = fo.Experiment(sample, beam.with_band_for(4), fo.Pattern.equal(4, grid=grid), time=8 * 3600)
    ref = fo.Experiment(sample, beam.with_band_for(1), fo.Pattern.single(), time=8 * 3600)
    R = exp.gain(reference=ref)
    assert np.isclose(R["thickness"], 4.0, rtol=0.03) and np.isclose(R["strain"], 4.0, rtol=0.03)


def test_wiener_ratio_half_period(setup):
    """With the open beam deconvolved by the same filter, two pulses half a
    period apart (a kernel with zeros) recover the transmission, up to the
    small overlap of the far tail of one pulse with the other (a real, small
    bias of the method, here below half a standard deviation)."""
    sample, grid, beam = setup
    exp = fo.Experiment(sample, beam, fo.Pattern.times([0.0, 25e-3]), time=8 * 3600)
    data = fo.Measurement(grid, exp.expected_counts(), exp.pattern, beam, exp.time)
    r = fo.WienerFit(noise_power=1e-3).fit(data, sample, theta0=sample.theta + [0.01, 2e-4])
    sig = exp.precision().sigma.values
    assert np.all(np.abs(r.theta - sample.theta) < 0.5 * sig)


def test_pattern_bias(setup):
    sample, grid, beam = setup
    p = fo.Pattern.times([0.0, 12e-3, 22.3e-3, 35.5e-3])
    exp = fo.Experiment(sample, beam, p, time=8 * 3600)
    same = fo.pattern_bias(exp, p)
    assert all(abs(v) < 1e-6 for v in same.values())
    late = fo.pattern_bias(exp, fo.Pattern.times(p.starts + 1e-6))
    assert late["strain"] > 0.5                         # a common delay looks like a longer wavelength


def test_optimise_two_pulses(setup):
    """Two pulses: the best second pulse is where the spectra do not overlap (R = 2)."""
    sample, grid, beam = setup
    exp = fo.Experiment(sample, beam, fo.Pattern.single(200e-6), time=8 * 3600)
    best = fo.optimise_pattern(exp, 2, target="strain", n_restarts=2, n_steps=300, seed=1)
    R = exp.with_pattern(best).gain()
    assert R["strain"] > 1.95 and not best.overlaps(grid.period)
    with pytest.raises(ValueError):
        fo.optimise_pattern(exp, 2, target="norm")


def test_survey_and_monte_carlo(setup):
    sample, grid, beam = setup
    exp = fo.Experiment(sample, beam, fo.Pattern.single(200e-6), time=8 * 3600)
    table = fo.survey(exp, {"one": fo.Pattern.single(200e-6), "two": fo.Pattern.equal(2, period=grid.period)})
    assert np.isclose(table.R_strain.iloc[0], 1.0) and table.R_strain.iloc[1] > 1.9
    est, summary = fo.monte_carlo(exp, fo.RietveldFit(), n=6, seed=3)
    assert summary.n_ok.iloc[0] == 6 and (est.error == "").all()
    assert np.all(np.abs(summary.bias / summary.bound) < 3)
    par, _ = fo.monte_carlo(exp, fo.RietveldFit(), n=6, seed=3, workers=2)   # same draws in parallel
    assert np.allclose(par[sample.free].values, est[sample.free].values)


def test_plan(setup):
    """plan: an optimised pattern, its gain, and time and precision that invert each other."""
    sample, grid, _ = setup
    ins = fo.Instrument(efficiency=0.2, count_variance=2.0)
    p = fo.plan(sample, target="strain", n_pulses=[1, 2], instrument=ins, n_restarts=1, n_steps=150, seed=1)
    assert p.pattern.n_pulses == 2 and p.gain["strain"] > 1.9 and not p.pattern.overlaps(grid.period)
    assert list(p.candidates.n_pulses) == [1, 2]
    t = p.time_for(bin_size=1.0, strain=500)
    assert np.isclose(p.precision(time=t, bin_size=1.0).set_index("parameter").sigma["strain"], 500)
    s1 = p.precision(8 * 3600, 1.0).set_index("parameter").sigma
    s2 = p.precision(8 * 3600, 2.0).set_index("parameter").sigma
    assert np.allclose(s1 / s2, 2.0)                      # one over the bin side at constant F
    exact = p.experiment(8 * 3600, 1.0).precision().set_index("parameter").sigma
    assert np.isclose(s1["strain"], exact["strain"] * 1e6, rtol=0.01)
    with pytest.warns(UserWarning):
        p.precision(time=60, bin_size=0.1)                # too few counts for the bound


def test_camera_count_variance():
    F = fo.camera_count_variance([0.109, 1.0, 3.5, 20.0])
    assert np.allclose(F[[0, 2, 3]], [1.10, 2.30, 2.30]) and 2.05 < F[1] < 2.28
    assert np.isclose(float(fo.Instrument.saraf_ii().F(1.0)), F[1])
