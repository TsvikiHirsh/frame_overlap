"""Tests with known answers for frame_overlap (run: pytest -q)."""
import numpy as np
import pytest

import frame_overlap as fo


# --------------------------------------------------------------- no nbragg

def test_grid():
    g = fo.TimeGrid(9.0, 20.0, 10e-6)
    assert g.n_bins == 5000
    assert np.isclose(g.wavelength_at(g.time_at(4.05)), 4.05)
    assert np.isclose(g.wavelength_at(12.5e-3), 5.494, atol=1e-3)


def test_pattern_arrays():
    g = fo.TimeGrid(9.0, 20.0, 10e-6)
    p = fo.Pattern.equal(4, 200e-6, g.period)
    assert p.times_ms == [0.0, 12.5, 25.0, 37.5]
    assert np.isclose(p.array(g).sum(), 4 * 200e-6)
    wrap = fo.Pattern([49.9e-3], 200e-6)              # wraps around the period
    a = wrap.array(g)
    assert np.isclose(a.sum(), 200e-6) and a[0] > 0 and a[-1] > 0
    frac = fo.Pattern([5e-6], 10e-6).array(g)            # half bins
    assert np.allclose(frac[:2], [5e-6, 5e-6])
    assert not p.overlaps(g.period) and fo.Pattern([0, 100e-6], 200e-6).overlaps(g.period)


def test_flat_sequences():
    for order in (2, 3, 4, 5):
        p = fo.Pattern.maximum_length(order)
        assert p.n_pulses == 2 ** (order - 1)
    assert fo.Pattern.quadratic_residue(11).n_pulses == 5
    assert fo.Pattern.quadratic_residue(11, complement=True).n_pulses == 6
    g = fo.TimeGrid(9.0, 20.0, 10e-6)
    for p in (fo.Pattern.maximum_length(4), fo.Pattern.quadratic_residue(7)):
        v = round(g.period / (p.starts[1] - p.starts[0])) if p.starts[1] > 0 else None
        assert p.n_pulses > 1 and v is not None
    assert fo.Pattern.poldi().n_pulses == 8


def test_open_beam():
    g = fo.TimeGrid(9.0, 20.0, 10e-6)
    b = fo.OpenBeam.maxwellian(g, flux=1e6, area=2.0, efficiency=0.5)
    assert np.isclose(b.total_rate(), 1e6, rtol=1e-6)
    band = b.with_band(1.0, 5.0)
    lam = g.wavelength
    assert np.all(band.rate[(lam < 1.0) | (lam >= 5.0)] == 0)
    counts = b.counts(fo.Pattern.single(200e-6), 3600)
    assert np.isclose(counts.sum(), 1e6 * 200e-6 * 20 * 3600, rtol=1e-6)
    short = fo.OpenBeam.from_counts(g, np.exp(-np.arange(2400) / 500.0), 1.0, 1.0)
    assert short.rate.shape == (5000,) and np.isclose(short.rate[2400], np.exp(-2400 / 500), rtol=0.05)


# ------------------------------------------------------------------ nbragg

nbragg = pytest.importorskip("nbragg")


@pytest.fixture(scope="module")
def setup():
    xs = nbragg.CrossSection(materials={"iron": {"mat": "Fe_sg229_Iron-alpha.ncmat", "weight": 1.0}})
    model = nbragg.TransmissionModel(xs, response="jorgensen_inv", tof_length=9.0)
    model.params["thickness"].set(value=2.0)
    sample = fo.Sample(model, free=["thickness", "strain"])
    grid = fo.TimeGrid(9.0, 20.0, 10e-6)
    beam = fo.OpenBeam.maxwellian(grid, flux=5e6, area=0.03)
    return sample, grid, beam


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
    sample, grid, beam = setup
    exp = fo.Experiment(sample, beam, fo.Pattern.times(starts), time=8 * 3600)
    data = fo.Measurement(grid, exp.expected_counts(), exp.pattern, beam, exp.time)
    sig = exp.precision().sigma.values
    r = fo.RietveldFit().fit(data, sample, theta0=sample.theta + sig)
    assert r.success
    assert np.all(np.abs(r.theta - sample.theta) < 0.05 * sig)


def test_two_separate_pulses_double_the_information(setup):
    sample, grid, beam = setup
    exp = fo.Experiment(sample, beam, fo.Pattern.times([0.0, 25e-3]), time=8 * 3600)
    R = exp.gain()
    assert np.isclose(R["thickness"], 2.0, rtol=0.02) and np.isclose(R["strain"], 2.0, rtol=0.02)


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


def test_survey_and_monte_carlo(setup):
    sample, grid, beam = setup
    exp = fo.Experiment(sample, beam, fo.Pattern.single(200e-6), time=8 * 3600)
    table = fo.survey(exp, {"one": fo.Pattern.single(200e-6), "two": fo.Pattern.equal(2, period=grid.period)})
    assert np.isclose(table.R_strain.iloc[0], 1.0) and table.R_strain.iloc[1] > 1.9
    est, summary = fo.monte_carlo(exp, fo.RietveldFit(), n=6, seed=3)
    assert summary.n_ok.iloc[0] == 6
    assert np.all(np.abs(summary.bias / summary.bound) < 3)
