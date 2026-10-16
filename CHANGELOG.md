# Changelog

## 1.0.0 (unreleased)

Complete rewrite on top of nbragg.

- `TimeGrid`, `OpenBeam`, `Pattern`, `Sample`, `Experiment`, `Measurement`: describe a measurement
  design with an nbragg material model, an open beam, a pulse pattern and a measurement time.
- `RietveldFit`: forward fit of (overlapped) counts through the pulse pattern with the Poisson
  likelihood, nbragg inside an analysis window and a free transmission outside it.
- `WienerFit`: Wiener deconvolution of sample and open beam as in frame-overlap Bragg-edge imaging.
- `precision`, `gain`, `survey`, `optimise_pattern`, `monte_carlo`, `pattern_bias`: design tools.
- `"strain"` as a smooth edge-position parameter (nbragg passes lattice parameters with four decimals).
- Tutorials in `notebooks/` on simulated data; tests in `tests/`.
- The version 0.x package, notebooks and apps are kept in `legacy/` and no longer installed.
