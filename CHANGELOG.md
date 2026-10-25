# Changelog

## 1.0.0 (unreleased)

Complete rewrite on top of nbragg, with nbragg's workflow, for the article *Bragg-edge imaging with
programmable overlapping neutron pulses at SARAF-II*.

- `TransmissionModel`: nbragg's TransmissionModel through a pulse pattern, built with the same
  arguments and `vary_*` switches, plus `pattern`, `frequency`, `tstep`, `vary_norm` (fixed by
  default: the open beam is known) and `vary_strain` (the edge position as a smooth strain). Methods
  `fit`, `simulate`, `expected_counts`, `precision`, `gain`, `optimise_pattern`, `monte_carlo`,
  `pattern_bias`, `plot`; nbragg's other model methods pass through. A given nbragg model is used
  with its own copy of the parameters.
- `fit(data, wlmin, wlmax, method="rietveld" | "wiener")`: the Rietveld fit of the overlapped counts
  (Poisson likelihood, nbragg inside the analysis window, a free transmission outside it), or Wiener
  deconvolution as in FOBI followed by a fit of the recovered transmission. Optional constant
  background, common time offset and relative pulse intensities.
- `FitResult` with the attributes of an lmfit ModelResult (`params` with the strain, `fit_report`,
  `redchi`, `best_fit`, `residual`), `transmission()` (the spectrum recovered from the counts) and an
  nbragg-style `plot()` (transmission, best fit and residuals in standard deviations).
- `Data`: overlapped counts with their pattern and open beam; `Data.from_counts` reads nbragg's file
  format; `count_variance` for event-mode cameras (errors scaled by its square root).
- `Pattern`: explicit times, `single`, `equal`, `random`, `maximum_length`, `quadratic_residue`,
  `poldi` (default period 50 ms).
- `plan`, `Plan`, `Instrument`: optimised pulse times for a material and a quantity, their gain, and
  the precision and measurement time against the bin size; `Instrument.saraf_ii()` and
  `camera_count_variance` for the station and camera of the article.
- `OpenBeam`, `TimeGrid` for advanced use (measured or reference spectra, bandwidth chopper bands).
- Three tutorials on simulated data; tests with known answers. The frame_overlap 0.x package,
  its notebooks and apps were removed (they remain in the git history).
