# FOBI Code & Paper Improvement Plan

**Goal**: Demonstrate in the SARAF-II paper that frame-overlap Bragg-edge imaging (FOBI)
with the digital chopper achieves a **large measurement-time reduction factor** at fixed
parameter precision, without the systematic effects that currently dominate the results.

**Repos**:
- Code: `/work/nuclear/frame_overlap`
- Paper: `/work/papers/SARAF_bragg_paper/manuscript.tex`

---

## Part 1 — Diagnosis: where the systematic effects come from

After reviewing `data_class.py`, `reconstruct.py`, `analysis_nbragg.py` and the
`.documents/` history, the "synthetic features" reported in the paper (cellulose fraction
biased from 2% to 8–9%, χ² growth, edge-position scatter) are traceable to five concrete
mismatches between the simulation, the deconvolution model, and the statistics. These are
not fundamental limits of FOBI — they are fixable model errors.

### 1.1 The overlap operator does not match the deconvolution kernel (CRITICAL)

- `Data._create_overlap(mode='superimpose')` shifts each frame and **drops** the part of
  the signal that falls past the end of the time window (no wrap-around).
- Wiener deconvolution via FFT (`Reconstruct._wiener_filter`, `_fobi_filter`) assumes
  **circular** convolution: `observed = signal ⊛ kernel (mod T)`.
- In a real continuously-pulsed instrument in steady state, frame overlap **is** circular:
  slow neutrons from pulse *k−1* arrive during the window of pulse *k*. So the physical
  operation is circular, the FFT model is circular, but the simulation is linear-truncated.
- Consequence: the deconvolution model is wrong precisely at the frame boundaries and for
  every frame whose tail leaves the window → low-frequency baseline warping over the whole
  spectrum → the fit absorbs it into the amorphous (cellulose) fraction and background.

**Fix**: make `superimpose` wrap contributions with modulo arithmetic
(`target_bin % n_bins`), exactly like the real steady-state physics and exactly like the
FFT assumes. Round-trip test: noiseless signal → overlap → Wiener with tiny noise_power
must recover the input to machine precision.

### 1.2 Sub-bin placement is inconsistent between overlap and kernel

- The overlap places frames at `int(round(frame_start/bin_width))` (whole bins).
- `_reconstruct_kernel(interpolate=True)` builds a fractional, two-bin interpolated kernel.
- If the simulation shifts by whole bins but the kernel says fractional bins (or vice
  versa), every frame has up to a half-bin phase error → ringing at Bragg edges.

**Fix**: one shared kernel-construction function used by *both* the overlap operation and
the reconstruction. When the simulation shifts by whole bins, the kernel must be discrete
deltas at the same bins. (Keep interpolated kernels for future real data where delays are
truly fractional.)

### 1.3 Poisson noise is applied at the wrong stage and is correlated between signal and openbeam

- Current recommended workflow is `convolute → poisson → overlap`. Overlap then *averages*
  N noisy copies, so the overlapped spectrum's noise is not Poisson-distributed for the
  counts it represents, and the "second poisson" band-aid (`poisson_seed`) double-counts noise.
- Real measurement: neutrons are counted **after** overlap. Correct order is
  `convolute → overlap (sum, not average) → scale to expected counts → poisson`.
- Additionally, `poisson_sample(seed=k)` resets the RNG to the same seed for the signal and
  the openbeam → their Poisson fluctuations are **correlated**, which artificially cancels
  noise in the transmission ratio and makes results look better than they are.
- The `/ n_frames` normalization in overlap (and `* n_frames` in reconstruct) hides the
  actual FOBI statistics: N frames per period = N× the counts per unit wall-clock time.
  That factor is the entire point of the method and must be explicit in the count budget.

**Fix**: overlap sums counts (no division); Poisson applied after overlap with independent
RNG streams (`np.random.default_rng(seed)` and `default_rng(seed + large_offset)` or a
`SeedSequence.spawn`); duty-cycle bookkeeping expressed as
`expected_counts = reference_counts × (flux ratio) × (time ratio) × n_pulses_per_period × (pulse_duration ratio)`.

### 1.4 Errors after deconvolution are wrong

- `Reconstruct.filter` sets `err = sqrt(counts)` on the deconvolved signal. Deconvolved
  bins are neither Poisson nor independent; the Wiener filter reshapes and correlates the
  noise. Feeding `sqrt(counts)` errors into nbragg biases χ² and every parameter
  uncertainty (and partially explains why χ²/dof looks reasonable while parameters are off).

**Fix**: propagate the per-bin variance through the filter:
`var_recon = IFFT-diag of |G_wiener(f)|² · PSD_noise` — for stationary Poisson noise, the
per-bin variance of the reconstruction is `var_x̂ ≈ (1/N) Σ_f |G(f)|² · mean(var_obs)`,
or do it exactly with a small Monte-Carlo (bootstrap ~50 Poisson replicas, take per-bin std).
Bootstrap is simple, exact, and cheap at 2400 bins. Off-diagonal correlations remain
unmodeled in the fit — acknowledge in the paper, or use forward fitting (1.5) which
avoids the problem entirely.

### 1.5 The methodological upgrade: forward-model fitting (no deconvolution at all)

Deconvolution is an ill-posed inverse step performed on the *data*; every regularization
choice (noise_power) trades bias against noise, which is exactly the systematic effect
described in the paper. The standard cure (same logic as Rietveld refinement: convolve the
model, don't deconvolve the data):

> **Fit the overlapped spectrum directly**: take the nbragg transmission model
> `T_model(λ)`, map to TOF, apply the *known* chopper kernel circularly (openbeam-weighted
> sum of shifted copies), and fit the raw overlapped counts with Poisson/χ² statistics.

- No regularization parameter, no synthetic features, exact likelihood, valid error bars.
- Works at any overlap density (even kernels that a Wiener filter cannot invert stably,
  e.g. near-singular |G(f)| ≈ 0 frequencies), because the fit only needs the forward map.
- The overlapped **openbeam is measured** (same kernel), so the model for the overlapped
  sample spectrum is `sum_i OB(t - t_i) · T_model(t - t_i)` — a weighted circular
  convolution. Implement as: `S_model(t) = Σ_i OB_shift_i(t) · T_model(t - t_i)`,
  normalized against `Σ_i OB_shift_i(t)`.
- Keep Wiener reconstruction in the paper as the "conventional FOBI" baseline and show the
  forward fit removes its bias → this *is* a paper-worthy result on its own.

Implementation: new module `src/frame_overlap/forward_fit.py` with class
`ForwardFit` (wraps `nbragg.TransmissionModel`-like fitting via lmfit):
1. Build wavelength→TOF grid from L and bin width (reuse nbragg conversion).
2. Model: `T(λ; θ)` from nbragg CrossSection (thickness, weights, background, response).
3. Overlap operator `A` (sparse circular shift-sum with openbeam weighting) applied to the
   model prediction.
4. Residuals `(S_obs - A[OB·T])/σ_obs` with `σ_obs = sqrt(S_obs)` — valid because S_obs
   are genuine counts. Fit with lmfit.

### 1.6 Smaller issues (fix opportunistically)

- `analysis_nbragg.py` line 8: `import numpy as pd` shadowed by `import pandas as pd` — a
  latent bug (works only by accident).
- `Reconstruct.optimize_noise` optimizes χ² against a *reference that exists only in
  simulation* — with real data this is impossible. Add a reference-free criterion or note
  the limitation (forward fit makes it moot).
- `_create_overlap` superimpose loop is O(N_frames × N_bins) in pure Python — vectorize
  with `np.add.at` / `np.roll` (needed anyway for the 50-realization sweeps).
- Wiener `noise_power` as an absolute constant is scale-dependent (counts² units); express
  it relative to `|G|²·SNR` or document units.

---

## Part 2 — Demonstrating the measurement-time reduction (the paper's new claim)

### 2.1 The experiment design (all simulation, LANSCE reference data as truth)

Baseline ("conventional"): single 200 µs pulse per 50 ms period (20 Hz), Poisson-scaled to
SARAF-II flux (m=7 guide column of Table 2), wall-clock time T.

FOBI: N pulses of the same 200 µs width per 50 ms period (N = 2…16, random min-gap
patterns), same wall-clock time T → N× the integrated counts. Analyze both with:
(a) Wiener deconvolution + nbragg fit (current method), and
(b) forward-model fit (new method).

### 2.2 Metrics and figures

For each (N, T): repeat over ≥30 Poisson realizations, record thickness, cellulose weight,
edge position, and their spread.

- **Fig A (key figure)**: parameter uncertainty σ(thickness) [and σ(edge position)] vs
  wall-clock time T, curves for conventional, FOBI-Wiener, FOBI-forward-fit. Equal-precision
  horizontal cut gives the **time-reduction factor** — expect ≈ N for the forward fit
  (√N in precision), saturation/bias floor for Wiener.
- **Fig B**: bias comparison — extracted cellulose fraction & thickness vs N for
  Wiener vs forward fit (shows the systematic is removed).
- **Fig C**: spectra illustration — overlapped raw data, Wiener reconstruction with
  artifacts, forward-model best fit overlaid on the overlapped data.
- Optional: kernel design study (random vs min-gap vs equal spacing) using the
  forward-fit Fisher information / condition number of A as the ranking metric.

### 2.3 Manuscript changes (`manuscript.tex`)

1. **Methods → new subsection** "Forward-model analysis of frame-overlap data": Eq. for the
   circular overlap operator, fit of raw overlapped counts, note on why deconvolution-based
   FOBI has an intrinsic bias-variance trade-off.
2. **Results → rewrite §"Frame-overlap Bragg-edge imaging (FOBI)"**: replace the current
   defensive text (seed sensitivity, Wiener noise-power sensitivity, 8–9% biased cellulose)
   with: (i) Wiener baseline reproduces literature FOBI incl. its artifacts; (ii) forward
   fit removes bias; (iii) Fig A time-reduction factor ~N× (state the achieved number,
   e.g. "a factor of ≈8 reduction in measurement time at equal thickness precision").
3. **Discussion**: replace "current reconstruction methods introduce synthetic features…"
   caveat with the demonstrated solution; keep adaptive-pattern outlook.
4. **Conclusions item 4**: upgrade from a limitation statement to a quantified capability.
5. **Abstract**: replace "can improve wavelength coverage" with the quantified time-reduction
   claim.

### 2.4 Honest-limitations section (keep the paper credible)

- Detector saturation/pile-up at N× instantaneous rate (LumaCam event mode mitigates).
- Kernel knowledge accuracy (chopper timing jitter) — digital chopper at SARAF is
  programmable and monitored, cite spec.
- Background is also overlapped — handled by the same operator in the forward model.
- Frame-overlap gain applies while the source is duty-cycle-limited (true for chopped CW).

---

## Part 3 — Execution checklist

- [ ] 1. `data_class.py`: circular superimpose (`wrap=True` default), sum instead of
      average, shared kernel placement helper, vectorized overlap.
- [ ] 2. `data_class.py`: Poisson after overlap workflow; independent RNG streams for
      signal/openbeam; explicit count-budget helper
      (`scale_counts(flux_ratio, time_ratio, n_frames, pulse_ratio)`).
- [ ] 3. `reconstruct.py`: kernel from shared helper; bootstrap error propagation
      (`n_boot` parameter); keep old behavior available for comparison.
- [ ] 4. New `forward_fit.py`: `ForwardFit` class (overlap operator + nbragg model + lmfit).
- [ ] 5. `tests/test_roundtrip_circular.py`: noiseless round-trip = machine precision;
      Poisson-scaling test: σ(param) ∝ 1/√T and ∝ 1/√N.
- [ ] 6. `paper_analysis/` script: run the (N, T) grid, ≥30 realizations, save CSV +
      figures A–C as JPEG into the paper's `figures/`.
- [ ] 7. Update `manuscript.tex` per §2.3.
- [ ] 8. Fix `analysis_nbragg.py` import bug; run existing test suite; note API changes in
      CHANGELOG.

Validation gates: step 5 must pass before running step 6; step 6 numbers go into the
manuscript only from the saved CSVs (no hand-copied values).
