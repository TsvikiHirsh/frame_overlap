"""
Forward-model fitting of frame-overlapped TOF spectra.

Instead of deconvolving the measured overlapped spectrum (an ill-posed inverse
problem whose regularization introduces synthetic spectral features), this
module fits the overlapped counts *directly*: the nbragg transmission model is
evaluated on the single-frame TOF grid, multiplied by the known single-frame
openbeam template, passed through the same circular frame-overlap operator
that the chopper applies, and compared to the raw overlapped counts with
Poisson statistics.

This is the same logic as Rietveld refinement: convolve the model with the
instrument function rather than deconvolving the data. It removes the
bias-variance trade-off of Wiener deconvolution entirely: no noise_power
parameter, no synthetic features, and statistically valid uncertainties.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit


class ForwardFit:
    """
    Fit frame-overlapped transmission data directly with a forward model.

    The model for the overlapped signal counts is

        S_ov(t) = sum_i  OB(t - t_i) * T(lambda(t - t_i); theta)   (mod T_window)

    where OB is the known single-frame openbeam template, t_i are the frame
    start times of the chopper pattern (data.kernel), and T is the nbragg
    transmission model with parameters theta (thickness, phase weights,
    background, response).

    Parameters
    ----------
    data : Data
        Data object processed with the correct workflow:
        convolute_response -> overlap(mode='superimpose') -> poisson_sample.
        The observed spectrum is data.table (overlapped + Poisson noise);
        the openbeam template is data.op_convolved_data scaled by the applied
        duty cycle.
    xs : str or nbragg.CrossSection, optional
        Cross-section specification, same options as Analysis
        (default 'iron_with_cellulose').
    L : float, optional
        Flight path length in meters. Default is 9.0.
    **model_kwargs
        Passed to Analysis / nbragg.TransmissionModel
        (vary_background, vary_weights, vary_response, response_kind, ...).

    Examples
    --------
    >>> data = Data('iron_powder.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)
    >>> data.convolute_response(200).overlap(8, freq=20, mode='random_min_gap',
    ...                                      min_gap=2.0, kernel_seed=1)
    >>> data.poisson_sample(duty_cycle=0.1, seed=42)
    >>> ff = ForwardFit(data, xs='iron_with_cellulose', vary_background=True)
    >>> result = ff.fit()
    >>> print(result.params['thickness'])
    >>> ff.plot()
    """

    def __init__(self, data, xs='iron_with_cellulose', L=9.0, wlmin=1.0, wlmax=6.0,
                 **model_kwargs):
        from .data_class import Data, frame_starts_to_kernel
        from .analysis_nbragg import Analysis

        if not isinstance(data, Data):
            raise TypeError("data must be a Data object")
        if data.kernel is None:
            raise ValueError("Data must have an overlap kernel (call data.overlap first)")
        if data.op_convolved_data is None and data.op_data is None:
            raise ValueError("Data must have openbeam loaded for the forward model")

        self.data = data
        self.L = L
        self.wlmin = wlmin
        self.wlmax = wlmax

        # Reuse Analysis for cross-section and TransmissionModel construction
        self._analysis = Analysis(xs=xs, **model_kwargs)
        self.model = self._analysis.model
        self.params = self.model.params.copy()
        self.result = None

        self._setup_grids()

    # ------------------------------------------------------------------ setup

    def _setup_grids(self):
        """Precompute TOF/wavelength grids, openbeam template and frame shifts."""
        import nbragg.utils as nb_utils
        import NCrystal as NC

        observed = self.data.table  # overlapped (+ Poisson) signal counts
        self.time_us = observed['time'].values.astype(float)
        self.observed_counts = observed['counts'].values.astype(float)
        self.observed_err = np.sqrt(np.maximum(self.observed_counts, 1.0))

        if len(self.time_us) > 1:
            self.bin_width = self.time_us[1] - self.time_us[0]
        else:
            self.bin_width = 10.0
        self.n_bins = len(self.time_us)

        # Wavelength of each single-frame TOF bin (same conversion as
        # nbragg.Data.from_counts: relativistic time2energy + ekin2wl)
        t_seconds = np.maximum(self.time_us, self.bin_width) * 1e-6
        energy = nb_utils.time2energy(t_seconds, self.L)
        self.wl = np.array([NC.ekin2wl(e) for e in energy])

        # Single-frame openbeam template scaled to expected counts per frame.
        duty = getattr(self.data, 'applied_duty_cycle', None) or 1.0
        ob_src = (self.data.op_convolved_data if self.data.op_convolved_data is not None
                  else self.data.op_data)
        self.openbeam_template = ob_src['counts'].values.astype(float) * duty

        # Frame start bins (identical placement to Data._create_overlap)
        kernel_us = np.array(self.data.kernel) * 1000.0
        frame_starts_us = np.cumsum(kernel_us)
        self.frame_shift_bins = [int(np.round(s / self.bin_width)) % self.n_bins
                                 for s in frame_starts_us]

        # Restrict the fit to bins where at least the direct frame carries
        # wavelength information inside [wlmin, wlmax]; by default use all bins
        # with positive openbeam. The overlapped spectrum mixes wavelengths, so
        # the natural choice is to fit the full time window.
        self.fit_mask = self.openbeam_template > 0

    # ------------------------------------------------------------ forward model

    def eval_overlapped(self, params=None):
        """
        Evaluate the forward model: overlapped signal counts for given params.

        Returns
        -------
        np.ndarray
            Predicted overlapped counts on the observed TOF grid.
        """
        if params is None:
            params = self.params

        # Transmission on the single-frame wavelength grid
        T = self.model.eval(params=params, wl=self.wl)

        # Single-frame expected signal counts
        s_single = self.openbeam_template * T

        # Circular frame-overlap operator (identical to Data._create_overlap)
        s_ov = np.zeros(self.n_bins)
        for shift in self.frame_shift_bins:
            s_ov += np.roll(s_single, shift)
        return s_ov

    def residual(self, params):
        """Weighted residuals of the overlapped counts (Poisson sigma)."""
        model_counts = self.eval_overlapped(params)
        resid = (self.observed_counts - model_counts) / self.observed_err
        return resid[self.fit_mask]

    # ------------------------------------------------------------------- fit

    def fit(self, params=None, method='leastsq', **fit_kwargs):
        """
        Fit the overlapped spectrum.

        Parameters
        ----------
        params : lmfit.Parameters, optional
            Starting parameters (default: the model's parameters).
        method : str, optional
            lmfit minimization method. Default 'leastsq'.

        Returns
        -------
        lmfit.MinimizerResult
            Fit result; also stored as self.result.
        """
        if params is None:
            params = self.params

        minimizer = lmfit.Minimizer(self.residual, params)
        self.result = minimizer.minimize(method=method, **fit_kwargs)
        self.params = self.result.params
        return self.result

    # ------------------------------------------------------------------ plots

    def plot(self, fontsize=14, figsize=(10, 8)):
        """Plot observed vs fitted overlapped counts with residuals."""
        model_counts = self.eval_overlapped(self.params)
        time_ms = self.time_us / 1000.0

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.3], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        axr = fig.add_subplot(gs[1], sharex=ax)

        ax.step(time_ms, self.observed_counts, where='mid', alpha=0.6,
                label='Observed (overlapped)')
        ax.plot(time_ms, model_counts, 'r-', lw=1.2, label='Forward model')
        ax.set_ylabel('Counts', fontsize=fontsize)
        redchi = getattr(self.result, 'redchi', None)
        title = f'Forward fit (χ²/dof = {redchi:.2f})' if redchi else 'Forward model'
        ax.legend(title=title, fontsize=fontsize - 2)
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.3)

        resid = (self.observed_counts - model_counts) / self.observed_err
        axr.step(time_ms, resid, where='mid', color='k')
        axr.axhline(0, color='r', ls='--', alpha=0.5)
        axr.set_xlabel('Time (ms)', fontsize=fontsize)
        axr.set_ylabel('Resid. (σ)', fontsize=fontsize)
        axr.grid(alpha=0.3)
        return fig

    def __repr__(self):
        fitted = self.result is not None
        redchi = f"{self.result.redchi:.2f}" if fitted else "N/A"
        return (f"ForwardFit(n_frames={len(self.data.kernel)}, "
                f"fitted={fitted}, redchi={redchi})")
