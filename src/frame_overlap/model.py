"""nbragg's TransmissionModel seen through a pulse pattern."""
import warnings
import numpy as np
import pandas as pd
from .grid import TimeGrid
from .pattern import Pattern
from .openbeam import OpenBeam
from ._sample import Sample
from ._experiment import Experiment
from . import design
from .fitting import RietveldFit, WienerFit

# inverse Jorgensen moderator response close to a thermal water moderator; nbragg's
# defaults are much broader. Calibrate it for your source.
DEFAULT_RESPONSE = {"α0": 3000.0, "β0": 50.0}
UNITS = {"thickness": (1e4, "um"), "strain": (1e6, "microstrain")}


class TransmissionModel:
    """
    nbragg's ``TransmissionModel`` for overlapping pulses.

    The expected counts of a pulse pattern are the nbragg transmission of one
    pulse, times the open beam, convolved with the pattern over the period:

        mu(t) = n_periods * [c (*) (o T)](t).

    ``fit`` determines the free parameters from overlapped counts by a
    Rietveld fit of the counts through the pattern (Poisson likelihood), or
    by Wiener deconvolution as in FOBI for comparison; ``simulate`` makes a
    projected measurement; ``precision``, ``gain`` and ``optimise_pattern``
    plan one. The model is built as in nbragg, with the same ``vary_*``
    switches, plus the pulse pattern.

    Parameters
    ----------
    cross_section : nbragg.CrossSection or nbragg.TransmissionModel
        The sample, as in nbragg; an nbragg model is used as it is.
    pattern : Pattern
        The pulse pattern (default: one 200 us pulse per period).
    tof_length : float
        Flight path [m] (default 9, or that of a given nbragg model).
    frequency : float
        Repetition rate of the pattern [Hz].
    tstep : float
        Time bin [s].
    response, background : str
        nbragg's response and background kinds. The response is applied with
        fixed parameters unless ``vary_response=True``; the default
        inverse-Jorgensen parameters suit a thermal water moderator.
    vary_basic : bool
        Fit the thickness (nbragg's default). The normalisation stays fixed:
        the open beam is known (see ``vary_norm``).
    vary_norm : bool
        Also fit the normalisation (costs most of the thickness precision).
    vary_strain : bool
        Fit the Bragg-edge position as a uniform strain of the wavelength axis
        (d lambda / lambda). nbragg passes lattice parameters to NCrystal with
        four decimals, so ``strain`` is the smooth way to fit edge positions.
    vary_weights, vary_background, vary_tof, vary_response, vary_lattice,
    vary_orientation, vary_extinction, vary_sans : bool
        As in nbragg.
    strain : float
        Strain of the model (the true value of a simulation, the starting
        value of a fit).
    substeps, fine_range, steps : advanced
        Wavelengths per time bin at which nbragg is evaluated inside
        ``fine_range`` [A] (the transmission is averaged over each bin), and
        finite-difference steps per parameter.
    **kwargs
        Passed to ``nbragg.TransmissionModel``.

    Examples
    --------
    >>> import nbragg, frame_overlap as fo                                  # doctest: +SKIP
    >>> xs = nbragg.CrossSection(iron="Fe_sg229_Iron-alpha.ncmat")
    >>> model = fo.TransmissionModel(xs, pattern=fo.Pattern.equal(2), vary_basic=True, vary_strain=True)
    >>> model.params["thickness"].set(value=2.0)
    >>> data = model.simulate(time=8 * 3600, flux=5e6, area=0.72, seed=1)
    >>> result = model.fit(data)
    >>> result.plot()
    """

    def __init__(self, cross_section, pattern=None, tof_length=None, frequency=20.0, tstep=10e-6,
                 response="jorgensen_inv", background="polynomial3", vary_basic=None, vary_norm=False,
                 vary_strain=False, vary_weights=None, vary_background=None, vary_tof=None, vary_response=False,
                 vary_lattice=None, vary_orientation=None, vary_extinction=None, vary_sans=None, strain=0.0,
                 substeps=16, fine_range=(0.5, 6.0), steps=None, **kwargs):
        import nbragg
        if isinstance(cross_section, nbragg.TransmissionModel):
            self.nbragg = cross_section
            self._params = cross_section.params.copy()     # its own copy: the given model stays as it is
            tof_length = tof_length or getattr(cross_section, "tof_length", 9.0)
        else:
            tof_length = tof_length or 9.0
            self.nbragg = nbragg.TransmissionModel(
                cross_section, response=response, background=background, tof_length=tof_length,
                vary_basic=vary_basic, vary_weights=vary_weights, vary_background=vary_background, vary_tof=vary_tof,
                vary_response=vary_response, vary_lattice=vary_lattice, vary_orientation=vary_orientation,
                vary_extinction=vary_extinction, vary_sans=vary_sans, **kwargs)
            p = self._params = self.nbragg.params
            if response == "jorgensen_inv" and vary_response is not None:
                for k, v in DEFAULT_RESPONSE.items():
                    if k in p:
                        p[k].set(value=v)
            if "norm" in p:
                p["norm"].set(vary=bool(vary_norm))
        if vary_lattice:
            warnings.warn("nbragg passes lattice parameters to NCrystal with four decimals; vary_strain=True fits the "
                          "edge position smoothly", stacklevel=2)
        self.grid = TimeGrid(tof_length, frequency, tstep)
        self.pattern = pattern if pattern is not None else Pattern.single(200e-6)
        self.pattern.check(self.grid)
        self.strain, self.vary_strain = float(strain), bool(vary_strain)
        self.substeps, self.fine_range, self.steps = int(substeps), tuple(fine_range), dict(steps or {})
        self.result = None

    # ------------------------------------------------------------ parameters
    @property
    def params(self):
        """The lmfit Parameters of the nbragg model (the strain is ``model.strain``)."""
        return self._params

    def copy(self, pattern=None):
        """A copy with its own parameters (and another pattern, if given)."""
        new = TransmissionModel.__new__(TransmissionModel)
        new.__dict__.update(self.__dict__)
        new._params = self._params.copy()
        new.steps = dict(self.steps)
        new.result = None
        if pattern is not None:
            new.pattern = pattern
            pattern.check(new.grid)
        return new

    @property
    def free(self):
        """Names of the parameters a fit determines (the varying nbragg parameters, and strain)."""
        names = [n for n, p in self.params.items() if p.vary and not p.expr]
        return names + (["strain"] if self.vary_strain else [])

    def __getattr__(self, name):                       # nbragg's other methods (show_available_params, ...)
        if name in ("nbragg", "_params", "__setstate__") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.nbragg, name)

    def _sample(self, free=None):
        free = self.free if free is None else list(free)
        if not free:
            raise ValueError("nothing to determine: switch on vary_basic, vary_strain, ... or set params[...].vary")
        return Sample(self.nbragg, params=self.params, free=free, substeps=self.substeps,
                      fine_range=self.fine_range, steps=self.steps, strain=self.strain)

    # ------------------------------------------------------------ the beam
    def openbeam(self, flux=5e6, area=1.0, efficiency=1.0, temperature=300.0, band=None):
        """
        Open beam of one pulse from a thermal (Maxwellian) moderator: a
        continuous flux [n/cm^2/s] at the sample, the detector area [cm^2]
        and efficiency, and optionally an ideal bandwidth chopper passing
        ``band = (lambda_min, lambda_max)`` [A]. For a measured spectrum use
        ``OpenBeam.from_counts`` or ``OpenBeam.from_reference`` on ``model.grid``.
        """
        ob = OpenBeam.maxwellian(self.grid, flux=flux, temperature=temperature, area=area, efficiency=efficiency)
        return ob.with_band(*band) if band is not None else ob

    def _experiment(self, time, openbeam=None, flux=5e6, area=1.0, efficiency=1.0, temperature=300.0, band=None,
                    count_variance=1.0, background=0.0, pattern=None, free=None):
        if openbeam is None:
            beam = self.openbeam(flux, area, efficiency, temperature, band)
        else:
            if openbeam.grid != self.grid:
                raise ValueError(f"the open beam is on {openbeam.grid!r}, the model on {self.grid!r}")
            beam = openbeam.with_band(*band) if band is not None else openbeam
        return Experiment(self._sample(free), beam, pattern or self.pattern, time, background, count_variance)

    # ------------------------------------------------------------ measurements
    def expected_counts(self, time=8 * 3600, **beam):
        """Expected counts per time bin of a measurement of ``time`` seconds (beam keywords as in ``simulate``)."""
        return self._experiment(time, **beam).expected_counts()

    def simulate(self, time=8 * 3600, seed=None, openbeam=None, flux=5e6, area=1.0, efficiency=1.0,
                 temperature=300.0, band=None, count_variance=1.0, background=0.0):
        """
        A projected measurement of ``time`` seconds with the model's values as
        the truth: Poisson counts (or ``count_variance`` times the Poisson
        variance, as recorded by event-mode cameras). The beam is a thermal
        spectrum of ``flux`` [n/cm^2/s] on ``area`` [cm^2] with the detector
        ``efficiency``, or a given ``openbeam``. Returns a ``Data`` object.
        """
        return self._experiment(time, openbeam, flux, area, efficiency, temperature, band, count_variance,
                                background).simulate(seed)

    def fit(self, data, wlmin=1.0, wlmax=5.0, method="rietveld", noise_power=1e-4, fit_background=False,
            fit_t0=False, fit_intensities=False, prior_sigma=1.0, knots=None, start=None, max_nfev=200):
        """
        Fit overlapped counts (a ``Data`` object).

        Parameters
        ----------
        data : Data
        wlmin, wlmax : float
            Analysis window [A]. Inside it the transmission is the nbragg
            model; outside it a free function of time (with a weak prior), so
            that model errors there cannot bias the result.
        method : "rietveld" or "wiener"
            ``"rietveld"``: the model is convolved with the pulse pattern and
            fitted to the counts with the Poisson likelihood (reaches the
            information limit for any pattern). ``"wiener"``: Wiener
            deconvolution of sample and open beam with ``noise_power``, as in
            FOBI, and a fit of their ratio with independent bins.
        fit_background, fit_t0, fit_intensities : bool
            Also determine a constant count background, a common time offset
            of the pulses or the relative pulse intensities (Rietveld fit).
        start : sequence, optional
            Starting values of the free parameters (default: the model's).

        Returns
        -------
        FitResult
            With ``params``, ``fit_report()``, ``plot()`` and ``transmission()``.
        """
        if abs(data.grid.flight_path - self.grid.flight_path) > 1e-9:
            warnings.warn(f"the data are at {data.grid.flight_path:g} m, the model at {self.grid.flight_path:g} m; "
                          "the data's flight path is used", stacklevel=2)
        m = method.lower()
        if m == "rietveld":
            engine = RietveldFit(window=(wlmin, wlmax), prior_sigma=prior_sigma, fit_background=fit_background,
                                 fit_t0=fit_t0, fit_intensities=fit_intensities, max_nfev=max_nfev, knots=knots)
        elif m == "wiener":
            engine = WienerFit(noise_power=noise_power, window=(wlmin, wlmax), max_nfev=max_nfev)
        else:
            raise ValueError('method must be "rietveld" or "wiener"')
        self.result = engine.fit(data, self._sample(), theta0=start)
        return self.result

    # ------------------------------------------------------------ planning
    def precision(self, time=8 * 3600, wlmin=1.0, wlmax=5.0, pattern=None, knots=None, background=False,
                  t0=False, intensities=False, **beam):
        """
        One standard deviation of each free parameter that a measurement of
        ``time`` seconds reaches (Cramer-Rao bound of the Rietveld fit), with
        the model's pattern or another ``pattern``. Beam keywords as in
        ``simulate`` (including ``count_variance``). ``background``, ``t0``,
        ``intensities``: treat these as unknown too.
        Returns a DataFrame (strain also in microstrain, thickness in um).
        """
        exp = self._experiment(time, pattern=pattern, **beam)
        df = design.precision(exp, (wlmin, wlmax), background=background, t0=t0, intensities=intensities,
                              knots=knots).drop(columns="sigma_microstrain")
        df["sigma_display"] = [s * UNITS.get(n, (1.0, ""))[0] for n, s in zip(df.parameter, df.sigma)]
        df["unit_display"] = [UNITS.get(n, (1.0, u))[1] for n, u in zip(df.parameter, df.unit)]
        return df

    def gain(self, reference=None, wlmin=1.0, wlmax=5.0, pattern=None, **beam):
        """
        Time-reduction factor R of each free parameter of the model's pattern
        (or ``pattern``) against ``reference`` (default: one pulse per period
        of the same width): the reference needs R times longer for the same
        precision.
        """
        exp = self._experiment(8 * 3600, pattern=pattern, **beam)
        ref = reference if reference is not None else Pattern.single(float(exp.pattern.widths[0]))
        return design.gain(exp, ref, (wlmin, wlmax))

    def optimise_pattern(self, n_pulses, target="strain", width=None, n_restarts=4, n_steps=800, seed=0, wlmin=1.0,
                         wlmax=5.0, verbose=False, **beam):
        """
        Pulse times that minimise the Cramer-Rao bound of ``target`` for
        ``n_pulses`` pulses of ``width`` (default: that of the model's
        pattern). A stochastic search: the result is a good pattern, not a
        proven optimum. Beam keywords as in ``simulate`` (only the shape of
        the spectrum matters). Returns a Pattern; set it with
        ``model.pattern = ...``.
        """
        if target not in self.free:
            raise ValueError(f"target {target!r} is not a free parameter; choose from {self.free}")
        exp = self._experiment(8 * 3600, **beam)
        w = float(self.pattern.widths[0]) if width is None else float(width)
        return design.optimise_pattern(exp, n_pulses, target=target, width=w, n_restarts=n_restarts,
                                       n_steps=n_steps, window=(wlmin, wlmax), seed=seed, verbose=verbose)

    def monte_carlo(self, n=100, time=8 * 3600, method="rietveld", seed=0, workers=1, wlmin=1.0, wlmax=5.0,
                    noise_power=1e-4, start_sd="bound", **beam):
        """
        Fit ``n`` simulated measurements and compare their scatter with the
        Cramer-Rao bound. Returns (estimates, summary). ``workers > 1`` runs
        in parallel by forking (Linux, macOS).
        """
        exp = self._experiment(time, **beam)
        fitter = (RietveldFit(window=(wlmin, wlmax)) if method.lower() == "rietveld"
                  else WienerFit(noise_power=noise_power, window=(wlmin, wlmax)))
        return design.monte_carlo(exp, fitter, n=n, seed=seed, start_sd=start_sd, workers=workers)

    def pattern_bias(self, true_pattern, time=8 * 3600, wlmin=1.0, wlmax=5.0, **beam):
        """
        Bias of the Rietveld fit, to first order, when the real pulse pattern
        is ``true_pattern`` but the fit assumes the model's pattern, in units
        of the standard deviation of each free parameter.
        """
        return design.pattern_bias(self._experiment(time, **beam), true_pattern, (wlmin, wlmax))

    # ------------------------------------------------------------ display
    def plot(self, time=8 * 3600, ax=None, **beam):
        """The expected counts of the model's pattern over the period, with the
        spectrum of each pulse and the pulse times."""
        import matplotlib.pyplot as plt
        ax = ax or plt.figure(figsize=(8, 3)).gca()
        exp = self._experiment(time, **beam)
        g = self.grid
        T = exp.sample.transmission(g)
        t_ms = g.t * 1e3
        for k, (t0, w) in enumerate(zip(self.pattern.starts, self.pattern.widths)):
            part = exp.n_periods * g.convolve(exp.beam.rate * T, Pattern([t0], w).array(g))
            ax.fill_between(t_ms, part, color=("#6baed6", "#9ecae1")[k % 2], lw=0, alpha=0.55)
        mu = exp.expected_counts()
        ax.plot(t_ms, mu, color="k", lw=0.7)
        top = 1.12 * mu.max()
        ax.plot(self.pattern.starts * 1e3, np.full(self.pattern.n_pulses, top / 1.06), "v", color="k", ms=4)
        ax.set_ylim(0, top)
        ax.set_xlim(0, g.period * 1e3)
        ax.set_xlabel("Time in the period (ms)")
        ax.set_ylabel(f"Expected counts per {g.bin_width * 1e6:g} µs")
        return ax

    def __repr__(self):
        p = self.pattern
        return (f"TransmissionModel({p.n_pulses} pulse(s) of {p.widths[0] * 1e6:g} us per {self.grid.period * 1e3:g} "
                f"ms at {self.grid.flight_path:g} m; free: {', '.join(self.free) or 'none'})")
