"""
Workflow class for method chaining and parameter sweeps.

This module provides a high-level interface for chaining data processing,
reconstruction, and analysis steps, with support for parameter sweeps.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Callable
from .data_class import Data
from .reconstruct import Reconstruct
from .analysis_nbragg import Analysis


class Workflow:
    """
    High-level workflow interface for method chaining.

    This class allows you to chain together all processing steps from
    data loading through reconstruction to analysis, with support for
    parameter sweeps and progress tracking.

    Parameters
    ----------
    signal_path : str
        Path to signal data file
    openbeam_path : str
        Path to openbeam data file

    Attributes
    ----------
    data : Data
        The Data object being processed
    recon : Reconstruct or None
        The Reconstruct object after reconstruction
    analysis : Analysis or None
        The Analysis object after fitting
    result : lmfit.ModelResult or None
        Fitting result after analysis

    Examples
    --------
    >>> from frame_overlap import Workflow
    >>> # Simple workflow
    >>> wf = Workflow('signal.csv', 'openbeam.csv')
    >>> result = (wf
    ...     .convolute(pulse_duration=200)
    ...     .poisson(flux=1e6, freq=60)
    ...     .overlap(kernel=[0, 25])
    ...     .reconstruct(kind='wiener', noise_power=0.01)
    ...     .analyze(xs='iron', vary_background=True, vary_response=True)
    ...     .result)
    >>>
    >>> # Parameter sweep
    >>> wf = Workflow('signal.csv', 'openbeam.csv')
    >>> results = (wf
    ...     .convolute(pulse_duration=200)
    ...     .poisson(flux=1e6, freq=60)
    ...     .overlap(kernel=[0, 25])
    ...     .sweep('noise_power', [0.001, 0.01, 0.1])
    ...     .reconstruct(kind='wiener')
    ...     .analyze(xs='iron')
    ...     .run())
    """

    def __init__(self, signal_path: str, openbeam_path: str,
                 flux: float = None, duration: float = None, freq: float = None):
        """
        Initialize Workflow with data file paths.

        Parameters
        ----------
        signal_path : str
            Path to signal data file
        openbeam_path : str
            Path to openbeam data file
        flux : float, optional
            Original flux in n/cm²/s
        duration : float, optional
            Original measurement duration in hours
        freq : float, optional
            Original pulse frequency in Hz
        """
        self.signal_path = signal_path
        self.openbeam_path = openbeam_path
        self.data = Data(signal_path, openbeam_path, flux=flux, duration=duration, freq=freq)
        self.recon = None
        self.analysis = None
        self.result = None

        # For parameter sweeps
        self._sweep_params = {}
        self._recon_params = {}
        self._analysis_params = {}

    def convolute(self, pulse_duration: float = None, bin_width: float = 10, **kwargs):
        """
        Convolute with response function.

        Parameters
        ----------
        pulse_duration : float, optional
            Pulse duration in microseconds. If None, will be set during parameter sweep.
        bin_width : float
            Bin width in microseconds (default 10)
        **kwargs
            Additional arguments for convolute_response()

        Returns
        -------
        self : Workflow
            Returns self for chaining
        """
        # Store parameters for reconstruction, don't apply yet if pulse_duration is None
        self._convolute_params = {'bin_width': bin_width, **kwargs}
        if pulse_duration is not None:
            self.data.convolute_response(pulse_duration, bin_width=bin_width, **kwargs)
        return self

    def poisson(self, flux: float = None, freq: float = None,
                duty_cycle: float = None, measurement_time: float = None,
                seed: int = None):
        """
        Apply Poisson sampling.

        Parameters
        ----------
        flux : float, optional
            New flux in n/cm²/s
        freq : float, optional
            Pulse frequency in Hz
        duty_cycle : float, optional
            Duty cycle (alternative to flux)
        measurement_time : float, optional
            Measurement time in minutes
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        self : Workflow
            Returns self for chaining
        """
        self.data.poisson_sample(
            flux=flux, freq=freq, duty_cycle=duty_cycle,
            measurement_time=measurement_time, seed=seed
        )
        return self

    def overlap(self, kernel: List[float], total_time: float = None,
                freq: float = None, bin_width: float = 10, poisson_seed: int = None):
        """
        Apply frame overlap.

        Parameters
        ----------
        kernel : list of float
            Frame start times in milliseconds
        total_time : float, optional
            Total time span in milliseconds
        freq : float, optional
            Pulse frequency in Hz
        bin_width : float, optional
            Bin width in microseconds
        poisson_seed : int, optional
            Random seed for optional second Poisson sampling

        Returns
        -------
        self : Workflow
            Returns self for chaining
        """
        self.data.overlap(kernel=kernel, total_time=total_time, freq=freq,
                         bin_width=bin_width, poisson_seed=poisson_seed)
        return self

    def reconstruct(self, kind: str = 'wiener', tmin: float = None,
                   tmax: float = None, **kwargs):
        """
        Reconstruct overlapped data.

        Parameters
        ----------
        kind : str
            Reconstruction method ('wiener', 'lucy', 'tikhonov')
        tmin : float, optional
            Minimum time (ms) for chi² calculation
        tmax : float, optional
            Maximum time (ms) for chi² calculation
        **kwargs
            Additional arguments for filter method

        Returns
        -------
        self : Workflow
            Returns self for chaining
        """
        if self._sweep_params:
            # Store parameters for later use in sweep
            self._recon_params = {'kind': kind, 'tmin': tmin, 'tmax': tmax, **kwargs}
        else:
            # Execute immediately
            self.recon = Reconstruct(self.data, tmin=tmin, tmax=tmax)
            self.recon.filter(kind=kind, **kwargs)
        return self

    def analyze(self, xs: str = 'iron', vary_weights: bool = False,
                vary_background: bool = True, L: float = 9.0,
                tstep: float = 10e-6, **kwargs):
        """
        Analyze reconstructed data with nbragg.

        Parameters
        ----------
        xs : str or object
            Cross-section specification
        vary_weights : bool
            Whether to vary material weights
        vary_background : bool
            Whether to vary background
        L : float
            Flight path length in meters
        tstep : float
            Time step in seconds
        **kwargs
            Additional arguments for Analysis and fit()

        Returns
        -------
        self : Workflow
            Returns self for chaining
        """
        if self._sweep_params:
            # Store parameters for later use in sweep
            self._analysis_params = {
                'xs': xs, 'vary_weights': vary_weights,
                'vary_background': vary_background, 'L': L,
                'tstep': tstep, **kwargs
            }
        else:
            # Execute immediately
            if self.recon is None:
                raise ValueError("Must call reconstruct() before analyze()")

            self.analysis = Analysis(
                xs=xs, vary_weights=vary_weights,
                vary_background=vary_background,
                **{k: v for k, v in kwargs.items() if k not in ['L', 'tstep']}
            )
            self.result = self.analysis.fit(
                self.recon, L=L, tstep=tstep,
                **{k: v for k, v in kwargs.items() if k in ['method', 'verbose']}
            )
        return self

    def sweep(self, param_name: str, values: Union[List, np.ndarray]):
        """
        Set up parameter sweep.

        Parameters
        ----------
        param_name : str
            Name of parameter to sweep (e.g., 'noise_power', 'pulse_duration')
        values : list or array
            Values to sweep over

        Returns
        -------
        self : Workflow
            Returns self for chaining

        Examples
        --------
        >>> wf.sweep('noise_power', [0.001, 0.01, 0.1])
        >>> wf.sweep('pulse_duration', np.arange(100, 300, 50))
        """
        self._sweep_params[param_name] = values
        return self

    def groupby(self, param_name: str, low: float, high: float,
                step: float = None, num: int = None):
        """
        Set up parameter sweep with range specification.

        Parameters
        ----------
        param_name : str
            Name of parameter to sweep
        low : float
            Lower bound
        high : float
            Upper bound
        step : float, optional
            Step size (use this OR num)
        num : int, optional
            Number of points (use this OR step)

        Returns
        -------
        self : Workflow
            Returns self for chaining

        Examples
        --------
        >>> wf.groupby('pulse_duration', low=100, high=300, step=50)
        >>> wf.groupby('noise_power', low=0.001, high=0.1, num=10)
        """
        if step is not None:
            values = np.arange(low, high + step/2, step)
        elif num is not None:
            values = np.linspace(low, high, num)
        else:
            raise ValueError("Must specify either 'step' or 'num'")

        return self.sweep(param_name, values)

    def run(self, progress_bar: bool = True):
        """
        Execute parameter sweep.

        Parameters
        ----------
        progress_bar : bool
            Whether to show tqdm progress bar

        Returns
        -------
        pd.DataFrame
            Results dataframe with columns for parameters and metrics

        Examples
        --------
        >>> results = wf.groupby('noise_power', 0.001, 0.1, num=10).run()
        >>> results.plot('noise_power', 'chi2')
        """
        if not self._sweep_params:
            raise ValueError("No sweep parameters defined. Use sweep() or groupby() first.")

        # Import tqdm
        try:
            from tqdm.notebook import tqdm
        except ImportError:
            try:
                from tqdm import tqdm
            except ImportError:
                # Fallback if tqdm not available
                def tqdm(x, **kwargs):
                    return x
                progress_bar = False

        # Get sweep parameter (currently only support single parameter sweep)
        if len(self._sweep_params) > 1:
            raise NotImplementedError("Multi-parameter sweeps not yet implemented")

        param_name = list(self._sweep_params.keys())[0]
        param_values = self._sweep_params[param_name]

        # Results storage
        results = []

        # Progress bar
        iterator = tqdm(param_values, desc=f"Sweeping {param_name}") if progress_bar else param_values

        for value in iterator:
            try:
                # Reload data for each iteration with original parameters
                data = Data(self.signal_path, self.openbeam_path,
                           flux=self.data.flux, duration=self.data.duration,
                           freq=self.data.freq)

                # Re-apply all previous steps up to sweep point
                # This is determined by which parameter is being swept
                if param_name == 'pulse_duration':
                    data.convolute_response(value)
                elif hasattr(self.data, 'pulse_duration') and self.data.pulse_duration is not None:
                    data.convolute_response(self.data.pulse_duration)

                # Apply Poisson if it was done
                if hasattr(self.data, 'poissoned_data') and self.data.poissoned_data is not None:
                    if param_name in ['flux', 'freq', 'duty_cycle']:
                        # Update the swept parameter
                        poisson_params = {
                            'flux': self.data.flux,
                            'freq': getattr(self.data, 'freq', None),
                            'duty_cycle': getattr(self.data, 'duty_cycle', None),
                            'measurement_time': self.data.duration
                        }
                        poisson_params[param_name] = value
                        data.poisson_sample(**{k: v for k, v in poisson_params.items() if v is not None})
                    else:
                        data.poisson_sample(
                            flux=self.data.flux,
                            freq=getattr(self.data, 'freq', None),
                            measurement_time=self.data.duration
                        )

                # Apply overlap if it was done
                if hasattr(self.data, 'kernel') and self.data.kernel is not None:
                    data.overlap(kernel=self.data.kernel)

                # Reconstruct with sweep parameter if applicable
                recon_params = self._recon_params.copy()
                if param_name in recon_params:
                    recon_params[param_name] = value

                recon = Reconstruct(
                    data,
                    tmin=recon_params.pop('tmin', None),
                    tmax=recon_params.pop('tmax', None)
                )
                recon.filter(**recon_params)

                # Analyze
                analysis_params = self._analysis_params.copy()
                L = analysis_params.pop('L', 9.0)
                tstep = analysis_params.pop('tstep', 10e-6)

                # Separate Analysis init params from fit params
                fit_params = {}
                for key in ['method', 'verbose']:
                    if key in analysis_params:
                        fit_params[key] = analysis_params.pop(key)

                analysis = Analysis(**analysis_params)
                result = analysis.fit(recon, L=L, tstep=tstep, **fit_params)

                # Store results
                result_dict = {
                    param_name: value,
                    'chi2': result.chisqr,
                    'redchi2': result.redchi,
                    'aic': result.aic,
                    'bic': result.bic
                }

                # Add fitted parameters
                for pname, param in result.params.items():
                    result_dict[f'param_{pname}'] = param.value
                    if param.stderr is not None:
                        result_dict[f'param_{pname}_err'] = param.stderr

                results.append(result_dict)

            except Exception as e:
                print(f"Error at {param_name}={value}: {e}")
                results.append({
                    param_name: value,
                    'chi2': np.nan,
                    'error': str(e)
                })

        return pd.DataFrame(results)

    def plot(self, **kwargs):
        """
        Plot current state.

        Parameters
        ----------
        **kwargs
            Arguments passed to data.plot() or result.plot()

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.result is not None:
            return self.result.plot(**kwargs)
        elif self.recon is not None:
            return self.recon.plot(**kwargs)
        else:
            return self.data.plot(**kwargs)

    def __repr__(self):
        """String representation of the Workflow object."""
        stages = []
        if hasattr(self.data, 'convolved_data') and self.data.convolved_data is not None:
            stages.append('convolved')
        if hasattr(self.data, 'poissoned_data') and self.data.poissoned_data is not None:
            stages.append('poissoned')
        if hasattr(self.data, 'overlapped_data') and self.data.overlapped_data is not None:
            stages.append('overlapped')
        if self.recon is not None:
            stages.append('reconstructed')
        if self.result is not None:
            stages.append('analyzed')

        stages_str = ' → '.join(stages) if stages else 'initialized'

        if self._sweep_params:
            sweep_str = f", sweep={list(self._sweep_params.keys())}"
        else:
            sweep_str = ""

        return f"Workflow({stages_str}{sweep_str})"
