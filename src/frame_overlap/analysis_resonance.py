"""
Resonance Analysis Module using nres package.

This module provides the ResonanceAnalysis class for fitting neutron absorption
resonances in epithermal/fast energy regions using the nres package.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union


class ResonanceAnalysis:
    """
    Resonance analysis using nres package for epithermal neutron absorption resonances.

    This class provides an interface to fit transmission data with neutron absorption
    resonances using the nres package. It's designed for analyzing epithermal and
    fast neutron data where resonance structures dominate.

    Parameters
    ----------
    material : str
        Material name for nres cross-section (e.g., 'Ta', 'U', 'W').
        Must be a material available in nres CrossSection database.
    vary_weights : bool or None, optional
        Allow resonance weights to vary during fitting. If None, uses nres default.
    vary_background : bool or None, optional
        Allow background to vary during fitting. Default is None (uses nres default).
    vary_response : bool or None, optional
        Allow response function to vary during fitting. If None, uses nres default.
    vary_tof : bool or None, optional
        Allow TOF calibration to vary during fitting. If None, uses nres default.
    **kwargs
        Additional keyword arguments passed to nres.TransmissionModel.

    Attributes
    ----------
    material : str
        Material name used for cross-section.
    xs : nres.CrossSection
        Cross-section object from nres.
    model : nres.TransmissionModel
        Transmission model from nres.
    data : nres.Data or None
        Data object after fitting.
    result : lmfit.ModelResult or None
        Fit result after fitting.

    Examples
    --------
    >>> from frame_overlap import Workflow, ResonanceAnalysis
    >>> wf = Workflow('ta_signal.csv', 'openbeam.csv')
    >>> wf.overlap(kernel=[0, 25]).reconstruct()
    >>>
    >>> # Optionally apply Cd filter
    >>> wf.data.apply_cd_filter(cutoff_energy=0.4)
    >>>
    >>> # Fit resonances
    >>> analysis = ResonanceAnalysis(material='Ta', vary_background=True)
    >>> result = analysis.fit(wf.recon, emin=4e5, emax=1.7e6)
    >>> analysis.plot_fit()
    """

    def __init__(
        self,
        material: str = 'Ta',
        vary_weights: Optional[bool] = None,
        vary_background: Optional[bool] = None,
        vary_response: Optional[bool] = None,
        vary_tof: Optional[bool] = None,
        **kwargs
    ):
        """Initialize ResonanceAnalysis with material and fitting options."""
        import nres

        self.material = material
        self.xs = nres.CrossSection(material)

        # Build kwargs for TransmissionModel, only including non-None values
        model_kwargs = {}
        if vary_weights is not None:
            model_kwargs['vary_weights'] = vary_weights
        if vary_background is not None:
            model_kwargs['vary_background'] = vary_background
        if vary_response is not None:
            model_kwargs['vary_response'] = vary_response
        if vary_tof is not None:
            model_kwargs['vary_tof'] = vary_tof

        # Add any additional kwargs
        model_kwargs.update(kwargs)

        self.model = nres.TransmissionModel(self.xs, **model_kwargs)
        self.data = None
        self.result = None

    def fit(
        self,
        reconstruct,
        L: float = 9.0,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        method: str = 'leastsq',
        **kwargs
    ):
        """
        Fit the reconstructed data with resonance model.

        Parameters
        ----------
        reconstruct : Reconstruct
            Reconstructed data object from frame_overlap.
        L : float, optional
            Flight path length in meters. Default is 9.0.
        emin : float or None, optional
            Minimum energy in eV for fitting range. If None, uses all data.
        emax : float or None, optional
            Maximum energy in eV for fitting range. If None, uses all data.
        method : str, optional
            Fitting method. Default is 'leastsq'. See lmfit documentation.
        **kwargs
            Additional keyword arguments passed to model.fit().

        Returns
        -------
        result : lmfit.ModelResult
            Fit result from nres model.

        Examples
        --------
        >>> analysis = ResonanceAnalysis(material='Ta')
        >>> result = analysis.fit(recon, emin=4e5, emax=1.7e6)
        >>> print(f"Reduced chi-squared: {result.redchi:.3f}")
        """
        # Convert reconstruct to nres Data format
        self.data = self._reconstruct_to_nres_data(reconstruct, L=L)

        # Fit the model
        self.result = self.model.fit(
            self.data,
            emin=emin,
            emax=emax,
            method=method,
            **kwargs
        )

        return self.result

    def _reconstruct_to_nres_data(self, reconstruct, L: float = 9.0):
        """
        Convert Reconstruct object to nres.Data format.

        Parameters
        ----------
        reconstruct : Reconstruct
            Reconstructed data from frame_overlap.
        L : float
            Flight path length in meters.

        Returns
        -------
        nres.Data
            Data object for nres fitting.
        """
        import nres

        # Get reconstructed transmission data
        df = reconstruct.reconstructed_data.copy()

        # Convert time (ms) to microseconds
        time_us = df['time'] * 1000.0

        # Convert time to energy using: E (eV) = 5227.0 * L^2 / t^2
        # where t is in microseconds
        energy_eV = 5227.0 * (L ** 2) / (time_us ** 2)

        # Get transmission and error
        trans = df['trans'].values
        err = df['err'].values

        # Create DataFrame for nres with energy, trans, err
        nres_df = pd.DataFrame({
            'energy': energy_eV,
            'trans': trans,
            'err': err
        })

        # Remove NaN and inf values
        nres_df = nres_df.replace([np.inf, -np.inf], np.nan).dropna()

        # Remove zero or negative errors
        nres_df = nres_df[nres_df['err'] > 0]

        # Sort by energy (ascending)
        nres_df = nres_df.sort_values('energy')

        # Create nres Data object
        # nres.Data expects columns: energy, trans, err
        data = nres.Data(nres_df)

        return data

    def plot_fit(
        self,
        figsize: tuple = (12, 8),
        show_residuals: bool = True,
        energy_units: str = 'eV',
        **kwargs
    ) -> plt.Figure:
        """
        Plot the fit results.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (12, 8).
        show_residuals : bool, optional
            Show residuals subplot. Default is True.
        energy_units : str, optional
            Energy units for x-axis. Default is 'eV'.
        **kwargs
            Additional keyword arguments passed to plotting functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.

        Examples
        --------
        >>> analysis.fit(recon)
        >>> fig = analysis.plot_fit()
        >>> plt.show()
        """
        if self.result is None:
            raise ValueError("No fit result available. Run fit() first.")

        # Use nres built-in plotting if available
        if hasattr(self.result, 'plot'):
            return self.result.plot()

        # Otherwise create custom plot
        if show_residuals:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                           gridspec_kw={'height_ratios': [3, 1]},
                                           sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        # Get data
        energy = self.data.table['energy'].values
        trans = self.data.table['trans'].values
        err = self.data.table['err'].values

        # Get best fit
        best_fit = self.result.best_fit

        # Plot data with error bars
        ax1.errorbar(energy, trans, yerr=err, fmt='o',
                     label='Data', markersize=4, alpha=0.6)

        # Plot best fit
        ax1.plot(energy, best_fit, 'r-', label='Best fit', linewidth=2)

        ax1.set_ylabel('Transmission')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Resonance Fit: {self.material} (χ²/dof = {self.result.redchi:.3f})')

        if show_residuals and hasattr(self.result, 'residual'):
            # Plot residuals
            residuals = self.result.residual
            ax2.plot(energy, residuals, 'o', markersize=3, alpha=0.6)
            ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel(f'Energy ({energy_units})')
            ax2.set_ylabel('Residuals')
            ax2.grid(True, alpha=0.3)
        else:
            ax1.set_xlabel(f'Energy ({energy_units})')

        plt.tight_layout()
        return fig

    def get_parameters(self) -> pd.DataFrame:
        """
        Get fitted parameters as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: name, value, stderr, vary.

        Examples
        --------
        >>> params = analysis.get_parameters()
        >>> print(params)
        """
        if self.result is None:
            raise ValueError("No fit result available. Run fit() first.")

        params_data = []
        for name, param in self.result.params.items():
            params_data.append({
                'name': name,
                'value': param.value,
                'stderr': param.stderr if param.stderr is not None else np.nan,
                'vary': param.vary
            })

        return pd.DataFrame(params_data)

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.result is not None else "not fitted"
        return f"ResonanceAnalysis(material='{self.material}', status='{status}')"
