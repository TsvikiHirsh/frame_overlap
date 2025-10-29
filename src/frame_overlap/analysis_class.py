"""
Analysis class for fitting reconstructed neutron ToF data.

This module provides the Analysis class for fitting reconstructed data using
various response functions and extracting material parameters like thickness
and composition weights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model, Parameters


class Analysis:
    """
    Analysis object for fitting reconstructed neutron ToF data.

    This class fits reconstructed data using various response functions
    (square, square-jorgensen) to extract material parameters such as
    thickness and composition weights.

    Parameters
    ----------
    reconstruct : Reconstruct
        Reconstruct object containing the reconstructed data

    Attributes
    ----------
    reconstruct : Reconstruct
        Reference to the input Reconstruct object
    fit_result : dict
        Dictionary containing fit results and parameters
    cross_section : CrossSection
        Cross section object defining the material composition

    Examples
    --------
    >>> from frame_overlap import Data, Reconstruct, Analysis
    >>> data = Data('signal.csv').convolute_response(200).overlap([0, 12, 10, 25])
    >>> recon = Reconstruct(data).filter(kind='wiener')
    >>> analysis = Analysis(recon)
    >>> analysis.fit(response='square')
    >>> analysis.plot_fit()
    """

    def __init__(self, reconstruct):
        """
        Initialize Analysis object with a Reconstruct object.

        Parameters
        ----------
        reconstruct : Reconstruct
            Reconstruct object with filtered data
        """
        from .reconstruct import Reconstruct

        if not isinstance(reconstruct, Reconstruct):
            raise TypeError("reconstruct must be a Reconstruct object")

        if reconstruct.reconstructed_table is None:
            raise ValueError("Reconstruct object must have reconstructed data")

        self.reconstruct = reconstruct
        self.fit_result = None
        self.cross_section = None

    def set_cross_section(self, materials=None, fractions=None):
        """
        Set the cross section material composition.

        Parameters
        ----------
        materials : list of str, optional
            List of material names. Default is ['Fe_alpha', 'Cellulose'].
        fractions : list of float, optional
            Fraction of each material. Default is [0.96, 0.04] (4% Cellulose).

        Returns
        -------
        self
            Returns self for method chaining
        """
        if materials is None:
            materials = ['Fe_alpha', 'Cellulose']
        if fractions is None:
            fractions = [0.96, 0.04]

        if len(materials) != len(fractions):
            raise ValueError("materials and fractions must have the same length")

        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("fractions must sum to 1.0")

        self.cross_section = CrossSection(materials, fractions)
        return self

    def fit(self, response='square', thickness_range=(0.1, 10.0),
            weights_range=(0.0, 1.0), **kwargs):
        """
        Fit the reconstructed data using specified response function.

        Parameters
        ----------
        response : str, optional
            Response function type:
            - 'square': Simple square response
            - 'square-jorgensen': Square response with Jorgensen correction
            Default is 'square'.
        thickness_range : tuple, optional
            Range for thickness parameter (min, max) in mm. Default is (0.1, 10.0).
        weights_range : tuple, optional
            Range for weight parameters (min, max). Default is (0.0, 1.0).
        **kwargs
            Additional keyword arguments for the fitting routine

        Returns
        -------
        self
            Returns self for method chaining

        Raises
        ------
        ValueError
            If response type is invalid or data is missing
        """
        if self.cross_section is None:
            # Set default cross section
            self.set_cross_section()

        response = response.lower()
        if response not in ['square', 'square-jorgensen']:
            raise ValueError(f"Unknown response '{response}'. "
                           f"Choose from: 'square', 'square-jorgensen'")

        # Get data from reconstruct object
        time = self.reconstruct.reconstructed_table['time'].values
        counts = self.reconstruct.reconstructed_table['counts'].values
        errors = self.reconstruct.reconstructed_table['err'].values

        # Create fit model
        if response == 'square':
            fit_func = self._square_response
        else:  # square-jorgensen
            fit_func = self._square_jorgensen_response

        # Set up lmfit model
        model = Model(fit_func, independent_vars=['time'])

        # Create parameters
        params = Parameters()
        params.add('thickness', value=1.0, min=thickness_range[0], max=thickness_range[1])
        params.add('amplitude', value=counts.max(), min=0)

        # Add weight parameters for each material (except last one which is determined)
        n_materials = len(self.cross_section.materials)
        for i in range(n_materials - 1):
            params.add(f'weight_{i}', value=self.cross_section.fractions[i],
                      min=weights_range[0], max=weights_range[1])

        # Perform fit
        try:
            result = model.fit(counts, params, time=time, weights=1.0/errors)

            # Extract results
            self.fit_result = {
                'success': True,
                'response': response,
                'thickness': result.params['thickness'].value,
                'thickness_err': result.params['thickness'].stderr if result.params['thickness'].stderr else 0,
                'amplitude': result.params['amplitude'].value,
                'amplitude_err': result.params['amplitude'].stderr if result.params['amplitude'].stderr else 0,
                'chi_square': result.chisqr,
                'reduced_chi_square': result.redchi,
                'aic': result.aic,
                'bic': result.bic,
                'n_data': len(time),
                'n_params': len(params),
                'fitted_counts': result.best_fit,
                'residuals': result.residual,
                'result_object': result
            }

            # Extract weight parameters
            weights = []
            for i in range(n_materials - 1):
                w = result.params[f'weight_{i}'].value
                w_err = result.params[f'weight_{i}'].stderr if result.params[f'weight_{i}'].stderr else 0
                weights.append((w, w_err))

            # Last weight is determined by constraint that sum = 1
            last_weight = 1.0 - sum(w[0] for w in weights)
            weights.append((last_weight, 0))  # Error propagation could be added

            self.fit_result['weights'] = weights
            self.fit_result['materials'] = self.cross_section.materials

        except Exception as e:
            self.fit_result = {
                'success': False,
                'error': str(e)
            }
            raise RuntimeError(f"Fit failed: {e}")

        return self

    def _square_response(self, time, thickness, amplitude, **weight_params):
        """
        Square response function for neutron transmission.

        Parameters
        ----------
        time : np.ndarray
            Time array in microseconds
        thickness : float
            Sample thickness in mm
        amplitude : float
            Signal amplitude
        **weight_params : dict
            Weight parameters for each material component

        Returns
        -------
        np.ndarray
            Calculated counts
        """
        # Simple exponential attenuation model
        # sigma_total = sum of cross sections weighted by composition
        sigma_total = self.cross_section.calculate_total_cross_section(**weight_params)

        # Transmission = exp(-n * sigma * thickness)
        # For ToF, we can use energy-dependent cross sections
        # Here's a simplified model
        transmission = np.exp(-sigma_total * thickness)

        # Apply to amplitude
        signal = amplitude * transmission * np.ones_like(time)

        return signal

    def _square_jorgensen_response(self, time, thickness, amplitude, **weight_params):
        """
        Square response with Jorgensen correction.

        This adds additional physical corrections based on the Jorgensen model
        for neutron transmission through samples.

        Parameters
        ----------
        time : np.ndarray
            Time array in microseconds
        thickness : float
            Sample thickness in mm
        amplitude : float
            Signal amplitude
        **weight_params : dict
            Weight parameters for each material component

        Returns
        -------
        np.ndarray
            Calculated counts
        """
        # Start with square response
        signal = self._square_response(time, thickness, amplitude, **weight_params)

        # Add Jorgensen corrections (simplified)
        # In a real implementation, this would include:
        # - Multiple scattering corrections
        # - Detector efficiency
        # - Geometric factors
        # - Energy-dependent corrections

        # Placeholder for Jorgensen correction factor
        jorgensen_factor = 1.0 + 0.01 * thickness  # Simplified correction

        return signal * jorgensen_factor

    def get_fit_results(self):
        """
        Get fit results as a dictionary.

        Returns
        -------
        dict
            Dictionary with fit results including parameters, errors, and statistics
        """
        if self.fit_result is None:
            raise ValueError("No fit results available. Call fit() first.")

        return {k: v for k, v in self.fit_result.items() if k != 'result_object'}

    def get_fit_report(self):
        """
        Get a formatted fit report string using lmfit's built-in report.

        Returns
        -------
        str
            Formatted fit report from lmfit
        """
        if self.fit_result is None:
            raise ValueError("No fit results available. Call fit() first.")

        if not self.fit_result['success']:
            return f"Fit failed: {self.fit_result.get('error', 'Unknown error')}"

        # Use lmfit's built-in fit_report method
        if 'result_object' in self.fit_result and self.fit_result['result_object'] is not None:
            return self.fit_result['result_object'].fit_report()
        else:
            # Fallback to basic report if result_object not available
            return (f"Fit completed with thickness={self.fit_result['thickness']:.3f} mm, "
                   f"reduced chi-square={self.fit_result['reduced_chi_square']:.3f}")

    def plot_fit(self):
        """
        Plot the fit result overlaid on the data.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.fit_result is None or not self.fit_result['success']:
            raise ValueError("No successful fit available. Call fit() first.")

        time = self.reconstruct.reconstructed_table['time'].values
        counts = self.reconstruct.reconstructed_table['counts'].values
        fitted = self.fit_result['fitted_counts']

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(time, counts, 'o', label='Reconstructed Data',
               alpha=0.6, markersize=4)
        ax.plot(time, fitted, 'r-', label='Fit', linewidth=2)

        title = (f"Fit Results (χ²/dof = {self.fit_result['reduced_chi_square']:.3f}, "
                f"thickness = {self.fit_result['thickness']:.2f} mm)")
        ax.set_title(title)
        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Counts')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_residuals(self):
        """
        Plot fit residuals.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.fit_result is None or not self.fit_result['success']:
            raise ValueError("No successful fit available. Call fit() first.")

        time = self.reconstruct.reconstructed_table['time'].values
        residuals = self.fit_result['residuals']

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(time, residuals, 'ko', alpha=0.5, markersize=4)
        ax.axhline(0, color='r', linestyle='--', linewidth=2)

        ax.set_title('Fit Residuals')
        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Residuals')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_materials(self):
        """
        Plot the fitted material composition.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.fit_result is None or not self.fit_result['success']:
            raise ValueError("No successful fit available. Call fit() first.")

        materials = self.fit_result['materials']
        weights = [w[0] for w in self.fit_result['weights']]
        errors = [w[1] for w in self.fit_result['weights']]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(range(len(materials)), weights, yerr=errors,
                     capsize=5, alpha=0.7)
        ax.set_xticks(range(len(materials)))
        ax.set_xticklabels(materials, rotation=45, ha='right')
        ax.set_ylabel('Weight Fraction')
        ax.set_title('Fitted Material Composition')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{weight*100:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def __repr__(self):
        """String representation of the Analysis object."""
        has_fit = self.fit_result is not None and self.fit_result.get('success', False)
        thickness = self.fit_result.get('thickness', None) if has_fit else None
        return (f"Analysis(has_fit={has_fit}, "
                f"thickness={thickness:.2f if thickness is not None else 'N/A'} mm)")


class CrossSection:
    """
    Cross section object for material composition.

    This class manages the neutron cross sections for different materials
    and their fractional composition.

    Parameters
    ----------
    materials : list of str
        List of material names (e.g., ['Fe_alpha', 'Cellulose'])
    fractions : list of float
        Fraction of each material (must sum to 1.0)

    Attributes
    ----------
    materials : list
        Material names
    fractions : list
        Material fractions
    cross_sections : dict
        Dictionary mapping material names to cross section values

    Examples
    --------
    >>> cs = CrossSection(['Fe_alpha', 'Cellulose'], [0.96, 0.04])
    >>> total_cs = cs.calculate_total_cross_section()
    """

    # Default cross section values (barns)
    # These are simplified values - in reality, cross sections are energy-dependent
    DEFAULT_CROSS_SECTIONS = {
        'Fe_alpha': 11.62,  # Iron alpha phase
        'Cellulose': 5.55,  # Cellulose
        'Al': 1.49,         # Aluminum
        'Cu': 8.03,         # Copper
        'Ni': 18.5,         # Nickel
        'H2O': 5.6,         # Water
    }

    def __init__(self, materials, fractions):
        """Initialize CrossSection with materials and fractions."""
        if len(materials) != len(fractions):
            raise ValueError("materials and fractions must have the same length")

        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("fractions must sum to 1.0")

        self.materials = materials
        self.fractions = fractions
        self.cross_sections = {}

        # Set cross sections from defaults or use placeholder
        for material in materials:
            if material in self.DEFAULT_CROSS_SECTIONS:
                self.cross_sections[material] = self.DEFAULT_CROSS_SECTIONS[material]
            else:
                # Use a default value if material not in database
                self.cross_sections[material] = 10.0
                print(f"Warning: Cross section for '{material}' not found. Using default value of 10.0 barns.")

    def calculate_total_cross_section(self, **weight_params):
        """
        Calculate the total cross section from weighted composition.

        Parameters
        ----------
        **weight_params : dict
            Weight parameters (weight_0, weight_1, etc.)

        Returns
        -------
        float
            Total cross section in barns
        """
        # Extract weights from parameters
        weights = []
        n_materials = len(self.materials)
        for i in range(n_materials - 1):
            key = f'weight_{i}'
            if key in weight_params:
                weights.append(weight_params[key])
            else:
                weights.append(self.fractions[i])

        # Last weight determined by constraint
        if len(weights) < n_materials:
            weights.append(1.0 - sum(weights))

        # Calculate weighted sum
        total_cs = sum(w * self.cross_sections[m]
                      for w, m in zip(weights, self.materials))

        return total_cs

    def __repr__(self):
        """String representation of the CrossSection object."""
        comp_str = ", ".join(f"{m}:{f:.1%}" for m, f in zip(self.materials, self.fractions))
        return f"CrossSection({comp_str})"
