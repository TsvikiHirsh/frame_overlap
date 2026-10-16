"""
Analysis class for fitting reconstructed neutron ToF data using nbragg.

This module provides the Analysis class as a wrapper around nbragg for
fitting reconstructed data and extracting material parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import nbragg
    HAS_NBRAGG = True
except ImportError:
    HAS_NBRAGG = False
    print("Warning: nbragg not installed. Analysis features will be limited.")
    print("Install with: pip install nbragg")


class Analysis:
    """
    Analysis object for fitting reconstructed neutron ToF data using nbragg.

    This class wraps nbragg functionality to fit reconstructed transmission data
    and extract material parameters. It uses nbragg's TransmissionModel with
    optional background and response variation.

    Parameters
    ----------
    reconstruct : Reconstruct
        Reconstruct object containing the reconstructed data
    cross_section_file : str, optional
        Path to NCrystal .ncmat file for cross section.
        Default is None (will use default material).
    cross_section_dict : dict, optional
        Dictionary mapping material names to .ncmat files.
        Example: {'iron': 'Fe_sg229_Iron-alpha.ncmat'}

    Attributes
    ----------
    reconstruct : Reconstruct
        Reference to the input Reconstruct object
    nbragg_data : nbragg.Data
        nbragg Data object created from transmission
    cross_section : nbragg.CrossSection
        nbragg CrossSection object
    model : nbragg.TransmissionModel
        nbragg transmission model
    result : nbragg fit result
        Result from model.fit()

    Examples
    --------
    >>> from frame_overlap import Data, Reconstruct, Analysis
    >>> data = Data('iron_powder.csv', 'openbeam.csv')
    >>> data.convolute_response(200).overlap([0, 12, 10]).poisson_sample()
    >>> recon = Reconstruct(data).filter(kind='wiener')
    >>> analysis = Analysis(recon, cross_section_dict={'iron': 'Fe_sg229_Iron-alpha.ncmat'})
    >>> result = analysis.fit(vary_background=True, vary_response=True)
    >>> result.plot()
    >>> print(result.fit_report())
    """

    def __init__(self, reconstruct, cross_section_file=None, cross_section_dict=None):
        """
        Initialize Analysis object with a Reconstruct object.

        Parameters
        ----------
        reconstruct : Reconstruct
            Reconstruct object with filtered data
        cross_section_file : str, optional
            Path to single .ncmat file
        cross_section_dict : dict, optional
            Dictionary of material names to .ncmat files
        """
        if not HAS_NBRAGG:
            raise ImportError(
                "nbragg is required for Analysis class. "
                "Install with: pip install nbragg"
            )

        from .reconstruct import Reconstruct

        if not isinstance(reconstruct, Reconstruct):
            raise TypeError("reconstruct must be a Reconstruct object")

        if reconstruct.reconstructed_table is None:
            raise ValueError("Reconstruct object must have reconstructed data")

        self.reconstruct = reconstruct
        self.nbragg_data = None
        self.cross_section = None
        self.model = None
        self.result = None

        # Set up cross section
        if cross_section_dict is not None:
            self.set_cross_section(**cross_section_dict)
        elif cross_section_file is not None:
            # Extract material name from filename
            import os
            material_name = os.path.splitext(os.path.basename(cross_section_file))[0]
            self.set_cross_section(**{material_name: cross_section_file})
        else:
            # Use default Fe-alpha + Cellulose mix
            self.set_cross_section(
                iron='Fe_sg229_Iron-alpha.ncmat',
                cellulose='C6H10O5_Cellulose.ncmat'
            )

    def set_cross_section(self, **materials):
        """
        Set the cross section material composition using nbragg.

        Parameters
        ----------
        **materials : dict
            Keyword arguments mapping material names to .ncmat file paths.
            Example: iron='Fe_sg229_Iron-alpha.ncmat', cellulose='C6H10O5_Cellulose.ncmat'

        Returns
        -------
        self
            Returns self for method chaining
        """
        if not HAS_NBRAGG:
            raise ImportError("nbragg is required")

        self.cross_section = nbragg.CrossSection(**materials)
        return self

    def prepare_data(self, save_csv=False, csv_path='temp_transmission.csv'):
        """
        Prepare nbragg Data object from reconstructed transmission.

        This converts the reconstructed data into a format suitable for nbragg.
        If both signal and openbeam are available, transmission is calculated.

        Parameters
        ----------
        save_csv : bool, optional
            If True, save transmission data to CSV file. Default is False.
        csv_path : str, optional
            Path to save CSV file if save_csv=True.

        Returns
        -------
        self
            Returns self for method chaining
        """
        if not HAS_NBRAGG:
            raise ImportError("nbragg is required")

        # Get reconstructed data
        recon_data = self.reconstruct.reconstructed_table

        # Check if we have reference data (openbeam) in the original Data object
        if (hasattr(self.reconstruct.data, 'op_data') and
            self.reconstruct.data.op_data is not None):

            # Calculate transmission from reconstructed signal and openbeam
            # Use the appropriate stage of openbeam (same as signal)
            if self.reconstruct.data.op_poissoned_data is not None:
                openbeam = self.reconstruct.data.op_poissoned_data
            elif self.reconstruct.data.op_overlapped_data is not None:
                openbeam = self.reconstruct.data.op_overlapped_data
            elif self.reconstruct.data.op_convolved_data is not None:
                openbeam = self.reconstruct.data.op_convolved_data
            else:
                openbeam = self.reconstruct.data.op_data

            # Align time points
            common_times = np.intersect1d(recon_data['time'].values, openbeam['time'].values)
            recon_subset = recon_data[recon_data['time'].isin(common_times)]
            openbeam_subset = openbeam[openbeam['time'].isin(common_times)]

            # Calculate transmission
            transmission = recon_subset['counts'].values / (openbeam_subset['counts'].values + 1e-10)

            # Error propagation
            rel_err_sig = recon_subset['err'].values / (recon_subset['counts'].values + 1e-10)
            rel_err_op = openbeam_subset['err'].values / (openbeam_subset['counts'].values + 1e-10)
            trans_err = transmission * np.sqrt(rel_err_sig**2 + rel_err_op**2)

            # Create DataFrame for nbragg
            df = pd.DataFrame({
                'stack': (recon_subset['time'].values / 10 + 1).astype(int),  # Convert time back to stack
                'counts': transmission,
                'err': trans_err
            })
        else:
            # No openbeam available, use reconstructed counts directly
            # Assume these are already transmission values
            df = pd.DataFrame({
                'stack': (recon_data['time'].values / 10 + 1).astype(int),
                'counts': recon_data['counts'].values,
                'err': recon_data['err'].values
            })

        if save_csv:
            df.to_csv(csv_path, index=False)
            self.nbragg_data = nbragg.Data.from_transmission(csv_path)
        else:
            # Create temporary file
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f, index=False)
                temp_path = f.name

            self.nbragg_data = nbragg.Data.from_transmission(temp_path)

            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass

        return self

    def fit(self, vary_background=True, vary_response=True, **kwargs):
        """
        Fit the reconstructed data using nbragg's TransmissionModel.

        Parameters
        ----------
        vary_background : bool, optional
            Whether to vary background in fit. Default is True.
        vary_response : bool, optional
            Whether to vary response in fit. Default is True.
        **kwargs
            Additional keyword arguments passed to model.fit()

        Returns
        -------
        nbragg fit result
            The nbragg fit result object with methods like plot() and fit_report()

        Examples
        --------
        >>> result = analysis.fit(vary_background=True, vary_response=True)
        >>> result.plot()
        >>> print(result.fit_report())
        """
        if not HAS_NBRAGG:
            raise ImportError("nbragg is required")

        if self.cross_section is None:
            raise ValueError("Cross section not set. Call set_cross_section() first.")

        # Prepare data if not already done
        if self.nbragg_data is None:
            self.prepare_data()

        # Create TransmissionModel
        self.model = nbragg.TransmissionModel(
            self.cross_section,
            vary_background=vary_background,
            vary_response=vary_response
        )

        # Perform fit

        # make sure nbragg_data has no NaNs
        self.nbragg_data.table = self.nbragg_data.table.dropna()
        
        self.result = self.model.fit(self.nbragg_data, **kwargs)

        return self.result

    def get_fit_report(self):
        """
        Get nbragg's fit report.

        Returns
        -------
        str
            Formatted fit report from nbragg
        """
        if self.result is None:
            raise ValueError("No fit results available. Call fit() first.")

        return self.result.fit_report()

    def plot(self, **kwargs):
        """
        Plot the fit results using nbragg's plot method.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to result.plot()

        Returns
        -------
        matplotlib figure
            The figure created by nbragg
        """
        if self.result is None:
            raise ValueError("No fit results available. Call fit() first.")

        return self.result.plot(**kwargs)

    def plot_html(self, filename='fit_report.html'):
        """
        Generate HTML fit report using nbragg.

        Parameters
        ----------
        filename : str, optional
            Output filename for HTML report. Default is 'fit_report.html'.

        Returns
        -------
        str
            Path to the generated HTML file
        """
        if self.result is None:
            raise ValueError("No fit results available. Call fit() first.")

        # nbragg should have an HTML export method
        # Check if it exists and use it
        if hasattr(self.result, 'to_html'):
            self.result.to_html(filename)
            return filename
        elif hasattr(self.result, 'save_html'):
            self.result.save_html(filename)
            return filename
        else:
            raise NotImplementedError(
                "HTML export not available in this version of nbragg. "
                "Use plot() or get_fit_report() instead."
            )

    def get_parameters(self):
        """
        Get fitted parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary of fitted parameters
        """
        if self.result is None:
            raise ValueError("No fit results available. Call fit() first.")

        # Extract parameters from nbragg result
        if hasattr(self.result, 'params'):
            return {name: param.value for name, param in self.result.params.items()}
        elif hasattr(self.result, 'best_values'):
            return self.result.best_values
        else:
            raise AttributeError("Cannot extract parameters from nbragg result")

    def __repr__(self):
        """String representation of the Analysis object."""
        has_data = self.nbragg_data is not None
        has_fit = self.result is not None
        return f"Analysis(nbragg_data={has_data}, has_fit={has_fit})"


# Legacy CrossSection class for backward compatibility
class CrossSection:
    """
    Legacy CrossSection class - deprecated.

    Use nbragg.CrossSection directly instead.
    """

    DEFAULT_CROSS_SECTIONS = {
        'Fe_alpha': 11.62,
        'Cellulose': 5.55,
        'Al': 1.49,
        'Cu': 8.03,
        'Ni': 18.5,
        'H2O': 5.6,
    }

    def __init__(self, materials, fractions):
        """Initialize CrossSection with materials and fractions."""
        print("Warning: This CrossSection class is deprecated. Use nbragg.CrossSection instead.")

        if len(materials) != len(fractions):
            raise ValueError("materials and fractions must have the same length")

        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("fractions must sum to 1.0")

        self.materials = materials
        self.fractions = fractions
        self.cross_sections = {}

        for material in materials:
            if material in self.DEFAULT_CROSS_SECTIONS:
                self.cross_sections[material] = self.DEFAULT_CROSS_SECTIONS[material]
            else:
                self.cross_sections[material] = 10.0

    def calculate_total_cross_section(self, **weight_params):
        """Calculate the total cross section from weighted composition."""
        weights = []
        n_materials = len(self.materials)
        for i in range(n_materials - 1):
            key = f'weight_{i}'
            if key in weight_params:
                weights.append(weight_params[key])
            else:
                weights.append(self.fractions[i])

        if len(weights) < n_materials:
            weights.append(1.0 - sum(weights))

        total_cs = sum(w * self.cross_sections[m]
                      for w, m in zip(weights, self.materials))

        return total_cs

    def __repr__(self):
        """String representation of the CrossSection object."""
        comp_str = ", ".join(f"{m}:{f:.1%}" for m, f in zip(self.materials, self.fractions))
        return f"CrossSection({comp_str}) [DEPRECATED - use nbragg.CrossSection]"
