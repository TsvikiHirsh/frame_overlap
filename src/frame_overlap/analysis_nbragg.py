"""
Analysis class for nbragg integration with frame_overlap reconstruction.

This module provides a simplified interface to nbragg for fitting transmission
models to reconstructed data.
"""

import numpy as pd
import pandas as pd


class Analysis:
    """
    Analysis class for fitting transmission models to reconstructed data.

    This class provides a simplified interface to nbragg for fitting,
    with predefined cross-section configurations and easy access to
    the underlying nbragg objects.

    Parameters
    ----------
    xs : str or object, optional
        Cross-section specification. Can be:
        - 'iron': Simple Fe_sg229_Iron-alpha (use with vary_background=True, vary_response=True)
        - 'iron_with_cellulose': Iron with cellulose background
        - 'iron_square_response': Iron with square response function
        - nbragg.CrossSection object: Custom cross-section
        - dict: Custom material dictionary for nbragg.CrossSection
    vary_weights : bool, optional
        Whether to vary material weights during fitting. Default is False.
    vary_background : bool, optional
        Whether to vary background during fitting. Default is True.
    **kwargs
        Additional keyword arguments passed to nbragg.TransmissionModel

    Attributes
    ----------
    xs : nbragg.CrossSection
        The cross-section object
    model : nbragg.TransmissionModel
        The underlying nbragg transmission model
    data : nbragg.Data or None
        The nbragg Data object after calling fit()
    result : lmfit.ModelResult or None
        Fitting result after calling fit()

    Examples
    --------
    >>> from frame_overlap import Data, Reconstruct, Analysis
    >>> # Create and reconstruct data
    >>> data = Data('signal.csv', 'openbeam.csv')
    >>> data.convolute_response(200).overlap([0, 25]).poisson_sample(duty_cycle=0.8)
    >>> recon = Reconstruct(data)
    >>> recon.filter(kind='wiener', noise_power=0.01)
    >>>
    >>> # Fit with 'iron' model (recommended)
    >>> analysis = Analysis(xs='iron', vary_background=True, vary_response=True)
    >>> analysis.model.params  # Access nbragg model parameters
    >>> result = analysis.fit(recon)
    >>> result.plot()
    >>>
    >>> # Or use 'iron_square_response'
    >>> analysis = Analysis(xs='iron_square_response', vary_background=True)
    >>> result = analysis.fit(recon)
    >>>
    >>> # Or use custom cross-section
    >>> import nbragg
    >>> xs = nbragg.CrossSection(iron=nbragg.materials["Fe_sg225_Iron-gamma"])
    >>> analysis = Analysis(xs=xs, vary_background=True)
    >>> result = analysis.fit(recon)
    """

    def __init__(self, xs='iron', vary_weights=False, vary_background=True,
                 vary_sans=False, vary_extinction=False, **kwargs):
        """
        Initialize Analysis with cross-section specification.

        Parameters
        ----------
        xs : str or object
            Cross-section specification. Available predefined options:
            - 'iron': Simple Fe_sg229_Iron-alpha (use with vary_background=True, vary_response=True)
            - 'iron_square_response': Iron with square response
            - 'iron_with_cellulose': Iron with cellulose background (2% cellulose, 98% Fe_alpha)
        vary_weights : bool
            Whether to vary material weights. Default is False.
        vary_background : bool
            Whether to vary background. Default is True.
        vary_sans : bool
            Whether to vary SANS parameters. Default is False.
        vary_extinction : bool
            Whether to include extinction parameters. Default is False.
        **kwargs
            Additional arguments for nbragg.TransmissionModel, including:
            - vary_response: Whether to vary response function (e.g., for 'iron' model)
            - thickness_guess: Initial guess for thickness in cm (default 1.95)
            - norm_guess: Initial guess for normalization (default 1.0)
        """
        try:
            import nbragg
        except ImportError:
            raise ImportError(
                "nbragg is required for Analysis class. "
                "Install with: pip install nbragg"
            )

        self.nbragg = nbragg
        self.vary_weights = vary_weights
        self.vary_background = vary_background
        self.vary_sans = vary_sans
        self.vary_extinction = vary_extinction
        self.kwargs = kwargs
        self.result = None
        self.data = None  # Will store nbragg.Data after fit()

        # Setup cross-section
        if isinstance(xs, str):
            self.xs = self._setup_predefined_xs(xs)
        elif hasattr(xs, '__class__') and 'CrossSection' in xs.__class__.__name__:
            # It's a CrossSection object
            self.xs = xs
        elif isinstance(xs, dict):
            # Custom material dictionary
            self.xs = nbragg.CrossSection(**xs)
        else:
            raise ValueError(
                f"xs must be a string, CrossSection object, or dict, got {type(xs)}"
            )

        # Extract thickness and norm guesses from kwargs
        thickness_guess = kwargs.pop('thickness_guess', 1.95)  # cm
        norm_guess = kwargs.pop('norm_guess', 1.0)

        # Create transmission model
        self.model = nbragg.TransmissionModel(
            self.xs,
            vary_background=vary_background,
            **kwargs
        )

        # Set initial parameter values
        if hasattr(self.model, 'params'):
            # Set thickness guess
            for param_name in self.model.params:
                if 'thickness' in param_name.lower() or param_name == 'L':
                    self.model.params[param_name].value = thickness_guess
                # Set norm to 1.0 and fix it
                if 'norm' in param_name.lower():
                    self.model.params[param_name].value = norm_guess
                    self.model.params[param_name].vary = False

        # Set parameter variations
        if vary_weights and hasattr(self.model, 'set_vary_weights'):
            self.model.set_vary_weights(True)

    def _setup_predefined_xs(self, name):
        """Setup predefined cross-section configurations."""
        if name == 'iron_with_cellulose':
            return self._iron_with_cellulose()
        elif name == 'iron_square_response':
            return self._iron_square_response()
        elif name == 'iron':
            return self._iron()
        else:
            raise ValueError(
                f"Unknown predefined cross-section '{name}'. "
                f"Choose from: 'iron_with_cellulose', 'iron_square_response', 'iron'"
            )

    def _iron_with_cellulose(self):
        """
        Create iron with cellulose cross-section (2% cellulose, 98% Fe_alpha).

        Registers cellulose from notebooks/Cellulose_C6O5H10.ncmat if not already available.
        If vary_extinction=True, adds extinction parameters using Uncorr_Sabine method.
        """
        import os
        from pathlib import Path

        try:
            # Register cellulose material if not already available
            cellulose_ncmat = Path("notebooks/Cellulose_C6O5H10.ncmat")
            if cellulose_ncmat.exists():
                # Register the cellulose material
                self.nbragg.register_material(str(cellulose_ncmat))

            # Use Fe_sg229_Iron-alpha as specified
            iron = "Fe_sg229_Iron-alpha.ncmat"
            cellulose = str(cellulose_ncmat) if cellulose_ncmat.exists() else "Cellulose_C6O5H10.ncmat"

            # Create base materials dict with 2% cellulose, 98% iron
            iron_mat = {'mat': iron, 'weight': 0.98}
            cellulose_mat = {'mat': cellulose, 'weight': 0.02}

            # Add extinction parameters to iron if requested
            if self.vary_extinction:
                # Add extinction parameters for Fe_alpha
                # ext_method: Uncorr_Sabine
                # ext_dist: tri (triangular distribution)
                # ext_L: 100000 µm = 10 cm
                # ext_l: 100 µm
                # ext_g: 100 µm
                iron_mat['ext_method'] = 'Uncorr_Sabine'
                iron_mat['ext_dist'] = 'tri'
                iron_mat['ext_L'] = 100000  # µm
                iron_mat['ext_l'] = 100     # µm
                iron_mat['ext_g'] = 100     # µm

            return self.nbragg.CrossSection(
                materials={
                    'iron': iron_mat,
                    'cellulose': cellulose_mat
                }
            )

        except Exception as e:
            raise ValueError(
                f"Failed to create iron_with_cellulose cross-section: {e}. "
                f"Make sure cellulose ncmat is in notebooks/ folder."
            )

    def _iron_square_response(self):
        """Create iron with square response cross-section."""
        try:
            iron = self.nbragg.materials.get("Fe_sg225_Iron-gamma")

            if iron is None:
                raise ValueError("Fe_sg225_Iron-gamma not found in nbragg materials")

            # Create cross-section with square response
            xs = self.nbragg.CrossSection(iron=iron)

            # Add square response if method exists
            if hasattr(xs, 'set_response'):
                xs.set_response('square', width=200)  # 200 µs width

            return xs
        except Exception as e:
            raise ValueError(
                f"Failed to create iron_square_response cross-section: {e}"
            )

    def _iron(self):
        """
        Create simple iron alpha cross-section.

        This creates a CrossSection using Fe_sg229_Iron-alpha.ncmat
        which is suitable for fitting with vary_background=True and vary_response=True.

        Usage
        -----
        >>> analysis = Analysis(xs='iron', vary_background=True, vary_response=True)
        >>> result = analysis.fit(recon)
        >>> result.plot()
        """
        try:
            # Use Fe_sg229_Iron-alpha as specified by user
            iron = "Fe_sg229_Iron-alpha.ncmat"

            return self.nbragg.CrossSection(iron=iron)
        except Exception as e:
            raise ValueError(
                f"Failed to create iron cross-section: {e}. "
                f"Make sure 'Fe_sg229_Iron-alpha.ncmat' is available in nbragg."
            )

    def fit(self, recon, L=9.0, tstep=10e-6, **fit_kwargs):
        """
        Fit the model to reconstructed data.

        Parameters
        ----------
        recon : Reconstruct
            Reconstruct object with reconstructed_data
        L : float, optional
            Flight path length in meters. Default is 9.0 m.
        tstep : float, optional
            Time step in seconds. Default is 10e-6 s (10 µs).
        **fit_kwargs
            Additional keyword arguments passed to model.fit()

        Returns
        -------
        lmfit.ModelResult
            Fitting result

        Raises
        ------
        ValueError
            If reconstruction has not been performed yet
        """
        if recon.reconstructed_data is None:
            raise ValueError(
                "No reconstructed data available. "
                "Call recon.filter() before fitting."
            )

        # Convert reconstructed data to nbragg format
        self.data = recon.to_nbragg(L=L, tstep=tstep)

        # Fit using nbragg
        self.result = self.model.fit(self.data, **fit_kwargs)

        return self.result

    def plot(self, **kwargs):
        """
        Plot the fitting result.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to result.plot()

        Returns
        -------
        matplotlib.figure.Figure
            The created figure

        Raises
        ------
        ValueError
            If fit() has not been called yet
        """
        if self.result is None:
            raise ValueError("No fitting result available. Call fit() first.")

        return self.result.plot(**kwargs)

    def __repr__(self):
        """String representation of the Analysis object."""
        has_result = self.result is not None
        if has_result:
            chi2_str = f"{self.result.redchi:.3f}"
        else:
            chi2_str = "N/A"
        return (f"Analysis(xs={self.xs.__class__.__name__}, "
                f"fitted={has_result}, "
                f"chi2={chi2_str})")
