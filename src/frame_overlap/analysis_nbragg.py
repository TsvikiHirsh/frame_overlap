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
                 vary_sans=False, vary_extinction=False, response_kind=None,
                 pulse_duration=None, **kwargs):
        """
        Initialize Analysis with cross-section specification.

        Parameters
        ----------
        xs : str or object
            Cross-section specification. Available predefined options:
            - 'iron': Simple Fe_sg229_Iron-alpha (use with vary_background=True, vary_response=True)
            - 'iron_square_response': Iron with square response
            - 'iron_with_cellulose': Iron with cellulose background (2% cellulose, 98% Fe_alpha)
            - 'iron_cellulose_fixed_response': Iron with cellulose, squared_jorgensen response fixed to pulse_duration
        vary_weights : bool
            Whether to vary material weights. Default is False.
        vary_background : bool
            Whether to vary background. Default is True.
        vary_sans : bool
            Whether to vary SANS parameters. Default is False.
        vary_extinction : bool
            Whether to include extinction parameters. Default is False.
        response_kind : str, optional
            Type of response function to use. Options:
            - 'jorgensen': Standard Jorgensen response (default)
            - 'squared': Squared response
            - 'full_jorgensen': Full Jorgensen response
            - 'squared_jorgensen': Squared Jorgensen response
            Default is None (uses nbragg default).
        pulse_duration : float, optional
            Pulse duration in microseconds. Used for 'iron_cellulose_fixed_response' model
            to set the fixed response width. Default is None.
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
        self.response_kind = response_kind
        self.pulse_duration = pulse_duration
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

        # Create transmission model with all vary parameters
        # Build the TransmissionModel kwargs
        model_kwargs = {
            'vary_background': vary_background,
            **kwargs
        }

        # Add vary_weights if specified (not None)
        if vary_weights is not None:
            model_kwargs['vary_weights'] = vary_weights

        # Add vary_sans if specified (not None)
        if vary_sans is not None:
            model_kwargs['vary_sans'] = vary_sans

        # Add vary_extinction if specified (not None)
        if vary_extinction is not None:
            model_kwargs['vary_extinction'] = vary_extinction

        # Add response_kind if specified (not None)
        if response_kind is not None:
            model_kwargs['response_kind'] = response_kind

        self.model = nbragg.TransmissionModel(
            self.xs,
            **model_kwargs
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

        # For iron_cellulose_fixed_response, set response width to pulse_duration and fix it
        if isinstance(xs, str) and xs == 'iron_cellulose_fixed_response':
            if hasattr(self.model, 'params'):
                # Look for response width parameters (e.g., 'width', 'sigma', etc.)
                for param_name in self.model.params:
                    if 'width' in param_name.lower() or 'sigma' in param_name.lower():
                        self.model.params[param_name].value = self.pulse_duration
                        self.model.params[param_name].vary = False
                        break

        # Set parameter variations
        if vary_weights and hasattr(self.model, 'set_vary_weights'):
            self.model.set_vary_weights(True)

    def _setup_predefined_xs(self, name):
        """Setup predefined cross-section configurations."""
        if name == 'iron_with_cellulose':
            return self._iron_with_cellulose()
        elif name == 'iron_cellulose_fixed_response':
            return self._iron_cellulose_fixed_response()
        elif name == 'iron_square_response':
            return self._iron_square_response()
        elif name == 'iron':
            return self._iron()
        else:
            raise ValueError(
                f"Unknown predefined cross-section '{name}'. "
                f"Choose from: 'iron_with_cellulose', 'iron_cellulose_fixed_response', 'iron_square_response', 'iron'"
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

    def _iron_cellulose_fixed_response(self):
        """
        Create iron with cellulose cross-section with fixed squared_jorgensen response.

        This model:
        - Uses iron_with_cellulose composition (2% cellulose, 98% Fe_alpha)
        - Sets response_kind='squared_jorgensen'
        - Fixes vary_response=False
        - Sets response width to pulse_duration (if provided)

        Requires pulse_duration to be set during initialization.
        """
        import os
        from pathlib import Path

        if self.pulse_duration is None:
            raise ValueError(
                "pulse_duration must be provided when using 'iron_cellulose_fixed_response' model. "
                "Pass pulse_duration=<value_in_microseconds> to Analysis()"
            )

        try:
            # Register cellulose material if not already available
            cellulose_ncmat = Path("notebooks/Cellulose_C6O5H10.ncmat")
            if cellulose_ncmat.exists():
                self.nbragg.register_material(str(cellulose_ncmat))

            # Use Fe_sg229_Iron-alpha
            iron = "Fe_sg229_Iron-alpha.ncmat"
            cellulose = str(cellulose_ncmat) if cellulose_ncmat.exists() else "Cellulose_C6O5H10.ncmat"

            # Create base materials dict with 2% cellulose, 98% iron
            iron_mat = {'mat': iron, 'weight': 0.98}
            cellulose_mat = {'mat': cellulose, 'weight': 0.02}

            # Add extinction parameters to iron if requested
            if self.vary_extinction:
                iron_mat['ext_method'] = 'Uncorr_Sabine'
                iron_mat['ext_dist'] = 'tri'
                iron_mat['ext_L'] = 100000  # µm
                iron_mat['ext_l'] = 100     # µm
                iron_mat['ext_g'] = 100     # µm

            # Override response_kind to squared_jorgensen for this model
            if 'response_kind' not in self.kwargs:
                self.kwargs['response_kind'] = 'squared_jorgensen'

            # Set vary_response to False for this model
            if 'vary_response' not in self.kwargs:
                self.kwargs['vary_response'] = False

            return self.nbragg.CrossSection(
                materials={
                    'iron': iron_mat,
                    'cellulose': cellulose_mat
                }
            )

        except Exception as e:
            raise ValueError(
                f"Failed to create iron_cellulose_fixed_response cross-section: {e}. "
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

    def get_params(self):
        """
        Get the current model parameters.

        Returns
        -------
        lmfit.Parameters
            The model's parameter object

        Examples
        --------
        >>> params = analysis.get_params()
        >>> print(params)
        >>> print(f"Thickness: {params['thickness'].value}, vary={params['thickness'].vary}")
        """
        return self.model.params

    def set_params(self, **param_settings):
        """
        Set model parameters before fitting.

        This is a convenience method to set parameter values and vary flags
        before calling fit().

        **IMPORTANT NOTE**: nbragg's internal Rietveld fitting may override
        some parameter constraints. If you find parameters are still being
        varied during fitting, you may need to:
        1. Fix the parameter after fitting and refit
        2. Use nbragg's fit options like `vary_params` or `fix_params`
        3. Access result.params directly and modify for subsequent fits

        Parameters
        ----------
        **param_settings : dict
            Parameter settings as keyword arguments. Each parameter can be:
            - A single value to set the parameter value
            - A dict with 'value' and/or 'vary' keys

        Examples
        --------
        >>> # Set thickness to 1.95 and fix it
        >>> analysis.set_params(thickness={'value': 1.95, 'vary': False})
        >>> result = analysis.fit(recon, params=analysis.model.params)
        >>>
        >>> # Set multiple parameters
        >>> analysis.set_params(
        ...     thickness={'value': 1.95, 'vary': False},
        ...     norm={'value': 1.0, 'vary': True},
        ...     temp={'vary': False}
        ... )
        >>>
        >>> # Shorthand: just set value (keeps existing vary flag)
        >>> analysis.set_params(thickness=1.95)
        >>>
        >>> # If parameters still vary, try fixing after first fit:
        >>> result = analysis.fit(recon)
        >>> result.params['thickness'].value = 1.95
        >>> result.params['thickness'].vary = False
        >>> result2 = analysis.model.fit(analysis.data, params=result.params)
        """
        if not hasattr(self.model, 'params'):
            raise ValueError("Model does not have params attribute")

        for param_name, setting in param_settings.items():
            if param_name not in self.model.params:
                raise ValueError(f"Parameter '{param_name}' not found in model. "
                               f"Available parameters: {list(self.model.params.keys())}")

            if isinstance(setting, dict):
                # Dict with 'value' and/or 'vary' keys
                if 'value' in setting:
                    self.model.params[param_name].value = setting['value']
                if 'vary' in setting:
                    self.model.params[param_name].vary = setting['vary']
            else:
                # Just a value
                self.model.params[param_name].value = setting

    def fit(self, recon, L=9.0, tstep=10e-6, params=None, **fit_kwargs):
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
        params : lmfit.Parameters, optional
            Custom parameters to use for fitting. If None, uses self.model.params.
            This allows you to override parameter settings before fitting.
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

        Notes
        -----
        To fix parameters before fitting, use the set_params() method or
        pass a modified params object:

        >>> analysis.set_params(thickness={'value': 1.95, 'vary': False})
        >>> result = analysis.fit(recon, params=analysis.model.params)
        """
        if recon.reconstructed_data is None:
            raise ValueError(
                "No reconstructed data available. "
                "Call recon.filter() before fitting."
            )

        # Convert reconstructed data to nbragg format
        self.data = recon.to_nbragg(L=L, tstep=tstep)

        # Fit using nbragg, passing params if provided
        if params is not None:
            self.result = self.model.fit(self.data, params=params, **fit_kwargs)
        else:
            self.result = self.model.fit(self.data, params=self.model.params, **fit_kwargs)

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
