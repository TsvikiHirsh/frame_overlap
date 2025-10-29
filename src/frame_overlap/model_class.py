"""
Model class for nbragg integration with frame_overlap reconstruction.

This module provides a simplified interface to nbragg for fitting transmission
models to reconstructed data.
"""

import numpy as np
import pandas as pd


class Model:
    """
    Model class for fitting transmission models to reconstructed data.

    This class provides a simplified interface to nbragg.TransmissionModel,
    with predefined cross-section configurations and automatic setup.

    Parameters
    ----------
    xs : str or object, optional
        Cross-section specification. Can be:
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
    model : nbragg.TransmissionModel
        The underlying nbragg transmission model
    result : lmfit.ModelResult or None
        Fitting result after calling fit()

    Examples
    --------
    >>> from frame_overlap import Data, Reconstruct, Model
    >>> # Create and reconstruct data
    >>> data = Data('signal.csv', 'openbeam.csv')
    >>> data.convolute_response(200).overlap([0, 12, 10, 25]).poisson_sample(duty_cycle=0.8)
    >>> recon = Reconstruct(data)
    >>> recon.filter(kind='wiener', noise_power=0.01)
    >>>
    >>> # Fit with predefined model
    >>> model = Model(xs='iron_with_cellulose', vary_weights=True)
    >>> result = model.fit(recon)
    >>>
    >>> # Or use custom cross-section
    >>> import nbragg
    >>> xs = nbragg.CrossSection(iron=nbragg.materials["Fe_sg225_Iron-gamma"])
    >>> model = Model(xs=xs, vary_background=True)
    >>> result = model.fit(recon)
    """

    def __init__(self, xs='iron_with_cellulose', vary_weights=False, vary_background=True, **kwargs):
        """
        Initialize Model with cross-section specification.

        Parameters
        ----------
        xs : str or object
            Cross-section specification
        vary_weights : bool
            Whether to vary material weights
        vary_background : bool
            Whether to vary background
        **kwargs
            Additional arguments for nbragg.TransmissionModel
        """
        try:
            import nbragg
        except ImportError:
            raise ImportError(
                "nbragg is required for Model class. "
                "Install with: pip install nbragg"
            )

        self.nbragg = nbragg
        self.vary_weights = vary_weights
        self.vary_background = vary_background
        self.kwargs = kwargs
        self.result = None

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

        # Create transmission model
        self.model = nbragg.TransmissionModel(
            self.xs,
            vary_background=vary_background,
            **kwargs
        )

        # Set parameter variations
        if vary_weights and hasattr(self.model, 'set_vary_weights'):
            self.model.set_vary_weights(True)

    def _setup_predefined_xs(self, name):
        """Setup predefined cross-section configurations."""
        if name == 'iron_with_cellulose':
            return self._iron_with_cellulose()
        elif name == 'iron_square_response':
            return self._iron_square_response()
        else:
            raise ValueError(
                f"Unknown predefined cross-section '{name}'. "
                f"Choose from: 'iron_with_cellulose', 'iron_square_response'"
            )

    def _iron_with_cellulose(self):
        """Create iron with cellulose cross-section."""
        try:
            iron = self.nbragg.materials.get("Fe_sg225_Iron-gamma")
            cellulose = self.nbragg.materials.get("C6H10O5_Cellulose")

            if iron is None or cellulose is None:
                raise ValueError("Required materials not found in nbragg")

            return self.nbragg.CrossSection(
                iron=iron,
                cellulose=cellulose
            )
        except Exception as e:
            raise ValueError(
                f"Failed to create iron_with_cellulose cross-section: {e}. "
                f"Make sure nbragg has the required materials."
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

    def fit(self, recon, **fit_kwargs):
        """
        Fit the model to reconstructed data.

        Parameters
        ----------
        recon : Reconstruct
            Reconstruct object with reconstructed_data
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

        # nbragg expects a Data-like object with table attribute
        # We can use recon directly since it has table property
        self.result = self.model.fit(recon, **fit_kwargs)

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
        """String representation of the Model object."""
        has_result = self.result is not None
        chi2 = self.result.redchi if has_result else None
        return (f"Model(xs={self.xs.__class__.__name__}, "
                f"fitted={has_result}, "
                f"chi2={chi2:.3f if chi2 is not None else 'N/A'})")
