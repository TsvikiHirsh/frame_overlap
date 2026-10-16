"""
Groupby functionality for parametric scans and analysis.

This module provides classes and functions for performing parametric scans
over various data processing and fitting parameters, allowing users to
explore sensitivity to different configurations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import copy


class ParametricScan:
    """
    Parametric scan for exploring sensitivity to different parameters.

    This class allows users to perform scans over data processing parameters
    (pulse duration, number of frames, noise power, etc.) and analyze the
    results across different configurations.

    Parameters
    ----------
    data_template : Data
        Template Data object to use as a starting point for scans
    reconstruct_params : dict, optional
        Default parameters for reconstruction
    analysis_params : dict, optional
        Default parameters for analysis

    Attributes
    ----------
    data_template : Data
        Template data object
    results : pandas.DataFrame
        DataFrame containing scan results
    scan_params : dict
        Dictionary of parameter ranges being scanned

    Examples
    --------
    >>> from frame_overlap import Data, ParametricScan
    >>> data = Data('signal.csv')
    >>> scan = ParametricScan(data)
    >>> scan.add_parameter('pulse_duration', [100, 200, 300, 400])
    >>> scan.add_parameter('n_frames', [2, 3, 4, 5])
    >>> scan.run()
    >>> scan.plot_parameter_sensitivity('pulse_duration', 'chi2_per_dof')
    """

    def __init__(self, data_template, reconstruct_params=None, analysis_params=None):
        """Initialize ParametricScan with template data."""
        from .data_class import Data

        if not isinstance(data_template, Data):
            raise TypeError("data_template must be a Data object")

        self.data_template = data_template
        self.reconstruct_params = reconstruct_params or {'kind': 'wiener', 'noise_power': 0.01}
        self.analysis_params = analysis_params or {'response': 'square'}

        self.scan_params = {}
        self.results = None

    def add_parameter(self, param_name, param_values):
        """
        Add a parameter to scan over.

        Parameters
        ----------
        param_name : str
            Name of the parameter. Can be:
            - Data parameters: 'pulse_duration', 'flux', 'duty_cycle'
            - Overlap parameters: 'n_frames', 'frame_spacing'
            - Reconstruction parameters: 'noise_power', 'filter_kind'
            - Analysis parameters: 'response', 'thickness_range'
        param_values : list or array
            Values to scan over for this parameter

        Returns
        -------
        self
            Returns self for method chaining
        """
        if not isinstance(param_values, (list, tuple, np.ndarray)):
            raise ValueError("param_values must be a list, tuple, or array")

        self.scan_params[param_name] = list(param_values)
        return self

    def run(self, verbose=True):
        """
        Run the parametric scan over all parameter combinations.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress information. Default is True.

        Returns
        -------
        self
            Returns self for method chaining
        """
        if not self.scan_params:
            raise ValueError("No parameters added. Use add_parameter() first.")

        # Generate all combinations of parameters
        param_names = list(self.scan_params.keys())
        param_values = list(self.scan_params.values())
        combinations = list(product(*param_values))

        results = []

        for i, combo in enumerate(combinations):
            if verbose:
                print(f"Running combination {i+1}/{len(combinations)}: "
                     f"{dict(zip(param_names, combo))}")

            try:
                result = self._run_single_combination(dict(zip(param_names, combo)))
                result['success'] = True
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Failed: {e}")
                result = dict(zip(param_names, combo))
                result['success'] = False
                result['error'] = str(e)
                results.append(result)

        self.results = pd.DataFrame(results)
        return self

    def _run_single_combination(self, params):
        """
        Run a single parameter combination.

        Parameters
        ----------
        params : dict
            Dictionary of parameter values for this combination

        Returns
        -------
        dict
            Dictionary with results for this combination
        """
        from .data_class import Data
        from .reconstruct import Reconstruct
        from .analysis_class import Analysis

        # Create a copy of the template data
        data = self.data_template.copy()

        # Apply data parameters
        if 'pulse_duration' in params:
            data.convolute_response(pulse_duration=params['pulse_duration'])

        if 'duty_cycle' in params:
            duty_cycle = params['duty_cycle']
        else:
            duty_cycle = 1.0

        # Apply overlap parameters
        if 'n_frames' in params or 'frame_spacing' in params:
            n_frames = params.get('n_frames', 4)
            frame_spacing = params.get('frame_spacing', 12)  # ms

            # Create sequence
            seq = [0] + [frame_spacing] * (n_frames - 1)
            data.overlap(seq)

        # Apply Poisson sampling
        if 'flux' in params:
            data.flux = params['flux']

        data.poisson_sample(duty_cycle=duty_cycle)

        # Reconstruct
        recon_params = self.reconstruct_params.copy()
        if 'noise_power' in params:
            recon_params['noise_power'] = params['noise_power']
        if 'filter_kind' in params:
            recon_params['kind'] = params['filter_kind']

        recon = Reconstruct(data)
        recon.filter(**recon_params)

        # Analysis
        analysis_params = self.analysis_params.copy()
        if 'response' in params:
            analysis_params['response'] = params['response']

        analysis = Analysis(recon)
        analysis.fit(**analysis_params)

        # Collect results
        result = params.copy()

        # Add reconstruction statistics
        recon_stats = recon.get_statistics()
        for key, value in recon_stats.items():
            result[f'recon_{key}'] = value

        # Add fit results
        if analysis.fit_result and analysis.fit_result['success']:
            fit_results = analysis.get_fit_results()
            result['fit_thickness'] = fit_results['thickness']
            result['fit_thickness_err'] = fit_results['thickness_err']
            result['fit_chi_square'] = fit_results['chi_square']
            result['fit_reduced_chi_square'] = fit_results['reduced_chi_square']

            # Add material weights
            for i, (material, (weight, err)) in enumerate(zip(fit_results['materials'],
                                                              fit_results['weights'])):
                result[f'weight_{material}'] = weight
                result[f'weight_{material}_err'] = err

        return result

    def get_results(self, successful_only=True):
        """
        Get the scan results as a DataFrame.

        Parameters
        ----------
        successful_only : bool, optional
            Whether to return only successful runs. Default is True.

        Returns
        -------
        pandas.DataFrame
            DataFrame with scan results
        """
        if self.results is None:
            raise ValueError("No results available. Call run() first.")

        if successful_only and 'success' in self.results.columns:
            return self.results[self.results['success']].copy()
        return self.results.copy()

    def plot_parameter_sensitivity(self, param_name, metric='fit_reduced_chi_square',
                                   groupby=None, **kwargs):
        """
        Plot sensitivity of a metric to a parameter.

        Parameters
        ----------
        param_name : str
            Name of the parameter to plot on x-axis
        metric : str, optional
            Name of the metric to plot on y-axis.
            Default is 'fit_reduced_chi_square'.
        groupby : str, optional
            Name of another parameter to group by (creates multiple lines)
        **kwargs
            Additional keyword arguments for matplotlib

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.results is None:
            raise ValueError("No results available. Call run() first.")

        df = self.get_results(successful_only=True)

        if param_name not in df.columns:
            raise ValueError(f"Parameter '{param_name}' not in results")
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not in results")

        fig, ax = plt.subplots(figsize=(10, 6))

        if groupby and groupby in df.columns:
            # Plot multiple lines grouped by another parameter
            for group_value in sorted(df[groupby].unique()):
                subset = df[df[groupby] == group_value]
                ax.plot(subset[param_name], subset[metric],
                       'o-', label=f'{groupby}={group_value}', **kwargs)
            ax.legend()
        else:
            # Single line plot
            ax.plot(df[param_name], df[metric], 'o-', **kwargs)

        ax.set_xlabel(param_name)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs {param_name}')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_heatmap(self, param_x, param_y, metric='fit_reduced_chi_square', **kwargs):
        """
        Plot a 2D heatmap of a metric versus two parameters.

        Parameters
        ----------
        param_x : str
            Parameter for x-axis
        param_y : str
            Parameter for y-axis
        metric : str, optional
            Metric to plot as color. Default is 'fit_reduced_chi_square'.
        **kwargs
            Additional keyword arguments for matplotlib

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.results is None:
            raise ValueError("No results available. Call run() first.")

        df = self.get_results(successful_only=True)

        if param_x not in df.columns or param_y not in df.columns:
            raise ValueError(f"Parameters '{param_x}' or '{param_y}' not in results")
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not in results")

        # Pivot data for heatmap
        pivot = df.pivot_table(values=metric, index=param_y, columns=param_x)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(pivot.values, aspect='auto', origin='lower',
                      extent=[pivot.columns.min(), pivot.columns.max(),
                             pivot.index.min(), pivot.index.max()],
                      **kwargs)

        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_title(f'{metric} Heatmap')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric)

        # Add value annotations if not too many cells
        if pivot.size <= 100:
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    text = ax.text(pivot.columns[j], pivot.index[i],
                                 f'{pivot.values[i, j]:.2f}',
                                 ha="center", va="center", color="w", fontsize=8)

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(self, parameters=None, **kwargs):
        """
        Plot correlation matrix between parameters and metrics.

        Parameters
        ----------
        parameters : list of str, optional
            List of parameters/metrics to include. If None, includes all.
        **kwargs
            Additional keyword arguments for matplotlib

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.results is None:
            raise ValueError("No results available. Call run() first.")

        df = self.get_results(successful_only=True)

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if parameters:
            numeric_cols = [c for c in numeric_cols if c in parameters]

        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation matrix")

        # Calculate correlation matrix
        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(corr.values, aspect='auto', cmap='RdBu_r',
                      vmin=-1, vmax=1, **kwargs)

        # Set ticks and labels
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax.set_yticklabels(numeric_cols)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')

        # Add value annotations
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                text = ax.text(j, i, f'{corr.values[i, j]:.2f}',
                             ha="center", va="center",
                             color="white" if abs(corr.values[i, j]) > 0.5 else "black",
                             fontsize=8)

        ax.set_title('Parameter Correlation Matrix')
        plt.tight_layout()
        return fig

    def plot_summary(self):
        """
        Plot a summary of all scan results.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.results is None:
            raise ValueError("No results available. Call run() first.")

        df = self.get_results(successful_only=True)

        # Create subplots for different metrics
        metrics = ['fit_reduced_chi_square', 'fit_thickness', 'recon_r_squared']
        metrics = [m for m in metrics if m in df.columns]

        n_metrics = len(metrics)
        if n_metrics == 0:
            raise ValueError("No suitable metrics found in results")

        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            values = df[metric].values
            ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {metric}')
            ax.grid(True, alpha=0.3, axis='y')

            # Add statistics
            mean_val = values.mean()
            std_val = values.std()
            ax.axvline(mean_val, color='r', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val - std_val, color='r', linestyle=':', alpha=0.5)
            ax.axvline(mean_val + std_val, color='r', linestyle=':', alpha=0.5)
            ax.legend()

        plt.tight_layout()
        return fig

    def __repr__(self):
        """String representation of the ParametricScan object."""
        n_params = len(self.scan_params)
        n_results = len(self.results) if self.results is not None else 0
        return f"ParametricScan(n_parameters={n_params}, n_results={n_results})"


def compare_configurations(configs, labels=None):
    """
    Compare results from multiple ParametricScan objects or configurations.

    Parameters
    ----------
    configs : list of ParametricScan or dict
        List of ParametricScan objects or result dictionaries to compare
    labels : list of str, optional
        Labels for each configuration

    Returns
    -------
    matplotlib.figure.Figure
        The created comparison figure
    """
    if labels is None:
        labels = [f'Config {i+1}' for i in range(len(configs))]

    if len(configs) != len(labels):
        raise ValueError("Number of configs must match number of labels")

    # Extract results from each config
    results_list = []
    for config in configs:
        if isinstance(config, ParametricScan):
            results_list.append(config.get_results(successful_only=True))
        elif isinstance(config, pd.DataFrame):
            results_list.append(config)
        else:
            raise TypeError("configs must be ParametricScan objects or DataFrames")

    # Find common metrics
    common_metrics = set(results_list[0].columns)
    for results in results_list[1:]:
        common_metrics &= set(results.columns)

    # Filter to numeric metrics
    numeric_metrics = []
    for metric in common_metrics:
        if results_list[0][metric].dtype in [np.float64, np.int64, np.float32, np.int32]:
            numeric_metrics.append(metric)

    if not numeric_metrics:
        raise ValueError("No common numeric metrics found")

    # Create comparison plots
    n_metrics = min(4, len(numeric_metrics))  # Limit to 4 metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(numeric_metrics[:n_metrics]):
        ax = axes[i]

        # Box plot for each configuration
        data_to_plot = [results[metric].values for results in results_list]
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(configs)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # Rotate x labels if many configs
        if len(configs) > 3:
            ax.set_xticklabels(labels, rotation=45, ha='right')

    # Remove extra subplots if fewer than 4 metrics
    for i in range(n_metrics, 4):
        fig.delaxes(axes[i])

    plt.tight_layout()
    return fig
