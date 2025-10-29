"""
Reconstruct/Filter class for deconvolution and signal reconstruction.

This module provides the Reconstruct class for applying various filtering
techniques to reconstruct original signals from overlapped frame data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal


class Reconstruct:
    """
    Reconstruct/Filter object for signal deconvolution and reconstruction.

    This class applies various filtering techniques (e.g., Wiener filter) to
    reconstruct the original signal from processed data. It provides statistical
    methods to quantify reconstruction quality and plotting methods for visualization.

    Parameters
    ----------
    data : Data
        Data object containing the signal to be reconstructed

    Attributes
    ----------
    data : Data
        Reference to the input Data object
    reconstructed_data : pandas.DataFrame
        DataFrame with reconstructed signal (time, counts, err)
    reference_data : pandas.DataFrame
        Reference data for comparison (poissoned data before reconstruction)
    statistics : dict
        Dictionary containing reconstruction quality statistics

    Examples
    --------
    >>> from frame_overlap import Data, Reconstruct, Model
    >>> # Load and process data
    >>> data = Data('signal.csv', 'openbeam.csv')
    >>> data.convolute_response(200).overlap([0, 12, 10, 25]).poisson_sample(duty_cycle=0.8)
    >>> # Reconstruct
    >>> recon = Reconstruct(data)
    >>> recon.filter(kind='wiener', noise_power=0.01)
    >>> # Plot with residuals
    >>> recon.plot()  # Default: transmission with residuals
    >>> recon.plot(kind='signal', ylim=(0, 1000), residual_ylim=(-5, 5))
    >>> recon.plot(kind='transmission', color='indianred', residual_color='blue')
    >>> print(recon.statistics)
    >>> # Fit with nbragg
    >>> import nbragg
    >>> xs = nbragg.CrossSection(iron=nbragg.materials["Fe_sg225_Iron-gamma"])
    >>> result = nbragg.TransmissionModel(xs).fit(recon)  # recon.table is used
    >>> # Or use simplified Model class
    >>> model = Model(xs='iron_with_cellulose', vary_weights=True)
    >>> result = model.fit(recon)
    """

    def __init__(self, data):
        """
        Initialize Reconstruct object with a Data object.

        Parameters
        ----------
        data : Data
            Data object to be reconstructed
        """
        from .data_class import Data

        if not isinstance(data, Data):
            raise TypeError("data must be a Data object")

        if data.table is None:
            raise ValueError("Data object must have loaded data")

        self.data = data
        self.reconstructed_data = None
        self.reference_data = None  # Will store poissoned data for comparison
        self.statistics = {}

    @property
    def table(self):
        """
        Get reconstructed data as table for nbragg compatibility.

        Returns
        -------
        pandas.DataFrame
            Reconstructed data with columns: time, counts, err
        """
        return self.reconstructed_data

    def filter(self, kind='wiener', noise_power=0.01, **kwargs):
        """
        Apply filtering to reconstruct the original signal.

        Parameters
        ----------
        kind : str, optional
            Type of filter to apply. Options:
            - 'wiener': Wiener deconvolution (default)
            - 'lucy': Richardson-Lucy deconvolution
            - 'tikhonov': Tikhonov regularization
        noise_power : float, optional
            Noise power parameter for regularization. Default is 0.01.
        **kwargs
            Additional keyword arguments for specific filters

        Returns
        -------
        self
            Returns self for method chaining

        Raises
        ------
        ValueError
            If kernel is not defined or filter kind is invalid
        """
        if self.data.kernel is None:
            raise ValueError("Data object must have a kernel defined (call data.overlap first)")

        # Store reference data if available (poissoned_data is the data we want to reconstruct to)
        if self.data.poissoned_data is not None:
            self.reference_data = self.data.poissoned_data.copy()
        elif self.data.overlapped_data is not None:
            # If no poisson sampling was done, use overlapped data as reference
            self.reference_data = self.data.overlapped_data.copy()

        kind = kind.lower()

        if kind == 'wiener':
            reconstructed = self._wiener_filter(noise_power, **kwargs)
        elif kind == 'lucy' or kind == 'richardson-lucy':
            reconstructed = self._lucy_richardson_filter(**kwargs)
        elif kind == 'tikhonov':
            reconstructed = self._tikhonov_filter(noise_power, **kwargs)
        else:
            raise ValueError(f"Unknown filter kind '{kind}'. "
                           f"Choose from: 'wiener', 'lucy', 'tikhonov'")

        # Create reconstructed data
        self.reconstructed_data = pd.DataFrame({
            'time': self.data.table['time'].values,
            'counts': reconstructed,
            'err': np.sqrt(np.maximum(reconstructed, 1))
        })

        # Calculate statistics
        self._calculate_statistics()

        return self

    def _wiener_filter(self, noise_power, **kwargs):
        """
        Apply Wiener deconvolution filter.

        Parameters
        ----------
        noise_power : float
            Noise power for Wiener filter regularization

        Returns
        -------
        np.ndarray
            Reconstructed signal
        """
        if noise_power <= 0:
            raise ValueError("noise_power must be positive")

        observed = self.data.table['counts'].values

        # Reconstruct the kernel from the stored sequence
        kernel = self._reconstruct_kernel()

        # Handle kernel length
        if len(kernel) > len(observed):
            # Truncate kernel if it's longer than observed
            kernel = kernel[:len(observed)]

        # Pad kernel to match observed signal length
        if len(kernel) < len(observed):
            kernel_padded = np.pad(kernel, (0, len(observed) - len(kernel)), 'constant')
        else:
            kernel_padded = kernel

        # Apply Wiener deconvolution in frequency domain
        H = np.fft.fft(kernel_padded)
        Y = np.fft.fft(observed)
        H_conj = np.conj(H)
        G = H_conj / (np.abs(H)**2 + noise_power)
        X_est = Y * G
        x_est = np.real(np.fft.ifft(X_est))

        return x_est

    def _lucy_richardson_filter(self, iterations=10, **kwargs):
        """
        Apply Richardson-Lucy deconvolution.

        Parameters
        ----------
        iterations : int, optional
            Number of iterations. Default is 10.

        Returns
        -------
        np.ndarray
            Reconstructed signal
        """
        observed = self.data.table['counts'].values
        kernel = self._reconstruct_kernel()

        # Handle kernel length - truncate if longer than observed
        if len(kernel) > len(observed):
            kernel = kernel[:len(observed)]

        # Normalize kernel
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()
        else:
            kernel = np.ones_like(kernel) / len(kernel)

        # Initialize with observed signal (ensure same length as observed)
        estimate = np.maximum(observed.copy(), 1e-10)

        # Richardson-Lucy iterations
        for _ in range(iterations):
            # Convolve estimate with kernel (same mode to keep length)
            convolved = np.convolve(estimate, kernel, mode='same')
            convolved = np.maximum(convolved, 1e-10)

            # Ensure convolved has same length as observed
            if len(convolved) != len(observed):
                convolved = convolved[:len(observed)]

            # Calculate ratio
            ratio = observed / convolved

            # Convolve ratio with flipped kernel
            correction = np.convolve(ratio, kernel[::-1], mode='same')

            # Ensure correction has same length
            if len(correction) != len(estimate):
                correction = correction[:len(estimate)]

            # Update estimate
            estimate = estimate * correction

        return estimate

    def _tikhonov_filter(self, alpha, **kwargs):
        """
        Apply Tikhonov regularization.

        Parameters
        ----------
        alpha : float
            Regularization parameter

        Returns
        -------
        np.ndarray
            Reconstructed signal
        """
        observed = self.data.table['counts'].values
        kernel = self._reconstruct_kernel()

        # Pad kernel to match observed signal length
        kernel_padded = np.pad(kernel, (0, len(observed) - len(kernel)), 'constant')

        # Apply Tikhonov regularization in frequency domain
        H = np.fft.fft(kernel_padded)
        Y = np.fft.fft(observed)
        H_conj = np.conj(H)
        G = H_conj / (np.abs(H)**2 + alpha)
        X_est = Y * G
        x_est = np.real(np.fft.ifft(X_est))

        return x_est

    def _reconstruct_kernel(self):
        """
        Reconstruct the convolution kernel from the frame overlap sequence.

        Returns
        -------
        np.ndarray
            The kernel array
        """
        if self.data.kernel is None:
            raise ValueError("Kernel not defined in Data object")

        # Create a simple rectangular kernel based on the overlap sequence
        # This is a simplified version - you may need to adjust based on your needs
        seq = np.array(self.data.kernel) * 100  # Convert ms to 10µs bins
        n_frames = len(seq)

        # Create kernel with rectangular pulses
        kernel_length = int(seq.sum()) + 100  # Add some buffer
        kernel = np.zeros(kernel_length)

        # Add rectangular pulses at the specified positions
        cumulative = 0
        pulse_width = 20  # Default pulse width in bins
        for i, delay in enumerate(seq):
            start = int(cumulative)
            end = start + pulse_width
            if end < len(kernel):
                kernel[start:end] = 1.0 / n_frames
            cumulative += delay

        # Normalize kernel
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()

        return kernel

    def _calculate_statistics(self):
        """Calculate reconstruction quality statistics."""
        if self.reconstructed_data is None:
            return

        # If we have reference data (poissoned data before reconstruction)
        if self.reference_data is not None:
            # Match lengths
            min_len = min(len(self.reference_data), len(self.reconstructed_data))
            reference = self.reference_data['counts'].values[:min_len]
            reconstructed = self.reconstructed_data['counts'].values[:min_len]
            errors = self.reference_data['err'].values[:min_len]

            # Calculate chi-squared
            residuals = reference - reconstructed
            errors_safe = np.maximum(errors, 1e-10)  # Avoid division by zero
            chi2 = np.sum((residuals / errors_safe)**2)
            chi2_per_dof = chi2 / min_len

            # Calculate R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((reference - reference.mean())**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Calculate RMSE
            rmse = np.sqrt(np.mean(residuals**2))

            # Calculate normalized RMSE
            nrmse = rmse / (reference.max() - reference.min()) if (reference.max() - reference.min()) > 0 else 0

            self.statistics = {
                'chi2': chi2,
                'chi2_per_dof': chi2_per_dof,
                'r_squared': r_squared,
                'rmse': rmse,
                'nrmse': nrmse,
                'n_points': min_len
            }
        else:
            # Without reference, just calculate basic statistics
            reconstructed = self.reconstructed_data['counts'].values
            self.statistics = {
                'mean': reconstructed.mean(),
                'std': reconstructed.std(),
                'min': reconstructed.min(),
                'max': reconstructed.max(),
                'n_points': len(reconstructed)
            }

    def get_statistics(self):
        """
        Get reconstruction quality statistics.

        Returns
        -------
        dict
            Dictionary with statistics: chi2, chi2_per_dof, r_squared, rmse, nrmse
        """
        return self.statistics.copy()

    def _format_chi2(self, chi2_val):
        """Format chi2 value with 2 significant figures or scientific notation."""
        if chi2_val is None:
            return "N/A"

        # Use scientific notation for very large or very small numbers
        if abs(chi2_val) >= 1000 or (abs(chi2_val) < 0.01 and abs(chi2_val) > 0):
            return f"{chi2_val:.2g}"
        else:
            # Round to 2 significant figures
            if chi2_val == 0:
                return "0.0"
            from math import log10, floor
            sig_figs = 2
            rounded = round(chi2_val, -int(floor(log10(abs(chi2_val)))) + (sig_figs - 1))
            return f"{rounded:.2f}" if rounded < 100 else f"{rounded:.0f}"

    def plot(self, kind='transmission', show_errors=True, fontsize=16, figsize=(10, 8),
             residual_height_ratio=0.3, **kwargs):
        """
        Plot reconstruction results with data and residuals in subplots.

        Parameters
        ----------
        kind : str, optional
            Type of plot:
            - 'transmission': Compare transmission (default)
            - 'signal': Compare signal counts
            - 'openbeam': Compare openbeam counts
            - 'statistics': Plot statistical summary (single plot)
        show_errors : bool, optional
            Whether to show error bars. Default is True.
        fontsize : int, optional
            Font size for labels, ticks, and legend. Default is 16.
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (10, 8).
        residual_height_ratio : float, optional
            Height ratio of residual plot to data plot. Default is 0.3.
        **kwargs
            Additional keyword arguments. Can use prefixes:
            - No prefix: applies to data plot (e.g., ylim=(0,1), color='red')
            - 'residual_': applies to residual plot (e.g., residual_ylim=(-5,5))

        Returns
        -------
        matplotlib.figure.Figure
            The created figure

        Raises
        ------
        ValueError
            If reconstruction has not been performed yet
        """
        if self.reconstructed_data is None:
            raise ValueError("No reconstruction available. Call filter() first.")

        # Set font sizes for all elements
        plt.rcParams.update({
            'font.size': fontsize,
            'axes.labelsize': fontsize,
            'axes.titlesize': fontsize + 2,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            'legend.fontsize': fontsize
        })

        # Statistics plot is a single plot
        if kind == 'statistics':
            fig, ax = plt.subplots(figsize=figsize)
            self._plot_statistics(ax)
            plt.tight_layout()
            return fig

        # Separate kwargs for data and residual plots
        data_kwargs = {}
        residual_kwargs = {}
        for key, value in kwargs.items():
            if key.startswith('residual_'):
                residual_kwargs[key.replace('residual_', '')] = value
            else:
                data_kwargs[key] = value

        # Create subplots with shared x-axis
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, residual_height_ratio], hspace=0.05)
        ax_data = fig.add_subplot(gs[0])
        ax_resid = fig.add_subplot(gs[1], sharex=ax_data)

        if kind == 'transmission':
            self._plot_transmission_with_residuals(ax_data, ax_resid, show_errors,
                                                   data_kwargs, residual_kwargs)
        elif kind == 'signal':
            self._plot_signal_with_residuals(ax_data, ax_resid, show_errors,
                                            data_kwargs, residual_kwargs)
        elif kind == 'openbeam':
            self._plot_openbeam_with_residuals(ax_data, ax_resid, show_errors,
                                              data_kwargs, residual_kwargs)
        else:
            raise ValueError(f"Unknown kind '{kind}'. Choose from: 'transmission', 'signal', "
                           f"'openbeam', 'statistics'")

        # Hide x-axis labels on upper plot
        ax_data.tick_params(labelbottom=False)

        # Add grid
        ax_data.grid(True, alpha=0.3)
        ax_resid.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_transmission_with_residuals(self, ax_data, ax_resid, show_errors, data_kwargs, residual_kwargs):
        """Plot transmission comparison with residuals below."""
        if self.reference_data is None:
            raise ValueError("No reference data available for transmission comparison.")

        # Get openbeam data
        if self.data.op_poissoned_data is not None:
            ref_openbeam_data = self.data.op_poissoned_data
        elif self.data.op_overlapped_data is not None:
            ref_openbeam_data = self.data.op_overlapped_data
        else:
            raise ValueError("No openbeam data available for transmission calculation.")

        # Match lengths
        min_len = min(len(self.reference_data), len(self.reconstructed_data), len(ref_openbeam_data))

        # Calculate transmissions
        ref_signal = self.reference_data['counts'].values[:min_len]
        recon_signal = self.reconstructed_data['counts'].values[:min_len]
        ref_openbeam = ref_openbeam_data['counts'].values[:min_len]

        ref_transmission = ref_signal / np.maximum(ref_openbeam, 1)
        recon_transmission = recon_signal / np.maximum(ref_openbeam, 1)

        time_ms = self.reference_data['time'].values[:min_len] / 1000

        # Plot data
        ref_df = pd.DataFrame({'time': time_ms, 'transmission': ref_transmission})
        recon_df = pd.DataFrame({'time': time_ms, 'transmission': recon_transmission})

        # Extract color for both plots if provided
        data_color = data_kwargs.pop('color', None)
        ref_df.set_index('time')['transmission'].plot(
            ax=ax_data, drawstyle='steps-mid', label='Poissoned', alpha=0.7,
            color=data_color, **data_kwargs)
        recon_df.set_index('time')['transmission'].plot(
            ax=ax_data, drawstyle='steps-mid', label='Reconstructed', alpha=0.7, **data_kwargs)

        # Error bars
        if show_errors:
            ref_signal_err = self.reference_data['err'].values[:min_len]
            ref_openbeam_err = ref_openbeam_data['err'].values[:min_len]
            recon_signal_err = self.reconstructed_data['err'].values[:min_len]

            ref_signal_safe = np.maximum(np.abs(ref_signal), 1)
            ref_openbeam_safe = np.maximum(np.abs(ref_openbeam), 1)
            recon_signal_safe = np.maximum(np.abs(recon_signal), 1)

            ref_trans_err = np.abs(ref_transmission) * np.sqrt(
                (ref_signal_err / ref_signal_safe)**2 + (ref_openbeam_err / ref_openbeam_safe)**2)
            recon_trans_err = np.abs(recon_transmission) * np.sqrt(
                (recon_signal_err / recon_signal_safe)**2 + (ref_openbeam_err / ref_openbeam_safe)**2)

            ax_data.errorbar(time_ms, ref_transmission, yerr=np.abs(ref_trans_err),
                           fmt='none', ecolor='0.5', capsize=2, alpha=0.5)
            ax_data.errorbar(time_ms, recon_transmission, yerr=np.abs(recon_trans_err),
                           fmt='none', ecolor='0.5', capsize=2, alpha=0.5)

        # Calculate residuals in sigma units: (reconstructed - poisson) / poisson_err
        ref_trans_err_safe = np.maximum(np.abs(ref_trans_err) if show_errors else 1, 1e-10)
        residuals_sigma = (recon_transmission - ref_transmission) / ref_trans_err_safe

        # Plot residuals
        residual_color = residual_kwargs.pop('color', 'black')
        resid_df = pd.DataFrame({'time': time_ms, 'residuals': residuals_sigma})
        resid_df.set_index('time')['residuals'].plot(
            ax=ax_resid, drawstyle='steps-mid', color=residual_color, **residual_kwargs)
        ax_resid.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)

        # Labels and title
        ax_data.set_ylabel('Transmission')
        ax_resid.set_xlabel('Time (ms)')
        ax_resid.set_ylabel('Residuals (σ)')

        # Add chi2 to legend title
        chi2_val = self.statistics.get('chi2_per_dof', None)
        chi2_str = self._format_chi2(chi2_val)
        ax_data.legend(title=f'χ²/dof = {chi2_str}')

    def _plot_signal_with_residuals(self, ax_data, ax_resid, show_errors, data_kwargs, residual_kwargs):
        """Plot signal comparison with residuals below."""
        if self.reference_data is None:
            raise ValueError("No reference data available.")

        # Match lengths
        min_len = min(len(self.reference_data), len(self.reconstructed_data))

        ref_counts = self.reference_data['counts'].values[:min_len]
        recon_counts = self.reconstructed_data['counts'].values[:min_len]
        ref_err = self.reference_data['err'].values[:min_len]
        time_ms = self.reference_data['time'].values[:min_len] / 1000

        # Plot data
        ref_df = pd.DataFrame({'time': time_ms, 'counts': ref_counts})
        recon_df = pd.DataFrame({'time': time_ms, 'counts': recon_counts})

        data_color = data_kwargs.pop('color', None)
        ref_df.set_index('time')['counts'].plot(
            ax=ax_data, drawstyle='steps-mid', label='Poissoned', alpha=0.7,
            color=data_color, **data_kwargs)
        recon_df.set_index('time')['counts'].plot(
            ax=ax_data, drawstyle='steps-mid', label='Reconstructed', alpha=0.7, **data_kwargs)

        # Error bars
        if show_errors:
            recon_err = self.reconstructed_data['err'].values[:min_len]
            ax_data.errorbar(time_ms, ref_counts, yerr=ref_err,
                           fmt='none', ecolor='0.5', capsize=2, alpha=0.5)
            ax_data.errorbar(time_ms, recon_counts, yerr=recon_err,
                           fmt='none', ecolor='0.5', capsize=2, alpha=0.5)

        # Calculate residuals in sigma units
        ref_err_safe = np.maximum(ref_err, 1e-10)
        residuals_sigma = (recon_counts - ref_counts) / ref_err_safe

        # Plot residuals
        residual_color = residual_kwargs.pop('color', 'black')
        resid_df = pd.DataFrame({'time': time_ms, 'residuals': residuals_sigma})
        resid_df.set_index('time')['residuals'].plot(
            ax=ax_resid, drawstyle='steps-mid', color=residual_color, **residual_kwargs)
        ax_resid.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)

        # Labels and title
        ax_data.set_ylabel('Signal Counts')
        ax_resid.set_xlabel('Time (ms)')
        ax_resid.set_ylabel('Residuals (σ)')

        # Add chi2 to legend title
        chi2_val = self.statistics.get('chi2_per_dof', None)
        chi2_str = self._format_chi2(chi2_val)
        ax_data.legend(title=f'χ²/dof = {chi2_str}')

    def _plot_openbeam_with_residuals(self, ax_data, ax_resid, show_errors, data_kwargs, residual_kwargs):
        """Plot openbeam comparison with residuals below."""
        # This plots the reconstructed openbeam if available
        # For now, we'll raise an error as openbeam reconstruction isn't standard
        raise NotImplementedError("Openbeam reconstruction plotting is not yet implemented. "
                                "Use kind='signal' or kind='transmission' instead.")

    def _plot_transmission(self, ax, show_errors, **kwargs):
        """Plot transmission comparison (reconstructed vs poissoned reference) using pandas plotting."""
        if self.reference_data is None:
            raise ValueError("No reference data available for transmission comparison. "
                           "Reference data is stored from poissoned_data or overlapped_data.")

        # Get openbeam data from the Data object (use same stage as signal reference)
        # If we have poissoned signal, use poissoned openbeam; otherwise use overlapped openbeam
        if self.data.op_poissoned_data is not None:
            ref_openbeam_data = self.data.op_poissoned_data
        elif self.data.op_overlapped_data is not None:
            ref_openbeam_data = self.data.op_overlapped_data
        else:
            raise ValueError("No openbeam data available for transmission calculation.")

        # Calculate transmissions
        # Match lengths for comparison
        min_len = min(len(self.reference_data), len(self.reconstructed_data),
                     len(ref_openbeam_data))

        # Reference transmission (poissoned signal / poissoned openbeam)
        ref_signal = self.reference_data['counts'].values[:min_len]
        ref_openbeam = ref_openbeam_data['counts'].values[:min_len]
        ref_transmission = ref_signal / np.maximum(ref_openbeam, 1)

        # Reconstructed transmission
        recon_signal = self.reconstructed_data['counts'].values[:min_len]
        recon_transmission = recon_signal / np.maximum(ref_openbeam, 1)

        # Time array (convert to ms)
        time_ms = self.reference_data['time'].values[:min_len] / 1000

        # Create DataFrames for plotting
        ref_df = pd.DataFrame({'time': time_ms, 'transmission': ref_transmission})
        recon_df = pd.DataFrame({'time': time_ms, 'transmission': recon_transmission})

        # Use pandas plot with step drawstyle
        ref_df.set_index('time')['transmission'].plot(
            ax=ax, drawstyle='steps-mid', label='Poissoned (Reference)', alpha=0.7, **kwargs)
        recon_df.set_index('time')['transmission'].plot(
            ax=ax, drawstyle='steps-mid', label='Reconstructed', alpha=0.7, **kwargs)

        # Add error bars if requested
        if show_errors:
            # Propagate errors for transmission
            ref_signal_err = self.reference_data['err'].values[:min_len]
            ref_openbeam_err = ref_openbeam_data['err'].values[:min_len]

            # Avoid division by zero and ensure positive errors
            ref_signal_safe = np.maximum(np.abs(ref_signal), 1)
            ref_openbeam_safe = np.maximum(np.abs(ref_openbeam), 1)

            ref_trans_err = np.abs(ref_transmission) * np.sqrt(
                (ref_signal_err / ref_signal_safe)**2 +
                (ref_openbeam_err / ref_openbeam_safe)**2
            )

            recon_signal_err = self.reconstructed_data['err'].values[:min_len]
            recon_signal_safe = np.maximum(np.abs(recon_signal), 1)

            recon_trans_err = np.abs(recon_transmission) * np.sqrt(
                (recon_signal_err / recon_signal_safe)**2 +
                (ref_openbeam_err / ref_openbeam_safe)**2
            )

            # Ensure errors are positive
            ref_trans_err = np.abs(ref_trans_err)
            recon_trans_err = np.abs(recon_trans_err)

            ax.errorbar(time_ms, ref_transmission, yerr=ref_trans_err,
                       fmt='none', ecolor='0.5', capsize=2, alpha=0.5)
            ax.errorbar(time_ms, recon_transmission, yerr=recon_trans_err,
                       fmt='none', ecolor='0.5', capsize=2, alpha=0.5)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Transmission')

        # Add statistics to title if available
        if 'chi2_per_dof' in self.statistics:
            ax.set_title(f'Transmission Comparison (χ²/dof = {self.statistics["chi2_per_dof"]:.3f}, '
                       f'R² = {self.statistics["r_squared"]:.3f})')
        else:
            ax.set_title('Transmission Comparison')

    def _plot_comparison(self, ax, show_errors, **kwargs):
        """Plot counts comparison (reconstructed vs reference) using pandas plotting."""
        # Convert time to ms for plotting
        recon_plot = self.reconstructed_data.copy()
        recon_plot['time'] = recon_plot['time'] / 1000

        # Use pandas plot with step drawstyle
        recon_plot.set_index('time')['counts'].plot(
            ax=ax, drawstyle='steps-mid', label='Reconstructed', alpha=0.7, **kwargs)

        # Add error bars
        if show_errors:
            ax.errorbar(recon_plot['time'].values, recon_plot['counts'].values,
                       yerr=recon_plot['err'].values, fmt='none', ecolor='0.5',
                       capsize=2, alpha=0.5)

        # Plot reference if available
        if self.reference_data is not None:
            ref_plot = self.reference_data.copy()
            ref_plot['time'] = ref_plot['time'] / 1000

            ref_plot.set_index('time')['counts'].plot(
                ax=ax, drawstyle='steps-mid', label='Reference (Poissoned)', alpha=0.7, **kwargs)

            if show_errors:
                ax.errorbar(ref_plot['time'].values, ref_plot['counts'].values,
                           yerr=ref_plot['err'].values, fmt='none', ecolor='0.5',
                           capsize=2, alpha=0.5)

            # Add statistics to title
            if 'chi2_per_dof' in self.statistics:
                ax.set_title(f'Counts Comparison (χ²/dof = {self.statistics["chi2_per_dof"]:.3f}, '
                           f'R² = {self.statistics["r_squared"]:.3f})')
        else:
            ax.set_title('Reconstructed Signal')

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Counts')

    def _plot_reconstructed(self, ax, show_errors, **kwargs):
        """Plot reconstructed signal only using pandas plotting."""
        # Convert time to ms for plotting
        recon_plot = self.reconstructed_data.copy()
        recon_plot['time'] = recon_plot['time'] / 1000

        # Use pandas plot with step drawstyle
        recon_plot.set_index('time')['counts'].plot(
            ax=ax, drawstyle='steps-mid', label='Reconstructed', **kwargs)

        # Add error bars
        if show_errors:
            ax.errorbar(recon_plot['time'].values, recon_plot['counts'].values,
                       yerr=recon_plot['err'].values, fmt='none', ecolor='0.5',
                       capsize=2, alpha=0.5)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Counts')
        ax.set_title('Reconstructed Signal')

    def _plot_residuals(self, ax, show_errors, **kwargs):
        """Plot residuals (reference - reconstructed) using pandas plotting."""
        if self.reference_data is None:
            raise ValueError("No reference data available for residual calculation.")

        # Match lengths
        min_len = min(len(self.reference_data), len(self.reconstructed_data))
        reference = self.reference_data['counts'].values[:min_len]
        reconstructed = self.reconstructed_data['counts'].values[:min_len]
        time_ms = self.reference_data['time'].values[:min_len] / 1000
        residuals = reference - reconstructed

        # Create DataFrame for plotting
        resid_df = pd.DataFrame({'time': time_ms, 'residuals': residuals})

        # Use pandas plot with step drawstyle
        resid_df.set_index('time')['residuals'].plot(
            ax=ax, drawstyle='steps-mid', label='Residuals', **kwargs)

        # Add zero line
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Residuals (Reference - Reconstructed)')
        ax.set_title(f'Residuals (RMSE = {self.statistics.get("rmse", 0):.2f})')

    def _plot_statistics(self, ax):
        """Plot statistical summary as a bar chart."""
        if not self.statistics:
            raise ValueError("No statistics available. Call filter() first.")

        # Filter out statistics that are suitable for bar plot
        plot_stats = {k: v for k, v in self.statistics.items()
                     if k not in ['n_points'] and isinstance(v, (int, float))}

        if not plot_stats:
            raise ValueError("No plottable statistics available.")

        bars = ax.bar(range(len(plot_stats)), list(plot_stats.values()))
        ax.set_xticks(range(len(plot_stats)))
        ax.set_xticklabels(list(plot_stats.keys()), rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title('Reconstruction Quality Statistics')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, plot_stats.values())):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    def __repr__(self):
        """String representation of the Reconstruct object."""
        has_recon = self.reconstructed_data is not None
        chi2_dof = self.statistics.get('chi2_per_dof', None)
        return (f"Reconstruct(reconstructed={has_recon}, "
                f"chi2_per_dof={chi2_dof:.3f if chi2_dof is not None else 'N/A'})")
