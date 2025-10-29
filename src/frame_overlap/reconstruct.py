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
    reconstructed_table : pandas.DataFrame
        Table with reconstructed signal (time, counts, err)
    reference_table : pandas.DataFrame
        Reference data for comparison (data after convolution but before overlap)
    statistics : dict
        Dictionary containing reconstruction quality statistics

    Examples
    --------
    >>> from frame_overlap import Data, Reconstruct
    >>> data = Data('signal.csv')
    >>> data.convolute_response(200).overlap([0, 12, 10, 25])
    >>> recon = Reconstruct(data)
    >>> recon.filter(kind='wiener', noise_power=0.01)
    >>> recon.plot_comparison()
    >>> print(recon.statistics)
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
        self.reconstructed_table = None
        self.reference_table = None  # Will store pre-overlap data for comparison
        self.statistics = {}

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

        # Store reference data if available (squared_data before overlap)
        if self.data.squared_data is not None:
            self.reference_table = self.data.squared_data.copy()

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

        # Create reconstructed table
        self.reconstructed_table = pd.DataFrame({
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
        if self.reconstructed_table is None:
            return

        # If we have reference data (convolved but not overlapped)
        if self.reference_table is not None:
            # Match lengths
            min_len = min(len(self.reference_table), len(self.reconstructed_table))
            reference = self.reference_table['counts'].values[:min_len]
            reconstructed = self.reconstructed_table['counts'].values[:min_len]
            errors = self.reference_table['err'].values[:min_len]

            # Calculate chi-squared
            residuals = reference - reconstructed
            chi2 = np.sum((residuals / errors)**2)
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
            reconstructed = self.reconstructed_table['counts'].values
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

    def plot_reconstruction(self):
        """
        Plot the reconstructed signal.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.reconstructed_table is None:
            raise ValueError("No reconstruction available. Call filter() first.")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.reconstructed_table['time'],
               self.reconstructed_table['counts'],
               'o-', label='Reconstructed', alpha=0.7)

        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Counts')
        ax.set_title('Reconstructed Signal')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_comparison(self):
        """
        Plot comparison between reference (if available) and reconstructed signal.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.reconstructed_table is None:
            raise ValueError("No reconstruction available. Call filter() first.")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot reconstructed
        ax.plot(self.reconstructed_table['time'],
               self.reconstructed_table['counts'],
               'r-', label='Reconstructed', alpha=0.7, linewidth=2)

        # Plot reference if available
        if self.reference_table is not None:
            ax.plot(self.reference_table['time'],
                   self.reference_table['counts'],
                   'b--', label='Reference (Pre-overlap)', alpha=0.7, linewidth=2)

            # Add statistics to title
            if 'chi2_per_dof' in self.statistics:
                ax.set_title(f'Reconstruction Comparison (χ²/dof = {self.statistics["chi2_per_dof"]:.3f}, '
                           f'R² = {self.statistics["r_squared"]:.3f})')
        else:
            ax.set_title('Reconstructed Signal')

        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Counts')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_residuals(self):
        """
        Plot residuals between reference and reconstructed signal.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure

        Raises
        ------
        ValueError
            If reference data is not available
        """
        if self.reconstructed_table is None:
            raise ValueError("No reconstruction available. Call filter() first.")

        if self.reference_table is None:
            raise ValueError("No reference data available for residual calculation.")

        # Match lengths
        min_len = min(len(self.reference_table), len(self.reconstructed_table))
        reference = self.reference_table['counts'].values[:min_len]
        reconstructed = self.reconstructed_table['counts'].values[:min_len]
        time = self.reference_table['time'].values[:min_len]
        residuals = reference - reconstructed

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(time, residuals, 'k-', alpha=0.7)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)

        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Residuals (Reference - Reconstructed)')
        ax.set_title(f'Residuals (RMSE = {self.statistics.get("rmse", 0):.2f})')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_statistics(self):
        """
        Plot statistical summary of the reconstruction quality.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if not self.statistics:
            raise ValueError("No statistics available. Call filter() first.")

        # Filter out statistics that are suitable for bar plot
        plot_stats = {k: v for k, v in self.statistics.items()
                     if k not in ['n_points'] and isinstance(v, (int, float))}

        if not plot_stats:
            raise ValueError("No plottable statistics available.")

        fig, ax = plt.subplots(figsize=(10, 6))

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

        plt.tight_layout()
        return fig

    def __repr__(self):
        """String representation of the Reconstruct object."""
        has_recon = self.reconstructed_table is not None
        chi2_dof = self.statistics.get('chi2_per_dof', None)
        return (f"Reconstruct(reconstructed={has_recon}, "
                f"chi2_per_dof={chi2_dof:.3f if chi2_dof is not None else 'N/A'})")
