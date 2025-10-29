"""
Data class for neutron Time-of-Flight measurements.

This module provides the Data class for loading, processing, and visualizing
neutron ToF data with support for convolution, frame overlap, and Poisson sampling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.stats import poisson


class Data:
    """
    Data object for neutron Time-of-Flight measurements.

    This class handles reading signal and openbeam data files, applying convolution
    with square response functions, creating frame overlap sequences, and performing
    Poisson sampling to simulate realistic neutron counting statistics.

    Parameters
    ----------
    signal_file : str, optional
        Path to signal data CSV file with columns: stack, counts, err
    openbeam_file : str, optional
        Path to openbeam data CSV file with columns: stack, counts, err
    flux : float, optional
        Expected neutron flux (neutrons/s). Default is None.
    duration : float, optional
        Total measurement duration (seconds). Default is None.
    threshold : float, optional
        Minimum stack value for filtering data. Default is None.
    max_stack : int, optional
        Maximum stack number for full frame. Default is 2400.

    Attributes
    ----------
    table : pandas.DataFrame
        Main data table with columns: time, counts, err
    openbeam_table : pandas.DataFrame
        Openbeam data table with columns: time, counts, err
    kernel : list
        Kernel sequence used for frame overlap (e.g., [0, 12, 10, 25])

    Examples
    --------
    >>> data = Data(signal_file='iron_powder.csv', openbeam_file='openbeam.csv')
    >>> data.convolute_response(pulse_duration=200)
    >>> data.overlap(seq=[0, 12, 10, 25])
    >>> data.poisson_sample(duty_cycle=0.8)
    >>> data.plot()
    """

    def __init__(self, signal_file=None, openbeam_file=None, flux=None,
                 duration=None, threshold=None, max_stack=2400):
        """Initialize Data object and load data files if provided."""
        self.signal_file = signal_file
        self.openbeam_file = openbeam_file
        self.flux = flux
        self.duration = duration
        self.threshold = threshold
        self.max_stack = max_stack

        # Initialize data storage
        self.table = None
        self.openbeam_table = None
        self.kernel = None
        self._original_data = None  # Store original data before processing
        self._convolved_data = None  # Store data after convolution but before overlap

        # Load data if files provided
        if signal_file:
            self.load_signal_data(signal_file, threshold)
        if openbeam_file:
            self.load_openbeam_data(openbeam_file, threshold)

    def load_signal_data(self, file_path, threshold=None):
        """
        Load signal data from CSV file.

        Parameters
        ----------
        file_path : str
            Path to the CSV file containing ToF data
        threshold : float, optional
            Minimum value for 'stack' column to filter data

        Raises
        ------
        FileNotFoundError
            If the specified file_path does not exist
        KeyError
            If required columns are missing in the CSV
        ValueError
            If data contains invalid values
        """
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path}' not found")

        required_columns = ['stack', 'counts', 'err']
        if not all(col in df.columns for col in required_columns):
            raise KeyError(f"CSV must contain columns: {required_columns}")

        if threshold is not None:
            if threshold < 0:
                raise ValueError("Threshold must be non-negative")
            df = df.loc[df['stack'] >= threshold]

        # Convert to time and validate
        df['time'] = (df['stack'] - 1) * 10  # Convert stack to time in µs

        if np.any(df['err'] <= 0):
            raise ValueError("Errors must be positive")
        if df[['time', 'counts', 'err']].isnull().any().any():
            raise ValueError("Data must not contain NaN values")

        # Store as DataFrame
        self.table = df[['time', 'counts', 'err']].copy()
        self._original_data = self.table.copy()
        self.signal_file = file_path

        return self

    def load_openbeam_data(self, file_path, threshold=None):
        """
        Load openbeam data from CSV file.

        Parameters
        ----------
        file_path : str
            Path to the CSV file containing openbeam data
        threshold : float, optional
            Minimum value for 'stack' column to filter data
        """
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path}' not found")

        required_columns = ['stack', 'counts', 'err']
        if not all(col in df.columns for col in required_columns):
            raise KeyError(f"CSV must contain columns: {required_columns}")

        if threshold is not None:
            if threshold < 0:
                raise ValueError("Threshold must be non-negative")
            df = df.loc[df['stack'] >= threshold]

        # Convert to time and validate
        df['time'] = (df['stack'] - 1) * 10  # Convert stack to time in µs

        if np.any(df['err'] <= 0):
            raise ValueError("Errors must be positive")
        if df[['time', 'counts', 'err']].isnull().any().any():
            raise ValueError("Data must not contain NaN values")

        # Store as DataFrame
        self.openbeam_table = df[['time', 'counts', 'err']].copy()
        self.openbeam_file = file_path

        return self

    def convolute_response(self, pulse_duration, bin_width=10):
        """
        Convolute data with a square response function.

        This mimics how the data would look when measured with an instrument
        having a longer pulse duration.

        Parameters
        ----------
        pulse_duration : float
            Duration of the square pulse in microseconds
        bin_width : float, optional
            Time bin width in microseconds. Default is 10.

        Returns
        -------
        self
            Returns self for method chaining

        Raises
        ------
        ValueError
            If table is not loaded or pulse_duration is invalid

        Examples
        --------
        >>> data = Data('signal.csv')
        >>> data.convolute_response(pulse_duration=200)
        """
        if self.table is None:
            raise ValueError("No data loaded. Call load_signal_data first.")

        if pulse_duration <= 0:
            raise ValueError("pulse_duration must be positive")

        # Create square pulse kernel
        pulse_length = int(pulse_duration / bin_width)
        kernel = np.ones(pulse_length) / pulse_length  # Normalized

        # Convolve signal and propagate errors
        counts_convolved = np.convolve(self.table['counts'].values, kernel, mode='same')

        # Error propagation for convolution (sum in quadrature, weighted by kernel)
        err_squared = self.table['err'].values ** 2
        err_convolved = np.sqrt(np.convolve(err_squared, kernel**2, mode='same'))

        # Update table
        self.table['counts'] = counts_convolved
        self.table['err'] = err_convolved
        self._convolved_data = self.table.copy()

        return self

    def overlap(self, seq, total_time=None, bin_width=10):
        """
        Create overlapping frame structure.

        This method duplicates the data into multiple frames with specified time
        delays, simulating frame overlap in Time-of-Flight measurements.

        Parameters
        ----------
        seq : list of float
            Time sequence for frames in milliseconds.
            For example, seq=[0, 12, 10, 25] means:
            - Frame 1 starts at t=0
            - Frame 2 starts at t=12 ms
            - Frame 3 starts at t=12+10=22 ms
            - Frame 4 starts at t=22+25=47 ms
        total_time : float, optional
            Total time frame in microseconds. If None, deduced automatically
            from the data length and sequence.
        bin_width : float, optional
            Time bin width in microseconds. Default is 10.

        Returns
        -------
        self
            Returns self for method chaining

        Raises
        ------
        ValueError
            If table is not loaded or seq is invalid

        Examples
        --------
        >>> data = Data('signal.csv')
        >>> data.overlap(seq=[0, 12, 10, 25])
        """
        if self.table is None:
            raise ValueError("No data loaded. Call load_signal_data first.")

        if not isinstance(seq, (list, tuple, np.ndarray)):
            raise ValueError("seq must be a list, tuple, or array")

        seq = np.array(seq)
        if len(seq) < 1:
            raise ValueError("seq must have at least one element")
        if np.any(seq < 0):
            raise ValueError("All elements in seq must be non-negative")

        # Save kernel for reconstruction
        self.kernel = seq.tolist()

        # Convert seq from milliseconds to microseconds
        seq_us = seq * 1000

        # Calculate cumulative start times
        frame_starts = np.cumsum(np.concatenate([[0], seq_us[:-1]]))

        # Determine total time frame
        if total_time is None:
            max_time_in_data = self.table['time'].max()
            total_time = frame_starts[-1] + max_time_in_data

        # Create new time axis
        n_bins = int(total_time / bin_width) + 1
        new_time = np.arange(0, n_bins * bin_width, bin_width)
        new_counts = np.zeros(n_bins)
        new_err_squared = np.zeros(n_bins)

        # Add each frame to the overlapped signal
        for frame_start in frame_starts:
            start_idx = int(frame_start / bin_width)
            for i, row in self.table.iterrows():
                time_idx = start_idx + int(row['time'] / bin_width)
                if time_idx < n_bins:
                    new_counts[time_idx] += row['counts']
                    new_err_squared[time_idx] += row['err']**2

        # Create new table
        new_err = np.sqrt(new_err_squared)
        self.table = pd.DataFrame({
            'time': new_time[:len(new_counts)],
            'counts': new_counts,
            'err': new_err
        })

        return self

    def poisson_sample(self, duty_cycle=1.0):
        """
        Apply Poisson sampling to simulate realistic neutron counting statistics.

        Parameters
        ----------
        duty_cycle : float, optional
            Duty cycle of the measurement (0 to 1). Default is 1.0.

        Returns
        -------
        self
            Returns self for method chaining

        Raises
        ------
        ValueError
            If table is not loaded or parameters are invalid

        Examples
        --------
        >>> data = Data('signal.csv')
        >>> data.poisson_sample(duty_cycle=0.8)
        """
        if self.table is None:
            raise ValueError("No data loaded. Call load_signal_data first.")

        if not 0 < duty_cycle <= 1:
            raise ValueError("duty_cycle must be between 0 and 1")

        # Calculate scaling factor based on flux, duration, and duty cycle
        if self.flux is not None and self.duration is not None:
            # Scale by flux * duration * duty_cycle
            scale_factor = self.flux * self.duration * duty_cycle / self.table['counts'].sum()
        else:
            scale_factor = duty_cycle

        # Apply Poisson sampling
        scaled_counts = self.table['counts'].values * scale_factor
        poisson_counts = poisson.rvs(np.clip(scaled_counts, 0, None))

        # Update table with Poisson-sampled data
        self.table['counts'] = poisson_counts
        self.table['err'] = np.sqrt(np.maximum(poisson_counts, 1))  # Poisson error

        return self

    def plot(self, show_errors=True, **kwargs):
        """
        Plot the current data.

        Parameters
        ----------
        show_errors : bool, optional
            Whether to show error bars. Default is True.
        **kwargs
            Additional keyword arguments passed to matplotlib.pyplot.plot

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.table is None:
            raise ValueError("No data loaded. Call load_signal_data first.")

        fig, ax = plt.subplots(figsize=(10, 6))

        if show_errors:
            ax.errorbar(self.table['time'], self.table['counts'],
                       yerr=self.table['err'], fmt='o-', capsize=3,
                       label='Signal', **kwargs)
        else:
            ax.plot(self.table['time'], self.table['counts'],
                   'o-', label='Signal', **kwargs)

        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Counts')
        ax.set_title('Neutron Time-of-Flight Data')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_comparison(self, other_data=None, labels=None):
        """
        Plot comparison between signal and openbeam or another Data object.

        Parameters
        ----------
        other_data : Data, optional
            Another Data object to compare with. If None, uses openbeam_table.
        labels : list of str, optional
            Labels for the plots. Default is ['Signal', 'Openbeam'].

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.table is None:
            raise ValueError("No data loaded.")

        if labels is None:
            labels = ['Signal', 'Openbeam']

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot signal
        ax.plot(self.table['time'], self.table['counts'],
               'o-', label=labels[0], alpha=0.7)

        # Plot comparison data
        if other_data is not None:
            if isinstance(other_data, Data):
                ax.plot(other_data.table['time'], other_data.table['counts'],
                       's-', label=labels[1], alpha=0.7)
            else:
                raise ValueError("other_data must be a Data object")
        elif self.openbeam_table is not None:
            ax.plot(self.openbeam_table['time'], self.openbeam_table['counts'],
                   's-', label=labels[1], alpha=0.7)
        else:
            raise ValueError("No comparison data available")

        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Counts')
        ax.set_title('Data Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def copy(self):
        """
        Create a deep copy of the Data object.

        Returns
        -------
        Data
            A new Data object with copied data
        """
        new_data = Data(flux=self.flux, duration=self.duration,
                       threshold=self.threshold, max_stack=self.max_stack)
        new_data.table = self.table.copy() if self.table is not None else None
        new_data.openbeam_table = self.openbeam_table.copy() if self.openbeam_table is not None else None
        new_data.kernel = self.kernel.copy() if self.kernel is not None else None
        new_data._original_data = self._original_data.copy() if self._original_data is not None else None
        new_data._convolved_data = self._convolved_data.copy() if self._convolved_data is not None else None
        new_data.signal_file = self.signal_file
        new_data.openbeam_file = self.openbeam_file
        return new_data

    def __repr__(self):
        """String representation of the Data object."""
        n_points = len(self.table) if self.table is not None else 0
        has_openbeam = self.openbeam_table is not None
        return (f"Data(n_points={n_points}, flux={self.flux}, duration={self.duration}, "
                f"has_openbeam={has_openbeam}, kernel={self.kernel})")
