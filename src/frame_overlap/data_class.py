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

    Both signal and openbeam data go through the same processing pipeline.

    Parameters
    ----------
    signal_file : str, optional
        Path to signal data CSV file with columns: stack, counts, err
    openbeam_file : str, optional
        Path to openbeam data CSV file with columns: stack, counts, err
    flux : float, optional
        Expected neutron flux in n/cm²/s. Default is None.
    duration : float, optional
        Total measurement duration in hours. Default is None.
    freq : float, optional
        Measurement frequency in Hz. Default is None.
    threshold : float, optional
        Minimum stack value for filtering data. Default is None.
    max_stack : int, optional
        Maximum stack number for full frame. Default is 2400.

    Attributes
    ----------
    data : pandas.DataFrame
        Original signal data (time, counts, err)
    convolved_data : pandas.DataFrame
        Signal after convolution with square response
    overlapped_data : pandas.DataFrame
        Signal after frame overlap
    poissoned_data : pandas.DataFrame
        Signal after Poisson sampling

    op_data : pandas.DataFrame
        Original openbeam data
    op_convolved_data : pandas.DataFrame
        Openbeam after convolution
    op_overlapped_data : pandas.DataFrame
        Openbeam after frame overlap
    op_poissoned_data : pandas.DataFrame
        Openbeam after Poisson sampling

    kernel : list
        Kernel sequence used for frame overlap (e.g., [0, 12, 10, 25])

    Examples
    --------
    >>> # Basic usage
    >>> data = Data(signal_file='iron_powder.csv', openbeam_file='openbeam.csv',
    ...             flux=5e6, duration=0.5, freq=20)
    >>> data.convolute_response(pulse_duration=200)
    >>> data.overlap(kernel=[0, 12, 10, 25])
    >>> data.poisson_sample(duty_cycle=0.8)
    >>> data.plot(kind='transmission')

    >>> # Wraparound example
    >>> data.overlap(kernel=[0, 5, 10, 10, 20], total_time=50)  # Excess wraps to start

    >>> # Poisson sampling with instrument parameters
    >>> data.poisson_sample(flux=1e4, measurement_time=5, freq=800)
    """

    def __init__(self, signal_file=None, openbeam_file=None, flux=None,
                 duration=None, threshold=None, max_stack=2400, freq=None):
        """Initialize Data object and load data files if provided."""
        self.signal_file = signal_file
        self.openbeam_file = openbeam_file
        self.flux = flux  # n/cm²/s
        self.duration = duration  # hours
        self.freq = freq  # Hz
        self.threshold = threshold
        self.max_stack = max_stack

        # Initialize data storage for signal at each stage
        self.data = None  # Original
        self.convolved_data = None  # After convolution
        self.overlapped_data = None  # After frame overlap
        self.poissoned_data = None  # After Poisson sampling

        # Initialize data storage for openbeam at each stage
        self.op_data = None
        self.op_convolved_data = None
        self.op_overlapped_data = None
        self.op_poissoned_data = None

        self.kernel = None
        self.n_overlapping_frames = None  # Number of overlapping frames

        # Legacy attributes for backward compatibility
        self.table = None
        self.openbeam_table = None

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
        # Convert stack to time in ms (each stack is 10 µs = 0.01 ms)
        df['time'] = (df['stack'] - 1) * 0.01

        if np.any(df['err'] <= 0):
            raise ValueError("Errors must be positive")
        if df[['time', 'counts', 'err']].isnull().any().any():
            raise ValueError("Data must not contain NaN values")

        # Store as DataFrame
        self.data = df[['time', 'counts', 'err']].copy()
        self.table = self.data  # For backward compatibility
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
        # Convert stack to time in ms (each stack is 10 µs = 0.01 ms)
        df['time'] = (df['stack'] - 1) * 0.01

        if np.any(df['err'] <= 0):
            raise ValueError("Errors must be positive")
        if df[['time', 'counts', 'err']].isnull().any().any():
            raise ValueError("Data must not contain NaN values")

        # Store as DataFrame
        self.op_data = df[['time', 'counts', 'err']].copy()
        self.openbeam_table = self.op_data  # For backward compatibility
        self.openbeam_file = file_path

        return self

    def convolute_response(self, pulse_duration, bin_width=0.01):
        """
        Convolute data with a square response function.

        This mimics how the data would look when measured with an instrument
        having a longer pulse duration. Applies to both signal and openbeam.

        Parameters
        ----------
        pulse_duration : float
            Duration of the square pulse in milliseconds
        bin_width : float, optional
            Time bin width in milliseconds. Default is 0.01 (10 µs).

        Returns
        -------
        self
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No signal data loaded. Call load_signal_data first.")

        if pulse_duration <= 0:
            raise ValueError("pulse_duration must be positive")

        # Create square pulse kernel (normalized)
        pulse_length = int(pulse_duration / bin_width)
        kernel = np.ones(pulse_length) / pulse_length

        # Convolve signal
        self.convolved_data = self._convolve_dataframe(self.data, kernel)
        self.table = self.convolved_data  # Update for legacy compatibility

        # Convolve openbeam if available
        if self.op_data is not None:
            self.op_convolved_data = self._convolve_dataframe(self.op_data, kernel)
            self.openbeam_table = self.op_convolved_data

        return self

    def _convolve_dataframe(self, df, kernel):
        """Helper to convolve a dataframe with a kernel."""
        counts_convolved = np.convolve(df['counts'].values, kernel, mode='same')

        # After convolution (moving average), recalculate error bars as sqrt(counts)
        # This is the proper statistical error for counting statistics
        err_convolved = np.sqrt(np.maximum(counts_convolved, 1))

        # Create new dataframe with updated values
        result = pd.DataFrame({
            'time': df['time'].values,
            'counts': counts_convolved,
            'err': err_convolved
        })
        return result

    def overlap(self, kernel, total_time=None, freq=None, bin_width=0.01):
        """
        Create overlapping frame structure.

        This method duplicates the data into multiple frames with specified time
        delays. Applies to both signal and openbeam.

        Parameters
        ----------
        kernel : list of float
            Time sequence for frames in milliseconds.
            For example, kernel=[0, 12, 10, 25] means:
            - Frame 1 starts at t=0
            - Frame 2 starts at t=12 ms
            - Frame 3 starts at t=12+10=22 ms
            - Frame 4 starts at t=22+25=47 ms
        total_time : float, optional
            Total time frame in milliseconds. If None, deduced automatically.
            If specified and shorter than the overlapped signal, the excess tail
            wraps around to the beginning of the spectrum.
        freq : float, optional
            Frequency in Hz. If provided, total_time = 1000/freq ms.
            For example, freq=20 Hz means total_time=50 ms.
        bin_width : float, optional
            Time bin width in milliseconds. Default is 0.01 (10 µs).

        Returns
        -------
        self
            Returns self for method chaining

        Examples
        --------
        >>> data.overlap(kernel=[0, 12, 10, 25])  # Auto total time
        >>> data.overlap(kernel=[0, 10], total_time=50)  # 50 ms total
        >>> data.overlap(kernel=[0, 10], freq=20)  # 20 Hz = 50 ms total
        >>> data.overlap(kernel=[0, 5, 10, 10, 20], total_time=50)  # Wraparound
        """
        # Use convolved_data if available, otherwise use data
        source_data = self.convolved_data if self.convolved_data is not None else self.data

        if source_data is None:
            raise ValueError("No data loaded. Call load_signal_data first.")

        if not isinstance(kernel, (list, tuple, np.ndarray)):
            raise ValueError("kernel must be a list, tuple, or array")

        kernel = np.array(kernel)
        if len(kernel) < 1:
            raise ValueError("kernel must have at least one element")
        if np.any(kernel < 0):
            raise ValueError("All elements in kernel must be non-negative")

        # Handle freq parameter
        if freq is not None:
            if total_time is not None:
                raise ValueError("Cannot specify both freq and total_time")
            total_time = 1000.0 / freq  # Convert Hz to ms

        # Save kernel and number of overlapping frames for reconstruction
        self.kernel = kernel.tolist()
        self.n_overlapping_frames = len(kernel)

        # Apply overlap to signal
        self.overlapped_data = self._create_overlap(source_data, kernel, total_time, bin_width)
        self.table = self.overlapped_data

        # Apply overlap to openbeam if available
        op_source = self.op_convolved_data if self.op_convolved_data is not None else self.op_data
        if op_source is not None:
            self.op_overlapped_data = self._create_overlap(op_source, kernel, total_time, bin_width)
            self.openbeam_table = self.op_overlapped_data

        return self

    def _create_overlap(self, df, kernel, total_time, bin_width):
        """Helper to create overlap for a dataframe with wraparound support."""
        # kernel is already in milliseconds
        kernel_ms = kernel

        # Calculate cumulative start times
        # kernel=[0, 12, 10, 25] in ms
        # cumsum gives [0, 12, 22, 47] which are the frame start times in ms
        frame_starts = np.cumsum(kernel_ms)

        # Determine total time frame (in milliseconds)
        if total_time is None:
            # Auto-calculate: last frame start + data length
            max_time_in_data = df['time'].max()
            total_time_ms = frame_starts[-1] + max_time_in_data
        else:
            # User provided total_time in milliseconds
            total_time_ms = total_time

        # Create new time axis
        n_bins = int(total_time_ms / bin_width) + 1
        new_time = np.arange(0, n_bins * bin_width, bin_width)
        new_counts = np.zeros(n_bins)
        new_err_squared = np.zeros(n_bins)

        # Add each frame to the overlapped signal with wraparound
        for frame_start in frame_starts:
            start_idx = int(frame_start / bin_width)
            for i, row in df.iterrows():
                time_idx = start_idx + int(row['time'] / bin_width)
                # Implement wraparound: if time_idx exceeds n_bins, wrap to beginning
                if time_idx >= n_bins:
                    time_idx = time_idx % n_bins
                new_counts[time_idx] += row['counts']
                new_err_squared[time_idx] += row['err']**2

        # Create new dataframe
        new_err = np.sqrt(new_err_squared)
        result = pd.DataFrame({
            'time': new_time[:len(new_counts)],
            'counts': new_counts,
            'err': new_err
        })
        return result

    def poisson_sample(self, duty_cycle=None, flux=None, measurement_time=None, freq=None):
        """
        Apply Poisson sampling to simulate realistic neutron counting statistics.

        The duty cycle can be specified directly or calculated from instrument parameters.
        When flux, measurement_time, and freq are provided, duty cycle is calculated as:
        duty_cycle = (flux_new / flux_orig) * (time_new / time_orig) * (freq_new / freq_orig)

        This simulates how the data would look with different instrument parameters
        while accounting for the number of overlapping frames.

        Applies to both signal and openbeam.

        Parameters
        ----------
        duty_cycle : float, optional
            Duty cycle of the measurement (0 to 1). If provided, flux, measurement_time,
            and freq are ignored. Default is None.
        flux : float, optional
            New neutron flux in n/cm²/s. Requires measurement_time and freq.
            Units: neutrons per square centimeter per second.
        measurement_time : float, optional
            New measurement time in hours. Requires flux and freq.
        freq : float, optional
            New measurement frequency in Hz. Requires flux and measurement_time.

        Returns
        -------
        self
            Returns self for method chaining

        Examples
        --------
        >>> # Direct duty cycle
        >>> data.poisson_sample(duty_cycle=0.8)

        >>> # Calculate duty cycle from instrument parameters
        >>> # Original: flux=5e6 n/cm²/s, 0.5 hours, 20 Hz
        >>> # New: flux=1e4 n/cm²/s, 5 hours, 800 Hz
        >>> data.poisson_sample(flux=1e4, measurement_time=5, freq=800)
        """
        # Use overlapped_data if available, otherwise convolved_data, otherwise data
        source_data = (self.overlapped_data if self.overlapped_data is not None
                      else self.convolved_data if self.convolved_data is not None
                      else self.data)

        if source_data is None:
            raise ValueError("No data loaded. Call load_signal_data first.")

        # Calculate duty cycle
        if duty_cycle is None:
            # Calculate from instrument parameters
            if flux is None or measurement_time is None or freq is None:
                raise ValueError(
                    "Must provide either duty_cycle OR all of (flux, measurement_time, freq)"
                )

            # Verify original parameters are set
            if self.flux is None or self.duration is None or self.freq is None:
                raise ValueError(
                    "Original flux, duration, and freq must be set in Data.__init__() "
                    "to calculate duty cycle from instrument parameters"
                )

            # Calculate duty cycle: (flux_new/flux_orig) * (time_new/time_orig) * (freq_new/freq_orig)
            duty_cycle = (flux / self.flux) * (measurement_time / self.duration) * (freq / self.freq)

            print(f"Calculated duty cycle: {duty_cycle:.4f}")
            print(f"  Flux ratio: {flux}/{self.flux} = {flux/self.flux:.4f}")
            print(f"  Time ratio: {measurement_time}/{self.duration} = {measurement_time/self.duration:.4f}")
            print(f"  Freq ratio: {freq}/{self.freq} = {freq/self.freq:.4f}")

        if not 0 < duty_cycle <= 10:  # Allow duty_cycle > 1 for higher flux scenarios
            raise ValueError("duty_cycle must be positive (typically between 0 and 1, but can exceed 1)")

        # Apply Poisson to signal
        self.poissoned_data = self._apply_poisson(source_data, duty_cycle)
        self.table = self.poissoned_data

        # Apply Poisson to openbeam if available
        op_source = (self.op_overlapped_data if self.op_overlapped_data is not None
                    else self.op_convolved_data if self.op_convolved_data is not None
                    else self.op_data)
        if op_source is not None:
            self.op_poissoned_data = self._apply_poisson(op_source, duty_cycle)
            self.openbeam_table = self.op_poissoned_data

        return self

    def _apply_poisson(self, df, duty_cycle):
        """Helper to apply Poisson sampling to a dataframe."""
        # Scale counts by duty cycle
        scaled_counts = df['counts'].values * duty_cycle

        # Apply Poisson sampling
        poisson_counts = poisson.rvs(np.maximum(scaled_counts, 0))

        # Poisson error is sqrt(counts)
        poisson_err = np.sqrt(np.maximum(poisson_counts, 1))

        result = df.copy()
        result['counts'] = poisson_counts.astype(float)
        result['err'] = poisson_err
        return result

    def plot(self, kind='auto', show_stages=False, show_errors=True, fontsize=16, figsize=(10, 6), **kwargs):
        """
        Plot the data using pandas plotting methods.

        Parameters
        ----------
        kind : str, optional
            Type of plot:
            - 'auto': Automatically choose best plot type (default)
            - 'transmission': Plot signal/openbeam ratio
            - 'signal': Plot signal only
            - 'openbeam': Plot openbeam only
            - 'both': Plot signal and openbeam on same axes
        show_stages : bool, optional
            If True, show all processing stages on the same plot. Default is False.
        show_errors : bool, optional
            Whether to show error bars. Default is True.
        fontsize : int, optional
            Font size for labels, ticks, and legend. Default is 16.
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (10, 6).
        **kwargs
            Additional keyword arguments passed to pandas plotting methods

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Set font sizes for all elements
        plt.rcParams.update({
            'font.size': fontsize,
            'axes.labelsize': fontsize,
            'axes.titlesize': fontsize + 2,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            'legend.fontsize': fontsize
        })

        # Auto-detect plot kind
        if kind == 'auto':
            if self.data is not None and self.op_data is not None:
                kind = 'transmission'
            elif self.data is not None:
                kind = 'signal'
            elif self.op_data is not None:
                kind = 'openbeam'
            else:
                raise ValueError("No data loaded")

        if kind == 'transmission':
            self._plot_transmission(ax, show_stages, show_errors, **kwargs)
        elif kind == 'signal':
            self._plot_signal(ax, show_stages, show_errors, **kwargs)
        elif kind == 'openbeam':
            self._plot_openbeam(ax, show_stages, show_errors, **kwargs)
        elif kind == 'both':
            self._plot_both(ax, show_stages, show_errors, **kwargs)
        else:
            raise ValueError(f"Unknown kind '{kind}'. Choose from: 'auto', 'transmission', 'signal', 'openbeam', 'both'")

        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig

    def _plot_transmission(self, ax, show_stages, show_errors, **kwargs):
        """Plot transmission (signal/openbeam ratio) using pandas plotting."""
        if self.data is None or self.op_data is None:
            raise ValueError("Both signal and openbeam data must be loaded for transmission plot")

        if show_stages:
            # Show all stages
            stages = [
                (self.data, self.op_data, 'Original'),
                (self.convolved_data, self.op_convolved_data, 'Convolved'),
                (self.overlapped_data, self.op_overlapped_data, 'Overlapped'),
                (self.poissoned_data, self.op_poissoned_data, 'Poissoned')
            ]

            for sig, op, label in stages:
                if sig is not None and op is not None:
                    trans = self._calculate_transmission(sig, op)
                    # Use pandas plot with step drawstyle
                    trans.set_index('time')['transmission'].plot(
                        ax=ax, drawstyle='steps-mid', label=label, alpha=0.7, **kwargs)
                    if show_errors:
                        ax.errorbar(trans['time'], trans['transmission'],
                                  yerr=trans['err'], fmt='none', ecolor='0.5',
                                  capsize=2, alpha=0.5)
        else:
            # Show current stage only
            sig = (self.poissoned_data if self.poissoned_data is not None
                  else self.overlapped_data if self.overlapped_data is not None
                  else self.convolved_data if self.convolved_data is not None
                  else self.data)
            op = (self.op_poissoned_data if self.op_poissoned_data is not None
                 else self.op_overlapped_data if self.op_overlapped_data is not None
                 else self.op_convolved_data if self.op_convolved_data is not None
                 else self.op_data)

            if sig is not None and op is not None:
                trans = self._calculate_transmission(sig, op)
                # Use pandas plot with step drawstyle
                trans.set_index('time')['transmission'].plot(
                    ax=ax, drawstyle='steps-mid', label='Transmission', **kwargs)
                if show_errors:
                    ax.errorbar(trans['time'], trans['transmission'],
                              yerr=trans['err'], fmt='none', ecolor='0.5',
                              capsize=2, alpha=0.5)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Transmission')
        ax.set_title('Neutron Transmission')

    def _plot_signal(self, ax, show_stages, show_errors, **kwargs):
        """Plot signal data using pandas plotting."""
        if self.data is None:
            raise ValueError("No signal data loaded")

        if show_stages:
            stages = [
                (self.data, 'Original'),
                (self.convolved_data, 'Convolved'),
                (self.overlapped_data, 'Overlapped'),
                (self.poissoned_data, 'Poissoned')
            ]
            for df, label in stages:
                if df is not None:
                    # Use pandas plot with step drawstyle
                    df.set_index('time')['counts'].plot(
                        ax=ax, drawstyle='steps-mid', label=label, alpha=0.7, **kwargs)
                    if show_errors:
                        ax.errorbar(df['time'], df['counts'], yerr=df['err'],
                                  fmt='none', ecolor='0.5', capsize=2, alpha=0.5)
        else:
            df = (self.poissoned_data if self.poissoned_data is not None
                 else self.overlapped_data if self.overlapped_data is not None
                 else self.convolved_data if self.convolved_data is not None
                 else self.data)
            # Use pandas plot with step drawstyle
            df.set_index('time')['counts'].plot(
                ax=ax, drawstyle='steps-mid', label='Signal', **kwargs)
            if show_errors:
                ax.errorbar(df['time'], df['counts'], yerr=df['err'],
                          fmt='none', ecolor='0.5', capsize=2, alpha=0.5)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Counts')
        ax.set_title('Neutron Signal')

    def _plot_openbeam(self, ax, show_stages, show_errors, **kwargs):
        """Plot openbeam data using pandas plotting."""
        if self.op_data is None:
            raise ValueError("No openbeam data loaded")

        if show_stages:
            stages = [
                (self.op_data, 'Original'),
                (self.op_convolved_data, 'Convolved'),
                (self.op_overlapped_data, 'Overlapped'),
                (self.op_poissoned_data, 'Poissoned')
            ]
            for df, label in stages:
                if df is not None:
                    # Use pandas plot with step drawstyle
                    df.set_index('time')['counts'].plot(
                        ax=ax, drawstyle='steps-mid', label=label, alpha=0.7, **kwargs)
                    if show_errors:
                        ax.errorbar(df['time'], df['counts'], yerr=df['err'],
                                  fmt='none', ecolor='0.5', capsize=2, alpha=0.5)
        else:
            df = (self.op_poissoned_data if self.op_poissoned_data is not None
                 else self.op_overlapped_data if self.op_overlapped_data is not None
                 else self.op_convolved_data if self.op_convolved_data is not None
                 else self.op_data)
            # Use pandas plot with step drawstyle
            df.set_index('time')['counts'].plot(
                ax=ax, drawstyle='steps-mid', label='Openbeam', **kwargs)
            if show_errors:
                ax.errorbar(df['time'], df['counts'], yerr=df['err'],
                          fmt='none', ecolor='0.5', capsize=2, alpha=0.5)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Counts')
        ax.set_title('Openbeam Data')

    def _plot_both(self, ax, show_stages, show_errors, **kwargs):
        """Plot both signal and openbeam using pandas plotting."""
        if self.data is None:
            raise ValueError("No signal data loaded")

        # Plot signal
        if show_stages:
            stages = [(self.data, 'Signal Original'), (self.convolved_data, 'Signal Convolved'),
                     (self.overlapped_data, 'Signal Overlapped'), (self.poissoned_data, 'Signal Poissoned')]
            for df, label in stages:
                if df is not None:
                    df.set_index('time')['counts'].plot(
                        ax=ax, drawstyle='steps-mid', label=label, alpha=0.7, **kwargs)
                    if show_errors:
                        ax.errorbar(df['time'], df['counts'], yerr=df['err'],
                                  fmt='none', ecolor='0.5', capsize=2, alpha=0.5)
        else:
            sig = (self.poissoned_data if self.poissoned_data is not None
                  else self.overlapped_data if self.overlapped_data is not None
                  else self.convolved_data if self.convolved_data is not None
                  else self.data)
            sig.set_index('time')['counts'].plot(
                ax=ax, drawstyle='steps-mid', label='Signal', alpha=0.7, **kwargs)
            if show_errors:
                ax.errorbar(sig['time'], sig['counts'], yerr=sig['err'],
                          fmt='none', ecolor='0.5', capsize=2, alpha=0.5)

        # Plot openbeam if available
        if self.op_data is not None:
            if show_stages:
                stages = [(self.op_data, 'Openbeam Original'), (self.op_convolved_data, 'Openbeam Convolved'),
                         (self.op_overlapped_data, 'Openbeam Overlapped'), (self.op_poissoned_data, 'Openbeam Poissoned')]
                for df, label in stages:
                    if df is not None:
                        df.set_index('time')['counts'].plot(
                            ax=ax, drawstyle='steps-mid', label=label, alpha=0.7, **kwargs)
                        if show_errors:
                            ax.errorbar(df['time'], df['counts'], yerr=df['err'],
                                      fmt='none', ecolor='0.5', capsize=2, alpha=0.5)
            else:
                op = (self.op_poissoned_data if self.op_poissoned_data is not None
                     else self.op_overlapped_data if self.op_overlapped_data is not None
                     else self.op_convolved_data if self.op_convolved_data is not None
                     else self.op_data)
                op.set_index('time')['counts'].plot(
                    ax=ax, drawstyle='steps-mid', label='Openbeam', alpha=0.7, **kwargs)
                if show_errors:
                    ax.errorbar(op['time'], op['counts'], yerr=op['err'],
                              fmt='none', ecolor='0.5', capsize=2, alpha=0.5)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Counts')
        ax.set_title('Signal and Openbeam')

    def _calculate_transmission(self, signal_df, openbeam_df):
        """Calculate transmission ratio with error propagation."""
        # Ensure same time points (interpolate if needed)
        if not np.array_equal(signal_df['time'].values, openbeam_df['time'].values):
            # Find common time points
            common_times = np.intersect1d(signal_df['time'].values, openbeam_df['time'].values)
            signal_df = signal_df[signal_df['time'].isin(common_times)]
            openbeam_df = openbeam_df[openbeam_df['time'].isin(common_times)]

        # Calculate transmission
        transmission = signal_df['counts'].values / (openbeam_df['counts'].values + 1e-10)

        # Error propagation: σ_T = T * sqrt((σ_S/S)^2 + (σ_O/O)^2)
        rel_err_sig = signal_df['err'].values / (signal_df['counts'].values + 1e-10)
        rel_err_op = openbeam_df['err'].values / (openbeam_df['counts'].values + 1e-10)
        trans_err = transmission * np.sqrt(rel_err_sig**2 + rel_err_op**2)

        return pd.DataFrame({
            'time': signal_df['time'].values,
            'transmission': transmission,
            'err': trans_err
        })

    def plot_comparison(self, other_data=None, labels=None, fontsize=14):
        """
        Plot comparison between signal and openbeam or another Data object.

        DEPRECATED: Use plot(kind='both') or plot(kind='transmission') instead.
        """
        return self.plot(kind='both', fontsize=fontsize)

    def copy(self):
        """
        Create a deep copy of the Data object.

        Returns
        -------
        Data
            A new Data object with copied data
        """
        new_data = Data(flux=self.flux, duration=self.duration, freq=self.freq,
                       threshold=self.threshold, max_stack=self.max_stack)

        # Copy all dataframes
        new_data.data = self.data.copy() if self.data is not None else None
        new_data.convolved_data = self.convolved_data.copy() if self.convolved_data is not None else None
        new_data.overlapped_data = self.overlapped_data.copy() if self.overlapped_data is not None else None
        new_data.poissoned_data = self.poissoned_data.copy() if self.poissoned_data is not None else None

        new_data.op_data = self.op_data.copy() if self.op_data is not None else None
        new_data.op_convolved_data = self.op_convolved_data.copy() if self.op_convolved_data is not None else None
        new_data.op_overlapped_data = self.op_overlapped_data.copy() if self.op_overlapped_data is not None else None
        new_data.op_poissoned_data = self.op_poissoned_data.copy() if self.op_poissoned_data is not None else None

        new_data.kernel = self.kernel.copy() if self.kernel is not None else None
        new_data.n_overlapping_frames = self.n_overlapping_frames
        new_data.signal_file = self.signal_file
        new_data.openbeam_file = self.openbeam_file

        # Update legacy table attributes
        new_data.table = (new_data.poissoned_data if new_data.poissoned_data is not None
                         else new_data.overlapped_data if new_data.overlapped_data is not None
                         else new_data.convolved_data if new_data.convolved_data is not None
                         else new_data.data)
        new_data.openbeam_table = (new_data.op_poissoned_data if new_data.op_poissoned_data is not None
                                   else new_data.op_overlapped_data if new_data.op_overlapped_data is not None
                                   else new_data.op_convolved_data if new_data.op_convolved_data is not None
                                   else new_data.op_data)

        return new_data

    def __repr__(self):
        """String representation of the Data object."""
        n_points = len(self.data) if self.data is not None else 0
        has_openbeam = self.op_data is not None
        current_stage = (
            'poissoned' if self.poissoned_data is not None
            else 'overlapped' if self.overlapped_data is not None
            else 'convolved' if self.convolved_data is not None
            else 'original'
        )
        return (f"Data(n_points={n_points}, stage='{current_stage}', "
                f"has_openbeam={has_openbeam}, kernel={self.kernel})")
