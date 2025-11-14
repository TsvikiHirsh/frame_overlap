"""
Event-mode data structures for adaptive TOF reconstruction.

This module provides data structures for handling neutron detection events
with multiple pulse timestamps, enabling frame overlap reconstruction.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import h5py


@dataclass
class NeutronEvent:
    """
    Single neutron detection event with multiple timestamps.

    Each event represents one detected neutron, which could have originated
    from any of several recent pulses (frame overlap ambiguity).

    Attributes
    ----------
    detector_id : int
        Detector pixel ID (for spatially-resolved detectors)
    detection_time : float
        Absolute time of detection in microseconds (µs)
    trigger_time : float
        Most recent pulse trigger time in µs
    previous_pulses : np.ndarray
        Array of previous pulse times in µs (up to 10 pulses)

    Examples
    --------
    >>> event = NeutronEvent(
    ...     detector_id=42,
    ...     detection_time=15250.0,
    ...     trigger_time=15000.0,
    ...     previous_pulses=np.array([12500.0, 10000.0, 7500.0])
    ... )
    >>> print(event.tof_candidates)
    [250.0, 2750.0, 5250.0, 7750.0]
    """

    detector_id: int
    detection_time: float
    trigger_time: float
    previous_pulses: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        """Validate event data."""
        if not isinstance(self.previous_pulses, np.ndarray):
            self.previous_pulses = np.array(self.previous_pulses)

        if self.detection_time < self.trigger_time:
            raise ValueError("Detection time must be >= trigger time")

        if len(self.previous_pulses) > 10:
            raise ValueError("Maximum 10 previous pulses allowed")

    @property
    def tof_candidates(self) -> np.ndarray:
        """
        Calculate all possible TOF values for this event.

        Returns
        -------
        np.ndarray
            Array of candidate TOF values in µs, one for each possible source pulse
        """
        all_pulses = np.concatenate([[self.trigger_time], self.previous_pulses])
        return self.detection_time - all_pulses

    @property
    def n_candidates(self) -> int:
        """Number of candidate source pulses."""
        return 1 + len(self.previous_pulses)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'detector_id': self.detector_id,
            'detection_time': self.detection_time,
            'trigger_time': self.trigger_time,
            'previous_pulses': self.previous_pulses.tolist()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeutronEvent':
        """Create event from dictionary."""
        return cls(
            detector_id=data['detector_id'],
            detection_time=data['detection_time'],
            trigger_time=data['trigger_time'],
            previous_pulses=np.array(data['previous_pulses'])
        )


@dataclass
class EventDataset:
    """
    Collection of neutron detection events.

    This class manages a set of events and provides methods for conversion
    to histogram format and I/O operations.

    Attributes
    ----------
    events : List[NeutronEvent]
        List of neutron events
    kernel : np.ndarray
        Frame overlap kernel pattern in milliseconds
    measurement_time : float
        Total measurement duration in hours
    flux : float
        Expected neutron flux in n/cm²/s
    metadata : dict
        Additional metadata (sample info, experimental conditions, etc.)

    Examples
    --------
    >>> events = [NeutronEvent(...), NeutronEvent(...), ...]
    >>> dataset = EventDataset(
    ...     events=events,
    ...     kernel=np.array([0, 25]),
    ...     measurement_time=1.0,
    ...     flux=5e6
    ... )
    >>> print(f"Total events: {dataset.n_events}")
    >>> hist = dataset.to_histogram(tof_bins=np.linspace(1000, 20000, 1000))
    """

    events: List[NeutronEvent]
    kernel: np.ndarray
    measurement_time: float
    flux: float
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate dataset."""
        if not isinstance(self.kernel, np.ndarray):
            self.kernel = np.array(self.kernel)

        if self.metadata is None:
            self.metadata = {}

    @property
    def n_events(self) -> int:
        """Total number of events."""
        return len(self.events)

    @property
    def max_tof(self) -> float:
        """Maximum TOF across all events."""
        return max(event.tof_candidates.max() for event in self.events)

    @property
    def min_tof(self) -> float:
        """Minimum TOF across all events."""
        return min(event.tof_candidates.min() for event in self.events)

    def to_histogram(
        self,
        tof_bins: np.ndarray,
        assignment: str = 'uniform',
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert events to histogram with specified assignment strategy.

        Parameters
        ----------
        tof_bins : np.ndarray
            TOF bin edges in µs
        assignment : str
            Strategy for assigning events to pulses:
            - 'uniform': Equal weight to all candidates
            - 'nearest': Assign to nearest bin (no ambiguity resolution)
            - 'weighted': Use provided weights
        weights : np.ndarray, optional
            Assignment weights for each event (shape: n_events x max_candidates)

        Returns
        -------
        np.ndarray
            Histogram counts in each TOF bin

        Examples
        --------
        >>> tof_bins = np.linspace(1000, 20000, 1000)
        >>> hist_uniform = dataset.to_histogram(tof_bins, assignment='uniform')
        >>> hist_nearest = dataset.to_histogram(tof_bins, assignment='nearest')
        """
        n_bins = len(tof_bins) - 1
        hist = np.zeros(n_bins)

        if assignment == 'uniform':
            # Equal weight to all candidates
            for event in self.events:
                candidates = event.tof_candidates
                weight = 1.0 / len(candidates)
                for tof in candidates:
                    bin_idx = np.searchsorted(tof_bins, tof) - 1
                    if 0 <= bin_idx < n_bins:
                        hist[bin_idx] += weight

        elif assignment == 'nearest':
            # Assign to most recent pulse (trigger time)
            for event in self.events:
                tof = event.tof_candidates[0]  # First candidate is from trigger
                bin_idx = np.searchsorted(tof_bins, tof) - 1
                if 0 <= bin_idx < n_bins:
                    hist[bin_idx] += 1.0

        elif assignment == 'weighted':
            if weights is None:
                raise ValueError("Must provide weights for 'weighted' assignment")
            if len(weights) != self.n_events:
                raise ValueError(f"Weights length {len(weights)} != n_events {self.n_events}")

            for event, event_weights in zip(self.events, weights):
                candidates = event.tof_candidates
                for tof, w in zip(candidates, event_weights):
                    bin_idx = np.searchsorted(tof_bins, tof) - 1
                    if 0 <= bin_idx < n_bins:
                        hist[bin_idx] += w

        else:
            raise ValueError(f"Unknown assignment strategy: {assignment}")

        return hist

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame with one row per event.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: detector_id, detection_time, trigger_time,
            n_candidates, tof_min, tof_max
        """
        data = []
        for event in self.events:
            candidates = event.tof_candidates
            data.append({
                'detector_id': event.detector_id,
                'detection_time': event.detection_time,
                'trigger_time': event.trigger_time,
                'n_candidates': len(candidates),
                'tof_min': candidates.min(),
                'tof_max': candidates.max(),
            })
        return pd.DataFrame(data)

    def save_hdf5(self, filepath: str):
        """
        Save event dataset to HDF5 file.

        Parameters
        ----------
        filepath : str
            Path to output HDF5 file

        Examples
        --------
        >>> dataset.save_hdf5('events.h5')
        """
        with h5py.File(filepath, 'w') as f:
            # Store events as structured array
            dt = np.dtype([
                ('detector_id', np.int32),
                ('detection_time', np.float64),
                ('trigger_time', np.float64),
                ('n_previous', np.int32),
            ])

            n_events = self.n_events
            event_array = np.zeros(n_events, dtype=dt)

            for i, event in enumerate(self.events):
                event_array[i] = (
                    event.detector_id,
                    event.detection_time,
                    event.trigger_time,
                    len(event.previous_pulses)
                )

            f.create_dataset('events', data=event_array)

            # Store previous pulses separately (variable length)
            max_previous = max(len(e.previous_pulses) for e in self.events)
            previous_array = np.full((n_events, max_previous), np.nan)
            for i, event in enumerate(self.events):
                n = len(event.previous_pulses)
                if n > 0:
                    previous_array[i, :n] = event.previous_pulses

            f.create_dataset('previous_pulses', data=previous_array)

            # Store metadata
            f.attrs['kernel'] = self.kernel
            f.attrs['measurement_time'] = self.measurement_time
            f.attrs['flux'] = self.flux
            for key, value in self.metadata.items():
                f.attrs[key] = value

    @classmethod
    def from_hdf5(cls, filepath: str) -> 'EventDataset':
        """
        Load event dataset from HDF5 file.

        Parameters
        ----------
        filepath : str
            Path to input HDF5 file

        Returns
        -------
        EventDataset
            Loaded dataset

        Examples
        --------
        >>> dataset = EventDataset.from_hdf5('events.h5')
        """
        with h5py.File(filepath, 'r') as f:
            event_array = f['events'][:]
            previous_array = f['previous_pulses'][:]

            events = []
            for i, event_data in enumerate(event_array):
                n_previous = event_data['n_previous']
                previous = previous_array[i, :n_previous]
                previous = previous[~np.isnan(previous)]

                events.append(NeutronEvent(
                    detector_id=int(event_data['detector_id']),
                    detection_time=float(event_data['detection_time']),
                    trigger_time=float(event_data['trigger_time']),
                    previous_pulses=previous
                ))

            # Load metadata
            kernel = f.attrs['kernel']
            measurement_time = f.attrs['measurement_time']
            flux = f.attrs['flux']

            metadata = {}
            for key in f.attrs.keys():
                if key not in ['kernel', 'measurement_time', 'flux']:
                    metadata[key] = f.attrs[key]

            return cls(
                events=events,
                kernel=kernel,
                measurement_time=measurement_time,
                flux=flux,
                metadata=metadata
            )

    def filter_by_tof_range(self, tof_min: float, tof_max: float) -> 'EventDataset':
        """
        Filter events by TOF range.

        Parameters
        ----------
        tof_min : float
            Minimum TOF in µs
        tof_max : float
            Maximum TOF in µs

        Returns
        -------
        EventDataset
            New dataset with filtered events
        """
        filtered_events = []
        for event in self.events:
            # Keep event if any candidate falls in range
            candidates = event.tof_candidates
            if np.any((candidates >= tof_min) & (candidates <= tof_max)):
                filtered_events.append(event)

        return EventDataset(
            events=filtered_events,
            kernel=self.kernel.copy(),
            measurement_time=self.measurement_time,
            flux=self.flux,
            metadata=self.metadata.copy() if self.metadata else None
        )

    def __repr__(self) -> str:
        return (f"EventDataset(n_events={self.n_events}, "
                f"kernel={self.kernel.tolist()}, "
                f"tof_range=({self.min_tof:.1f}, {self.max_tof:.1f}) µs)")


@dataclass
class ReconstructionResult:
    """
    Result of spectrum reconstruction from events.

    Attributes
    ----------
    spectrum : np.ndarray
        Reconstructed TOF spectrum (counts per bin)
    tof_bins : np.ndarray
        TOF bin centers in µs
    uncertainty : np.ndarray
        Uncertainty (standard error) per bin
    chi2 : float
        Chi-squared goodness-of-fit (if reference available)
    iterations : int
        Number of iterations performed
    convergence : bool
        Whether algorithm converged
    computation_time : float
        Wall-clock time in seconds
    event_probabilities : np.ndarray, optional
        Assignment probabilities for EM-based methods (n_events x n_candidates)
    metadata : dict
        Additional reconstruction metadata

    Examples
    --------
    >>> result = reconstructor.reconstruct(events)
    >>> result.plot()
    >>> df = result.to_dataframe()
    >>> result.save('reconstruction.h5')
    """

    spectrum: np.ndarray
    tof_bins: np.ndarray
    uncertainty: np.ndarray
    chi2: float
    iterations: int
    convergence: bool
    computation_time: float
    event_probabilities: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame compatible with existing frame_overlap code.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: time (µs), counts, err
        """
        return pd.DataFrame({
            'time': self.tof_bins,
            'counts': self.spectrum,
            'err': self.uncertainty
        })

    def plot(self, show_uncertainty: bool = True, **kwargs):
        """
        Plot reconstructed spectrum.

        Parameters
        ----------
        show_uncertainty : bool
            Whether to show error bars
        **kwargs
            Additional plotting arguments passed to plt.plot()
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Convert to ms for plotting
        tof_ms = self.tof_bins / 1000.0

        ax.plot(tof_ms, self.spectrum, drawstyle='steps-mid', **kwargs)

        if show_uncertainty:
            ax.errorbar(tof_ms, self.spectrum, yerr=self.uncertainty,
                       fmt='none', ecolor='0.5', capsize=2, alpha=0.5)

        ax.set_xlabel('Time-of-Flight (ms)')
        ax.set_ylabel('Counts')
        ax.set_title(f'Reconstructed Spectrum (χ²={self.chi2:.2f}, '
                    f'{self.iterations} iterations)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save(self, filepath: str):
        """Save result to HDF5 file."""
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('spectrum', data=self.spectrum)
            f.create_dataset('tof_bins', data=self.tof_bins)
            f.create_dataset('uncertainty', data=self.uncertainty)

            f.attrs['chi2'] = self.chi2
            f.attrs['iterations'] = self.iterations
            f.attrs['convergence'] = self.convergence
            f.attrs['computation_time'] = self.computation_time

            if self.event_probabilities is not None:
                f.create_dataset('event_probabilities', data=self.event_probabilities)

            if self.metadata:
                for key, value in self.metadata.items():
                    f.attrs[key] = value

    @classmethod
    def from_file(cls, filepath: str) -> 'ReconstructionResult':
        """Load result from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            return cls(
                spectrum=f['spectrum'][:],
                tof_bins=f['tof_bins'][:],
                uncertainty=f['uncertainty'][:],
                chi2=f.attrs['chi2'],
                iterations=int(f.attrs['iterations']),
                convergence=bool(f.attrs['convergence']),
                computation_time=float(f.attrs['computation_time']),
                event_probabilities=f['event_probabilities'][:] if 'event_probabilities' in f else None,
                metadata={k: v for k, v in f.attrs.items()
                         if k not in ['chi2', 'iterations', 'convergence', 'computation_time']}
            )
