# Quick Start: Adaptive TOF Reconstruction

This guide will get you up and running with adaptive frame overlap reconstruction in 30 minutes.

## Installation

The adaptive module is part of the frame_overlap package:

```bash
cd /home/user/frame_overlap
pip install -e .
```

## Phase 1 Implementation Checklist

### Week 1: Event Data Structures (3 days)

**Day 1: Core Data Structures**
- [ ] `event_data.py`: `NeutronEvent` class ✅ (template ready)
- [ ] `event_data.py`: `EventDataset` class ✅ (template ready)
- [ ] `event_data.py`: `ReconstructionResult` class ✅ (template ready)
- [ ] Unit tests for data structures

**Day 2: I/O and Conversion**
- [ ] HDF5 save/load (already in template)
- [ ] Histogram conversion methods
- [ ] Integration with existing `Data` class
- [ ] CSV import (for compatibility)

**Day 3: Testing and Documentation**
- [ ] Comprehensive unit tests
- [ ] Docstring completion
- [ ] Usage examples
- [ ] Performance profiling

### Week 2: Simulation Framework (5 days)

**Day 4-5: Spectrum Generator**
```python
# File: src/frame_overlap/adaptive/simulation.py

def generate_bragg_edge_spectrum(
    tof_range=(1000, 20000),
    n_bins=1000,
    material='iron',
    thickness=1.0,
    temperature=300
):
    """Generate realistic Bragg edge spectrum."""
    # Use existing nbragg if available, else analytical
    pass

def add_poisson_noise(spectrum, scale=1.0, seed=None):
    """Add Poisson counting statistics."""
    pass
```

**Day 6-7: Event Simulator**
```python
def generate_events_from_spectrum(
    true_spectrum,
    tof_bins,
    kernel,
    n_events=10000,
    pulse_period=50,  # ms
    seed=None
):
    """
    Generate synthetic events with frame overlap.

    This simulates the data collection process with overlapping frames.
    """
    events = []

    # Sample events from true spectrum (Poisson)
    for bin_idx, counts in enumerate(true_spectrum):
        n_samples = np.random.poisson(counts)
        tof = tof_bins[bin_idx]

        for _ in range(n_samples):
            # Assign to random pulse (frame)
            frame_idx = np.random.choice(len(kernel))
            trigger_offset = sum(kernel[:frame_idx+1])

            # Create detection time
            detection_time = tof + trigger_offset * 1000  # kernel in ms

            # Add jitter
            detection_time += np.random.normal(0, 1)  # 1 µs timing jitter

            # Generate previous pulse times
            previous_pulses = []
            for j in range(1, min(10, frame_idx+1)):
                prev_time = detection_time - pulse_period * 1000 * j
                previous_pulses.append(prev_time)

            event = NeutronEvent(
                detector_id=0,
                detection_time=detection_time,
                trigger_time=detection_time - tof,
                previous_pulses=np.array(previous_pulses)
            )
            events.append(event)

    return EventDataset(
        events=events,
        kernel=kernel,
        measurement_time=1.0,
        flux=1e6
    )
```

**Day 8: Validation**
- [ ] Compare simulated spectra with nbragg
- [ ] Verify event statistics (Poisson check)
- [ ] Benchmark against existing `Data.overlap()`

### Week 3: Baseline Reconstructor (3 days)

**Day 9: Baseline Implementation**
```python
# File: src/frame_overlap/adaptive/reconstructors/baseline.py

class BaselineReconstructor(BaseReconstructor):
    """
    Baseline: Direct binning with no frame overlap (separated frames).

    This reconstructor assumes no frame overlap and directly bins events
    using only the most recent pulse (trigger time).
    """

    def reconstruct(self, event_data, **kwargs):
        start_time = time.time()

        # Simple: use only trigger time (no ambiguity)
        hist = np.zeros(self.n_bins)
        bin_edges = np.linspace(self.tof_range[0], self.tof_range[1], self.n_bins + 1)

        for event in event_data.events:
            tof = event.tof_candidates[0]  # First = from trigger
            bin_idx = np.searchsorted(bin_edges, tof) - 1
            if 0 <= bin_idx < self.n_bins:
                hist[bin_idx] += 1

        # Poisson uncertainty
        uncertainty = np.sqrt(np.maximum(hist, 1))

        self.spectrum = hist
        self.uncertainty = uncertainty
        self.iterations = 1
        self.n_events_processed = event_data.n_events

        return ReconstructionResult(
            spectrum=hist,
            tof_bins=self.tof_bins,
            uncertainty=uncertainty,
            chi2=0.0,
            iterations=1,
            convergence=True,
            computation_time=time.time() - start_time
        )

    def update(self, new_events, **kwargs):
        # Online update: just add to histogram
        bin_edges = np.linspace(self.tof_range[0], self.tof_range[1], self.n_bins + 1)

        if self.spectrum is None:
            self.spectrum = np.zeros(self.n_bins)
            self.uncertainty = np.zeros(self.n_bins)

        for event in new_events:
            tof = event.tof_candidates[0]
            bin_idx = np.searchsorted(bin_edges, tof) - 1
            if 0 <= bin_idx < self.n_bins:
                self.spectrum[bin_idx] += 1

        self.uncertainty = np.sqrt(np.maximum(self.spectrum, 1))
        self.n_events_processed += len(new_events)
```

**Day 10-11: Testing and Benchmarking**
- [ ] Unit tests
- [ ] Compare with `Data.overlap(kernel=[0])`
- [ ] Performance baseline

### Week 4: Wiener Reconstructor (4 days)

**Day 12-13: Implementation**
```python
# File: src/frame_overlap/adaptive/reconstructors/wiener_event.py

class WienerEventReconstructor(BaseReconstructor):
    """
    Wiener filter reconstruction for event data with fixed kernel.

    Uses uniform initial assignment, then applies Wiener deconvolution.
    """

    def reconstruct(self, event_data, noise_power=0.01, **kwargs):
        start_time = time.time()

        # Step 1: Create histogram with uniform assignment
        hist = event_data.to_histogram(
            tof_bins=self.tof_bins,
            assignment='uniform'
        )

        # Step 2: Apply Wiener deconvolution
        spectrum = self._wiener_filter(hist, event_data.kernel, noise_power)

        # Step 3: Calculate uncertainty (approximate)
        uncertainty = np.sqrt(np.maximum(spectrum, 1))

        self.spectrum = spectrum
        self.uncertainty = uncertainty
        self.iterations = 1
        self.n_events_processed = event_data.n_events

        return ReconstructionResult(
            spectrum=spectrum,
            tof_bins=self.tof_bins,
            uncertainty=uncertainty,
            chi2=0.0,
            iterations=1,
            convergence=True,
            computation_time=time.time() - start_time
        )

    def _wiener_filter(self, observed, kernel, noise_power):
        """Apply Wiener deconvolution in frequency domain."""
        # Build kernel from frame overlap pattern
        kernel_signal = self._build_kernel(kernel)

        # Pad kernel to match observed length
        if len(kernel_signal) < len(observed):
            kernel_padded = np.pad(kernel_signal, (0, len(observed) - len(kernel_signal)))
        else:
            kernel_padded = kernel_signal[:len(observed)]

        # FFT-based Wiener filtering
        H = np.fft.fft(kernel_padded)
        Y = np.fft.fft(observed)
        H_conj = np.conj(H)
        G = H_conj / (np.abs(H)**2 + noise_power)
        X_est = Y * G
        x_est = np.real(np.fft.ifft(X_est))

        return x_est

    def _build_kernel(self, kernel_ms):
        """Build convolution kernel from frame timing."""
        # Convert kernel from ms to bins
        bin_width = (self.tof_range[1] - self.tof_range[0]) / self.n_bins  # µs
        kernel_us = np.array(kernel_ms) * 1000  # ms to µs
        kernel_bins = (kernel_us / bin_width).astype(int)

        # Create delta functions at frame starts
        max_bin = kernel_bins.max() + 1 if len(kernel_bins) > 0 else 1
        kernel_signal = np.zeros(max_bin)

        for bin_idx in kernel_bins:
            if bin_idx < len(kernel_signal):
                kernel_signal[bin_idx] = 1.0 / len(kernel_ms)

        return kernel_signal

    def update(self, new_events, **kwargs):
        # For online Wiener, would need sliding window approach
        # For now, raise NotImplementedError
        raise NotImplementedError("Online Wiener update not yet implemented")
```

**Day 14-15: Testing**
- [ ] Unit tests
- [ ] Compare with existing `Reconstruct.filter(kind='wiener')`
- [ ] Noise power optimization

## Quick Example

Once Phase 1 is complete, you can:

```python
from frame_overlap.adaptive import (
    generate_synthetic_events,
    BaselineReconstructor,
    WienerEventReconstructor,
)

# Generate synthetic data
events = generate_synthetic_events(
    material='iron',
    kernel=np.array([0, 25]),  # 2-frame, 25 ms spacing
    n_events=50000,
    flux=5e6,
    measurement_time=0.5
)

# Baseline reconstruction (no overlap handling)
baseline = BaselineReconstructor(tof_range=(1000, 20000), n_bins=1000)
result_baseline = baseline.reconstruct(events)
result_baseline.plot()

# Wiener reconstruction (handles overlap)
wiener = WienerEventReconstructor(tof_range=(1000, 20000), n_bins=1000)
result_wiener = wiener.reconstruct(events, noise_power=0.01)
result_wiener.plot()

# Compare
print(f"Baseline χ²: {result_baseline.chi2:.2f}")
print(f"Wiener χ²: {result_wiener.chi2:.2f}")
```

## Testing Strategy

### Unit Tests Template

```python
# tests/adaptive/test_event_data.py

import pytest
import numpy as np
from frame_overlap.adaptive import NeutronEvent, EventDataset

class TestNeutronEvent:
    def test_creation(self):
        event = NeutronEvent(
            detector_id=0,
            detection_time=1000.0,
            trigger_time=900.0,
            previous_pulses=np.array([850.0, 800.0])
        )
        assert event.n_candidates == 3
        assert len(event.tof_candidates) == 3

    def test_tof_calculation(self):
        event = NeutronEvent(
            detector_id=0,
            detection_time=1000.0,
            trigger_time=900.0,
            previous_pulses=np.array([850.0])
        )
        expected_tofs = np.array([100.0, 150.0])
        np.testing.assert_array_almost_equal(event.tof_candidates, expected_tofs)

    def test_invalid_times(self):
        with pytest.raises(ValueError):
            NeutronEvent(
                detector_id=0,
                detection_time=900.0,
                trigger_time=1000.0,  # Invalid: trigger after detection
                previous_pulses=np.array([])
            )


class TestEventDataset:
    def test_histogram_uniform(self):
        events = [
            NeutronEvent(0, 1000, 900, np.array([850])),
            NeutronEvent(0, 1100, 1000, np.array([950])),
        ]
        dataset = EventDataset(
            events=events,
            kernel=np.array([0, 50]),
            measurement_time=1.0,
            flux=1e6
        )

        tof_bins = np.linspace(0, 200, 21)  # 20 bins, 10 µs each
        hist = dataset.to_histogram(tof_bins, assignment='uniform')

        # Each event contributes 0.5 to two bins
        assert hist.sum() == len(events)

    def test_save_load_hdf5(self, tmp_path):
        events = [NeutronEvent(0, 1000, 900, np.array([850]))]
        dataset = EventDataset(
            events=events,
            kernel=np.array([0, 25]),
            measurement_time=1.0,
            flux=1e6
        )

        filepath = tmp_path / "test.h5"
        dataset.save_hdf5(str(filepath))

        loaded = EventDataset.from_hdf5(str(filepath))
        assert loaded.n_events == dataset.n_events
        np.testing.assert_array_equal(loaded.kernel, dataset.kernel)
```

## Performance Targets (Phase 1)

| Metric | Target | Status |
|--------|--------|--------|
| Event creation | < 1 µs per event | ⏳ |
| Histogram conversion | < 100 µs per 1k events | ⏳ |
| HDF5 save | < 1 s per 100k events | ⏳ |
| HDF5 load | < 0.5 s per 100k events | ⏳ |
| Baseline reconstruction | < 10 ms per 10k events | ⏳ |
| Wiener reconstruction | < 100 ms per 10k events | ⏳ |

## Next Steps

After completing Phase 1:
1. Review and validate against existing code
2. Create performance benchmarks
3. Generate example datasets
4. Write user documentation
5. Begin Phase 2: EM Reconstruction

## Getting Help

- **Documentation**: See `ADAPTIVE_TOF_IMPLEMENTATION_PLAN.md`
- **Examples**: Check `notebooks/adaptive/`
- **Tests**: Look at `tests/adaptive/` for usage patterns
- **Issues**: File on GitHub issue tracker
