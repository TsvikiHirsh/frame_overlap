# Adaptive Frame Overlap TOF Reconstruction - Implementation Plan

## Executive Summary

This plan outlines the implementation of an **adaptive frame overlap TOF reconstruction system** that can dynamically adjust frame overlap kernels during data collection and reconstruct spectra faster than traditional separated-frames methods. The system will handle event-mode neutron data where each event is tagged with multiple pulse timestamps.

**Key Innovation**: Use adaptive/ML algorithms to exploit the fact that frame overlap increases effective flux, allowing faster convergence to accurate spectra in narrow regions of interest (e.g., around Bragg edges in 1-10 Å range).

---

## 1. Problem Formulation

### 1.1 Mathematical Framework

#### Traditional Approach (Baseline)
- **Low frequency pulsing**: Pulses separated enough to avoid overlap (e.g., 10-20 Hz)
- **Single timestamp per event**: Each neutron unambiguously associated with one pulse
- **Reconstruction**: Direct binning into TOF histogram
- **Time to target accuracy**: High (limited by flux)

#### Adaptive Frame Overlap Approach
- **Higher frequency pulsing**: Intentional overlap (e.g., 60-200 Hz)
- **Multiple timestamps per event**: Each neutron carries current trigger + up to 10 previous pulse times
- **Reconstruction problem**: Probabilistic assignment of each event to correct source pulse

**Forward Model:**
```
Observed events with timestamp sets → True TOF spectrum

For each neutron event i:
  - Current trigger time: t_0^(i)
  - Previous pulse times: {t_-1^(i), t_-2^(i), ..., t_-N^(i)} where N ≤ 10
  - Possible TOF values: {TOF_0^(i), TOF_1^(i), ..., TOF_N^(i)}
    where TOF_j^(i) = t_0^(i) - t_-j^(i)
```

**Objective**: Assign each event to the most probable source pulse to reconstruct true TOF spectrum S(t).

#### Mathematical Formulation

Let:
- `S(t)` = True TOF spectrum (counts vs time-of-flight)
- `K(t)` = Frame overlap kernel (pulse timing pattern)
- `O(t)` = Observed overlapped spectrum
- `E = {e_1, e_2, ..., e_M}` = Set of M neutron events
- `e_i = {t_0, {t_-1, ..., t_-N}}` = Event i with timestamp set
- `p_ij` = Probability that event i came from pulse j

**Forward model (convolution):**
```
O(t) = (S ⊗ K)(t) = Σ_j S(t - Δt_j)
```
where `Δt_j` are pulse-to-pulse separations in kernel K.

**Event-level forward model:**
```
P(e_i observed) = Σ_j p_ij * S(TOF_j^(i))
```

**Inverse problem:**
Reconstruct S(t) from event set E, given:
1. Kernel K (may be adaptive/time-varying)
2. Prior information about S (e.g., smoothness, expected features)

### 1.2 Using Multiple Timestamps Per Event

Each event provides multiple hypotheses for its true TOF:

```
Event i at detector time T_detect:
  Candidate TOFs: {T_detect - t_0, T_detect - t_-1, ..., T_detect - t_-N}
```

**Probabilistic Assignment Strategies:**

1. **Bayesian Assignment** (most rigorous):
   ```
   p_ij ∝ S_prior(TOF_j^(i)) * flux_model(pulse_j)
   ```
   Where `S_prior` is current estimate of spectrum.

2. **Maximum Likelihood**:
   Assign each event to pulse that maximizes likelihood given current S estimate.

3. **Expectation-Maximization (EM)**:
   - E-step: Calculate expected assignments given current S
   - M-step: Update S given expected assignments
   - Iterate until convergence

4. **Weighted Assignment**:
   Fractionally assign each event across all candidate pulses.

### 1.3 Objective Functions to Optimize

**Primary objective**: Minimize reconstruction error while maximizing data collection efficiency.

1. **Chi-squared (goodness-of-fit)**:
   ```
   χ² = Σ_t [(S_recon(t) - S_true(t))² / σ²(t)]
   ```

2. **Kullback-Leibler Divergence** (for Poisson data):
   ```
   KL(S_true || S_recon) = Σ_t S_true(t) log(S_true(t) / S_recon(t))
   ```

3. **Time-to-target accuracy**:
   ```
   T_target = min{T : χ²(T) < χ²_threshold}
   ```

4. **Information gain per measurement** (for adaptive kernel selection):
   ```
   IG(K) = H(S) - H(S | measurements with kernel K)
   ```
   Where H is entropy.

5. **Reconstruction uncertainty**:
   ```
   U(t) = σ_S(t) / S(t)  (relative uncertainty)
   ```

**Multi-objective optimization** (for adaptive kernel):
- Maximize information gain in regions of interest (ROI)
- Minimize total measurement time
- Maintain acceptable uncertainty outside ROI

---

## 2. Algorithmic Approaches (Ranked by Feasibility)

### Rank 1: Traditional Baseline (Separated Frames) ⭐⭐⭐⭐⭐
**Feasibility**: Very High | **Priority**: Essential

**Description**: Low-frequency pulsing with no frame overlap.

**Implementation**:
- Single timestamp per event
- Direct histogram binning
- No reconstruction needed
- Serves as ground truth for validation

**Advantages**:
- Simple, robust, well-understood
- No ambiguity in event assignment
- Provides reference for comparison

**Disadvantages**:
- Low effective flux (slow data collection)
- Long time to reach target accuracy

**Implementation complexity**: Low (1-2 days)
**Scientific risk**: None

---

### Rank 2: Fixed-Kernel Wiener Reconstruction ⭐⭐⭐⭐⭐
**Feasibility**: Very High | **Priority**: Essential

**Description**: Fixed frame overlap pattern throughout measurement. Standard Wiener deconvolution.

**Implementation**:
- Already partially implemented in existing codebase!
- Extend to handle event-mode data with multiple timestamps
- Use histogram binning with initial uniform assignment
- Apply Wiener filter as in current `Reconstruct` class

**Algorithm**:
```python
# Pseudo-code
def fixed_kernel_reconstruction(events, kernel):
    # Initial assignment: distribute events uniformly
    hist = create_histogram_uniform_assignment(events, kernel)

    # Apply Wiener deconvolution
    S_recon = wiener_filter(hist, kernel, noise_power)

    return S_recon
```

**Advantages**:
- Builds on existing code
- Well-studied, stable
- Higher flux than baseline
- Predictable performance

**Disadvantages**:
- Fixed kernel may not be optimal for all ROIs
- No adaptation during measurement

**Implementation complexity**: Low-Medium (3-5 days)
**Scientific risk**: Low

**Key Extensions**:
1. Event-mode data handling
2. Multiple timestamp processing
3. Initial event assignment strategies

---

### Rank 3: EM-based Iterative Reconstruction ⭐⭐⭐⭐
**Feasibility**: High | **Priority**: High

**Description**: Expectation-Maximization algorithm for probabilistic event assignment.

**Algorithm**:
```python
def em_reconstruction(events, kernel, max_iter=50):
    # Initialize spectrum estimate (uniform or from prior)
    S = initialize_spectrum()

    for iteration in range(max_iter):
        # E-step: Compute assignment probabilities
        for event in events:
            for j, tof_candidate in enumerate(event.tof_candidates):
                # Probability ∝ current spectrum value
                p[event][j] = S[tof_candidate] * kernel_weight[j]
            # Normalize probabilities
            p[event] /= sum(p[event])

        # M-step: Update spectrum based on expected assignments
        S_new = zeros_like(S)
        for event in events:
            for j, tof_candidate in enumerate(event.tof_candidates):
                S_new[tof_candidate] += p[event][j]

        # Check convergence
        if ||S_new - S|| < tolerance:
            break

        S = S_new

    return S
```

**Advantages**:
- Principled probabilistic approach
- Guaranteed convergence to local maximum
- Handles uncertainty naturally
- Can incorporate priors

**Disadvantages**:
- Slower than direct methods
- May converge to local optima
- Requires good initialization

**Implementation complexity**: Medium (1-2 weeks)
**Scientific risk**: Low-Medium

**Key Components**:
1. Event probability calculator
2. Spectrum updater
3. Convergence checker
4. Prior specification interface

---

### Rank 4: Adaptive Kernel Selection (Active Learning) ⭐⭐⭐⭐
**Feasibility**: Medium-High | **Priority**: High

**Description**: Dynamically adjust kernel during measurement to maximize information gain in ROI.

**Strategy**: Bayesian Experimental Design approach
- Maintain posterior distribution over S(t)
- Select next kernel to maximize expected information gain
- Update posterior after each measurement batch

**Algorithm**:
```python
def adaptive_kernel_reconstruction(
    events_stream,
    roi_ranges,
    kernel_library,
    update_frequency
):
    # Initialize
    S_posterior = initialize_posterior()
    kernel_history = []

    batch_count = 0
    for event_batch in events_stream:
        # Process current batch with current kernel
        S_posterior = update_posterior(S_posterior, event_batch, current_kernel)

        # Every N batches, reassess kernel
        if batch_count % update_frequency == 0:
            # Calculate uncertainty in ROIs
            uncertainty_roi = compute_roi_uncertainty(S_posterior, roi_ranges)

            # Select kernel that maximizes info gain in highest-uncertainty ROI
            current_kernel = select_kernel(
                S_posterior,
                uncertainty_roi,
                kernel_library
            )
            kernel_history.append(current_kernel)

            # Send kernel update to acquisition system
            update_acquisition_kernel(current_kernel)

        batch_count += 1

    return S_posterior.mean(), kernel_history
```

**Kernel Selection Strategies**:

1. **Uncertainty Sampling**:
   Select kernel that reduces uncertainty in ROI with highest current uncertainty.

2. **Expected Information Gain**:
   ```
   K* = argmax_K E[IG(S | measurements with K)]
   ```

3. **Thompson Sampling**:
   Sample from posterior, select kernel optimal for that sample.

4. **Hybrid**: Start with high-overlap (high flux), transition to low-overlap (high resolution) as spectrum sharpens.

**Advantages**:
- Optimizes measurement for specific ROIs
- Can significantly reduce time-to-target
- Adapts to unexpected features
- Scientifically interesting

**Disadvantages**:
- Requires real-time decision making
- More complex infrastructure
- Needs simulation validation first

**Implementation complexity**: Medium-High (3-4 weeks)
**Scientific risk**: Medium

**Key Components**:
1. Uncertainty quantification module
2. Information gain estimator
3. Kernel library manager
4. Real-time kernel selection engine
5. Interface to acquisition system

---

### Rank 5: Kernel Optimization via Bayesian Optimization ⭐⭐⭐
**Feasibility**: Medium | **Priority**: Medium

**Description**: Use Bayesian optimization to find optimal fixed kernel for given sample/ROI.

**Algorithm**:
```python
def optimize_kernel_bayesian(
    sample_type,
    roi_ranges,
    measurement_budget,
    n_trials=20
):
    # Define kernel parameter space
    kernel_space = {
        'n_pulses': (2, 10),
        'pulse_spacing': (5, 50),  # ms
        'pattern_type': ['uniform', 'fibonacci', 'exponential']
    }

    # Objective: time to reach target accuracy in ROI
    def objective(kernel_params):
        kernel = construct_kernel(**kernel_params)
        time_to_target = simulate_measurement(
            kernel,
            sample_type,
            roi_ranges,
            target_accuracy=0.05
        )
        return time_to_target

    # Run Bayesian optimization
    optimizer = BayesianOptimizer(objective, kernel_space)
    best_kernel = optimizer.maximize(n_trials=n_trials)

    return best_kernel
```

**Advantages**:
- Finds near-optimal kernel efficiently
- Sample-specific optimization
- Offline pre-experiment planning

**Disadvantages**:
- Requires accurate forward model
- Fixed for entire measurement
- Computationally expensive

**Implementation complexity**: Medium-High (2-3 weeks)
**Scientific risk**: Medium

---

### Rank 6: ML-Based Direct Reconstruction (Neural Networks) ⭐⭐
**Feasibility**: Low-Medium | **Priority**: Low-Medium

**Description**: Train neural network to directly map event sets to spectra.

**Architecture Options**:

1. **Event-Set Network** (PointNet-style):
   ```
   Input: Set of events {(t_0, {t_-1, ..., t_-N})}
   → Per-event embedding network
   → Permutation-invariant aggregation (max pooling)
   → Spectrum decoder network
   Output: S(t)
   ```

2. **Hybrid Classical-ML**:
   - Use EM for initial reconstruction
   - Train network to refine/denoise
   - Input: (initial S, event statistics) → Output: refined S

**Training Requirements**:
- Large dataset of (events, true_spectrum) pairs
- Simulate various kernels, noise levels, sample types
- ~10,000-100,000 training examples

**Advantages**:
- Potentially very fast at inference time
- Can learn complex patterns
- Could handle non-standard kernels

**Disadvantages**:
- Requires extensive training data
- Black-box (hard to interpret)
- Generalization uncertain
- High development overhead

**Implementation complexity**: Very High (6-8 weeks)
**Scientific risk**: High

**Recommendation**: Defer to Phase 4 or future work. Focus on interpretable methods first.

---

### Rank 7: Kalman Filter / Sequential Bayesian ⭐⭐⭐
**Feasibility**: Medium | **Priority**: Medium

**Description**: Treat reconstruction as online state estimation problem.

**Framework**:
```python
class SpectrumKalmanFilter:
    def __init__(self, n_bins):
        # State: TOF spectrum S(t)
        self.state = np.zeros(n_bins)
        self.covariance = np.eye(n_bins) * initial_variance

    def predict(self, kernel):
        # State transition (no dynamics, spectrum is static)
        # Covariance growth due to process noise
        self.covariance += process_noise

    def update(self, event_batch, kernel):
        # Measurement model: events are noisy samples from S ⊗ K
        for event in event_batch:
            # Measurement Jacobian
            H = compute_measurement_jacobian(event, kernel)

            # Kalman gain
            K = self.covariance @ H.T @ inv(H @ self.covariance @ H.T + R)

            # Update state
            innovation = event.measurement - H @ self.state
            self.state += K @ innovation

            # Update covariance
            self.covariance = (I - K @ H) @ self.covariance
```

**Advantages**:
- Online/streaming capability
- Uncertainty quantification
- Principled handling of noise
- Can incorporate dynamics if needed

**Disadvantages**:
- Linearization approximations
- Covariance matrix large (n_bins × n_bins)
- Measurement model nonlinear (need EKF/UKF)

**Implementation complexity**: Medium-High (3-4 weeks)
**Scientific risk**: Medium

---

## 3. Software Architecture

### 3.1 Core Classes and Modules

```
src/frame_overlap/adaptive/
├── __init__.py
├── event_data.py          # Event-mode data structures
├── event_processor.py      # Event assignment and binning
├── reconstructors/
│   ├── __init__.py
│   ├── base.py            # Base reconstructor interface
│   ├── em_reconstructor.py         # EM algorithm
│   ├── wiener_event.py             # Wiener for event data
│   ├── kalman_reconstructor.py     # Kalman filter
│   └── ml_reconstructor.py         # Neural network (future)
├── kernel_manager.py       # Kernel library and selection
├── adaptive_controller.py  # Adaptive kernel selection logic
├── uncertainty.py          # Uncertainty quantification
├── simulation.py           # Synthetic data generation
└── evaluation.py           # Metrics and benchmarking
```

### 3.2 Data Structures

#### Event-Mode Data

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class NeutronEvent:
    """Single neutron detection event with multiple timestamps."""

    detector_id: int          # Which detector pixel
    detection_time: float     # Absolute time of detection (µs)
    trigger_time: float       # Current pulse trigger time (µs)
    previous_pulses: np.ndarray  # Up to 10 previous pulse times (µs)

    @property
    def tof_candidates(self) -> np.ndarray:
        """Calculate all possible TOF values for this event."""
        all_pulses = np.concatenate([[self.trigger_time], self.previous_pulses])
        return self.detection_time - all_pulses

    @property
    def n_candidates(self) -> int:
        return len(self.tof_candidates)


@dataclass
class EventDataset:
    """Collection of neutron events."""

    events: List[NeutronEvent]
    kernel: np.ndarray        # Frame overlap pattern used (ms)
    measurement_time: float   # Total time (hours)
    flux: float              # Expected flux (n/cm²/s)
    metadata: dict = None

    @property
    def n_events(self) -> int:
        return len(self.events)

    def to_histogram(
        self,
        tof_bins: np.ndarray,
        assignment: str = 'uniform'
    ) -> np.ndarray:
        """Convert events to histogram with specified assignment strategy."""
        pass  # Implementation below


@dataclass
class ReconstructionResult:
    """Result of spectrum reconstruction."""

    spectrum: np.ndarray          # Reconstructed S(t)
    tof_bins: np.ndarray         # TOF bin centers (µs)
    uncertainty: np.ndarray      # Uncertainty per bin

    # Diagnostics
    chi2: float
    iterations: int
    convergence: bool
    computation_time: float

    # Event assignments (for EM-based methods)
    event_probabilities: Optional[np.ndarray] = None

    def to_dataframe(self):
        """Convert to pandas DataFrame for compatibility with existing code."""
        import pandas as pd
        return pd.DataFrame({
            'time': self.tof_bins,
            'counts': self.spectrum,
            'err': self.uncertainty
        })
```

### 3.3 Base Reconstructor Interface

```python
from abc import ABC, abstractmethod

class BaseReconstructor(ABC):
    """Abstract base class for all reconstruction algorithms."""

    def __init__(self, tof_range: tuple, n_bins: int):
        self.tof_range = tof_range  # (min, max) in µs
        self.n_bins = n_bins
        self.tof_bins = np.linspace(tof_range[0], tof_range[1], n_bins)

    @abstractmethod
    def reconstruct(
        self,
        event_data: EventDataset,
        **kwargs
    ) -> ReconstructionResult:
        """
        Reconstruct spectrum from event data.

        Parameters
        ----------
        event_data : EventDataset
            Input events with multiple timestamps

        Returns
        -------
        ReconstructionResult
            Reconstructed spectrum with uncertainty
        """
        pass

    @abstractmethod
    def update(self, new_events: List[NeutronEvent]):
        """Update reconstruction with new events (for online methods)."""
        pass

    def compute_chi2(
        self,
        reconstructed: np.ndarray,
        reference: np.ndarray,
        uncertainty: np.ndarray
    ) -> float:
        """Compute chi-squared between reconstructed and reference spectra."""
        return np.sum(((reconstructed - reference) / uncertainty) ** 2)
```

### 3.4 Kernel Manager

```python
class KernelManager:
    """Manages kernel library and selection strategies."""

    def __init__(self):
        self.kernel_library = self._build_kernel_library()

    def _build_kernel_library(self) -> Dict[str, np.ndarray]:
        """Build library of pre-defined kernels."""
        return {
            'separated': np.array([0]),  # No overlap (baseline)
            'two_frame_25ms': np.array([0, 25]),
            'two_frame_12ms': np.array([0, 12]),
            'three_frame_uniform': np.array([0, 16, 16, 16]),
            'fibonacci': np.array([0, 5, 8, 13, 21]),
            'exponential': np.array([0, 5, 10, 20, 40]),
            'high_flux': np.array([0, 5, 5, 5, 5, 5]),  # Many short frames
            'high_res': np.array([0, 40]),  # Few long frames
        }

    def select_kernel(
        self,
        current_spectrum: np.ndarray,
        uncertainty: np.ndarray,
        roi_ranges: List[tuple],
        strategy: str = 'uncertainty_sampling'
    ) -> np.ndarray:
        """
        Select next kernel based on strategy.

        Parameters
        ----------
        current_spectrum : np.ndarray
            Current best estimate of spectrum
        uncertainty : np.ndarray
            Uncertainty in each bin
        roi_ranges : List[tuple]
            Regions of interest [(tof_min, tof_max), ...]
        strategy : str
            Selection strategy

        Returns
        -------
        np.ndarray
            Selected kernel
        """
        if strategy == 'uncertainty_sampling':
            return self._select_by_uncertainty(uncertainty, roi_ranges)
        elif strategy == 'info_gain':
            return self._select_by_info_gain(current_spectrum, uncertainty, roi_ranges)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _select_by_uncertainty(
        self,
        uncertainty: np.ndarray,
        roi_ranges: List[tuple]
    ) -> np.ndarray:
        """Select kernel that reduces uncertainty in highest-uncertainty ROI."""
        # Calculate average uncertainty in each ROI
        roi_uncertainties = []
        for tof_min, tof_max in roi_ranges:
            mask = (self.tof_bins >= tof_min) & (self.tof_bins <= tof_max)
            roi_uncertainties.append(np.mean(uncertainty[mask]))

        # Find ROI with highest uncertainty
        worst_roi = np.argmax(roi_uncertainties)

        # Select kernel optimized for that ROI
        # (Simple heuristic: use high-overlap for high TOF, low-overlap for low TOF)
        roi_center = np.mean(roi_ranges[worst_roi])

        if roi_center > 15000:  # High TOF (slow neutrons)
            return self.kernel_library['high_flux']
        else:  # Low TOF (fast neutrons)
            return self.kernel_library['two_frame_12ms']
```

### 3.5 Adaptive Controller

```python
class AdaptiveController:
    """Controls adaptive kernel selection during measurement."""

    def __init__(
        self,
        reconstructor: BaseReconstructor,
        kernel_manager: KernelManager,
        roi_ranges: List[tuple],
        update_interval: int = 1000  # events between kernel updates
    ):
        self.reconstructor = reconstructor
        self.kernel_manager = kernel_manager
        self.roi_ranges = roi_ranges
        self.update_interval = update_interval

        self.event_count = 0
        self.kernel_history = []

    def process_event(self, event: NeutronEvent) -> Optional[np.ndarray]:
        """
        Process single event and optionally return new kernel.

        Returns
        -------
        Optional[np.ndarray]
            New kernel if update triggered, None otherwise
        """
        # Update reconstruction
        self.reconstructor.update([event])
        self.event_count += 1

        # Check if kernel update needed
        if self.event_count % self.update_interval == 0:
            result = self.reconstructor.get_current_result()

            new_kernel = self.kernel_manager.select_kernel(
                result.spectrum,
                result.uncertainty,
                self.roi_ranges,
                strategy='uncertainty_sampling'
            )

            self.kernel_history.append({
                'event_count': self.event_count,
                'kernel': new_kernel
            })

            return new_kernel

        return None
```

---

## 4. Implementation Strategy

### Phase 1: Foundation & Baseline (2-3 weeks)

**Goals**:
- Event-mode data structures
- Baseline separated-frames reconstruction
- Fixed-kernel Wiener reconstruction
- Simulation framework

**Tasks**:

1. **Event Data Infrastructure** (3 days)
   - [ ] Implement `NeutronEvent` class
   - [ ] Implement `EventDataset` class
   - [ ] Event I/O (load from CSV, HDF5)
   - [ ] Conversion to/from histogram mode
   - [ ] Unit tests

2. **Simulation Framework** (5 days)
   - [ ] Synthetic spectrum generator (Bragg edges, resonances)
   - [ ] Event simulator with Poisson statistics
   - [ ] Multiple-timestamp tagging
   - [ ] Noise models
   - [ ] Validation against existing `Data` class

3. **Baseline Reconstructor** (3 days)
   - [ ] Implement separated-frames (direct binning)
   - [ ] Performance metrics
   - [ ] Integration tests

4. **Fixed-Kernel Wiener for Events** (4 days)
   - [ ] Extend `Reconstruct` class for event data
   - [ ] Uniform event assignment
   - [ ] Wiener deconvolution
   - [ ] Comparison with baseline
   - [ ] Unit tests

**Deliverables**:
- `adaptive/event_data.py` - Complete
- `adaptive/simulation.py` - Complete
- `adaptive/reconstructors/baseline.py` - Complete
- `adaptive/reconstructors/wiener_event.py` - Complete
- Tests passing
- Example notebooks showing simulation

---

### Phase 2: Iterative Reconstruction (2-3 weeks)

**Goals**:
- EM-based reconstruction
- Uncertainty quantification
- Performance benchmarking

**Tasks**:

1. **EM Reconstructor** (7 days)
   - [ ] Implement E-step (probability calculation)
   - [ ] Implement M-step (spectrum update)
   - [ ] Convergence criteria
   - [ ] Prior specification (uniform, smoothness)
   - [ ] Unit tests

2. **Uncertainty Module** (3 days)
   - [ ] Bootstrap uncertainty estimation
   - [ ] Poisson uncertainty propagation
   - [ ] ROI uncertainty calculation
   - [ ] Visualization tools

3. **Benchmarking Suite** (4 days)
   - [ ] Comparison framework (baseline vs Wiener vs EM)
   - [ ] Time-to-target accuracy measurement
   - [ ] Chi-squared vs iteration plots
   - [ ] Flux efficiency calculations
   - [ ] Automated test suite

**Deliverables**:
- `adaptive/reconstructors/em_reconstructor.py` - Complete
- `adaptive/uncertainty.py` - Complete
- `adaptive/evaluation.py` - Complete
- Comprehensive benchmarks
- Performance report

---

### Phase 3: Adaptive Kernel Selection (3-4 weeks)

**Goals**:
- Dynamic kernel selection
- Information gain estimation
- Closed-loop simulation

**Tasks**:

1. **Kernel Manager** (4 days)
   - [ ] Kernel library implementation
   - [ ] Selection strategies (uncertainty, info gain)
   - [ ] Thompson sampling
   - [ ] Unit tests

2. **Adaptive Controller** (5 days)
   - [ ] Online reconstruction with kernel updates
   - [ ] Event stream processing
   - [ ] Kernel history tracking
   - [ ] Integration with EM reconstructor

3. **Information Gain Estimator** (4 days)
   - [ ] Expected information gain calculation
   - [ ] Fisher information approximation
   - [ ] ROI-weighted information
   - [ ] Computational optimization

4. **Closed-Loop Simulation** (5 days)
   - [ ] Simulated measurement loop
   - [ ] Adaptive vs fixed kernel comparison
   - [ ] Time-to-target experiments
   - [ ] Sensitivity analysis

5. **Real-Time Interface** (Optional, 3 days)
   - [ ] ZMQ/REST API for kernel updates
   - [ ] Acquisition system integration
   - [ ] Latency testing

**Deliverables**:
- `adaptive/kernel_manager.py` - Complete
- `adaptive/adaptive_controller.py` - Complete
- Adaptive simulation examples
- Performance vs fixed kernel comparison
- Real-time interface (if implemented)

---

### Phase 4: Advanced Methods & Validation (3-4 weeks)

**Goals**:
- Kalman filter (if time permits)
- Bayesian optimization for kernel design
- Real data validation
- Publication-quality results

**Tasks**:

1. **Kalman Filter** (Optional, 7 days)
   - [ ] State-space formulation
   - [ ] EKF/UKF implementation
   - [ ] Comparison with EM

2. **Kernel Optimization** (5 days)
   - [ ] Bayesian optimization framework
   - [ ] Sample-specific kernel design
   - [ ] Multi-objective optimization
   - [ ] Case studies (Fe, Ta, etc.)

3. **Real Data Validation** (7 days)
   - [ ] Convert real measured data to event format
   - [ ] Apply reconstructors
   - [ ] Compare with traditional analysis
   - [ ] Error analysis

4. **Documentation & Publishing** (5 days)
   - [ ] API documentation
   - [ ] User guide
   - [ ] Example gallery
   - [ ] Methods paper draft

**Deliverables**:
- All optional methods implemented
- Real data case studies
- Complete documentation
- Publication draft

---

## 5. Evaluation Metrics

### 5.1 Reconstruction Quality

1. **Chi-Squared**:
   ```python
   def chi_squared(S_recon, S_true, uncertainty):
       return np.sum(((S_recon - S_true) / uncertainty) ** 2)
   ```

2. **Normalized RMSE**:
   ```python
   def normalized_rmse(S_recon, S_true):
       rmse = np.sqrt(np.mean((S_recon - S_true) ** 2))
       return rmse / (S_true.max() - S_true.min())
   ```

3. **ROI-Specific MSE**:
   ```python
   def roi_mse(S_recon, S_true, roi_mask):
       return np.mean((S_recon[roi_mask] - S_true[roi_mask]) ** 2)
   ```

4. **Feature Detection Accuracy**:
   - Bragg edge position error
   - Resonance peak position error
   - Edge width error

### 5.2 Efficiency Metrics

1. **Time-to-Target Accuracy**:
   ```python
   def time_to_target(reconstruction_history, target_chi2):
       for t, result in enumerate(reconstruction_history):
           if result.chi2 < target_chi2:
               return t
       return np.inf
   ```

2. **Flux Efficiency**:
   ```python
   def flux_efficiency(kernel):
       # Ratio of effective to nominal flux
       n_frames = len(kernel)
       total_time = sum(kernel)
       return n_frames * nominal_pulse_period / total_time
   ```

3. **Information Rate**:
   ```python
   def information_rate(spectrum, uncertainty, measurement_time):
       # Bits of information per unit time
       info = -np.sum(spectrum * np.log2(spectrum))  # Entropy
       return info / measurement_time
   ```

### 5.3 Adaptive Performance

1. **Kernel Efficiency**:
   ```python
   def kernel_efficiency(adaptive_history, fixed_history, target_accuracy):
       t_adaptive = time_to_target(adaptive_history, target_accuracy)
       t_fixed = time_to_target(fixed_history, target_accuracy)
       return t_fixed / t_adaptive  # >1 means adaptive is better
   ```

2. **ROI Coverage**:
   ```python
   def roi_coverage(spectrum, uncertainty, roi_ranges, threshold):
       # Fraction of ROI bins with uncertainty below threshold
       n_good = 0
       n_total = 0
       for tof_min, tof_max in roi_ranges:
           mask = (tof_bins >= tof_min) & (tof_bins <= tof_max)
           n_good += np.sum(uncertainty[mask] < threshold)
           n_total += np.sum(mask)
       return n_good / n_total
   ```

### 5.4 Robustness

1. **Noise Sensitivity**:
   - Test with varying background levels
   - Test with varying flux levels

2. **Kernel Robustness**:
   - Performance across different kernel choices
   - Sensitivity to kernel timing errors

3. **Feature Sensitivity**:
   - Performance on different materials
   - Performance with/without Cd filter

---

## 6. File Structure

### 6.1 New Directory Structure

```
frame_overlap/
├── src/frame_overlap/
│   ├── adaptive/                    # NEW MODULE
│   │   ├── __init__.py
│   │   ├── event_data.py            # Event structures
│   │   ├── event_processor.py       # Event→histogram conversion
│   │   ├── reconstructors/
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # Base interface
│   │   │   ├── baseline.py          # Separated frames
│   │   │   ├── wiener_event.py      # Wiener for events
│   │   │   ├── em_reconstructor.py  # EM algorithm
│   │   │   ├── kalman_reconstructor.py  # Kalman filter
│   │   │   └── ml_reconstructor.py  # Neural net (future)
│   │   ├── kernel_manager.py        # Kernel library & selection
│   │   ├── adaptive_controller.py   # Adaptive logic
│   │   ├── uncertainty.py           # Uncertainty quantification
│   │   ├── simulation.py            # Synthetic data
│   │   └── evaluation.py            # Metrics
│   ├── data_class.py                # EXISTING (may extend)
│   ├── reconstruct.py               # EXISTING (may extend)
│   ├── workflow.py                  # EXISTING (may extend)
│   └── ...
├── tests/
│   ├── adaptive/                    # NEW TESTS
│   │   ├── test_event_data.py
│   │   ├── test_simulation.py
│   │   ├── test_reconstructors.py
│   │   ├── test_kernel_manager.py
│   │   ├── test_adaptive_controller.py
│   │   └── test_benchmarks.py
│   └── ...
├── notebooks/
│   ├── adaptive/                    # NEW NOTEBOOKS
│   │   ├── 01_event_mode_intro.ipynb
│   │   ├── 02_baseline_comparison.ipynb
│   │   ├── 03_em_reconstruction.ipynb
│   │   ├── 04_adaptive_kernels.ipynb
│   │   ├── 05_benchmarking.ipynb
│   │   └── 06_real_data_validation.ipynb
│   └── ...
├── data/
│   ├── synthetic/                   # NEW SYNTHETIC DATA
│   │   ├── iron_events_baseline.h5
│   │   ├── iron_events_overlap.h5
│   │   └── ...
│   └── measured/
│       └── ...
└── docs/
    ├── adaptive_reconstruction.md   # NEW DOCUMENTATION
    ├── api_adaptive.rst
    └── ...
```

### 6.2 Integration with Existing Code

**Minimal Changes to Existing Code**:
- Keep existing `Data`, `Reconstruct`, `Workflow` classes unchanged
- New `adaptive` module is self-contained
- Bridge classes for interoperability:

```python
# In adaptive/__init__.py

def event_dataset_to_data(event_dataset: EventDataset) -> Data:
    """Convert EventDataset to Data for use with existing workflows."""
    hist = event_dataset.to_histogram()
    # Create Data object from histogram
    # ...

def data_to_event_dataset(data: Data, kernel: np.ndarray) -> EventDataset:
    """Simulate event-mode data from histogram Data object."""
    # ...
```

### 6.3 Configuration Files

```yaml
# configs/adaptive_default.yaml
reconstruction:
  method: 'em'
  max_iterations: 50
  convergence_threshold: 1e-4

kernel:
  update_interval: 1000  # events
  selection_strategy: 'uncertainty_sampling'
  library:
    - [0]           # Baseline
    - [0, 25]       # 2-frame
    - [0, 12, 12]   # 3-frame

roi:
  bragg_edges:
    - [3000, 3500]   # µs
    - [7000, 8000]
  resonances:
    - [15000, 18000]

simulation:
  flux: 5e6         # n/cm²/s
  duration: 0.5     # hours
  noise_level: 0.1
```

---

## 7. Testing Strategy

### 7.1 Unit Tests
- Each module has `test_<module>.py`
- Test coverage > 80%
- Focus on edge cases, error handling

### 7.2 Integration Tests
- End-to-end workflows
- Simulation → Reconstruction → Analysis
- Comparison with existing code

### 7.3 Validation Tests
- Synthetic data with known ground truth
- Real measured data comparisons
- Statistical tests (chi-squared, KS test)

### 7.4 Performance Tests
- Time-to-target benchmarks
- Memory usage profiling
- Scalability tests (1k to 1M events)

---

## 8. Success Criteria

### Minimum Viable Product (MVP)
- [ ] Event-mode data handling
- [ ] Baseline (separated frames) reconstruction
- [ ] Fixed-kernel Wiener reconstruction for events
- [ ] EM reconstruction
- [ ] Simulation framework
- [ ] Basic benchmarking
- [ ] Time-to-target < 2x baseline for fixed overlap

### Full Success
- [ ] All MVP criteria
- [ ] Adaptive kernel selection working
- [ ] Time-to-target < 0.5x baseline (2x speedup) for adaptive
- [ ] Uncertainty quantification validated
- [ ] Real data case studies
- [ ] Publication-quality documentation

### Stretch Goals
- [ ] Kalman filter implementation
- [ ] Bayesian kernel optimization
- [ ] ML-based reconstruction
- [ ] Real-time interface to acquisition system
- [ ] Multi-detector support

---

## 9. Risk Assessment & Mitigation

### Technical Risks

1. **Risk**: EM convergence issues with high overlap
   - **Mitigation**: Multiple initialization strategies, regularization

2. **Risk**: Adaptive kernel selection overhead too high
   - **Mitigation**: Efficient approximations, caching, update less frequently

3. **Risk**: Uncertainty estimation inaccurate
   - **Mitigation**: Bootstrap validation, comparison with analytical propagation

4. **Risk**: Real data has artifacts not in simulation
   - **Mitigation**: Robust noise models, outlier rejection, conservative assumptions

### Scientific Risks

1. **Risk**: Adaptive approach may not outperform fixed optimal kernel
   - **Mitigation**: Focus on scenarios where no single fixed kernel is optimal (multiple ROIs, unknown features)

2. **Risk**: Method limitations unclear without extensive testing
   - **Mitigation**: Systematic sensitivity analysis, document failure modes

### Project Risks

1. **Risk**: Scope too large for timeline
   - **Mitigation**: Phased approach, focus on MVP first, defer advanced methods

2. **Risk**: Integration difficulties with existing codebase
   - **Mitigation**: Self-contained module, clear interfaces, minimal dependencies

---

## 10. Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1: Foundation | 2-3 weeks | Event data, simulation, baseline & Wiener |
| Phase 2: Iterative | 2-3 weeks | EM reconstruction, uncertainty, benchmarks |
| Phase 3: Adaptive | 3-4 weeks | Kernel selection, adaptive controller, closed-loop sim |
| Phase 4: Advanced | 3-4 weeks | Kalman, optimization, real data, docs |
| **Total** | **10-14 weeks** | **Full system with real data validation** |

---

## 11. Next Steps (Immediate Actions)

1. **Review & Approve Plan** (1 day)
   - Stakeholder review
   - Prioritize features
   - Adjust timeline

2. **Setup Development Environment** (1 day)
   - Create `adaptive` module structure
   - Setup test framework
   - Configure CI/CD

3. **Start Phase 1, Task 1** (Day 3)
   - Implement `NeutronEvent` class
   - Write unit tests
   - Document data format

---

## 12. References & Resources

### Key Papers
1. **Frame Overlap in TOF**: [To be added based on your domain]
2. **EM for Spectral Deconvolution**: Veklerov & Llacer (1987)
3. **Bayesian Experimental Design**: Chaloner & Verdinelli (1995)
4. **Active Learning for Spectroscopy**: [Recent ML papers]

### Software Tools
- `numpy`, `scipy`: Numerical computation
- `lmfit`: Parameter optimization
- `scikit-learn`: ML utilities (if needed)
- `h5py`: Event data storage
- `pytest`: Testing
- `jupyter`: Notebooks

### Existing Codebase
- Build on `Data`, `Reconstruct`, `Workflow` classes
- Reuse convolution, Poisson sampling, Wiener filter logic
- Leverage parameter sweep framework from `groupby`

---

## Appendix A: Example Usage

### Example 1: Fixed-Kernel Reconstruction

```python
from frame_overlap.adaptive import EventDataset, WienerEventReconstructor

# Load event data
events = EventDataset.from_hdf5('iron_events.h5')

# Reconstruct with fixed kernel
recon = WienerEventReconstructor(tof_range=(1000, 20000), n_bins=1000)
result = recon.reconstruct(events, noise_power=0.01)

# Plot
result.plot()
print(f"Chi-squared: {result.chi2:.2f}")
```

### Example 2: EM Reconstruction

```python
from frame_overlap.adaptive import EMReconstructor

recon = EMReconstructor(tof_range=(1000, 20000), n_bins=1000)
result = recon.reconstruct(
    events,
    max_iterations=50,
    prior='smooth',
    convergence_threshold=1e-4
)

# Compare with true spectrum
result.plot_comparison(true_spectrum)
```

### Example 3: Adaptive Kernel Selection

```python
from frame_overlap.adaptive import (
    AdaptiveController,
    KernelManager,
    EMReconstructor
)

# Setup
recon = EMReconstructor(tof_range=(1000, 20000), n_bins=1000)
kernel_mgr = KernelManager()
controller = AdaptiveController(
    recon,
    kernel_mgr,
    roi_ranges=[(3000, 3500), (7000, 8000)],  # Bragg edge ROIs
    update_interval=1000
)

# Simulate measurement with adaptive kernel
for event in event_stream:
    new_kernel = controller.process_event(event)
    if new_kernel is not None:
        print(f"Kernel updated at event {controller.event_count}")
        # In real system: send new_kernel to acquisition system

# Get final result
final_result = controller.reconstructor.get_current_result()
controller.plot_kernel_history()
```

---

## Appendix B: Kernel Design Principles

### Kernel Library Design

Good kernels balance:
1. **Flux efficiency**: More frames = higher effective flux
2. **Resolvability**: Frames too close → assignment ambiguity
3. **ROI coverage**: Different TOF regions need different overlap

**Heuristics**:
- **Fast neutrons (low TOF)**: Less overlap (higher resolution needed)
- **Slow neutrons (high TOF)**: More overlap (higher flux needed)
- **Bragg edges**: Medium overlap, focus on edge region
- **Resonances**: High overlap (peaks are narrow, need statistics)

**Example Kernels**:
```python
kernels = {
    'no_overlap': [0],                      # Baseline
    'low_overlap': [0, 40],                 # 2 frames, 40 ms spacing
    'medium_overlap': [0, 25],              # Standard
    'high_overlap': [0, 12, 12, 12],        # 4 frames, tight
    'fibonacci': [0, 5, 8, 13, 21],        # Logarithmic spacing
    'bragg_optimized': [0, 20, 15],        # Tuned for Bragg edges
    'resonance_optimized': [0, 8, 8, 8, 8], # High flux for resonances
}
```

---

**END OF IMPLEMENTATION PLAN**

This plan provides a comprehensive, actionable roadmap for implementing adaptive frame overlap TOF reconstruction. The phased approach allows for incremental progress and validation, while the modular architecture keeps the new code separate from the existing, well-tested codebase.

**Key Recommendations**:
1. Start with Phase 1 to build solid foundations
2. Prioritize EM reconstruction (Rank 3) before adaptive selection
3. Focus on interpretable, scientifically rigorous methods
4. Defer ML approaches until classical methods are validated
5. Maintain extensive benchmarking throughout development

The MVP (Phases 1-2) should provide immediate scientific value even before adaptive selection is implemented.
