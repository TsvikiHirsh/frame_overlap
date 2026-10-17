# Kernel Generation and UI Updates

## Summary

This document describes two major updates:
1. **nbragg fit UI reorganization** - Improved tri-state vary parameter controls
2. **Kernel generation modes** - Added random, equal, and blue_noise kernel generation to overlap method

---

## 1. nbragg Fit UI Reorganization

### Improved Vary Parameters UI

**Previous implementation:**
- Two-column layout with enable checkbox + True/False radio per parameter
- Verbose and took up too much space

**New implementation:**
- Single tri-state radio button per parameter with three options:
  - "None (not set)" - Parameter not passed to nbragg (uses model default)
  - "True (vary)" - Parameter explicitly set to vary during fit
  - "False (fixed)" - Parameter explicitly fixed during fit

### Benefits

✅ **Cleaner** - Much more compact, single radio button instead of checkbox + radio
✅ **Clearer** - Three states visible at once, easier to understand
✅ **Correct** - Properly supports nbragg's None/True/False semantics

### UI Code

```python
def tristate_radio(label, default_index, key, help_text):
    """Create a tri-state radio button that can be None, True, or False"""
    options = ["None (not set)", "True (vary)", "False (fixed)"]
    selection = st.radio(label, options=options, index=default_index,
                        horizontal=True, key=key, help=help_text)
    if "None" in selection:
        return None
    elif "True" in selection:
        return True
    else:
        return False

# Usage
vary_background = tristate_radio(
    "vary_background",
    default_index=1,  # Default to True
    key="vary_bg_tristate",
    help_text="Control background parameter"
)
```

---

## 2. Kernel Generation Modes

### New Feature: Auto-Generate Kernels

Added support for passing an integer as `kernel` parameter to automatically generate frame timing sequences.

### Usage

**Equal spacing:**
```python
data.overlap(kernel=5, total_time=50, mode='equal')
# Generates: [0.0, 10.0, 20.0, 30.0, 40.0]
```

**Random spacing:**
```python
data.overlap(kernel=5, total_time=50, mode='random', kernel_seed=42)
# Generates: [0.0, 18.73, 29.93, 36.60, 47.54] (sorted)
```

**Blue noise spacing:**
```python
data.overlap(kernel=5, total_time=50, mode='blue_noise', kernel_seed=42)
# Generates: [0.0, 9.80, 22.80, 34.21, 47.54] (more uniform than random)
```

**Manual kernel (existing behavior):**
```python
data.overlap(kernel=[0, 8, 22, 18], total_time=50, mode='extend')
# Uses explicit frame positions
```

### Implementation Details

#### New `_generate_kernel()` Method

Added helper method in `data_class.py`:

```python
def _generate_kernel(self, n_frames, total_time, mode, seed=None):
    """
    Generate a kernel (frame timing sequence) based on the specified mode.

    Parameters
    ----------
    n_frames : int
        Number of frames to generate
    total_time : float
        Total time window in milliseconds
    mode : str
        Generation mode: 'random', 'equal', or 'blue_noise'
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    numpy.ndarray
        Array of frame start times in milliseconds
    """
```

**Equal mode:**
- Uses `np.linspace(0, total_time, n_frames, endpoint=False)`
- Perfectly evenly spaced frames

**Random mode:**
- Generates random positions in [0, total_time]
- Sorts and ensures first frame at t=0
- Fully random distribution

**Blue noise mode:**
- Uses Mitchell's best-candidate algorithm
- Generates multiple candidates per point
- Chooses candidate with maximum minimum distance to existing points
- Results in more uniform spacing than pure random
- Better for avoiding clustering

#### Modified `overlap()` Method

Updated signature:
```python
def overlap(self, kernel, total_time=None, freq=None, bin_width=10,
           poisson_seed=None, mode='superimpose', kernel_seed=None):
```

New logic:
```python
# Generate kernel if integer is provided
if isinstance(kernel, int):
    n_frames = kernel
    if mode not in ['random', 'equal', 'blue_noise']:
        raise ValueError(f"When kernel is an integer, mode must be 'random', 'equal', or 'blue_noise', got '{mode}'")
    if total_time is None:
        raise ValueError("total_time must be specified when generating kernel from integer")

    kernel = self._generate_kernel(n_frames, total_time, mode, kernel_seed)
    print(f"Generated {mode} kernel with {n_frames} frames: {kernel}")
    # After kernel generation, switch to 'extend' mode for actual overlap
    mode = 'extend'
```

### Test Results

Created comprehensive test suite in `tests/test_kernel_generation.py`:

**Test 1: Equal spacing ✅**
```
Generated kernel: [0.0, 10.0, 20.0, 30.0, 40.0]
✅ PASS: Equal spacing is correct
```

**Test 2: Random ✅**
```
Generated kernel: [0.0, 18.73, 29.93, 36.60, 47.54]
✅ PASS: Correct number of frames (5)
✅ PASS: First frame at t=0
✅ PASS: Frames are sorted
✅ PASS: All frames within [0, 50)
✅ PASS: Reproducible with same seed
```

**Test 3: Blue noise ✅**
```
Generated kernel: [0.0, 9.80, 22.80, 34.21, 47.54]
✅ PASS: Blue noise has lower variance than random (more uniform)
   Inter-frame distance variance: 1.98
   Random inter-frame distance variance: 18.85
```

**Test 4: Visual comparison ✅**
```
EQUAL (10 frames):
  Mean spacing: 5.00 ms, Std spacing: 0.00 ms
  Min: 5.00 ms, Max: 5.00 ms

RANDOM (10 frames):
  Mean spacing: 5.28 ms, Std spacing: 4.05 ms
  Min: 0.00 ms, Max: 11.21 ms

BLUE_NOISE (10 frames):
  Mean spacing: 5.28 ms, Std spacing: 1.30 ms
  Min: 3.69 ms, Max: 7.09 ms
```

**Key observation:** Blue noise has much lower variance than random (1.30 vs 4.05), demonstrating more uniform spacing.

---

## 3. Streamlit App Updates

### Updated Frame Overlap Controls

**Spacing Type:**
- Now includes "Blue Noise" option in addition to "Equal" and "Random"

**Kernel Seed:**
- For Random and Blue Noise modes, users can:
  - Enable seed for reproducibility (checkbox)
  - Set seed value (number input, default: 42)

**Pipeline Integration:**
```python
if apply_overlap:
    # Handle both manual kernel and auto-generated kernel
    if kernel_mode == "Manual":
        data.overlap(kernel=kernel_absolute, total_time=total_time)
    else:  # Auto-generate
        data.overlap(kernel=kernel, total_time=total_time,
                    mode=kernel_gen_mode, kernel_seed=kernel_seed)
    st.sidebar.success(f"✓ Overlap ({n_frames} frames)")
```

### UI Flow

1. **Choose kernel input mode:** Auto-generate or Manual
2. **If Auto-generate:**
   - Select spacing type: Equal, Random, or Blue Noise
   - Choose number of frames (1-20 slider)
   - For Random/Blue Noise: optionally set seed
3. **If Manual:**
   - Enter comma-separated time differences
   - App converts to absolute frame positions

---

## Modified Files

### Core Library
- **src/frame_overlap/data_class.py**
  - Lines 273-336: Added `_generate_kernel()` method
  - Lines 338-446: Updated `overlap()` method signature and logic

### Tests
- **tests/test_kernel_generation.py** (new file)
  - Comprehensive tests for all three generation modes
  - Validation of properties (sorted, first=0, within range, reproducibility)
  - Variance comparison between modes

### Streamlit App
- **streamlit_app.py**
  - Lines 831-892: Redesigned vary parameters UI (tri-state radio buttons)
  - Lines 637-667: Updated overlap controls (added Blue Noise, kernel seed)
  - Lines 956-961: Updated overlap pipeline call to support both modes

### Documentation
- **NBRAGG_UI_REORGANIZATION.md** - Updated with tri-state radio implementation
- **KERNEL_GENERATION_AND_UI_UPDATES.md** - This file

---

## Examples

### Example 1: Equal Spacing (Python API)
```python
from frame_overlap import Data

data = Data('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)
data.convolute_response(200.0)
data.poisson_sample(flux=1e6, freq=20, measurement_time=480, seed=42)

# Generate 5 equally spaced frames in 50 ms window
data.overlap(kernel=5, total_time=50, mode='equal')

print(data.kernel)  # [0.0, 10.0, 20.0, 30.0, 40.0]
```

### Example 2: Random Spacing (Python API)
```python
# Generate 10 randomly spaced frames with reproducible seed
data.overlap(kernel=10, total_time=50, mode='random', kernel_seed=123)

print(data.kernel)  # [0.0, 4.32, 12.18, 19.45, ...]  (sorted, random)
```

### Example 3: Blue Noise (Python API)
```python
# Generate 10 blue noise frames (more uniform than random)
data.overlap(kernel=10, total_time=50, mode='blue_noise', kernel_seed=456)

print(data.kernel)  # [0.0, 5.12, 10.83, 16.24, ...]  (more evenly spaced)
```

### Example 4: Streamlit App Workflow
1. Load iron powder data
2. Apply instrument response
3. Apply Poisson sampling
4. **Frame Overlap:**
   - Kernel Input Mode: Auto-generate
   - Spacing Type: Blue Noise
   - Number of Frames: 5
   - Use seed: Yes, seed=42
5. Apply Wiener reconstruction
6. Fit with nbragg using tri-state vary parameters

---

## Benefits

### 1. Kernel Generation

✅ **Convenience** - No need to manually calculate frame positions
✅ **Reproducibility** - Seed support for random/blue_noise modes
✅ **Flexibility** - Three distribution types for different use cases
✅ **Quality** - Blue noise provides better coverage than pure random

### 2. UI Improvements

✅ **Compact** - Tri-state radio much cleaner than checkbox + radio
✅ **Clear** - All three states visible at once
✅ **Correct** - Properly implements nbragg's None/True/False semantics
✅ **User-friendly** - Streamlit integration with seed control

---

## Use Cases

### When to use each kernel generation mode:

**Equal:**
- Baseline measurements
- Maximum uniformity needed
- Predictable reconstruction performance

**Random:**
- Testing reconstruction robustness
- Simulating jittered timing
- Avoiding systematic artifacts

**Blue Noise:**
- Better spatial/temporal coverage than random
- Avoiding clustering while maintaining randomness
- Optimal sampling theory applications
- Adaptive measurement strategies

### When to use each vary parameter state:

**None (not set):**
- Let nbragg model decide default behavior
- Don't know if parameter should vary
- Using standard model configuration

**True (vary):**
- Explicitly want parameter to vary during fit
- Fitting for this parameter value
- Need model flexibility

**False (fixed):**
- Know parameter value from independent measurement
- Want to constrain fit
- Reduce number of free parameters

---

## Technical Notes

### Blue Noise Algorithm

The blue noise implementation uses Mitchell's best-candidate algorithm:

1. Start with first point at t=0
2. For each subsequent point:
   - Generate N random candidates (N increases with number of placed points)
   - For each candidate, calculate minimum distance to all existing points
   - Choose candidate with maximum minimum distance (best-candidate)
3. Sort final kernel

This produces a Poisson disk distribution - points are randomly distributed but maintain minimum separation, avoiding clustering.

### Backward Compatibility

All existing code continues to work:
- Passing list/array as kernel works as before
- Manual mode in Streamlit app unchanged
- Default mode='superimpose' unchanged
- No breaking changes to API

### Performance

- Equal mode: O(n) - simple linspace
- Random mode: O(n log n) - sort dominates
- Blue noise: O(n² × m) where m is candidates per point (typically 20-200)
  - Fast enough for practical n_frames (1-20)
  - May be slow for very large n_frames (>100)

---

## Future Enhancements

Potential improvements:
1. **Adaptive kernels** - Generate kernel based on Bragg edge positions
2. **Optimization-based** - Find optimal kernel for specific measurement goals
3. **Custom distributions** - Support user-defined probability distributions
4. **Visualization** - Plot kernel distributions in Streamlit app
5. **Kernel library** - Pre-computed optimal kernels for common scenarios

---

**Date**: 2025-11-18
**Status**: ✅ Implemented and tested
**Branch**: `claude/bragg-edge-adaptive-chopper-01NTdCs5m5xCwqaKVyjq4LtB`
