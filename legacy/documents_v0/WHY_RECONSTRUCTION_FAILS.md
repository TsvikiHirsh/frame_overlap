# Why Your Reconstruction Isn't Working

**Date**: 2025-11-10
**Issue**: Poor reconstruction quality even for "non-overlapping" frames
**Root Cause**: Fundamental mismatch between synthetic overlap and real FOBI physics

---

## The Problem You're Facing

You said: *"I can't show a good reconstruction, even for 2 frames that are not even overlapping"*

But actually, you have **TWO** problems:

1. **Your "non-overlapping" frames ARE overlapping** (misunderstanding of frame period)
2. **Your overlap simulation doesn't match real FOBI physics** (fundamental algorithm issue)

---

## Problem 1: Misunderstanding Frame Overlap

### What You Thought:

With `kernel=[0, 25]` at 20 Hz:
- Frame 1 starts at 0ms
- Frame 2 starts at 25ms
- They don't overlap (if frames are short pulses)

### Reality:

At 20 Hz, frame period = **50ms**:
```
Frame 1: [0ms, 50ms]    ████████████████████████████
Frame 2: [25ms, 75ms]            ████████████████████████████
                              ▲        ▲
                            25ms     50ms
                           OVERLAP = 25ms (50% of frame!)
```

**For truly non-overlapping frames at 20 Hz**: Use `kernel=[0, 50]` or larger!

---

## Problem 2: Wrong Overlap Simulation (CRITICAL!)

### Real FOBI (Neutron Instrument):

```
Single neutron pulse from source
         ↓
Chopper with multiple openings (θ₁, θ₂, ...)
         ↓
SAME neutrons measured through MULTIPLE windows
         ↓
Detector sees SUPERPOSITION within FIXED time window
```

**Math**: `observed[t] = Σᵢ signal[t - tᵢ]` for `t ∈ [0, T]`

**Key properties**:
- ✅ Time window is FIXED: [0, T]
- ✅ Multiple contributions SUPERIMPOSE
- ✅ Counts INCREASE in overlap regions
- ✅ Deconvolution recovers original signal

### Your Current Approach (Synthetic):

```
Complete measured signal [0, 24ms]
         ↓
Extend time window to [0, 50ms]
         ↓
Add shifted copies sequentially
         ↓
Creates signal with EXTENDED time range
```

**What happens**:
```python
# Your code does something like:
extended_time = [0, 50ms]          # EXTENDS from 24ms to 50ms
extended_signal = zeros(5000)      # Zero padding
extended_signal[0:2400] = original # Place original
extended_signal[2500:4900] += original # Add shifted copy
```

**Result**:
- ❌ Time window EXTENDS from 24ms → 50ms
- ❌ Length changes: 2400 points → 5000 points
- ❌ Zero padding creates boundary artifacts
- ❌ Reference signal (24ms) doesn't match overlapped signal (50ms)
- ❌ Deconvolution math doesn't work!

---

## Why This Breaks Reconstruction

### The Math Doesn't Match:

**Real FOBI deconvolution**:
```
Given: observed[0:T] = signal[0:T] * kernel
Solve for: signal[0:T]
```

**Your current approach**:
```
Given: observed[0:2T] = extended_signal with padding
Reference: signal[0:T] (shorter!)
Solve for: ??? (dimensions don't match!)
```

### Statistical Properties Are Wrong:

**Real overlap**:
- Mean counts INCREASE (superposition adds counts)
- Poisson statistics maintained
- SNR can improve with multiple measurements

**Your approach**:
- Mean counts DECREASE (normalization + padding)
- Poisson statistics broken (zero padding)
- Boundary artifacts dominate

---

## How to Fix It

### Option 1: Fix the `overlap()` Method ⚡ (Recommended)

Modify `data.overlap()` to keep SAME time window:

```python
def overlap_correct(signal, kernel_ms, bin_width_us=10):
    """
    Simulate real frame overlap by superimposing within SAME window.

    This matches real FOBI measurement physics.
    """
    n_bins = len(signal)
    observed = np.zeros(n_bins, dtype=float)

    for delay_ms in kernel:
        delay_bins = int((delay_ms * 1000) / bin_width_us)

        if delay_bins == 0:
            # No shift
            observed += signal
        elif delay_bins < n_bins:
            # Shift and add (within same window)
            observed[delay_bins:] += signal[:n_bins - delay_bins]
        # else: delay too large, no contribution

    observed /= len(kernel)  # Normalize
    return observed
```

**Key differences**:
- ✅ Output length = input length (no extension!)
- ✅ Superposition within same window
- ✅ Proper boundary handling
- ✅ Matches real FOBI physics

### Option 2: Use Longer Measurement Times

If you want to simulate multi-frame FOBI:

1. Measure for **longer duration** (e.g., 100ms for 20Hz = 2 full frames)
2. Your signal naturally contains information from multiple frames
3. Apply overlap operation within this longer window
4. Deconvolution separates frame contributions

**Example**:
```python
# Measure for 100ms (2 frames at 20Hz)
data = Data('signal.csv', 'openbeam.csv',
            flux=5e6, duration=0.5, freq=20)
data.convolute_response(200, bin_width=10)

# Extend measurement time to capture multiple frames
data.poisson_sample(flux=1e6, freq=20, measurement_time=100)  # 100ms

# Now apply overlap within this window
data.overlap(kernel=[0, 25], total_time=100)  # Both frames fit in 100ms
```

### Option 3: Understand What You're Actually Trying to Do

**Clarify your goal**:

A) **Simulate FOBI-style chopper measurements**?
   → Use Option 1 (fix overlap to superimpose)
   → Match real neutron instrument physics

B) **Evaluate benefit of frame overlap for flux enhancement**?
   → Use Option 2 (longer measurements)
   → Show how overlap improves statistics

C) **Test deconvolution algorithms on synthetic data**?
   → Need to ensure synthetic data matches expected structure
   → Current approach creates incorrect boundary conditions

D) **Something else**?
   → Let's discuss what you're trying to achieve!

---

## What Real FOBI Data Looks Like

In a real FOBI measurement:

```
Single neutron pulse from spallation source
    ↓
Sample: iron powder (Bragg edges in TOF spectrum)
    ↓
POLDI chopper: 8 slits at angles [0°, 9.36°, 21.47°, ...]
    ↓
Each slit opening allows neutrons through at different times
    ↓
Detector measures SUPERPOSITION of 8 time-shifted signals
    ↓
Observed[t] = Σᵢ Signal[t - τᵢ] for τᵢ from chopper angles
    ↓
Wiener deconvolution separates contributions
    ↓
Recovered signal has 8× better statistics!
```

**Key insight**: The SAME neutron pulse is measured MULTIPLE times through different chopper slits, all within ONE measurement cycle.

Your current code is trying to simulate this, but the `overlap()` method extends time instead of superimposing within the same window.

---

## Expected Reconstruction Quality

Once you fix the overlap method:

| Configuration | Overlap | Expected χ²/dof | Quality |
|--------------|---------|-----------------|---------|
| `[0]` | 0% | < 0.001 | Perfect (identity) |
| `[0, 50]` | 0% | < 0.01 | Excellent |
| `[0, 25]` | 50% | 0.1 - 1.0 | Good |
| `[0, 12]` | 76% | 1 - 10 | Fair |
| `[0, 5]` | 90% | > 10 | Poor |

**Your current results** ([0, 25] gives χ²/dof ~22) indicate the overlap simulation is fundamentally wrong, not just that the deconvolution is hard.

---

## Next Steps

1. **Fix the `overlap()` method** to superimpose within same window
2. **Test with corrected overlap** - should get χ²/dof < 1 for non-overlapping
3. **Validate against real FOBI data** if available
4. **Document what you're actually simulating** for future reference

Let me know which option you want to pursue, and I can help you implement it!

---

**End of Document**
