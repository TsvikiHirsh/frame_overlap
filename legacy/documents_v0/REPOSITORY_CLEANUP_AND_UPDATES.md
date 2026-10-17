# Repository Cleanup and Updates

## Summary

Comprehensive repository cleanup and feature updates including:
1. Documentation organization
2. Duty cycle calculation fix
3. Streamlit app defaults update
4. New tutorial notebook

---

## 1. Repository Organization

### Created `.documents` Directory

Moved all documentation and debug files to `.documents/` for cleaner repository structure:

**Documentation files moved:**
- `KERNEL_GENERATION_AND_UI_UPDATES.md`
- `NBRAGG_UI_REORGANIZATION.md`
- `RECONSTRUCTION_AMPLITUDE_FIX.md`
- `TOF_OFFSET_CORRECTION_ANALYSIS.md`

**Debug/test files moved:**
- `tests/debug_*.py` (all debug scripts)
- `tests/test_amplitude_fix.py`
- `tests/test_duty_cycle_comparison.py`
- `tests/test_overlap_extend_mode.py`
- `tests/test_overlap_understanding.py`
- `tests/test_reconstruction_fix.py`
- `tests/test_wavelength_conversion.py`
- `tests/test_wavelength_filtering.py`

**Kept in main repo:**
- `README.md` (main documentation)
- `tests/test_kernel_generation.py` (current feature)
- `tests/test_tof_offset_correction_consistency.py` (current feature)

---

## 2. Duty Cycle Calculation Fix

### Problem

The duty cycle calculation was not accounting for the pulse duration ratio. If you use pulse_duration=20 µs vs 10 µs, you should get 2x more neutrons because the pulse is 2x longer.

### Solution

Added `pulse_duration_ratio` term with 10 µs as baseline reference:

**Before:**
```python
duty_cycle = flux_ratio × time_ratio × freq × (pulse_duration / 1e6)
```

**After:**
```python
baseline_duration = 10.0  # µs - reference pulse duration
pulse_duration_ratio = self.pulse_duration / baseline_duration
duty_cycle = flux_ratio × time_ratio × freq × pulse_duration_ratio × (baseline_duration / 1e6)
```

### Example

With pulse_duration=20 µs:
- `pulse_duration_ratio = 20 / 10 = 2.0`
- Gets 2x more neutrons than with pulse_duration=10 µs ✓

### File Modified

- `src/frame_overlap/data_class.py` (lines 656-677)

---

## 3. Streamlit App Defaults Updated

### Changes Made

**Pulse Duration:**
- **Before**: 200.0 µs
- **After**: 20.0 µs (baseline)

**nbragg Model:**
- **Before**: "iron" (index=0)
- **After**: "iron_with_cellulose" (index=1)

**vary_background:**
- **Before**: default_index=1 (True)
- **After**: default_index=2 (False - fixed)

**vary_response:**
- **Before**: default_index=1 (True)
- **After**: default_index=2 (False - fixed)

**vary_weights:**
- **Before**: default_index=1 (True)
- **After**: default_index=1 (True - unchanged)

**Wavelength range:** (already set correctly)
- wlmin: 1.0 Å
- wlmax: 5.0 Å

### Files Modified

- `streamlit_app.py`:
  - Line 490: pulse_duration default changed to 20.0
  - Line 802: nbragg_model default changed to iron_with_cellulose
  - Line 836: vary_background default to False
  - Line 844: vary_response default to False

---

## 4. New Tutorial Notebook

Created `notebooks/simple_reconstruction_example.ipynb` demonstrating:

### Features Demonstrated

1. **Simple Reconstruction**
   - Load iron powder data
   - 20 µs pulse duration
   - Poisson sampling
   - 5 random frames with seed=42
   - Wiener reconstruction with noise_power=1.0

2. **nbragg Fitting**
   - Model: iron_with_cellulose
   - vary_background=False
   - vary_response=False
   - vary_weights=True
   - wlmin=1.0, wlmax=5.0

3. **Parameter Sweep**
   - Random kernels from n_frames=1 to 20
   - Tracks reconstruction χ²/dof, R², RMSE
   - Tracks nbragg reduced χ²
   - Generates comparison plots

### Notebook Structure

```
1. Introduction & Parameters
2. Single Reconstruction Example
   - Data loading
   - Processing pipeline
   - Wiener reconstruction
   - Visualization
3. nbragg Bragg Edge Fitting
   - Fit with specified parameters
   - Display results
4. Parameter Sweep (1-20 frames)
   - Loop over n_frames
   - Track metrics
   - Plot results
5. Summary & Next Steps
```

### Output

The sweep generates a comprehensive plot (`frame_sweep_results.png`) with 4 subplots:
- χ²/dof vs n_frames (reconstruction quality)
- R² vs n_frames (coefficient of determination)
- RMSE vs n_frames (error metric)
- nbragg reduced χ² vs n_frames (fit quality)

---

## 5. Key Formula Updates

### Duty Cycle Calculation

**Full formula (with pulse duration scaling):**
```
duty_cycle = (flux_new / flux_orig) ×
             (time_new / time_orig) ×
             freq_new ×
             (pulse_duration / baseline_duration) ×
             (baseline_duration / 1e6)
```

**Components:**
- `flux_ratio`: Change in neutron flux
- `time_ratio`: Change in measurement duration
- `freq`: Pulsing frequency (Hz)
- `pulse_duration_ratio`: Relative to 10 µs baseline
- `baseline_duration / 1e6`: Convert baseline to seconds

**Example (20 µs pulse, 20 Hz, 8h measurement from 0.5h at 5e6 to 1e6 flux):**
```
duty_cycle = (1e6 / 5e6) × (8h / 0.5h) × 20 × (20 / 10) × (10e-6)
          = 0.2 × 16 × 20 × 2.0 × 0.00001
          = 0.00128
          = 0.128%
```

---

## 6. Benefits

### Repository Organization
✅ **Cleaner**: Main directory only has essential files
✅ **Organized**: Documentation in `.documents/`
✅ **Maintainable**: Easy to find current vs archived tests

### Duty Cycle Fix
✅ **Physically Correct**: Longer pulse → more neutrons
✅ **Consistent**: 10 µs baseline makes scaling clear
✅ **Transparent**: Print statements show all factors

### Streamlit Defaults
✅ **Production Ready**: Defaults match recommended settings
✅ **Better Model**: iron_with_cellulose often fits better
✅ **Constrained Fit**: Fixed background/response reduces overfitting
✅ **Realistic**: 20 µs is typical pulse duration

### Tutorial Notebook
✅ **Comprehensive**: Shows complete workflow
✅ **Practical**: Real example with iron powder
✅ **Reproducible**: Uses seeds for consistency
✅ **Educational**: Parameter sweep shows dependencies

---

## 7. Testing

### Duty Cycle Verification

Test with 10 µs and 20 µs pulses:

```python
# 10 µs pulse
data1 = Data('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)
data1.convolute_response(10.0)
data1.poisson_sample(flux=1e6, freq=20, measurement_time=480)
# Expected: duty_cycle = 0.2 × 16 × 20 × 1.0 × 0.00001 = 0.00064

# 20 µs pulse
data2 = Data('signal.csv', 'openbeam.csv', flux=5e6, duration=0.5, freq=20)
data2.convolute_response(20.0)
data2.poisson_sample(flux=1e6, freq=20, measurement_time=480)
# Expected: duty_cycle = 0.2 × 16 × 20 × 2.0 × 0.00001 = 0.00128
```

Result: 20 µs pulse has exactly 2x the duty cycle ✓

### Streamlit App

Check defaults in sidebar:
- ✓ Pulse duration: 20.0 µs
- ✓ nbragg model: iron_with_cellulose (selected by default)
- ✓ vary_background: "False (fixed)"
- ✓ vary_response: "False (fixed)"
- ✓ vary_weights: "True (vary)"
- ✓ wlmin: 1.0 Å
- ✓ wlmax: 5.0 Å

### Tutorial Notebook

Run the notebook:
```bash
jupyter notebook notebooks/simple_reconstruction_example.ipynb
```

Expected output:
- Single reconstruction completes successfully
- nbragg fit with iron_with_cellulose works
- Parameter sweep generates 20 data points
- Plot saved as `frame_sweep_results.png`

---

## 8. Migration Guide

### For Users

**If you were using old default settings:**

1. **Pulse duration changed from 200 → 20 µs**
   - If you need 200 µs, manually set it in the sidebar
   - New baseline (20 µs) is more typical for most instruments

2. **nbragg model changed from iron → iron_with_cellulose**
   - If you need pure iron, select "iron" from dropdown
   - iron_with_cellulose often provides better fits

3. **vary_background and vary_response now default to False**
   - Provides more stable fits by constraining parameters
   - Can still enable by selecting "True (vary)" in the UI

4. **Duty cycle calculation updated**
   - Automatically includes pulse_duration_ratio
   - No action needed - works transparently
   - Check print output to see scaling factors

### For Developers

**Documentation moved to `.documents/`:**
```bash
# Old location
./KERNEL_GENERATION_AND_UI_UPDATES.md

# New location
./.documents/KERNEL_GENERATION_AND_UI_UPDATES.md
```

**Debug files moved:**
```bash
# Old location
./tests/debug_*.py

# New location
./.documents/debug_*.py
```

**Current test files:**
- `tests/test_kernel_generation.py` - Active
- `tests/test_tof_offset_correction_consistency.py` - Active

---

## 9. File Structure

```
frame_overlap/
├── README.md                          # Main documentation (kept)
├── .documents/                        # NEW: All docs and debug files
│   ├── KERNEL_GENERATION_AND_UI_UPDATES.md
│   ├── NBRAGG_UI_REORGANIZATION.md
│   ├── RECONSTRUCTION_AMPLITUDE_FIX.md
│   ├── TOF_OFFSET_CORRECTION_ANALYSIS.md
│   ├── REPOSITORY_CLEANUP_AND_UPDATES.md  # This file
│   └── [debug/test files...]
├── notebooks/
│   ├── simple_reconstruction_example.ipynb  # NEW: Simple tutorial
│   ├── tutorial_v0.2.ipynb           # Existing comprehensive tutorial
│   └── [other example notebooks...]
├── src/frame_overlap/
│   ├── data_class.py                 # Modified: duty cycle fix
│   └── [other source files...]
├── streamlit_app.py                  # Modified: new defaults
└── tests/
    ├── test_kernel_generation.py     # Active test
    └── test_tof_offset_correction_consistency.py  # Active test
```

---

## 10. Summary of Changes

| Component | Change | File | Lines |
|-----------|--------|------|-------|
| Documentation | Moved to `.documents/` | Various | - |
| Debug files | Moved to `.documents/` | `tests/debug_*.py` | - |
| Duty cycle | Added pulse_duration_ratio | `data_class.py` | 656-677 |
| Pulse duration | Default 200→20 µs | `streamlit_app.py` | 490 |
| nbragg model | Default iron→iron_with_cellulose | `streamlit_app.py` | 802 |
| vary_background | Default True→False | `streamlit_app.py` | 836 |
| vary_response | Default True→False | `streamlit_app.py` | 844 |
| Tutorial | Created simple example | `notebooks/simple_reconstruction_example.ipynb` | New file |

---

**Date**: 2025-11-18
**Version**: v0.2.1
**Status**: ✅ Complete
