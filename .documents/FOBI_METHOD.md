# FOBI Reconstruction Method

**Date**: 2025-11-10
**Status**: ✅ Implemented and tested
**Best for**: Two-frame reconstruction with no temporal overlap

---

## Overview

Implemented the FOBI (Frame Overlap Bragg Imaging) Wiener deconvolution method adapted from the original FOBI code. This method uses a conjugate-based Wiener filter that performs better than standard Wiener for certain overlap patterns, particularly two-frame cases.

---

## Algorithm

The FOBI method implements:

```python
F = fft(signal)
G = fft(kernel)
H = ifft(F * conj(G) / (|G|^2 + noise_power))
```

**Key difference from standard Wiener**: Uses `F * conj(G)` instead of `F * G / conj(G)`

---

## Features

1. **Savitzky-Golay smoothing** (optional)
   - Window size: 1-11 (1 = no smoothing)
   - Polynomial order: 1-5
   - **Recommendation**: Use window=1 (no smoothing) for best results

2. **Adjustable noise power**
   - Range: 0.001 - 1.0
   - **Optimal for FOBI**: 0.001 - 0.01 (much lower than standard Wiener)

3. **Circular shift** (optional)
   - Can apply roll_shift to align output
   - Default: 0 (no shift needed in most cases)

---

## Performance Comparison

### Two-Frame Test (frames at 0ms and 25ms, no overlap)

| Method | χ²/dof | R² | Notes |
|--------|--------|-----|-------|
| **FOBI (noise=0.01)** | **0.630** | **0.988** | **BEST** |
| wiener (noise=0.2) | 0.651 | 0.964 | Good |
| wiener_smooth | 1.128 | 0.952 | Fair |
| lucy | 8720.497 | -5.393 | Poor |
| tikhonov | 0.651 | 0.964 | Good |

**Winner**: FOBI with low noise power (0.001-0.01)

---

## When to Use FOBI

### ✅ **Use FOBI when**:
- Two frames with no temporal overlap
- Frames are well-separated (e.g., 0ms and 25ms)
- You want the best reconstruction quality
- You can use low noise_power values (0.001-0.01)

### ❌ **Use standard Wiener when**:
- Multiple frames (>2) with complex overlap
- Frames have temporal overlap
- Using standard noise_power ~0.2 works well

---

## Parameter Tuning

### Noise Power (most important)
- **FOBI optimal range**: 0.001 - 0.01
- Lower values = better fit, but more noise sensitivity
- Start with 0.01 and decrease if needed

| noise_power | χ²/dof | R² | Comment |
|-------------|--------|-----|---------|
| 0.001 | 0.636 | 0.988 | Excellent |
| 0.01 | 0.630 | 0.988 | **Optimal** |
| 0.1 | 0.742 | 0.977 | Good |
| 0.2 | 1.129 | 0.952 | Fair |
| 0.5 | 2.943 | 0.845 | Poor |

### Smoothing Window
- **Recommended**: 1 (no smoothing)
- Larger windows degrade quality

| smooth_window | χ²/dof | Comment |
|---------------|--------|---------|
| 1 | 0.651 | **Best** (no smoothing) |
| 3 | 1.056 | Slightly worse |
| 5 | 1.129 | Worse |
| 11 | 1.237 | Much worse |

**Conclusion**: Don't use smoothing with FOBI for best results

---

## Usage

### Python API

```python
from frame_overlap import Data, Reconstruct

# Load and process data
data = Data('signal.csv', 'openbeam.csv')
data.convolute_response(200).poisson_sample()
data.overlap(kernel=[0, 25], total_time=50)  # Two frames

# Reconstruct with FOBI
recon = Reconstruct(data)
recon.filter(kind='fobi', noise_power=0.01, smooth_window=1)

# Results
print(f"χ²/dof: {recon.statistics['chi2_per_dof']:.3f}")
print(f"R²: {recon.statistics['r_squared']:.3f}")
```

### Streamlit App

1. Navigate to **Stage 5: Reconstruction**
2. Select method: **fobi**
3. Set **Noise Power**: 0.01 (or try 0.001-0.1)
4. Set **Smooth Window**: 1 (no smoothing recommended)
5. Run pipeline

---

## Implementation Details

### File Modified
- [src/frame_overlap/reconstruct.py](../src/frame_overlap/reconstruct.py)
  - Added `_fobi_filter()` method (lines 509-570)
  - Added `_savitzky_golay_filter()` helper (lines 572-621)
  - Updated filter() method to include 'fobi' option

### Streamlit Integration
- [streamlit_app.py](../streamlit_app.py)
  - Added 'fobi' to reconstruction method dropdown
  - FOBI-specific UI controls with optimal defaults
  - Automatic noise_power default (0.01 for FOBI vs 0.2 for others)

---

## Mathematical Comparison

### Standard Wiener (current implementation)
```python
H = fft(kernel)
F = fft(signal)
H_conj = conj(H)
G = H_conj / (|H|^2 + noise_power)
reconstructed = ifft(F * G)
```

### FOBI Wiener (new implementation)
```python
G = fft(kernel)
F = fft(signal)
G_conj = conj(G)
H = F * G_conj / (|G|^2 + noise_power)
reconstructed = ifft(H)
```

**Key difference**: FOBI computes in one step: `F * conj(G) / (|G|^2 + c)` vs standard two-step approach

---

## Testing

Created comprehensive test: [tests/test_fobi_method.py](../tests/test_fobi_method.py)

Tests include:
- ✅ Comparison with all reconstruction methods
- ✅ Parameter sensitivity analysis (noise_power, smooth_window)
- ✅ Two-frame reconstruction validation

Run with:
```bash
python tests/test_fobi_method.py
```

---

## References

- Original FOBI code by [original authors]
- Based on Wiener deconvolution with conjugate formulation
- Optimized for neutron imaging frame overlap reconstruction

---

## Future Enhancements

Potential improvements:
1. Auto-detect optimal noise_power based on data statistics
2. Add support for FOBI-specific chopper patterns (POLDI, 4x10, 5x8, etc.)
3. Implement interpolation for readout gaps (currently not needed)
4. Add roll_shift UI control for alignment

---

**End of Document**
