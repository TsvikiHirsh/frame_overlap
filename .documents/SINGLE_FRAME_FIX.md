# Single-Frame Reconstruction Fix

**Date**: 2025-11-10
**Issue**: Single-frame reconstruction had 20% difference from original signal
**Status**: ✅ FIXED

---

## Problem Description

When using only one frame (no overlap), the reconstruction was showing a 20% difference from the original signal instead of returning the data unchanged. This was reported as a critical bug.

**Expected Behavior**: With a single frame, there's no overlap to undo, so reconstruction should return the exact same signal (identity operation).

**Observed Behavior**: Reconstruction was showing differences of ~1-10% depending on the method used.

---

## Root Cause Analysis

The issue was in [src/frame_overlap/reconstruct.py](../src/frame_overlap/reconstruct.py):

1. **Single-frame kernel**: When using `kernel=[0]`, the reconstruction kernel is correctly constructed as `[1.0]` (a delta function at bin 0)

2. **Unnecessary deconvolution**: Even though the kernel is `[1.0]` (identity in convolution), the deconvolution process was still being applied

3. **Numerical errors**: The FFT-based deconvolution, even with a delta function kernel, introduced small numerical errors (~1%)

4. **Smoothing effects**: Some methods like `wiener_smooth` applied additional smoothing before deconvolution, causing larger errors (~10%)

---

## Solution

Modified the `filter()` method in `Reconstruct` class to detect single-frame cases and skip deconvolution entirely:

```python
# Check if we have a single-frame case (kernel = [0])
is_single_frame = (len(self.data.kernel) == 1 and self.data.kernel[0] == 0)

if is_single_frame:
    # Single frame: no overlap, no deconvolution needed
    # Just return the overlapped data as-is
    reconstructed_signal = self.data.table['counts'].values.copy()
else:
    # Multiple frames: apply deconvolution
    # ... (existing deconvolution code)
```

The same logic was applied to both signal and openbeam reconstruction.

---

## Test Results

Created comprehensive test: [tests/test_single_frame_reconstruction.py](../tests/test_single_frame_reconstruction.py)

### Before Fix
```
wiener              :   0.99% diff, χ²/dof=  0.00 - ✓ PASS (barely)
wiener_smooth       :  10.03% diff, χ²/dof=  0.62 - ✗ FAIL
lucy                :   0.00% diff, χ²/dof=  0.00 - ✓ PASS
tikhonov            :   0.99% diff, χ²/dof=  0.00 - ✓ PASS
```

### After Fix
```
wiener              :   0.00% diff, χ²/dof=  0.00 - ✓ PASS
wiener_smooth       :   0.00% diff, χ²/dof=  0.00 - ✓ PASS
lucy                :   0.00% diff, χ²/dof=  0.00 - ✓ PASS
tikhonov            :   0.00% diff, χ²/dof=  0.00 - ✓ PASS
```

**Result**: All methods now return **perfect identity** (0.00% difference) for single-frame case!

---

## Verification

1. ✅ **Single-frame reconstruction**: Perfect identity (0.00% diff)
2. ✅ **Multi-frame reconstruction**: Still works correctly (χ²/dof = 0.651)
3. ✅ **All reconstruction methods**: wiener, wiener_smooth, lucy, tikhonov all pass
4. ✅ **Existing tests**: 32 tests pass in test_frame_overlap.py

---

## Files Modified

1. **src/frame_overlap/reconstruct.py** (lines 329-390)
   - Added single-frame detection
   - Skip deconvolution for single-frame case
   - Apply same logic to both signal and openbeam

2. **tests/test_single_frame_reconstruction.py** (NEW)
   - Comprehensive test for single-frame reconstruction
   - Tests all reconstruction methods
   - Ensures 0% difference requirement

---

## Usage

The fix is transparent to users. Single-frame reconstruction now works correctly:

```python
from frame_overlap import Data, Reconstruct

# Load and process data
data = Data('signal.csv', 'openbeam.csv')
data.convolute_response(200).poisson_sample()

# Single frame - no overlap
data.overlap(kernel=[0], total_time=50)

# Reconstruct - returns identity (perfect)
recon = Reconstruct(data)
recon.filter(kind='wiener', noise_power=0.2)

# Result: reconstructed signal == original signal (0% difference)
assert recon.statistics['chi2_per_dof'] < 0.001  # Perfect fit!
```

---

## Performance Impact

**Positive**: Single-frame reconstruction is now faster since it skips the FFT-based deconvolution entirely.

**No negative impact**: Multi-frame reconstruction performance unchanged.

---

## Backward Compatibility

✅ **Fully backward compatible**: The fix only affects single-frame cases, which previously had incorrect results. Multi-frame cases behave identically.

---

## Related Issues

- User reported: "20% difference in reconstruction even if I use one frame"
- Root cause: Unnecessary deconvolution with delta function kernel
- Fix: Skip deconvolution for single-frame case (identity operation)

---

**End of Document**
