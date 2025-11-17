# TOF Shift Correction Validation Report

## Executive Summary

This report validates the TOF (Time-of-Flight) shift correction implementation against the formulas presented in the Frame Overlap Bragg Edge Imaging article (Nature Scientific Reports, 2020). All core formulas have been validated and simple test cases have been created to help diagnose potential issues with reconstruction parameters.

---

## Validation Results

### ✅ Core Formulas - All Validated

All three key formulas from the article have been tested and validated:

#### 1. Time Delays from Chopper Parameters (Equation 1)

**Article Formula:**
```
τₖ = θₖ / (360° × f)
```

Where:
- `τₖ` = time delay for slit k (seconds)
- `θₖ` = angular position of slit k (degrees)
- `f` = chopper rotation frequency (Hz)

**Status:** ✅ **VALIDATED**
- Formula correctly converts slit angles to time delays
- Inverse relationship with frequency verified (doubling frequency halves delays)
- POLDI chopper parameters from article tested successfully

#### 2. Kernel Reconstruction (Time Structure Function)

**Article Formula:**
```
τ(t') = Σₖ aₖ δ(t' - τₖ)
```

Where:
- `aₖ = 1/n_frames` for equal-width slits
- `δ(t' - τₖ)` = Dirac delta function at time delay τₖ

**Status:** ✅ **VALIDATED**
- Discrete kernel: Delta functions correctly placed at rounded bin positions
- Interpolated kernel: Linear interpolation between adjacent bins implemented
- Weight distribution follows article's specification:
  - Floor bin weight: `(1 - frac) / n_frames`
  - Ceiling bin weight: `frac / n_frames`
- Normalization verified (kernel sum = 1.0)

#### 3. TOF to Wavelength Conversion (Equation 3)

**Article Formula:**
```
λ = (h/m_n) × t / L_sd
```

Where:
- `h` = Planck constant = 6.62607015×10⁻³⁴ J·s
- `m_n` = neutron mass = 1.674927498×10⁻²⁷ kg
- `t` = time-of-flight (s)
- `L_sd` = source-to-detector distance (m)

**Status:** ✅ **VALIDATED**
- Conversion formula implemented correctly
- h/m_n constant = 3956.034 Å·m/s verified
- Reverse calculation (wavelength → TOF) verified

---

## Key Implementation Details

### Frame Delay Calculation

The kernel parameter represents **frame-to-frame delays**:

```python
kernel = [0, 12, 10, 25]  # ms (delays between consecutive frames)
```

These are converted to **absolute frame start times** using cumulative sum:

```python
frame_starts = np.cumsum(kernel)  # [0, 12, 22, 47] ms
```

### Bin Position Calculation

Frame start times are converted to bin indices:

```python
frame_starts_us = frame_starts_ms * 1000  # Convert to microseconds
frame_starts_bins = frame_starts_us / bin_width  # Fractional bin positions
```

### Kernel Reconstruction Methods

#### Discrete Method
```python
for bin_float in frame_starts_bins:
    bin_int = int(np.round(bin_float))
    kernel[bin_int] = 1.0 / n_frames
```

#### Interpolated Method (FOBI-style)
```python
for bin_float in frame_starts_bins:
    bin_floor = int(np.floor(bin_float))
    frac = bin_float - bin_floor

    kernel[bin_floor] += (1.0 - frac) / n_frames
    kernel[bin_floor + 1] += frac / n_frames
```

---

## Potential Issues & Diagnostics

### 1. TOF Offset After Wiener Deconvolution

**Issue:**
After Wiener deconvolution, the reconstructed signal may have a TOF offset relative to the expected position. This is a **known effect** in frequency-domain deconvolution.

**Cause:**
- The offset is related to the phase of the Fourier transform
- Depends on the position of the kernel peak relative to the array center
- Can be calculated as: `offset = peak_position - center_position`

**Solution:**
The codebase already has a `TOFOffsetCorrector` class in `tof_offset_correction.py` that handles this. The correction methods include:
- Cross-correlation with reference signal
- Edge position alignment
- Kernel peak alignment
- Optimization-based correction

### 2. Fractional Bin Positions

**Issue:**
When frame delays don't align exactly to integer bin positions, using discrete kernel reconstruction can cause small errors.

**Detection:**
```python
# Check if delays have fractional bin components
fractional_parts = frame_starts_bins - np.floor(frame_starts_bins)
has_fractional = any(frac > 1e-6 for frac in fractional_parts)
```

**Recommendation:**
Use `interpolate_kernel=True` in the FOBI reconstruction method when fractional bins are detected.

### 3. Large TOF Offsets

**Issue:**
For certain parameter sets, the expected TOF offset can be very large (> 1000 bins), which may cause alignment issues.

**Detection:**
```python
peak_idx = np.argmax(kernel)
center = len(kernel) // 2
offset = peak_idx - center

if abs(offset) > 1000:
    print(f"Warning: Large offset detected: {offset} bins")
```

**Recommendation:**
Use the `TOFOffsetCorrector` to automatically detect and correct large offsets.

### 4. Insufficient Kernel Length

**Issue:**
If the kernel array is too short for the maximum frame delay, reconstruction will fail or produce artifacts.

**Detection:**
```python
max_delay_bins = int(np.max(frame_starts_bins))
recommended_length = max_delay_bins + 1000  # Add buffer
```

**Recommendation:**
Ensure kernel length is at least `max_delay + buffer` bins.

---

## Diagnostic Tool Usage

A diagnostic tool has been created to help identify potential issues with your parameters:

### Using the Diagnostic Tool

```python
from tests.tof_diagnostic import diagnose_parameters

# Test your parameters
results = diagnose_parameters(
    kernel=[0, 25],      # Your frame delays in ms
    bin_width=10         # Your bin width in µs
)
```

### From Chopper Parameters

```python
from tests.tof_diagnostic import diagnose_from_chopper

# Calculate kernel from chopper specs
results = diagnose_from_chopper(
    angles=[0, 45, 90, 180],  # Slit angles in degrees
    frequency=50,              # Chopper frequency in Hz
    bin_width=10               # Bin width in µs
)
```

### Command Line Usage

```bash
python tests/tof_diagnostic.py
```

This will run diagnostics on several example cases and show what issues can be detected.

---

## Test Files Created

### 1. `tests/test_tof_formulas_simple.py`
- Validates all article formulas independently
- Tests time delay calculation, kernel reconstruction, wavelength conversion
- Tests frequency scaling, bin width scaling, cumulative delays
- **Result:** All 7 tests pass ✅

### 2. `tests/test_implementation_validation.py`
- Validates that the codebase implementation matches article formulas
- Tests kernel reconstruction, chopper parameter conversion
- **Result:** Core validations pass ✅

### 3. `tests/tof_diagnostic.py`
- Diagnostic tool for parameter validation
- Identifies potential issues:
  - Invalid kernel format (not starting at 0)
  - Fractional bin positions
  - Large TOF offsets
  - Insufficient kernel length
  - Chopper parameter mismatches
- Provides recommendations for each issue

---

## Recommendations

### For General Use

1. **Always validate your parameters** using the diagnostic tool before running reconstruction
   ```bash
   python tests/tof_diagnostic.py
   ```

2. **Use interpolated kernel** when you have fractional bin positions
   ```python
   reconstructor.filter(kind='fobi', interpolate_kernel=True)
   ```

3. **Check TOF offset** after reconstruction and apply correction if needed
   ```python
   from frame_overlap.tof_offset_correction import TOFOffsetCorrector

   corrector = TOFOffsetCorrector(reconstructed_data, reference_data)
   corrected = corrector.correct_by_cross_correlation()
   ```

### For Debugging Reconstruction Issues

If reconstruction looks incorrect for certain parameters:

1. **Run diagnostics** on your specific parameters
2. **Check for warnings** about large offsets or fractional bins
3. **Verify kernel normalization** (should sum to 1.0)
4. **Test with simpler parameters** first (e.g., 2 frames, integer bins)
5. **Compare different reconstruction methods**:
   - Standard Wiener
   - FOBI-style Wiener
   - Discrete vs interpolated kernel

### For the POLDI Chopper Configuration

The article's POLDI chopper has been tested and validated:

```python
angles = [0, 9.363, 21.475, 37.039, 50.417, 56.664, 67.422, 75.406]  # degrees
frequency = 50  # Hz

# This correctly produces:
kernel_delays = [0, 0.520, 0.673, 0.865, 0.743, 0.347, 0.598, 0.444]  # ms
```

All formulas match the article specification ✅

---

## Conclusion

The TOF shift correction implementation in the codebase correctly follows the formulas from the Frame Overlap Bragg Edge Imaging article. The core algorithms for:
- Time delay calculation
- Kernel reconstruction (both discrete and interpolated)
- TOF-to-wavelength conversion

...are all properly implemented and validated.

If you're experiencing issues with specific parameter sets, please:
1. Run the diagnostic tool on your parameters
2. Check the warnings and recommendations
3. Verify that TOF offset correction is being applied
4. Test with both discrete and interpolated kernel modes

The diagnostic tool will help identify the specific issue causing problems with your reconstruction.

---

## Running the Tests

### Validate Formulas
```bash
python tests/test_tof_formulas_simple.py
```
Expected: All 7 tests pass

### Run Diagnostics
```bash
python tests/tof_diagnostic.py
```
Expected: See diagnostic output for example cases

### Test Your Parameters
```python
from tests.tof_diagnostic import diagnose_parameters

diagnose_parameters(
    kernel=[your, parameters, here],
    bin_width=10
)
```

---

## References

**Frame overlap Bragg edge imaging**
Matteo Busi, Jan Čapek, Markus Strobl
Scientific Reports, Volume 10, Article 14867 (2020)
https://www.nature.com/articles/s41598-020-71705-4

**Key Equations:**
- Equation (1): Time structure function τ(t') = Σₖ aₖ δ(t' - τₖ)
- Equation (3): TOF to wavelength λ = (h/m_n) × t / L_sd
- Chopper formula: τₖ = θₖ / (360° × f)

---

*Report generated: 2025-11-17*
