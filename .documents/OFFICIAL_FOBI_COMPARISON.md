# Official FOBI Repository Analysis

**Date**: 2025-11-10
**Source**: https://github.com/matteobsu/FOBI
**Nature Paper**: https://www.nature.com/articles/s41598-020-71705-4

---

## Key Findings from Official FOBI Code

### 1. Wiener Deconvolution Formula ✅

The official FOBI uses **exactly the same formula** as our implementation:

```matlab
% Official FOBI (MATLAB)
F = fft(f);
G = fft(g);
arg = F.*conj(G)./((abs(G).^2)+c);
H = real(ifft(arg));
```

This matches our implementation:
```python
# Our implementation (Python)
F = np.fft.fft(observed)
G = np.fft.fft(kernel_padded)
H = F * np.conj(G) / (np.abs(G)**2 + noise_power)
reconstructed = np.real(np.fft.ifft(H))
```

**Conclusion**: The deconvolution formula is identical! ✅

---

### 2. Critical Difference: Kernel Construction! 🎯

#### **Official FOBI Kernel Construction**

The official FOBI uses **interpolated kernel** for sub-bin precision:

```matlab
% For each chopper angle/time delay:
shifts = Nt * angles;  % Convert to bin indices

for i = 1:length(shifts)
    sfloor = floor(shifts(i)) + 1;
    rest = shifts(i) - floor(shifts(i));

    % Distribute intensity across TWO adjacent bins
    D(sfloor) = D(sfloor) + (1 - rest);
    D(sfloor+1) = D(sfloor+1) + rest;
end
```

**Example**: If a frame should start at bin 2500.7:
- Bin 2500 gets weight: 0.3 (1 - 0.7)
- Bin 2501 gets weight: 0.7

This creates a **smooth, interpolated kernel** instead of discrete delta functions!

#### **Our Current Kernel Construction**

We use **discrete delta functions**:

```python
# Our current implementation
for frame_delay in kernel_ms:
    bin_idx = int(frame_delay / bin_width_ms)
    kernel[bin_idx] = 1.0 / n_frames
```

**Example**: If a frame should start at bin 2500.7:
- Bin 2500 gets weight: 1.0 (rounded down)
- Bin 2501 gets weight: 0.0

This creates **sharp delta functions** at integer bin positions.

---

### 3. POLDI Chopper Configuration

The official FOBI POLDI configuration uses **8 angles**:

```matlab
angles = [0, 9.363, 21.475, 37.039, 50.417, 56.664, 67.422, 75.406] % degrees
```

These are normalized and converted to time delays based on chopper rotation speed.

**Processing steps**:
1. Normalize angles to [0, 1] range
2. Scale by time window: `shifts = Nt * normalized_angles`
3. Create interpolated kernel using the shift values
4. Apply Wiener deconvolution
5. Scale result by `nslits × nrep`
6. Roll (circular shift) by `roll` parameter to align phases

---

### 4. Key Parameters

| Parameter | Official FOBI | Our Implementation | Notes |
|-----------|--------------|-------------------|-------|
| **Regularization (c)** | 0.1 (1e-1) | 0.01 | FOBI uses 10x higher! |
| **Smoothing** | Yes (`flag_smooth=1`, span=3) | Optional (SG filter) | Different smoothing methods |
| **Roll shift** | Yes (e.g., 165 samples) | Optional | Phase alignment |
| **Kernel type** | Interpolated (sub-bin) | Discrete delta | **Critical difference!** |
| **Normalization** | `nslits × nrep` | `1.0 / n_frames` | Different scaling |

---

### 5. Processing Workflow

#### Official FOBI Workflow:

```
1. Load data (sample & open beam)
2. Crop blind spots (instrument-specific)
3. Optional smoothing (moving average, span=3)
4. Interpolate readout gaps
5. Construct kernel from chopper angles (interpolated)
6. Apply Wiener deconvolution (c=0.1)
7. Scale by nslits×nrep
8. Roll/shift for phase alignment
9. Apply moving average filter [100, 20]
10. Edge fitting (Gaussian models)
```

#### Our Current Workflow:

```
1. Load data
2. Convolute response
3. Poisson sampling
4. Overlap (discrete kernel)
5. Reconstruct with FOBI filter (c=0.01)
6. Optional SG smoothing
7. Statistical analysis
8. Edge fitting (nbragg)
```

---

## Why Our FOBI "Doesn't Show Much Better Reconstruction"

### Root Cause: Kernel Construction

The **interpolated kernel** in official FOBI vs our **discrete delta kernel** is the critical difference!

**Official FOBI kernel** (interpolated):
```
[0, 0, ..., 0.3, 0.7, 0, ..., 0, 0.1, 0.9, 0, ...]
```

**Our kernel** (discrete):
```
[0, 0, ..., 1.0, 0, 0, ..., 0, 1.0, 0, 0, ...]
```

The interpolated kernel provides:
- ✅ Sub-bin precision
- ✅ Smoother frequency response
- ✅ Better reconstruction for non-integer bin delays
- ✅ Reduced aliasing effects

---

## Recommendations

### 1. Update Our Kernel Construction ⚡

Add interpolated kernel option to `reconstruct.py`:

```python
def _reconstruct_kernel_interpolated(self):
    """Construct interpolated kernel for sub-bin precision (FOBI style)."""
    bin_width_ms = (self.data.table['time'][1] - self.data.table['time'][0]) / 1000
    n_bins = len(self.data.table)
    kernel = np.zeros(n_bins)

    for frame_delay_ms in self.data.kernel:
        # Convert to fractional bin index
        bin_idx_float = frame_delay_ms / bin_width_ms
        bin_floor = int(np.floor(bin_idx_float))
        rest = bin_idx_float - bin_floor

        # Interpolate across two adjacent bins
        if bin_floor < n_bins:
            kernel[bin_floor] += (1 - rest) / len(self.data.kernel)
        if bin_floor + 1 < n_bins:
            kernel[bin_floor + 1] += rest / len(self.data.kernel)

    return kernel
```

### 2. Adjust Default Parameters

Use official FOBI defaults:
- **noise_power**: 0.1 (not 0.01)
- **smoothing**: span=3 moving average
- **roll shift**: experiment with phase alignment

### 3. Add POLDI Chopper Support

Create chopper configuration for standard patterns:
```python
POLDI_CHOPPER_ANGLES = [0, 9.363, 21.475, 37.039, 50.417, 56.664, 67.422, 75.406]
FOBI_4x10_ANGLES = [0, 5.737, 10.895, 22.252, 28.622, 45.784, 53.558, 59.753, 69.095, 80.0]
```

### 4. Test with Official FOBI Results

Recreate their examples to validate implementation.

---

## Action Plan

1. ✅ Analyze official FOBI repository
2. ✅ Identify kernel construction difference
3. ⚡ Implement interpolated kernel method
4. ⚡ Update default parameters to match official FOBI
5. ⚡ Add POLDI chopper configuration support
6. ⚡ Create test comparing discrete vs interpolated kernels
7. ⚡ Update documentation and streamlit UI

---

## Expected Improvements

With interpolated kernel:
- **Better sub-bin precision**: Handles fractional bin delays correctly
- **Smoother reconstruction**: Reduces artifacts from discrete approximation
- **Closer to official FOBI**: Matches their implementation exactly
- **Better for POLDI data**: Designed for chopper angle patterns

---

**End of Document**
