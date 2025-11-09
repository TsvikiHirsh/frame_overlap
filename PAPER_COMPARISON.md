# Paper Implementation Comparison

## Reference Paper
**Frame overlap Bragg edge imaging**
Busi et al., Scientific Reports (2020)
DOI: 10.1038/s41598-020-71705-4
https://www.nature.com/articles/s41598-020-71705-4

## MATLAB Code Analysis

Based on the provided MATLAB code from the paper, here are the key implementation details:

### Original MATLAB Approach

```matlab
% Key steps from the paper's code:
1. Load data (20sim.txt)
2. Create kernel from 8-point kernel repeated 20 times
3. Convolve signal with kernel: y0 = conv(kernel_repeated, x0)
4. **Apply smoothing**: y0 = smooth(y0)  ← KEY STEP
5. Wiener deconvolution: x0_rec = wiener_deconvolution(y0, k8(:,2), 1)
```

### Critical Details

1. **Smoothing Before Deconvolution**
   - MATLAB's `smooth()` function (moving average)
   - Applied to overlapped signal BEFORE deconvolution
   - This is the key difference from standard Wiener filter

2. **Kernel Handling**
   - Uses 8-point base kernel
   - Repeats it 20 times for convolution
   - Final reconstruction uses only original kernel length

3. **Signal Processing**
   - Works directly on signal counts
   - Does NOT separate signal and openbeam reconstruction
   - Applies filter to overlapped signal

## Our Implementation

### What We've Implemented

#### 1. **Standard Wiener Filter** (`kind='wiener'`)
```python
# Frequency domain Wiener deconvolution
H = fft(kernel)
Y = fft(observed)
G = H* / (|H|^2 + noise_power)
X = Y * G
```

#### 2. **Smoothed Wiener Filter** (`kind='wiener_smooth'`) ⭐ **PAPER METHOD**
```python
# Apply smoothing first (like MATLAB smooth())
from scipy.ndimage import uniform_filter1d
observed_smooth = uniform_filter1d(observed, size=smooth_window)

# Then apply Wiener deconvolution
x_reconstructed = wiener_deconvolution(observed_smooth, kernel, noise_power)
```

**This matches the paper's approach!**

#### 3. **Adaptive Wiener Filter** (`kind='wiener_adaptive'`)
```python
# Scipy adaptive noise estimation
filtered = scipy.signal.wiener(observed, mysize=mysize)

# Then deconvolution
x_reconstructed = wiener_deconvolution(filtered, kernel, noise_power)
```

### Key Improvements Over MATLAB Code

✅ **We Do BETTER:**
1. **Separate Signal & Openbeam Reconstruction**
   - Reconstruct signal independently
   - Reconstruct openbeam independently
   - Calculate transmission: T = signal/openbeam
   - **Why better**: Proper uncertainty propagation, physically accurate

2. **Multiple Filter Options**
   - Standard Wiener
   - Smoothed Wiener (paper method)
   - Adaptive Wiener (scipy)
   - Lucy-Richardson
   - Tikhonov

3. **Comprehensive Testing**
   - 7 test cases for smoothed filter
   - Tests with 32 random frames in 30ms
   - Quality metrics and validation

4. **Streamlit UI**
   - Interactive parameter tuning
   - Real-time visualization
   - Parameter sweeps (GroupBy)

### What MATLAB Does

❌ **Limitations:**
1. Works only on signal (no separate openbeam)
2. Single method (smoothed Wiener only)
3. No interactive interface
4. Limited to specific dataset format

## Recommendations for Users

### For Paper-Like Results

Use **`wiener_smooth`** filter:
```python
from frame_overlap import Data, Reconstruct

data = Data('signal.csv', 'openbeam.csv')
data.overlap(kernel=[0, 0.9375, 0.9375, ...])  # 32 random frames

recon = Reconstruct(data)
recon.filter(kind='wiener_smooth',
             noise_power=1.0,  # Try different values
             smooth_window=5)   # Paper uses ~5
```

### Streamlit App

1. Select **Reconstruction Method**: `wiener_smooth`
2. Set **Noise Power**: Start with 0.2-1.0
3. Set **Smooth Window**: 5 (paper default)
4. For 32 frames: Use random kernel generator

### Parameter Tuning

**Smooth Window** (moving average size):
- Smaller (3): Less smoothing, more detail, more noise
- Medium (5-7): Balanced (recommended, paper uses ~5)
- Larger (11+): More smoothing, less noise, less detail

**Noise Power** (regularization):
- Lower (0.01-0.1): Less regularization, sharper but noisier
- Medium (0.2-0.5): Balanced
- Higher (1.0+): More regularization, smoother but may lose features

## Implementation Differences

### Signal vs Transmission

**MATLAB (Paper):**
```
Signal only → Overlap → Smooth → Wiener → Reconstructed Signal
```

**Our Implementation:**
```
Signal → Overlap → Smooth → Wiener → Reconstructed Signal
OpenBeam → Overlap → Smooth → Wiener → Reconstructed OpenBeam
                                      ↓
                            Transmission = Signal/OpenBeam
```

**Why our approach is better:**
- Accounts for openbeam variations
- Proper error propagation
- Standard practice in neutron imaging
- Compatible with nbragg/nres analysis

### Kernel Handling

**MATLAB:**
- Creates repeated kernel for convolution
- Uses original kernel for deconvolution

**Our Implementation:**
- Stores kernel as time differences
- Reconstructs full kernel on demand
- Handles arbitrary number of frames
- Supports both absolute times and differences

## Testing

We've tested all scenarios from the paper:

✅ **4 frames** (original tests)
✅ **32 random frames in 30ms** (paper scenario)
✅ **Different smoothing windows** (3, 5, 7, 11)
✅ **Different noise levels** (0.001 - 1.0)
✅ **Comparison with other methods** (lucy, tikhonov)

All tests pass with good reconstruction quality.

## Conclusion

Our implementation:
1. ✅ **Includes the paper's smoothing approach** (`wiener_smooth`)
2. ✅ **Improves upon it** (separate signal/openbeam)
3. ✅ **Adds more options** (adaptive, lucy, tikhonov)
4. ✅ **Provides better interface** (Streamlit UI)
5. ✅ **Comprehensive testing** (7 test cases)

**Use `wiener_smooth` for results closest to the paper!**
