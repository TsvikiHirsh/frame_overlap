# Wavelength Filtering Feature

## Summary

Added wavelength range filtering to the Streamlit app's data loading stage. Users can now specify minimum and maximum wavelength values (in Angstroms) to filter the neutron data before processing.

## Feature Details

### User Interface

**Location**: Sidebar → Stage 1: Data Loading → Wavelength Range

**Controls**:
- **Min λ (Å)**: Minimum wavelength (default: 1.0 Å)
- **Max λ (Å)**: Maximum wavelength (default: 10.0 Å)
- **Caption**: Shows converted time-of-flight range in milliseconds

**Screenshot of UI**:
```
┌─ Wavelength Range ─────────────┐
│  Min λ (Å)  │  Max λ (Å)      │
│     1.0     │     10.0        │
└────────────────────────────────┘
Time range: 2.28 - 22.75 ms
```

### Default Values

- **Minimum wavelength**: 1.0 Å
- **Maximum wavelength**: 10.0 Å
- **Flight path length**: 9.0 m (used for conversion)

These defaults correspond to a time-of-flight range of approximately 2.3 ms to 22.8 ms.

### Implementation

#### 1. Wavelength to Time-of-Flight Conversion

Uses the de Broglie relation to convert neutron wavelength to time-of-flight:

```python
# λ = h / (m * v)
# v = h / (m * λ)
# t = L / v

def wavelength_to_tof(wavelength_angstrom, flight_path_length_m):
    wavelength_m = wavelength_angstrom * 1e-10
    velocity = PLANCK_CONSTANT / (NEUTRON_MASS_KG * wavelength_m)
    tof_seconds = flight_path_length_m / velocity
    return tof_seconds * 1e6  # microseconds
```

**Constants**:
- `PLANCK_CONSTANT = 6.62607015e-34` J·s
- `NEUTRON_MASS_KG = 1.67492749804e-27` kg

#### 2. Data Filtering

After loading the data, both signal and openbeam datasets are filtered:

```python
# Convert wavelength range to TOF
tof_min_us = wavelength_to_tof(lambda_min, flight_path_m)
tof_max_us = wavelength_to_tof(lambda_max, flight_path_m)

# Filter signal data
mask_signal = (data.data['time'] >= tof_min_us) & (data.data['time'] <= tof_max_us)
data.data = data.data[mask_signal].copy()

# Filter openbeam data
mask_openbeam = (data.op_data['time'] >= tof_min_us) & (data.op_data['time'] <= tof_max_us)
data.op_data = data.op_data[mask_openbeam].copy()
```

#### 3. Applied in Two Locations

The filtering is applied in:

1. **Main Pipeline** ([streamlit_app.py:819-831](streamlit_app.py#L819-L831))
   - When user clicks "Run Pipeline" button
   - Filters data before processing stages

2. **Parameter Sweep (GroupBy Tab)** ([streamlit_app.py:1344-1354](streamlit_app.py#L1344-L1354))
   - Applied to each data instance in the sweep loop
   - Ensures consistent filtering across all parameter combinations

## Use Cases

### 1. Focus on Thermal Neutrons
Set `λ_min = 0.5 Å` and `λ_max = 2.0 Å` to analyze only thermal neutrons.

### 2. Exclude Low-Energy Neutrons
Set `λ_min = 1.0 Å` to filter out very cold neutrons (default behavior).

### 3. Extended Range Analysis
Set `λ_min = 1.0 Å` and `λ_max = 15.0 Å` for broader wavelength coverage.

### 4. Narrow Band Analysis
Set `λ_min = 4.0 Å` and `λ_max = 6.0 Å` to focus on a specific wavelength range.

## Test Results

### Test Coverage

Three comprehensive test scripts verify the functionality:

#### 1. **test_wavelength_filtering.py**
Tests the conversion and filtering logic:

```
✅ Wavelength Conversion: PASS
   - 1.0 Å → 2275.01 µs (2.28 ms)
   - 10.0 Å → 22750.06 µs (22.75 ms)
   - Reversible conversion verified

✅ Data Filtering: PASS
   - Original: 2400 points
   - Filtered (1-10 Å): 2048 points (85.3%)
   - All filtered points within range
```

#### 2. **test_streamlit_wavelength.py**
Tests the complete pipeline with filtering:

```
✅ Pipeline: PASS
   - Data loading and filtering
   - Convolution
   - Poisson sampling
   - Frame overlap
   - Reconstruction (χ²/dof: 717.6)
   - nbragg analysis (χ²/dof: 1.06)
```

#### 3. **Validation with Different Ranges**

| Wavelength Range | Points Kept | Percentage |
|-----------------|-------------|------------|
| 0.5 - 2.0 Å     | 342         | 14.2%      |
| 1.0 - 5.0 Å     | 910         | 37.9%      |
| 1.0 - 10.0 Å    | 2048        | 85.3%      |
| 2.0 - 15.0 Å    | 1944        | 81.0%      |

## Benefits

1. **Data Quality**: Remove unwanted wavelength ranges before processing
2. **Performance**: Fewer data points → faster processing
3. **Analysis Focus**: Concentrate on relevant energy ranges
4. **Flexibility**: Easy to adjust range via UI sliders
5. **Consistency**: Filtering applied to both signal and openbeam data
6. **Transparency**: Shows converted time range for verification

## Technical Notes

### Wavelength-Time Relationship

For neutrons at a 9.0 m flight path:
- **Shorter wavelengths** → **shorter times** (faster neutrons)
- **Longer wavelengths** → **longer times** (slower neutrons)

Example conversions:
- 1 Å → 2.28 ms (thermal neutrons)
- 5 Å → 11.38 ms (cold neutrons)
- 10 Å → 22.75 ms (very cold neutrons)

### Data Range

The original iron powder dataset spans:
- **Time**: 0 - 23,990 µs (0 - 23.99 ms)
- **Wavelength**: 0 - 10.55 Å

Default filtering (1-10 Å) captures most of the useful data while excluding:
- Very short wavelengths (< 1 Å): High-energy neutrons, often noisy
- Time = 0: Boundary effects and division-by-zero issues

### Integration with nbragg

The wavelength filtering works seamlessly with nbragg analysis:

1. Data is filtered by wavelength range
2. Processing pipeline applies (convolution, Poisson, overlap)
3. Reconstruction performs deconvolution
4. `to_nbragg()` converts TOF to wavelength internally
5. nbragg fits the transmission data in wavelength space

The filtered time range determines which wavelengths are available for fitting.

## Future Enhancements

Possible improvements:

1. **Flight Path Control**: Allow user to specify L (currently fixed at 9.0 m)
2. **Energy Units**: Option to input range in energy (meV) instead of wavelength
3. **Preset Ranges**: Quick-select buttons for common ranges (thermal, cold, etc.)
4. **Visual Indicator**: Show filtered region on plots
5. **Auto-Range**: Suggest optimal wavelength range based on data statistics

## Files Modified

### Main Application
- **[streamlit_app.py](streamlit_app.py)**
  - Lines 450-479: Added wavelength range inputs to Data Loading stage
  - Lines 819-831: Applied filtering in main pipeline
  - Lines 1344-1354: Applied filtering in parameter sweep loop

### Test Scripts (New)
- **[tests/test_wavelength_filtering.py](tests/test_wavelength_filtering.py)**
  - Tests conversion functions
  - Tests data filtering logic
  - Validates with multiple wavelength ranges

- **[tests/test_streamlit_wavelength.py](tests/test_streamlit_wavelength.py)**
  - End-to-end pipeline test
  - Verifies integration with all processing stages
  - Tests nbragg analysis with filtered data

### Documentation
- **[tests/README_NBRAGG_TESTS.md](tests/README_NBRAGG_TESTS.md)**
  - Added documentation for new test scripts

- **[WAVELENGTH_FILTERING.md](WAVELENGTH_FILTERING.md)** (this file)
  - Complete feature documentation

## Usage Example

```python
# In Streamlit app UI:

1. Open sidebar → "📁 1. Data Loading"

2. Set wavelength range:
   Min λ (Å): 1.0
   Max λ (Å): 10.0

3. Observe converted time range:
   "Time range: 2.28 - 22.75 ms"

4. Configure other pipeline stages as needed

5. Click "🚀 Run Pipeline"

6. Data is automatically filtered before processing

7. All results (plots, statistics, nbragg fits)
   use the filtered wavelength range
```

## Validation

To verify the feature works correctly:

```bash
# Test conversion and filtering
python tests/test_wavelength_filtering.py

# Test full pipeline integration
python tests/test_streamlit_wavelength.py

# Run Streamlit app
streamlit run streamlit_app.py
```

All tests should pass with:
```
✅ Wavelength Conversion: PASS
✅ Data Filtering: PASS
✅ Pipeline: PASS
```

---

**Implementation Date**: 2025-11-10
**Status**: ✅ Complete and tested
**Default Range**: 1.0 - 10.0 Å
