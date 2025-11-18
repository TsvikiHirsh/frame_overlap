# nbragg Fit UI Reorganization

## Summary of Changes

Reorganized the nbragg fit controls in the Streamlit app to provide better organization and support for tri-state vary parameters (None, True, False).

---

## Changes Made

### 1. Vary Parameters - Collapsible Single Column

**Previous implementation:**
- Vary checkboxes displayed in two columns
- Simple True/False logic (checkbox on/off)
- Did not support None state (parameter not set)

**New implementation:**
- All vary parameters in **collapsible expander** "⚙️ Vary Parameters"
- **Single column layout** for better organization
- **Tri-state logic** for each parameter:
  - **Not set (None)**: Checkbox unchecked → parameter not passed to nbragg
  - **True (vary)**: Checkbox checked + radio button "True" → parameter varies during fit
  - **False (fixed)**: Checkbox checked + radio button "False" → parameter fixed during fit

**Parameters with tri-state control:**
- vary_background (default: enabled, True)
- vary_response (default: enabled, True)
- vary_weights (default: enabled, True)
- vary_sans (default: disabled, None)
- vary_extinction (default: disabled, None)

### 2. Advanced Fit Parameters - New Collapsible Menu

**New section:** "🔧 Advanced Fit Parameters"

**Wavelength Range Controls:**
- **λ min (Å)**: Minimum wavelength for fitting (default: 1.0 Å)
- **λ max (Å)**: Maximum wavelength for fitting (default: 5.0 Å)
- Controls displayed side-by-side in two columns
- Range: 0.1 - 20.0 Å

**Other Parameters:**
- Thickness Guess (cm): Moved here from previous location
- Fix Normalization to 1.0: Moved here from previous location

### 3. Wavelength Filtering Applied to nbragg Fits

**Main pipeline fitting:**
```python
# Filter wavelength range for fitting
wavelength_mask = (nbragg_data.table['wl'] >= wlmin) & (nbragg_data.table['wl'] <= wlmax)
nbragg_data.table = nbragg_data.table[wavelength_mask].copy()
```

**GroupBy parameter sweep:**
```python
# Convert to nbragg format and filter wavelength range
nbragg_data_sweep = recon_sweep.to_nbragg(L=9.0, tstep=10e-6)
wavelength_mask_sweep = (nbragg_data_sweep.table['wl'] >= wlmin) & (nbragg_data_sweep.table['wl'] <= wlmax)
nbragg_data_sweep.table = nbragg_data_sweep.table[wavelength_mask_sweep].copy()
```

---

## UI Layout

### Before:
```
🔬 6. Analysis (nbragg)
├── Apply nbragg Analysis [checkbox]
├── Material Model [selectbox]
├── Fitting Options
│   ├── Column 1:
│   │   ├── Vary Background [checkbox]
│   │   ├── Vary Response [checkbox]
│   │   └── Vary Weights [checkbox]
│   └── Column 2:
│       ├── Vary SANS [checkbox]
│       └── Vary Extinction [checkbox]
└── Advanced Parameters [expander]
    ├── Thickness Guess
    └── Fix Normalization
```

### After:
```
🔬 6. Analysis (nbragg)
├── Apply nbragg Analysis [checkbox]
├── Material Model [selectbox]
├── ⚙️ Vary Parameters [expander, collapsed by default]
│   ├── Background:
│   │   ├── Enable vary_background [checkbox] (default: on)
│   │   └── vary_background [radio: True/False] (default: True)
│   ├── Response:
│   │   ├── Enable vary_response [checkbox] (default: on)
│   │   └── vary_response [radio: True/False] (default: True)
│   ├── Weights:
│   │   ├── Enable vary_weights [checkbox] (default: on)
│   │   └── vary_weights [radio: True/False] (default: True)
│   ├── SANS:
│   │   ├── Enable vary_sans [checkbox] (default: off)
│   │   └── vary_sans [radio: True/False]
│   └── Extinction:
│       ├── Enable vary_extinction [checkbox] (default: off)
│       └── vary_extinction [radio: True/False]
└── 🔧 Advanced Fit Parameters [expander, collapsed by default]
    ├── Wavelength Range:
    │   ├── λ min (Å) [number input] (default: 1.0)
    │   └── λ max (Å) [number input] (default: 5.0)
    └── Other Parameters:
        ├── Thickness Guess (cm)
        └── Fix Normalization to 1.0
```

---

## Technical Implementation Details

### Tri-State Parameter Logic

For each vary parameter:

```python
# Example: vary_background
enable_vary_background = st.checkbox("Enable vary_background", value=True, key="enable_vary_bg")
if enable_vary_background:
    vary_background = st.radio("vary_background", [True, False], index=0, horizontal=True,
                              key="vary_bg_radio")
else:
    vary_background = None
```

**Result:**
- `enable_vary_background=False` → `vary_background=None` → Not passed to Analysis constructor
- `enable_vary_background=True`, radio=True → `vary_background=True` → Passed as True
- `enable_vary_background=True`, radio=False → `vary_background=False` → Passed as False

### Default Values

When analysis is **not enabled**, defaults are set in the else block:

```python
else:
    nbragg_model = "iron"
    vary_background = True
    vary_response = True
    vary_weights = True
    vary_sans = False
    vary_extinction = False
    thickness_guess = 1.95
    norm_fixed = True
    wlmin = 1.0
    wlmax = 5.0
```

### Parameter Filtering

Analysis kwargs are filtered to remove None values before passing to Analysis:

```python
analysis_kwargs = {
    'vary_background': vary_background,
    'vary_response': vary_response,
    'vary_weights': vary_weights,
    'vary_sans': vary_sans,
    'vary_extinction': vary_extinction,
    'thickness_guess': thickness_guess,
    'norm_guess': 1.0 if norm_fixed else None
}
# Remove None values
analysis_kwargs = {k: v for k, v in analysis_kwargs.items() if v is not None}
```

---

## Benefits

### 1. Better Organization
- Single column layout is cleaner and easier to scan
- Related controls grouped in logical expanders
- Less visual clutter in sidebar

### 2. Tri-State Support
- Properly supports nbragg's None/True/False distinction for vary parameters
- None = use nbragg's default behavior (don't set parameter)
- True = explicitly vary the parameter
- False = explicitly fix the parameter

### 3. Wavelength Range Control
- Users can now restrict fitting to specific wavelength range
- Useful for excluding noisy regions at edges
- Default 1-5 Å covers typical Bragg edge range for iron
- Applied consistently to both main pipeline and GroupBy sweeps

### 4. Flexibility
- Advanced users can customize all parameters
- Default settings remain sensible for typical use
- Collapsed expanders keep UI clean for basic usage

---

## Usage Examples

### Example 1: Default Settings (Quick Fit)
1. Enable "Apply nbragg Analysis"
2. Select material model (default: iron)
3. Run pipeline
   - vary_background = True
   - vary_response = True
   - vary_weights = True
   - vary_sans = None (not set)
   - vary_extinction = None (not set)
   - λ range = 1.0 - 5.0 Å

### Example 2: Fix Background
1. Enable "Apply nbragg Analysis"
2. Expand "⚙️ Vary Parameters"
3. Under Background:
   - Keep "Enable vary_background" checked
   - Select "False" radio button
4. Run pipeline
   - vary_background = False (explicitly fixed)

### Example 3: Don't Set Background Parameter
1. Enable "Apply nbragg Analysis"
2. Expand "⚙️ Vary Parameters"
3. Under Background:
   - Uncheck "Enable vary_background"
4. Run pipeline
   - vary_background = None (not passed to nbragg, uses model default)

### Example 4: Custom Wavelength Range
1. Enable "Apply nbragg Analysis"
2. Expand "🔧 Advanced Fit Parameters"
3. Set λ min = 2.0 Å, λ max = 6.0 Å
4. Run pipeline
   - Fits only data between 2-6 Å
   - Excludes noisy regions outside this range

### Example 5: Enable SANS
1. Enable "Apply nbragg Analysis"
2. Expand "⚙️ Vary Parameters"
3. Under SANS:
   - Check "Enable vary_sans"
   - Select "True" radio button
4. Run pipeline
   - vary_sans = True (SANS parameters will vary)

---

## Modified Files

- **streamlit_app.py**
  - Lines 830-935: New vary parameter UI (tri-state, single column, collapsible)
  - Lines 897-935: New advanced fit parameters section with wavelength range
  - Lines 936-946: Updated else block with wlmin/wlmax defaults
  - Lines 1032-1034: Apply wavelength filtering to main pipeline nbragg fit
  - Lines 1614-1622: Apply wavelength filtering to GroupBy sweep nbragg fits

---

## Testing Recommendations

1. **Test tri-state logic:**
   - Try None (unchecked enable), True, and False for each vary parameter
   - Verify parameters are correctly passed/not passed to Analysis constructor

2. **Test wavelength filtering:**
   - Set different wlmin/wlmax ranges
   - Verify fit only uses data within specified range
   - Check fit results change appropriately

3. **Test GroupBy sweep:**
   - Enable nbragg analysis in GroupBy sweep
   - Verify wavelength filtering applies to all sweep iterations
   - Check nbragg metrics (redchi, thickness) in sweep results

4. **Test default behavior:**
   - Verify defaults match expected behavior (vary_bg/resp/wts=True, sans/ext=None)
   - Check wavelength range default is 1.0-5.0 Å

---

## Known Considerations

### nbragg Parameter Semantics

According to nbragg documentation:
- `vary_parameter=None` → Use model's default behavior (parameter may or may not vary depending on model)
- `vary_parameter=True` → Explicitly allow parameter to vary during fit
- `vary_parameter=False` → Explicitly fix parameter during fit

The new tri-state UI correctly implements this semantic distinction.

### Wavelength Range

The wavelength filter is applied to the nbragg data table **after** conversion from time-of-flight. This ensures:
- Filtering happens in wavelength space (more intuitive for Bragg edge analysis)
- Data cleaning (NaN, inf removal) still processes full dataset first
- Filter applies before fitting, reducing computational cost for narrow ranges

### Performance

- Wavelength filtering reduces fit time for narrow ranges
- Collapsible expanders reduce visual clutter
- Default collapsed state speeds up navigation for users who don't need advanced controls

---

**Date**: 2025-11-18
**Branch**: `claude/bragg-edge-adaptive-chopper-01NTdCs5m5xCwqaKVyjq4LtB`
**Status**: ✅ Implemented and syntax-checked
