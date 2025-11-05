# nbragg Integration Tests and Debug Scripts

This folder contains test and debug scripts for the nbragg analysis integration in the Streamlit app.

## 🧪 Main Test Scripts

### **test_nbragg_final.py** ⭐ (RECOMMENDED)
The definitive test that verifies nbragg integration works with proper data cleaning.
- Tests exact Streamlit default settings
- Includes NaN/Inf removal
- Shows fitted parameters
- **Run this to verify everything works!**

```bash
python tests/test_nbragg_final.py
```

Expected output:
```
✅ FIT SUCCEEDED!
- Reduced χ²: 1.0680
- Success: True
- Number of parameters: 8
```

### **test_streamlit_workflow.py**
Complete end-to-end workflow test that mimics Streamlit's behavior.
- Tests all tabs (Reconstruction, Statistics)
- Verifies session state management
- Tests plot generation with nbragg overlay

```bash
python tests/test_streamlit_workflow.py
```

### **test_streamlit_nbragg_fix.py**
Tests the nbragg integration with wavelength-to-TOF conversion.
- Verifies proper time conversion
- Tests plot overlay
- Tests statistics display

```bash
python tests/test_streamlit_nbragg_fix.py
```

### **test_analysis_integration.py**
Basic nbragg Analysis class functionality test.
- Tests Analysis object creation
- Tests fit() method
- Verifies result attributes

```bash
python tests/test_analysis_integration.py
```

## 📊 Plot Testing Scripts

### **test_final_plot.py**
Tests nbragg fit overlay on reconstruction plot.
- Generates matplotlib figure
- Adds nbragg fit curve
- Saves to `/tmp/reconstruction_with_nbragg_fit.png`

### **test_wavelength_conversion.py**
Tests wavelength to time-of-flight conversion.
- Verifies conversion formula
- Checks data range alignment

### **test_stats_display.py**
Tests statistics tab display functionality.
- Tests parameter table generation
- Tests fit report display
- Handles None stderr values

## 🐛 Debug Scripts

### **debug_streamlit_settings.py** ⭐ (MOST USEFUL)
Comprehensive debugging script that shows exactly where NaN/Inf values come from.
- Tests with exact Streamlit default settings
- Shows data at each processing stage
- Identifies problematic values (found the 45 inf values!)
- Shows step-by-step data cleaning

```bash
python tests/debug_streamlit_settings.py
```

Output shows:
- Data shape at each stage
- NaN/Inf counts
- Sample values
- Final cleaned data statistics

### **debug_nbragg.py**
Quick debug script to understand nbragg data structure.
- Shows nbragg.Data.table columns
- Shows wavelength vs stack indexing
- Useful for understanding nbragg API

### **debug_nan_issue.py**
Investigates NaN issues in reconstructed data.
- Checks for NaN at each processing stage
- Tests with/without time filter
- Shows where NaN values originate

## 📝 Test Results Summary

All tests pass with the fixed implementation:

| Test | Status | Key Output |
|------|--------|------------|
| test_nbragg_final.py | ✅ PASS | redchi: 1.0680 |
| test_streamlit_workflow.py | ✅ PASS | 5 lines in plot |
| test_streamlit_nbragg_fix.py | ✅ PASS | 1084 fit points |
| test_analysis_integration.py | ✅ PASS | redchi: 6.5813 |
| debug_streamlit_settings.py | ✅ PASS | Found 45 inf values |

## 🔧 Key Fixes Applied

1. **Data Cleaning** (streamlit_app.py:650-661)
   ```python
   nbragg_data.table = nbragg_data.table.dropna()
   nbragg_data.table = nbragg_data.table[~np.isinf(nbragg_data.table['trans'])]
   nbragg_data.table = nbragg_data.table[~np.isinf(nbragg_data.table['err'])]
   nbragg_data.table = nbragg_data.table[nbragg_data.table['err'] > 0]
   ```

2. **Proper Wavelength Conversion** (streamlit_app.py:32-59)
   - Uses de Broglie relation: λ = h / (m * v)
   - Converts Angstroms → TOF in microseconds

3. **Direct Model Fitting** (streamlit_app.py:664-666)
   - Calls `analysis.model.fit(nbragg_data)` directly
   - Avoids Analysis.fit() wrapper that may have issues

## 📚 Other Test Files

- **test_frame_overlap.py** - Original frame_overlap tests
- **test_groupby.py** - GroupBy functionality tests
- **test_new_features.py** - New features tests
- **test_poisson_normalization.py** - Poisson sampling tests

## 🚀 Quick Start

To verify nbragg integration works:

```bash
# Run the main test
python tests/test_nbragg_final.py

# If it passes, run Streamlit
streamlit run streamlit_app.py

# Then:
# 1. Enable "Apply nbragg Analysis" in sidebar
# 2. Click "Run Pipeline"
# 3. See green fit line in Reconstruction tab
# 4. See fit results in Statistics tab
```

## ⚠️ Troubleshooting

If tests fail:

1. **Run debug_streamlit_settings.py** - It will show exactly where the issue is
2. Check for NaN/Inf values in the output
3. Verify nbragg is installed: `pip install nbragg`
4. Check data files exist: `notebooks/iron_powder.csv` and `notebooks/openbeam.csv`

## 📖 Related Documentation

- See main README.md for overall project documentation
- See streamlit_app.py comments for implementation details
- See src/frame_overlap/analysis_nbragg.py for Analysis class API
