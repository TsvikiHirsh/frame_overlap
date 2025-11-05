# Data Cleaning with Hydration/Dehydration Denoising

This feature adds support for loading TIFF stack files with advanced denoising using the mbirjax library's hydration/dehydration method (HSNT - Hyperspectral Nonnegative with Total Variation).

## Overview

The data cleaning stage allows you to:
1. Load multi-frame TIFF stacks (signal and openbeam)
2. Apply region-of-interest (ROI) masking (ImageJ format or default circular)
3. Denoise using mbirjax's `hyper_denoise` function with dehydration/rehydration
4. Compute TOF spectra from the denoised data

This preprocessing step can significantly improve data quality before frame overlap analysis.

## Usage

### Python API

```python
from frame_overlap import Data

# Create Data object
data = Data()

# Load signal TIFF with default circular ROI (1 cm diameter, 1.4 cm FOV)
data.load_signal_tiff(
    'signal_stack.tif',
    fov_cm=1.4,
    roi_diameter_cm=1.0,
    subspace_dimension=10  # Number of dehydration dimensions
)

# Load openbeam TIFF with custom ImageJ ROI
data.load_openbeam_tiff(
    'openbeam_stack.tif',
    roi_path='my_roi.roi',
    subspace_dimension=10
)

# Continue with normal pipeline
data.convolute_response(pulse_duration=200)
data.overlap(kernel=[0, 12, 10, 25])
# ... etc
```

### Streamlit UI

1. Open the Streamlit app: `streamlit run streamlit_app.py`
2. In the sidebar, expand **"🧹 0. Data Cleaning (Optional)"**
3. Check **"Use TIFF with Denoising"**
4. Upload your signal and openbeam TIFF files
5. Configure ROI:
   - Use custom ImageJ ROI file, or
   - Use default circular ROI (specify FOV and diameter)
6. Configure denoising:
   - Auto-estimate subspace dimension (recommended), or
   - Manually set the number of dehydration dimensions
7. Select dataset type (attenuation or transmission)
8. Click **"🚀 Run Pipeline"**

## Parameters

### ROI Configuration

- **Field of View (FOV)**: Total imaging area in cm (default: 1.4 × 1.4 cm)
- **ROI Diameter**: Diameter of circular ROI in cm (default: 1.0 cm)
- **Custom ROI**: Upload ImageJ .roi or .zip file with named ROI

### Denoising Parameters

- **Subspace Dimension**: Number of dehydration dimensions
  - Lower values = more aggressive denoising
  - Higher values = preserve more detail
  - `None` = automatic estimation (recommended)
- **Dataset Type**:
  - `'attenuation'`: For attenuation data (default)
  - `'transmission'`: For transmission data

## How it Works

The mbirjax `hyper_denoise` function uses a three-step process:

1. **Dehydration**: Compress the hyperspectral (multi-frame) data onto a low-dimensional subspace
   - Reduces dimensionality from N_k frames to N_s dimensions
   - The subspace dimension N_s is the key denoising parameter

2. **Denoising**: Denoise in the compressed subspace
   - Uses Bayesian estimation with appropriate loss function
   - Separates signal from noise more effectively in reduced space

3. **Rehydration**: Reconstruct the full hyperspectral data from the denoised subspace
   - Returns data with original dimensions but reduced noise

## Requirements

New dependencies added to `requirements.txt`:
- `mbirjax>=0.1.0` - For hydration/dehydration denoising
- `tifffile>=2021.0.0` - For reading TIFF stacks
- `read-roi>=1.6.0` - For ImageJ ROI file support

Install with:
```bash
pip install -r requirements.txt
```

## Example Workflow

```python
from frame_overlap import Data, Reconstruct

# 1. Data Cleaning: Load TIFF with denoising
data = Data()
data.load_signal_tiff('signal.tif', subspace_dimension=10)
data.load_openbeam_tiff('openbeam.tif', subspace_dimension=10)

# 2. Instrument Response
data.convolute_response(pulse_duration=200)

# 3. Poisson Sampling
data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)

# 4. Frame Overlap
data.overlap(kernel=[0, 12, 10, 25], total_time=50)

# 5. Reconstruction
recon = Reconstruct(data, tmin=3.7, tmax=11.0)
recon.filter(kind='wiener', noise_power=0.2)

# 6. Analysis
recon.plot_fit()
```

## Technical Details

### TIFF Stack Format

- **Expected format**: 3D array (n_frames, height, width)
- **Supported types**: .tif, .tiff
- **Multi-page TIFF**: Automatically handles stacked TIFF files

### ROI Masking

Supported ImageJ ROI types:
- Oval (circular/elliptical)
- Rectangle
- Polygon
- Freehand
- Traced

Default circular ROI:
- Centered at image center
- Specified by diameter in physical units (cm)
- Scaled based on FOV and image dimensions

### Memory Considerations

The denoising process:
- Only processes pixels within the ROI
- Uses batching for large datasets
- Temporary memory ~2-3x the ROI pixel data size

For very large TIFF stacks:
- Use a smaller ROI
- Increase subspace dimension (less compression)
- Process signal and openbeam separately

## References

- mbirjax documentation: https://mbirjax.readthedocs.io/
- HSNT method: Hydration-Shepp-Vardi Nonnegative with Total Variation
- ImageJ ROI format: https://imagej.nih.gov/ij/docs/guide/146-30.html
