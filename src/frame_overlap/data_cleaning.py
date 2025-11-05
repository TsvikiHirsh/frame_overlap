"""
Data cleaning module for neutron ToF measurements using mbirjax dehydration/rehydration.

This module provides functionality for:
- Loading TIFF stacks (signal and openbeam)
- Handling ImageJ ROI files with circular default
- Applying mbirjax hyper_denoise for denoising
- Computing TOF spectra from denoised data
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

try:
    import tifffile
except ImportError:
    warnings.warn("tifffile not installed. TIFF loading will not be available.")
    tifffile = None

try:
    from read_roi import read_roi_file, read_roi_zip
except ImportError:
    warnings.warn("read-roi not installed. ROI loading will not be available.")
    read_roi_file = None
    read_roi_zip = None

try:
    from mbirjax import hyper_denoise
except ImportError:
    warnings.warn("mbirjax not installed. Dehydration/rehydration will not be available.")
    hyper_denoise = None


def load_tiff_stack(tiff_path):
    """
    Load a TIFF stack file.

    Parameters
    ----------
    tiff_path : str or Path
        Path to TIFF stack file

    Returns
    -------
    numpy.ndarray
        3D array with shape (n_frames, height, width)

    Raises
    ------
    ImportError
        If tifffile is not installed
    FileNotFoundError
        If TIFF file does not exist
    """
    if tifffile is None:
        raise ImportError("tifffile is required for loading TIFF stacks. "
                         "Install with: pip install tifffile")

    tiff_path = Path(tiff_path)
    if not tiff_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {tiff_path}")

    with tifffile.TiffFile(tiff_path) as tif:
        stack = tif.asarray()

    # Ensure 3D array (n_frames, height, width)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    elif stack.ndim != 3:
        raise ValueError(f"Expected 2D or 3D TIFF stack, got {stack.ndim}D")

    return stack


def create_circular_roi(image_shape, fov_cm=1.4, roi_diameter_cm=1.0):
    """
    Create a circular ROI mask centered in the image.

    Parameters
    ----------
    image_shape : tuple
        Shape of the image (height, width)
    fov_cm : float, optional
        Field of view in cm (default: 1.4 cm for 1.4x1.4 cm FOV)
    roi_diameter_cm : float, optional
        Diameter of circular ROI in cm (default: 1.0 cm)

    Returns
    -------
    numpy.ndarray
        Boolean mask with shape (height, width), True inside ROI
    """
    height, width = image_shape

    # Calculate ROI radius in pixels
    pixels_per_cm = height / fov_cm  # Assuming square pixels and FOV
    roi_radius_px = (roi_diameter_cm / 2) * pixels_per_cm

    # Create coordinate grids centered at image center
    y_center, x_center = height / 2, width / 2
    y, x = np.ogrid[:height, :width]

    # Create circular mask
    distance_from_center = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    mask = distance_from_center <= roi_radius_px

    return mask


def load_imagej_roi(roi_path, image_shape=None):
    """
    Load an ImageJ ROI file and create a binary mask.

    Supports both .roi files (single ROI) and .zip files (multiple ROIs).
    If multiple ROIs are present, the first named ROI is used.

    Parameters
    ----------
    roi_path : str or Path
        Path to .roi or .zip file
    image_shape : tuple, optional
        Shape (height, width) for the output mask. If None, uses ROI bounds.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True inside ROI region(s)
    dict
        Metadata about the loaded ROI(s)

    Raises
    ------
    ImportError
        If read-roi is not installed
    FileNotFoundError
        If ROI file does not exist
    """
    if read_roi_file is None or read_roi_zip is None:
        raise ImportError("read-roi is required for loading ImageJ ROI files. "
                         "Install with: pip install read-roi")

    roi_path = Path(roi_path)
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI file not found: {roi_path}")

    # Load ROI file
    if roi_path.suffix.lower() == '.zip':
        rois = read_roi_zip(str(roi_path))
    else:
        rois = read_roi_file(str(roi_path))

    if len(rois) == 0:
        raise ValueError(f"No ROIs found in {roi_path}")

    # Get first ROI
    roi_name = list(rois.keys())[0]
    roi = rois[roi_name]

    # Determine image shape
    if image_shape is None:
        # Use ROI bounds to determine size
        if 'width' in roi and 'height' in roi:
            image_shape = (roi['height'] + roi.get('top', 0),
                          roi['width'] + roi.get('left', 0))
        else:
            # For polygon ROIs, use coordinate bounds
            x_coords = np.array(roi.get('x', [0]))
            y_coords = np.array(roi.get('y', [0]))
            image_shape = (int(y_coords.max()) + 1, int(x_coords.max()) + 1)

    # Create mask
    mask = np.zeros(image_shape, dtype=bool)

    # Handle different ROI types
    roi_type = roi.get('type', 'polygon')

    if roi_type == 'oval':
        # Circular/elliptical ROI
        left = roi.get('left', 0)
        top = roi.get('top', 0)
        width = roi.get('width', image_shape[1])
        height = roi.get('height', image_shape[0])

        center_x = left + width / 2
        center_y = top + height / 2
        radius_x = width / 2
        radius_y = height / 2

        y, x = np.ogrid[:image_shape[0], :image_shape[1]]
        mask = ((x - center_x) / radius_x)**2 + ((y - center_y) / radius_y)**2 <= 1

    elif roi_type == 'rectangle':
        # Rectangular ROI
        left = roi.get('left', 0)
        top = roi.get('top', 0)
        width = roi.get('width', image_shape[1])
        height = roi.get('height', image_shape[0])

        mask[top:top+height, left:left+width] = True

    elif roi_type in ['polygon', 'freehand', 'traced']:
        # Polygon ROI
        from matplotlib.path import Path as MplPath
        x_coords = np.array(roi['x'])
        y_coords = np.array(roi['y'])

        # Create polygon path
        vertices = np.column_stack([x_coords, y_coords])
        path = MplPath(vertices)

        # Test all pixels
        y, x = np.mgrid[:image_shape[0], :image_shape[1]]
        points = np.column_stack([x.ravel(), y.ravel()])
        mask = path.contains_points(points).reshape(image_shape)

    else:
        warnings.warn(f"Unsupported ROI type '{roi_type}', creating empty mask")

    metadata = {
        'name': roi_name,
        'type': roi_type,
        'n_pixels': mask.sum()
    }

    return mask, metadata


def apply_dehydration_denoising(tiff_stack, roi_mask=None, subspace_dimension=None,
                                 dataset_type='attenuation', verbose=1):
    """
    Apply mbirjax hyper_denoise to a TIFF stack using dehydration/rehydration.

    Parameters
    ----------
    tiff_stack : numpy.ndarray
        3D array with shape (n_frames, height, width)
    roi_mask : numpy.ndarray, optional
        Boolean mask with shape (height, width). If provided, only pixels
        within the ROI are processed (others set to zero).
    subspace_dimension : int, optional
        Number of dehydration dimensions. If None, automatically estimated.
    dataset_type : str, optional
        Either 'attenuation' or 'transmission' (default: 'attenuation')
    verbose : int, optional
        Verbosity level (default: 1)

    Returns
    -------
    numpy.ndarray
        Denoised TIFF stack with same shape as input
    dict
        Metadata about the denoising process

    Raises
    ------
    ImportError
        If mbirjax is not installed
    """
    if hyper_denoise is None:
        raise ImportError("mbirjax is required for dehydration/rehydration. "
                         "Install with: pip install mbirjax")

    n_frames, height, width = tiff_stack.shape

    # Apply ROI mask if provided
    if roi_mask is not None:
        if roi_mask.shape != (height, width):
            raise ValueError(f"ROI mask shape {roi_mask.shape} does not match "
                           f"image shape {(height, width)}")

        # Mask out pixels outside ROI
        tiff_stack_masked = tiff_stack.copy()
        tiff_stack_masked[:, ~roi_mask] = 0
    else:
        tiff_stack_masked = tiff_stack
        roi_mask = np.ones((height, width), dtype=bool)

    # Reshape to (n_pixels, n_frames) for processing only ROI pixels
    roi_indices = np.where(roi_mask)
    n_roi_pixels = len(roi_indices[0])

    # Extract ROI pixel time series: shape (n_roi_pixels, n_frames)
    roi_time_series = tiff_stack_masked[:, roi_indices[0], roi_indices[1]].T

    # Apply hyper_denoise
    # hyper_denoise expects spectral axis in last dimension
    denoised_roi = hyper_denoise(
        roi_time_series,
        dataset_type=dataset_type,
        subspace_dimension=subspace_dimension,
        verbose=verbose
    )

    # Reconstruct full image stack
    denoised_stack = np.zeros_like(tiff_stack)
    denoised_stack[:, roi_indices[0], roi_indices[1]] = denoised_roi.T

    metadata = {
        'n_frames': n_frames,
        'n_roi_pixels': n_roi_pixels,
        'subspace_dimension': subspace_dimension,
        'dataset_type': dataset_type
    }

    return denoised_stack, metadata


def compute_tof_spectrum(denoised_stack, roi_mask=None):
    """
    Compute time-of-flight spectrum by summing over spatial dimensions.

    Parameters
    ----------
    denoised_stack : numpy.ndarray
        3D array with shape (n_frames, height, width)
    roi_mask : numpy.ndarray, optional
        Boolean mask with shape (height, width). If provided, only sum
        pixels within the ROI.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: stack, counts, err
    """
    if roi_mask is not None:
        # Sum only over ROI pixels
        tof_counts = denoised_stack[:, roi_mask].sum(axis=1)
    else:
        # Sum over all spatial pixels
        tof_counts = denoised_stack.sum(axis=(1, 2))

    # Create DataFrame matching Data class format
    tof_spectrum = pd.DataFrame({
        'stack': np.arange(len(tof_counts)),
        'counts': tof_counts,
        'err': np.sqrt(np.maximum(tof_counts, 1))  # Poisson error
    })

    return tof_spectrum


def process_tiff_to_tof(tiff_path, roi_path=None, fov_cm=1.4, roi_diameter_cm=1.0,
                        subspace_dimension=None, dataset_type='attenuation', verbose=1):
    """
    Complete pipeline: load TIFF stack, apply ROI, denoise, compute TOF spectrum.

    Parameters
    ----------
    tiff_path : str or Path
        Path to TIFF stack file
    roi_path : str or Path, optional
        Path to ImageJ ROI file (.roi or .zip). If None, creates default
        circular ROI.
    fov_cm : float, optional
        Field of view in cm (default: 1.4)
    roi_diameter_cm : float, optional
        Diameter of default circular ROI in cm (default: 1.0)
    subspace_dimension : int, optional
        Number of dehydration dimensions. If None, automatically estimated.
    dataset_type : str, optional
        Either 'attenuation' or 'transmission' (default: 'attenuation')
    verbose : int, optional
        Verbosity level (default: 1)

    Returns
    -------
    pandas.DataFrame
        TOF spectrum with columns: stack, counts, err
    dict
        Metadata about the processing pipeline
    """
    # Load TIFF stack
    if verbose:
        print(f"Loading TIFF stack: {tiff_path}")
    tiff_stack = load_tiff_stack(tiff_path)

    # Load or create ROI
    if roi_path is not None:
        if verbose:
            print(f"Loading ROI: {roi_path}")
        roi_mask, roi_metadata = load_imagej_roi(roi_path, image_shape=tiff_stack.shape[1:])
    else:
        if verbose:
            print(f"Creating default circular ROI: {roi_diameter_cm} cm diameter")
        roi_mask = create_circular_roi(tiff_stack.shape[1:], fov_cm, roi_diameter_cm)
        roi_metadata = {
            'type': 'default_circular',
            'diameter_cm': roi_diameter_cm,
            'n_pixels': roi_mask.sum()
        }

    # Apply denoising
    if verbose:
        print(f"Applying hyper_denoise with subspace_dimension={subspace_dimension}")
    denoised_stack, denoise_metadata = apply_dehydration_denoising(
        tiff_stack, roi_mask, subspace_dimension, dataset_type, verbose
    )

    # Compute TOF spectrum
    if verbose:
        print("Computing TOF spectrum")
    tof_spectrum = compute_tof_spectrum(denoised_stack, roi_mask)

    # Combine metadata
    metadata = {
        'tiff_path': str(tiff_path),
        'roi': roi_metadata,
        'denoising': denoise_metadata,
        'tof_length': len(tof_spectrum)
    }

    return tof_spectrum, metadata
