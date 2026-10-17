"""
Debug script to test with EXACT streamlit default settings.
This will show us where the NaN values are coming from.
"""

import sys
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd

from frame_overlap import Data, Reconstruct, Analysis

print("="*70)
print("DEBUGGING NBRAGG FIT WITH STREAMLIT DEFAULT SETTINGS")
print("="*70)

# Use EXACT default settings from streamlit_app.py
flux_orig = 5e6
duration_orig = 0.5
freq_orig = 20

pulse_duration = 200.0
bin_width = 10

flux_new = 1e6
freq_new = 20
measurement_time = 8 * 60  # 8 hours
seed_poisson = 42

kernel_absolute = [0, 25]
total_time = 50

recon_method = 'wiener'
noise_power = 0.2
tmin = None  # Check if using time filter
tmax = None

nbragg_model = 'iron'
vary_background = True
vary_response = True

print("\nSettings:")
print(f"  tmin: {tmin}, tmax: {tmax}")
print(f"  kernel: {kernel_absolute}")
print(f"  nbragg_model: {nbragg_model}")
print(f"  vary_background: {vary_background}, vary_response: {vary_response}")

# Pipeline
print("\n" + "="*70)
print("STEP 1: Data Processing")
print("="*70)

data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
           flux=flux_orig, duration=duration_orig, freq=freq_orig)
print("✓ Data loaded")

data.convolute_response(pulse_duration, bin_width=bin_width)
print("✓ Convolution applied")

data.poisson_sample(flux=flux_new, freq=freq_new,
                   measurement_time=measurement_time, seed=seed_poisson)
print("✓ Poisson sampling applied")

data.overlap(kernel=kernel_absolute, total_time=total_time)
print("✓ Overlap applied")

# Check for NaN in overlapped data
print(f"\nChecking overlapped data:")
print(f"  Shape: {data.overlapped_data.shape}")
print(f"  NaN in counts: {data.overlapped_data['counts'].isna().sum()}")
print(f"  Inf in counts: {np.isinf(data.overlapped_data['counts']).sum()}")
print(f"  NaN in err: {data.overlapped_data['err'].isna().sum()}")
print(f"  Inf in err: {np.isinf(data.overlapped_data['err']).sum()}")

if data.op_overlapped_data is not None:
    print(f"\nChecking openbeam overlapped data:")
    print(f"  Shape: {data.op_overlapped_data.shape}")
    print(f"  NaN in counts: {data.op_overlapped_data['counts'].isna().sum()}")
    print(f"  Inf in counts: {np.isinf(data.op_overlapped_data['counts']).sum()}")

# Reconstruction
print("\n" + "="*70)
print("STEP 2: Reconstruction")
print("="*70)

recon = Reconstruct(data, tmin=tmin, tmax=tmax)
recon.filter(kind=recon_method, noise_power=noise_power)
print("✓ Reconstruction complete")

print(f"\nChecking reconstructed data:")
print(f"  Shape: {recon.reconstructed_data.shape}")
print(f"  NaN in counts: {recon.reconstructed_data['counts'].isna().sum()}")
print(f"  Inf in counts: {np.isinf(recon.reconstructed_data['counts']).sum()}")
print(f"  NaN in err: {recon.reconstructed_data['err'].isna().sum()}")
print(f"  Inf in err: {np.isinf(recon.reconstructed_data['err']).sum()}")

if recon.reconstructed_openbeam is not None:
    print(f"\nChecking reconstructed openbeam:")
    print(f"  Shape: {recon.reconstructed_openbeam.shape}")
    print(f"  NaN in counts: {recon.reconstructed_openbeam['counts'].isna().sum()}")
    print(f"  Inf in counts: {np.isinf(recon.reconstructed_openbeam['counts']).sum()}")

# Convert to nbragg format
print("\n" + "="*70)
print("STEP 3: Convert to nbragg format")
print("="*70)

print("Calling recon.to_nbragg(L=9.0, tstep=10e-6)...")
nbragg_data = recon.to_nbragg(L=9.0, tstep=10e-6)
print("✓ Conversion complete")

print(f"\nChecking nbragg data BEFORE dropna:")
print(f"  Shape: {nbragg_data.table.shape}")
print(f"  Columns: {nbragg_data.table.columns.tolist()}")
print(f"  NaN in wavelength: {nbragg_data.table['wavelength'].isna().sum()}")
print(f"  NaN in trans: {nbragg_data.table['trans'].isna().sum()}")
print(f"  NaN in err: {nbragg_data.table['err'].isna().sum()}")
print(f"  Inf in trans: {np.isinf(nbragg_data.table['trans']).sum()}")
print(f"  Inf in err: {np.isinf(nbragg_data.table['err']).sum()}")

# Show some sample values
print(f"\nSample values (first 10 rows):")
print(nbragg_data.table.head(10))

# Show problematic values if any
bad_trans = (nbragg_data.table['trans'].isna()) | (np.isinf(nbragg_data.table['trans']))
bad_err = (nbragg_data.table['err'].isna()) | (np.isinf(nbragg_data.table['err'])) | (nbragg_data.table['err'] <= 0)

if bad_trans.sum() > 0:
    print(f"\n⚠️ Found {bad_trans.sum()} bad transmission values:")
    print(nbragg_data.table[bad_trans].head())

if bad_err.sum() > 0:
    print(f"\n⚠️ Found {bad_err.sum()} bad error values:")
    print(nbragg_data.table[bad_err].head())

# Clean the data
print("\nCalling nbragg_data.table.dropna()...")
nbragg_data.table = nbragg_data.table.dropna()
print(f"✓ After dropna: {nbragg_data.table.shape[0]} rows")

# Also remove zero or negative errors
print("\nRemoving zero/negative errors...")
nbragg_data.table = nbragg_data.table[nbragg_data.table['err'] > 0]
print(f"✓ After removing bad errors: {nbragg_data.table.shape[0]} rows")

# Remove inf values
print("\nRemoving inf values...")
nbragg_data.table = nbragg_data.table[~np.isinf(nbragg_data.table['trans'])]
nbragg_data.table = nbragg_data.table[~np.isinf(nbragg_data.table['err'])]
print(f"✓ After removing inf: {nbragg_data.table.shape[0]} rows")

# Final check
print(f"\nFinal data check:")
print(f"  NaN anywhere: {nbragg_data.table.isna().any().any()}")
print(f"  Inf anywhere: {np.isinf(nbragg_data.table).any().any()}")
print(f"  All errors > 0: {(nbragg_data.table['err'] > 0).all()}")
print(f"  Trans range: [{nbragg_data.table['trans'].min():.6f}, {nbragg_data.table['trans'].max():.6f}]")

# Try fitting
print("\n" + "="*70)
print("STEP 4: nbragg Fitting")
print("="*70)

try:
    analysis = Analysis(xs=nbragg_model, vary_background=vary_background,
                       vary_response=vary_response)
    print("✓ Analysis object created")

    result = analysis.model.fit(nbragg_data)
    print(f"✓ Fit completed!")
    print(f"  redchi: {result.redchi:.4f}")
    print(f"  success: {result.success}")

except Exception as e:
    print(f"✗ Fit failed: {e}")
    import traceback
    traceback.print_exc()

    # Try to diagnose the issue
    print("\n" + "="*70)
    print("DIAGNOSTIC INFORMATION")
    print("="*70)

    # Check if model can evaluate
    try:
        print("\nTrying to evaluate model at first wavelength point...")
        test_wl = nbragg_data.table['wavelength'].iloc[0]
        test_result = analysis.model.eval(wl=np.array([test_wl]))
        print(f"  Model eval result: {test_result}")
    except Exception as e2:
        print(f"  Model eval failed: {e2}")
