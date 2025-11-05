"""Debug NaN issue in nbragg fitting."""

import sys
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd

from frame_overlap import Data, Reconstruct, Analysis

# Same parameters as in streamlit (default settings)
data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
           flux=5e6, duration=0.5, freq=20)

data.convolute_response(200, bin_width=10)
data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
data.overlap(kernel=[0, 25])

# Check data after overlap
print("After overlap:")
print(f"  overlapped_data shape: {data.overlapped_data.shape}")
print(f"  NaN in counts: {data.overlapped_data['counts'].isna().sum()}")
print(f"  Inf in counts: {np.isinf(data.overlapped_data['counts']).sum()}")
print(f"  NaN in err: {data.overlapped_data['err'].isna().sum()}")
print(f"  Inf in err: {np.isinf(data.overlapped_data['err']).sum()}")

# Reconstruction with default parameters (NO time filter)
print("\nReconstruction WITHOUT time filter:")
recon = Reconstruct(data)
recon.filter(kind='wiener', noise_power=0.2)

print(f"  reconstructed_data shape: {recon.reconstructed_data.shape}")
print(f"  NaN in counts: {recon.reconstructed_data['counts'].isna().sum()}")
print(f"  Inf in counts: {np.isinf(recon.reconstructed_data['counts']).sum()}")
print(f"  NaN in err: {recon.reconstructed_data['err'].isna().sum()}")
print(f"  Inf in err: {np.isinf(recon.reconstructed_data['err']).sum()}")

# Check openbeam
if hasattr(data, 'op_overlapped_data') and data.op_overlapped_data is not None:
    print(f"\n  op_overlapped_data shape: {data.op_overlapped_data.shape}")
    print(f"  NaN in counts: {data.op_overlapped_data['counts'].isna().sum()}")
    print(f"  Inf in counts: {np.isinf(data.op_overlapped_data['counts']).sum()}")

# Try nbragg analysis
print("\nTrying nbragg analysis...")
try:
    analysis = Analysis(xs='iron', vary_background=True, vary_response=True)

    # Check the data before fitting
    print("\nBefore fit, checking recon.to_nbragg()...")
    nbragg_data = recon.to_nbragg(L=9.0, tstep=10e-6)

    print(f"  nbragg_data.table shape: {nbragg_data.table.shape}")
    print(f"  Columns: {nbragg_data.table.columns.tolist()}")
    print(f"  NaN in wavelength: {nbragg_data.table['wavelength'].isna().sum()}")
    print(f"  NaN in trans: {nbragg_data.table['trans'].isna().sum()}")
    print(f"  NaN in err: {nbragg_data.table['err'].isna().sum()}")
    print(f"  Inf in trans: {np.isinf(nbragg_data.table['trans']).sum()}")

    # Show some sample values
    print(f"\n  Sample trans values: {nbragg_data.table['trans'].head(10).values}")
    print(f"  Sample err values: {nbragg_data.table['err'].head(10).values}")

    # Check for problematic values
    print(f"\n  Min trans: {nbragg_data.table['trans'].min()}")
    print(f"  Max trans: {nbragg_data.table['trans'].max()}")
    print(f"  Trans > 1: {(nbragg_data.table['trans'] > 1).sum()}")
    print(f"  Trans < 0: {(nbragg_data.table['trans'] < 0).sum()}")

    result = analysis.fit(recon)
    print(f"\n✓ Fit succeeded! redchi = {result.redchi:.4f}")

except Exception as e:
    print(f"\n✗ Fit failed with error: {e}")
    import traceback
    traceback.print_exc()
