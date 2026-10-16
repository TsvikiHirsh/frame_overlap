"""
FINAL TEST - Tests nbragg fitting with exact Streamlit settings including inf removal.
Run this to verify nbragg integration works.
"""

import sys
sys.path.insert(0, 'src')
import numpy as np

from frame_overlap import Data, Reconstruct, Analysis

print("="*70)
print("FINAL NBRAGG INTEGRATION TEST")
print("="*70)

# Exact streamlit default settings
print("\n1. Processing data with Streamlit default settings...")
data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
           flux=5e6, duration=0.5, freq=20)
data.convolute_response(200.0, bin_width=10)
data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
data.overlap(kernel=[0, 25], total_time=50)
print("   ✓ Data processing complete")

print("\n2. Running reconstruction...")
recon = Reconstruct(data, tmin=None, tmax=None)
recon.filter(kind='wiener', noise_power=0.2)
stats = recon.get_statistics()
print(f"   ✓ Reconstruction complete (χ²/dof: {stats['chi2_per_dof']:.1f})")

print("\n3. Preparing nbragg data...")
analysis = Analysis(xs='iron', vary_background=True, vary_response=True)

nbragg_data = recon.to_nbragg(L=9.0, tstep=10e-6)
print(f"   - Initial data: {nbragg_data.table.shape[0]} rows")

# Clean NaN values
nbragg_data.table = nbragg_data.table.dropna()
print(f"   - After dropna: {nbragg_data.table.shape[0]} rows")

# Remove inf values
nbragg_data.table = nbragg_data.table[~np.isinf(nbragg_data.table['trans'])]
nbragg_data.table = nbragg_data.table[~np.isinf(nbragg_data.table['err'])]
print(f"   - After removing inf: {nbragg_data.table.shape[0]} rows")

# Remove zero or negative errors
nbragg_data.table = nbragg_data.table[nbragg_data.table['err'] > 0]
print(f"   - After removing bad errors: {nbragg_data.table.shape[0]} rows")

print("\n4. Running nbragg fit...")
try:
    result = analysis.model.fit(nbragg_data)
    analysis.result = result
    analysis.data = nbragg_data

    if hasattr(result, 'redchi') and not np.isnan(result.redchi):
        print(f"   ✅ FIT SUCCEEDED!")
        print(f"   - Reduced χ²: {result.redchi:.4f}")
        print(f"   - Success: {result.success}")
        print(f"   - Number of parameters: {len(result.params)}")
        print(f"   - Number of data points: {len(result.best_fit)}")

        # Show parameters
        print(f"\n5. Fitted parameters:")
        for name, param in result.params.items():
            stderr = f"{param.stderr:.4e}" if param.stderr is not None else "N/A"
            print(f"   - {name:12s}: {param.value:12.4e} ± {stderr}")

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED - nbragg integration works!")
        print("="*70)
        print("\nYou can now:")
        print("  1. Run: streamlit run streamlit_app.py")
        print("  2. Enable 'Apply nbragg Analysis' in sidebar")
        print("  3. Click 'Run Pipeline'")
        print("  4. See green fit line in Reconstruction tab")
        print("  5. See fit results in Statistics tab")

    else:
        print("   ❌ FIT FAILED - invalid redchi")
        sys.exit(1)

except Exception as e:
    print(f"   ❌ FIT FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
