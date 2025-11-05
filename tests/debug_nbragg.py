"""Debug script to understand nbragg data structure."""

import sys
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct, Analysis

# Quick setup
data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
           flux=5e6, duration=0.5, freq=20)
data.convolute_response(200, bin_width=10)
data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
data.overlap(kernel=[0, 25])

recon = Reconstruct(data, tmin=3.7, tmax=11.0)
recon.filter(kind='wiener', noise_power=0.2)

print("Reconstruction data columns:", recon.reconstructed_data.columns.tolist())
print("Reconstruction data shape:", recon.reconstructed_data.shape)
print("First few rows:")
print(recon.reconstructed_data.head())

analysis = Analysis(xs='iron', vary_background=True, vary_response=True)
result = analysis.fit(recon)

print("\n" + "="*60)
print("nbragg data structure:")
print("="*60)
print("analysis.data type:", type(analysis.data))
print("analysis.data.table columns:", analysis.data.table.columns.tolist())
print("analysis.data.table shape:", analysis.data.table.shape)
print("\nFirst few rows of analysis.data.table:")
print(analysis.data.table.head())

print("\n" + "="*60)
print("Result structure:")
print("="*60)
print("result.best_fit shape:", result.best_fit.shape)
print("result.best_fit[:5]:", result.best_fit[:5])

print("\nresult.params:")
for name, param in result.params.items():
    print(f"  {name}: value={param.value:.6e}, stderr={param.stderr}")
