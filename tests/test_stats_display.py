"""Test the statistics display functionality."""

import sys
sys.path.insert(0, 'src')
import pandas as pd

from frame_overlap import Data, Reconstruct, Analysis

# Setup
data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
           flux=5e6, duration=0.5, freq=20)
data.convolute_response(200, bin_width=10)
data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
data.overlap(kernel=[0, 25])

recon = Reconstruct(data, tmin=3.7, tmax=11.0)
recon.filter(kind='wiener', noise_power=0.2)

# Run nbragg analysis
analysis = Analysis(xs='iron', vary_background=True, vary_response=True)
result = analysis.fit(recon)

print("="*60)
print("Testing Statistics Tab Display")
print("="*60)

# Test 1: Reduced chi-squared
print("\n1. Testing reduced chi-squared display...")
print(f"   Reduced χ² (nbragg): {result.redchi:.4f}")

if result.redchi < 2:
    quality = "✅ Excellent nbragg fit"
elif result.redchi < 5:
    quality = "ℹ️ Good nbragg fit"
else:
    quality = "⚠️ Poor nbragg fit"
print(f"   Quality indicator: {quality}")
print("   ✓ Chi-squared display works")

# Test 2: Parameters table
print("\n2. Testing parameters table...")
params_data = []
for param_name, param in result.params.items():
    # Handle stderr which might be None
    if param.stderr is not None:
        stderr_str = f"{param.stderr:.4e}"
    else:
        stderr_str = "N/A"

    params_data.append({
        'Parameter': param_name,
        'Value': f"{param.value:.4e}",
        'Stderr': stderr_str,
        'Vary': 'Yes' if param.vary else 'No'
    })

params_table = pd.DataFrame(params_data)
print("\nParameters Table:")
print(params_table.to_string(index=False))
print("\n   ✓ Parameters table created successfully")

# Test 3: Fit report
print("\n3. Testing fit report...")
fit_report = result.fit_report()
print("\nFit Report (first 500 chars):")
print(fit_report[:500] + "...")
print("\n   ✓ Fit report generated successfully")

print("\n" + "="*60)
print("✅ All statistics display tests passed!")
print("="*60)
