#!/usr/bin/env python3
"""
Example: How to fix parameters during nbragg fitting

This example shows how to set and fix parameters before fitting,
ensuring they don't vary during the fit.
"""

from frame_overlap import Data, Reconstruct, Analysis

# Load and process data
data = Data(
    signal_file='notebooks/iron_powder.csv',
    openbeam_file='notebooks/openbeam.csv',
    flux=5e6,
    duration=0.5,
    freq=20
)

# Apply processing pipeline
data.convolute_response(pulse_duration=20.0)
data.poisson_sample(flux=1e6, freq=20, measurement_time=8.0)
data.overlap(kernel=1, total_time=50, mode='equal')

# Reconstruct
recon = Reconstruct(data, tmin=None, tmax=None)
recon.filter(kind='wiener', noise_power=1.0)

print("="*70)
print("EXAMPLE: Fixing parameters during nbragg fitting")
print("="*70)

# Create Analysis object
analysis = Analysis(
    xs='iron',
    vary_background=True,
    vary_response=False,
    vary_weights=True,
)

# Method 1: Using set_params() - RECOMMENDED
print("\nMethod 1: Using set_params() method")
print("-" * 70)

analysis.set_params(
    thickness={'value': 1.95, 'vary': False},  # Fix thickness at 1.95 cm
    norm={'value': 1.0, 'vary': True},         # Allow norm to vary
    temp={'vary': False}                        # Fix temperature
)

print("Parameters before fitting:")
print(f"  thickness = {analysis.model.params['thickness'].value:.3f}, vary={analysis.model.params['thickness'].vary}")
print(f"  norm = {analysis.model.params['norm'].value:.3f}, vary={analysis.model.params['norm'].vary}")

# Fit - parameters will be respected!
result = analysis.fit(recon, wlmin=1.0, wlmax=4)

print("\nResults after fitting:")
print(f"  Reduced χ² = {result.redchi:.4f}")
print(f"  thickness = {result.params['thickness'].value:.6f} (fixed: {not result.params['thickness'].vary})")
print(f"  norm = {result.params['norm'].value:.6f} ± {result.params['norm'].stderr:.6f}")

# Method 2: Using model.params directly - ALSO WORKS
print("\n" + "="*70)
print("Method 2: Direct params manipulation")
print("-" * 70)

analysis2 = Analysis(xs='iron', vary_background=True, vary_weights=True)

# Set parameters directly
analysis2.model.params['thickness'].value = 1.95
analysis2.model.params['thickness'].vary = False
analysis2.model.params['norm'].vary = True
analysis2.model.params['temp'].vary = False

print("Parameters before fitting:")
print(f"  thickness = {analysis2.model.params['thickness'].value:.3f}, vary={analysis2.model.params['thickness'].vary}")

result2 = analysis2.fit(recon, wlmin=1.0, wlmax=4)

print("\nResults after fitting:")
print(f"  Reduced χ² = {result2.redchi:.4f}")
print(f"  thickness = {result2.params['thickness'].value:.6f} (fixed: {not result2.params['thickness'].vary})")

print("\n" + "="*70)
print("✅ Both methods successfully fix parameters during fitting!")
print("="*70)
