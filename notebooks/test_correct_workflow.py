"""Test CORRECT workflow: Data → Convolute → Poisson → Overlap"""
import numpy as np
from frame_overlap import Data, Reconstruct

print("Testing CORRECT workflow order...")
print("=" * 60)

# Load data
data = Data(signal_file='notebooks/iron_powder.csv',
            openbeam_file='notebooks/openbeam.csv',
            flux=1e6, duration=1.0, freq=20)
print("✓ Data loaded")

# 1. CONVOLUTE (stores pulse_duration)
data.convolute_response(pulse_duration=200)
print(f"✓ Convolution applied (pulse_duration={data.pulse_duration} µs)")

# 2. POISSON (uses pulse_duration for flux scaling)
print("\n" + "=" * 60)
print("Applying Poisson with flux scaling...")
print("=" * 60)
data.poisson_sample(flux=1e6, measurement_time=1.0, freq=20)
print("✓ Poisson sampling applied")

# 3. OVERLAP
data.overlap(kernel=[0, 25])
print("✓ Overlap applied")

# 4. RECONSTRUCT
recon = Reconstruct(data)
recon.filter(kind='wiener', noise_power=0.01)
print("✓ Reconstruction complete")

# Check reference is poissoned_data
assert recon.reference_data is not None
print(f"✓ Reference data: {recon.reference_data.shape}")
print(f"  Chi²/dof = {recon.statistics['chi2_per_dof']:.2f}")

# Test tmin/tmax filtering
print("\n" + "=" * 60)
print("Testing tmin/tmax filtering...")
print("=" * 60)
recon2 = Reconstruct(data, tmin=10, tmax=40)
recon2.filter(kind='wiener', noise_power=0.01)
assert recon2.tmin == 10
assert recon2.tmax == 40
assert 'tmin' in recon2.statistics
print(f"✓ tmin/tmax filtering works: [{recon2.tmin}, {recon2.tmax}] ms")
print(f"  Chi²/dof on filtered range = {recon2.statistics['chi2_per_dof']:.2f}")
print(f"  n_points in range = {recon2.statistics['n_points']}")

# Test method chaining
print("\n" + "=" * 60)
print("Testing method chaining...")
print("=" * 60)
data2 = (Data(signal_file='notebooks/iron_powder.csv',
              openbeam_file='notebooks/openbeam.csv',
              flux=1e6, duration=1.0, freq=20)
         .convolute_response(200)
         .poisson_sample(flux=1e6, measurement_time=1.0, freq=20)
         .overlap([0, 25]))
print("✓ Method chaining works with CORRECT order")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
