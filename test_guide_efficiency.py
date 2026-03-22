"""
Test neutron guide efficiency implementation
"""
import numpy as np
import matplotlib.pyplot as plt
from frame_overlap import Data

# Create test data with known TOF range
print("Creating test data...")
test_data = {
    'stack': np.arange(1, 1001),  # 1000 data points
    'counts': np.ones(1000) * 100,  # Uniform counts
    'err': np.ones(1000) * 10
}

import pandas as pd
pd.DataFrame(test_data).to_csv('/tmp/test_signal.csv', index=False)
pd.DataFrame(test_data).to_csv('/tmp/test_openbeam.csv', index=False)

# Load data
data = Data('/tmp/test_signal.csv', '/tmp/test_openbeam.csv',
            flux=1e6, duration=1.0, freq=20)

print(f"\nOriginal data:")
print(f"  TOF range: {data.data['time'].min():.1f} - {data.data['time'].max():.1f} µs")
print(f"  Mean counts: {data.data['counts'].mean():.1f}")

# Apply Poisson sampling (required before guide efficiency)
data.poisson_sample(duty_cycle=1.0, seed=42)

print(f"\nAfter Poisson:")
print(f"  Mean counts: {data.poissoned_data['counts'].mean():.1f}")

# Apply guide efficiency
data.apply_guide_efficiency(guide_length=5.0, m_value=3, flight_path=9.0)

print(f"\nAfter guide efficiency:")
print(f"  Mean counts: {data.poissoned_data['counts'].mean():.1f}")

# Check that longer wavelengths have higher counts (better transmission)
# TOF is proportional to wavelength, so later times = longer wavelengths
early_time_mask = data.poissoned_data['time'] < 5000  # First 5ms
late_time_mask = data.poissoned_data['time'] > 5000   # After 5ms

early_counts = data.poissoned_data[early_time_mask]['counts'].mean()
late_counts = data.poissoned_data[late_time_mask]['counts'].mean()

print(f"\nTransmission check (should be higher at longer wavelengths):")
print(f"  Early TOF (<5ms) mean counts: {early_counts:.1f}")
print(f"  Late TOF (>5ms) mean counts: {late_counts:.1f}")
print(f"  Ratio (late/early): {late_counts/early_counts:.3f}")

if late_counts > early_counts:
    print("  ✓ PASS: Longer wavelengths have higher transmission!")
else:
    print("  ✗ FAIL: Expected higher transmission at longer wavelengths")

# Plot the efficiency curve
h_over_m = 3.95603e-3
wavelength = h_over_m * data.poissoned_data['time'].values / 9.0
efficiency = data.poissoned_data['counts'].values / 100.0  # Normalize by original counts

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot efficiency vs wavelength
ax1.plot(wavelength, efficiency, 'b-', linewidth=2)
ax1.set_xlabel('Wavelength (Å)', fontsize=14)
ax1.set_ylabel('Guide Transmission Efficiency', fontsize=14)
ax1.set_title('Neutron Guide Efficiency vs Wavelength (L=5m, m=3)', fontsize=16)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.1])

# Plot counts vs TOF
ax2.plot(data.poissoned_data['time'], data.poissoned_data['counts'], 'r-', linewidth=2)
ax2.set_xlabel('Time-of-Flight (µs)', fontsize=14)
ax2.set_ylabel('Counts', fontsize=14)
ax2.set_title('Counts after Guide Efficiency Applied', fontsize=16)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/guide_efficiency_test.png', dpi=150)
print(f"\nPlot saved to: /tmp/guide_efficiency_test.png")

print("\n✓ Test complete!")
